"""Export original SAGO SAM2 memory path. Offline tool, never a web backend."""
import hashlib
import json
from pathlib import Path
import runpy
import types

import torch
from torch import nn

base = runpy.run_path(str(Path(__file__).with_name('export-sam2.py')))
ROOT = base['ROOT']


def pointer_forward(self, object_token, mask_tokens):
    score = self.score_mlp(object_token)
    pointers = torch.where((score > 0).unsqueeze(-1), self.pointer_mlp(mask_tokens), self.no_ptr)
    return score, pointers


class MemoryEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.memory_encoder

    def forward(self, features, mask, is_prompt):
        e = self.encoder
        d = e.mask_downsampler
        hires = nn.functional.interpolate(mask, (1024, 1024), mode='bilinear', align_corners=False)
        values = torch.where(is_prompt > 0, (hires > 0).float(), torch.sigmoid(hires))
        encoded_mask = d.out_proj(d.downsample(values * d.mask_scale + d.mask_bias))
        return e.out_proj(e.channel_mixer(e.image_proj(features) + encoded_mask))


class MemoryFusion(nn.Module):
    def __init__(self, model):
        super().__init__()
        f = model.memory_fusion
        self.layers = f.layers
        self.out_norm = f.out_norm
        self.register_buffer('image_position', f.imgposenc(torch.zeros(1, 256, 64, 64)))
        self.register_buffer('memory_positions', torch.cat([f.memconcat.memposenc((1,64,64,64), i) for i in range(7)]))
        # Real arithmetic equivalent to complex RoPE; constants use the original encoder.
        for layer in self.layers:
            for attention in [layer.image_selfattn.attn, layer.image_crossattn.attn]:
                vectors = attention.rotposenc.get_rotation_vectors((64, 64))
                attention.register_buffer('rope_cos', vectors.real.clone())
                attention.register_buffer('rope_sin', vectors.imag.clone())

    @staticmethod
    def rotate(x, a):
        paired = x.reshape(1, a.num_heads, -1, 4096, a.features_per_head // 2, 2)
        real, imag = paired[..., 0], paired[..., 1]
        c, s = a.rope_cos.unsqueeze(2), a.rope_sin.unsqueeze(2)
        return torch.stack([real*c-imag*s, real*s+imag*c], dim=-1).reshape(1, a.num_heads, -1, a.features_per_head)

    def attention(self, a, q, k, v, spatial_count, cross=False):
        q = a.q_proj(q).reshape(1, -1, a.num_heads, a.features_per_head).transpose(1, 2)
        k = a.k_proj(k).reshape(1, -1, a.num_heads, a.features_per_head).transpose(1, 2)
        v = a.v_proj(v).reshape(1, -1, a.num_heads, a.features_per_head).transpose(1, 2)
        q = self.rotate(q, a)
        if cross:
            k = torch.cat([self.rotate(k[:, :, :spatial_count], a), k[:, :, spatial_count:]], dim=2)
        else:
            k = self.rotate(k, a)
        # Keep each attention matrix below WebGPU's minimum storage-buffer limit.
        chunks = [nn.functional.scaled_dot_product_attention(q[:, :, start:start+256], k, v) for start in range(0, 4096, 256)]
        result = torch.cat(chunks, dim=2).transpose(1, 2).reshape(1, 4096, 256)
        return a.out_proj(result)

    def forward(self, features, memories, memory_indices, pointers):
        spatial = memories.flatten(2).transpose(1, 2).reshape(1, -1, 64)
        positions = self.memory_positions[memory_indices].flatten(2).transpose(1, 2).reshape(1, -1, 64)
        ptr = pointers.reshape(1, -1, 64)
        tokens = torch.cat([spatial, ptr], dim=1)
        positions = torch.cat([positions, torch.zeros_like(ptr)], dim=1)  # SAM2.0 pointer positions are zero.
        image = (features + self.image_position).flatten(2).transpose(1, 2)
        for layer in self.layers:
            norm = layer.image_selfattn.norm(image)
            image = image + self.attention(layer.image_selfattn.attn, norm, norm, norm, 4096)
            norm = layer.image_crossattn.norm(image)
            image = image + self.attention(layer.image_crossattn.attn, norm, tokens+positions, tokens, spatial.shape[1], True)
            image = layer.image_mlp(image)
        return self.out_norm(image).transpose(1, 2).reshape(1, 256, 64, 64)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=str(ROOT.parent/'sago_core/sam2/checkpoints/sam2_hiera_large.pt'))
    parser.add_argument('--output', default=str(ROOT/'static/models/sam2-large'))
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(0)
    out = Path(args.output)
    manifest = json.loads((out/'manifest.json').read_text())
    digest = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    if digest != manifest['checkpointSha256']:
        raise ValueError('Memory and image model checkpoints must be identical')
    _, model = base['make_sam_from_state_dict'](args.checkpoint)
    model.eval()
    model.mask_decoder.objptrgen.forward = types.MethodType(pointer_forward, model.mask_decoder.objptrgen)
    # Upgrade the decoder to retain pointer/object-score outputs, including the dynamic absent-object branch.
    decoder = base['Decoder'](model)
    def decode(self, features, hires2, hires4, point_coords, point_labels):
        positional = self.coords(point_coords)
        prompts = torch.where((point_labels == 4).unsqueeze(-1), torch.zeros_like(positional), positional) + self.labels[point_labels]
        return self.decoder([features,hires2,hires4], prompts, self.position, blank_promptless_output=False)
    decoder.forward = types.MethodType(decode, decoder)
    features = (torch.zeros(1,256,64,64),torch.zeros(1,64,128,128),torch.zeros(1,32,256,256))
    exports = []
    with torch.inference_mode():
        exports.append(('decoder', decoder, (*features,torch.tensor([[[.3,.5],[0.,0.]]]),torch.tensor([[1,4]])),
            ['features','hires2','hires4','point_coords','point_labels'], ['masks','scores','pointers','object_score'],
            {'point_coords':{1:'num_points'},'point_labels':{1:'num_points'}}))
        exports.append(('memoryEncoder',MemoryEncoder(model), (features[0],torch.zeros(1,1,256,256),torch.ones(1)),
            ['features','mask','is_prompt'], ['memory'], {}))
        exports.append(('memoryFusion',MemoryFusion(model), (features[0],torch.zeros(2,64,64,64),torch.tensor([6,0]),torch.zeros(1,2,256)),
            ['features','memories','memory_indices','pointers'], ['features'],
            {'memories':{0:'memory_count'},'memory_indices':{0:'memory_count'},'pointers':{1:'pointer_count'}}))
        for name, module, inputs, names, outputs, dynamic in exports:
            print('Exporting',name,flush=True)
            # Input/output may not share a name (fusion features must remain an input).
            output_names = ['fused_features'] if name=='memoryFusion' else outputs
            path=out/f'{name}.onnx'
            torch.onnx.export(module.eval(), inputs, str(path), opset_version=17,input_names=names,output_names=output_names,dynamic_axes=dynamic)
            manifest[name]=base['save_chunked'](module,path)
    manifest['memory']={'version':1,'promptFrames':1,'recentMemories':6,'recentPointers':15,'promptPositionIndex':6,'defaultTrackingLevels':[1,2,3]}
    parts=['encoder','decoder','memoryEncoder','memoryFusion']
    files=[out/f for part in parts for f in [manifest[part]['file'],*manifest[part]['externalData']]]
    manifest['files']={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Memory export complete',flush=True)


if __name__=='__main__': main()
