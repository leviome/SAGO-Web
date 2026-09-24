"""Offline export of SAGO's exact SAM2 Large image/prompt path (no web backend)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import onnx
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get('SAGO_ROOT', str(ROOT.parent)))
from sago_core.CrossViewTracker.lib.make_sam import make_sam_from_state_dict


class Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.coords = model.coordinate_encoder
        self.decoder = model.mask_decoder
        p = model.prompt_encoder
        self.register_buffer('labels', torch.cat([p.point_encoder.bg_embed, p.point_encoder.fg_embed,
            p.box_encoder.tl_embed, p.box_encoder.br_embed, p.point_encoder.not_a_point_embed]))
        self.register_buffer('position', self.coords.get_grid_position_encoding((64, 64)).clone())

    def forward(self, features, hires2, hires4, point_coords, point_labels):
        positional = self.coords(point_coords)
        prompts = torch.where((point_labels == 4).unsqueeze(-1), torch.zeros_like(positional), positional)
        prompts = prompts + self.labels[point_labels]
        masks, scores, _, _ = self.decoder([features, hires2, hires4], prompts, self.position,
                                           blank_promptless_output=False)
        return masks, scores


def save_chunked(model, target):
    """Small static files work with HTTPS tunnels and avoid giant fetch buffers."""
    graph = onnx.load(str(target))
    # WebGPU guarantees only eight storage bindings per shader. A flat concat
    # of split attention heads can exceed this (one binding per input + output).
    nodes = []
    for node in graph.graph.node:
        if node.op_type == 'Concat' and len(node.input) > 4:
            axis = next(a.i for a in node.attribute if a.name == 'axis')
            inputs = list(node.input)
            stage = 0
            while len(inputs) > 4:
                outputs = []
                for i in range(0, len(inputs), 4):
                    name = f'{node.output[0]}_bounded_{stage}_{i}'
                    nodes.append(onnx.helper.make_node('Concat', inputs[i:i+4], [name], axis=axis))
                    outputs.append(name)
                inputs = outputs
                stage += 1
            del node.input[:]
            node.input.extend(inputs)
        nodes.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    chunks = []
    data = bytearray()
    name = f'{target.stem}.data.{len(chunks)}'
    for tensor in graph.graph.initializer:
        if not tensor.HasField('raw_data') or len(tensor.raw_data) < 1024:
            continue
        if data and len(data) + len(tensor.raw_data) > 32 * 1024**2:
            (target.parent / name).write_bytes(data)
            chunks.append(name)
            data = bytearray()
            name = f'{target.stem}.data.{len(chunks)}'
        raw = tensor.raw_data
        onnx.external_data_helper.set_external_data(tensor, location=name, offset=len(data), length=len(raw))
        tensor.ClearField('raw_data')
        data.extend(raw)
    if data:
        (target.parent / name).write_bytes(data)
        chunks.append(name)
    onnx.save(graph, str(target))
    onnx.checker.check_model(str(target))
    return {'file': target.name, 'externalData': chunks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=str(ROOT.parent / 'sago_core/sam2/checkpoints/sam2_hiera_large.pt'))
    parser.add_argument('--output', default=str(ROOT / 'static/models/sam2-large'))
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(0)
    config, model = make_sam_from_state_dict(args.checkpoint)
    if sum(p.numel() for p in model.parameters()) != 224430113:
        raise ValueError("Expected SAGO SAM2 Hiera Large (224,430,113 parameters).")
    model.eval()
    # Bound attention intermediates to one head: WebGPU storage buffers have finite limits.
    original_attention = nn.functional.scaled_dot_product_attention
    def attention(q, k, v, *a, **kw):
        if q.shape[-2] >= 1024 and k.shape[-2] >= 1024:
            return torch.cat([original_attention(q[:, i:i+1], k[:, i:i+1], v[:, i:i+1], *a, **kw)
                              for i in range(q.shape[1])], dim=1)
        return original_attention(q, k, v, *a, **kw)
    nn.functional.scaled_dot_product_attention = attention
    with torch.inference_mode():
        pixels = torch.zeros(1, 3, 1024, 1024)
        model.image_encoder.posenc._scale_to_patch_grid((256, 256))
        print('Exporting original FP32 Hiera Large encoder...', flush=True)
        torch.onnx.export(model.image_encoder, pixels, str(out / 'encoder.onnx'), opset_version=17,
                          input_names=['pixel_values'], output_names=['features', 'hires2', 'hires4'])
        encoder = save_chunked(model.image_encoder, out / 'encoder.onnx')
        features = (torch.zeros(1, 256, 64, 64), torch.zeros(1, 64, 128, 128), torch.zeros(1, 32, 256, 256))
        decoder = Decoder(model).eval()
        prompts = (torch.tensor([[[.3, .5], [0., 0.]]]), torch.tensor([[1, 4]]))
        print('Exporting native prompts and all four mask levels...', flush=True)
        torch.onnx.export(decoder, (*features, *prompts), str(out / 'decoder.onnx'), opset_version=17,
            input_names=['features', 'hires2', 'hires4', 'point_coords', 'point_labels'],
            output_names=['masks', 'scores'], dynamic_axes={'point_coords': {1: 'num_points'}, 'point_labels': {1: 'num_points'}})
        decoder_info = save_chunked(decoder, out / 'decoder.onnx')
    manifest = {'id': 'sago/sam2-hiera-large', 'checkpoint': Path(args.checkpoint).name,
        'checkpointSha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        'dtype': 'fp32', 'onnxRuntimeWeb': '1.23.2', 'inputSize': 1024, 'defaultLevel': 0, 'levels': 4,
        'preprocessing': 'RGB bilinear antialias square 1024; ImageNet mean/std',
        'coordinateSpace': 'normalized-original-image', 'paddingLabel': 4,
        'encoder': encoder, 'decoder': decoder_info, 'license': 'Apache-2.0'}
    files = [out / item for m in [encoder, decoder_info] for item in [m['file'], *m['externalData']]]
    manifest['files'] = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
    (out / 'LICENSE').write_bytes((ROOT.parent / 'sago_core/sam2/LICENSE').read_bytes())
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Export complete: {sum(p.stat().st_size for p in files) / 1024**2:.1f} MiB', flush=True)


if __name__ == '__main__':
    main()
