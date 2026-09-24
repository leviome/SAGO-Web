"""Write original PyTorch memory-path fixtures for browser/CPU ONNX comparison."""
import json
from pathlib import Path
import runpy
import torch
import cv2

base=runpy.run_path(str(Path(__file__).with_name('export-sam2.py')))
root=base['ROOT']; out=root/'artifacts/memory-validation'; out.mkdir(parents=True,exist_ok=True)
_,model=base['make_sam_from_state_dict'](str(root.parent/'sago_core/sam2/checkpoints/sam2_hiera_large.pt'))
model.eval().to('cuda'); torch.set_num_threads(8)
files={}
def save(name,tensor):
    value=tensor.detach().cpu().contiguous().numpy();value.tofile(out/f'{name}.bin')
    files[name]={'dims':list(value.shape),'dtype':str(value.dtype)}
with torch.inference_mode():
    image=cv2.imread(str(root.parent/'debug/first_view.png'))
    features,hw,_=model.encode_image(image)
    for name,t in zip(['features','hires2','hires4'],features):save(name,t)
    prompts=model.encode_prompts([],[(.292,.424)],[])
    masks,scores,pointers,object_score=model.mask_decoder(features,prompts,model.coordinate_encoder.get_grid_position_encoding(hw))
    mask=masks[:,0:1]; ptr=pointers[:,0:1]
    save('mask',mask);save('pointer',ptr);save('object-score',object_score)
    seed=model.memory_encoder(features[0],mask,object_score,is_prompt_encoding=True)
    save('memory-prompt',seed)
    save('memory-tracked',model.memory_encoder(features[0],mask,object_score,is_prompt_encoding=False))
    for count,pcount in [(1,1),(3,3),(7,16)]:
        memories=[seed]+[seed*(1+i*.01) for i in range(count-1)]
        ptrs=[ptr]+[ptr*(1+i*.01) for i in range(pcount-1)]
        save(f'memories-{count}',torch.cat(memories,dim=0))
        save(f'indices-{count}',torch.tensor([6]+list(range(count-1)),dtype=torch.int64))
        save(f'pointers-{count}',torch.cat(ptrs,dim=1))
        fused=model.memory_fusion(features[0],memories[:1],ptrs[:1],memories[1:],ptrs[1:])
        save(f'fused-{count}',fused)
        prediction=model.mask_decoder([fused,*features[1:]],model.prompt_encoder.create_video_no_prompt_encoding(),model.coordinate_encoder.get_grid_position_encoding(hw),blank_promptless_output=False)
        for name,t in zip(['masks','scores','pointers','object-score'],prediction):save(f'track-{count}-{name}',t)
        print('Original memory history',count,'pointers',pcount,flush=True)
(out/'fixtures.json').write_text(json.dumps(files,indent=2))
