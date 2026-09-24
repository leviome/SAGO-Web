"""Compare the original SAGO SAM2 checkpoint on the exact browser diagnostic image.

Development-only benchmark. This does not add a Python backend to the web tool.
Run after debug-segmentation.mjs with the existing SAGO Python environment.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from sago_core.CrossViewTracker.lib.make_sam import make_sam_from_state_dict


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def overlay(image, mask, title, points=None, box=None):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    colored = rgb.copy()
    colored[mask] = (rgb[mask] * .55 + np.array([40, 245, 145]) * .45).astype(np.uint8)
    im = Image.fromarray(colored).resize((360, 360))
    d = ImageDraw.Draw(im)
    for p in points or []:
        x, y = p['x'] * 360, p['y'] * 360
        d.ellipse((x-3, y-3, x+3, y+3), fill='lime' if p['label'] else 'red', outline='white')
    if box:
        d.rectangle((box['x0']*360, box['y0']*360, box['x1']*360, box['y1']*360), outline='yellow', width=2)
    tile = Image.new('RGB', (360, 388), '#111b23')
    tile.paste(im, (0, 28))
    ImageDraw.Draw(tile).text((8, 8), title, fill='white')
    return tile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default=str(ROOT / 'artifacts/sam2-large-debug'))
    parser.add_argument('--checkpoint', default=str(ROOT.parent / 'sago_core/sam2/checkpoints/sam2_hiera_large.pt'))
    args = parser.parse_args()
    folder = Path(args.directory)
    report = json.loads((folder / 'web-report.json').read_text())
    image = cv2.imread(str(folder / 'input.png'))
    h, w = image.shape[:2]
    config, model = make_sam_from_state_dict(args.checkpoint)
    model = model.eval().to('cuda', dtype=torch.float32)
    parameter_count = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        pixels = model.image_encoder.prepare_image(image, max_side_length=1024, use_square_sizing=True)
        web_pixels = np.fromfile(folder / 'pixel-values.f32', dtype=np.float32).reshape(1, 3, 1024, 1024)
        pixel_error = np.abs(pixels.cpu().numpy() - web_pixels)
        features, token_hw, _ = model.encode_image(image, max_side_length=1024, use_square_sizing=True)
        position = model.coordinate_encoder.get_grid_position_encoding(token_hw)
        rows = []
        results = []
        for case in report['cases']:
            points = case['points']
            box = case.get('box')
            prompts = model.encode_prompts(
                [] if not box else [[(box['x0'], box['y0']), (box['x1'], box['y1'])]],
                [(p['x'], p['y']) for p in points if p['label']],
                [(p['x'], p['y']) for p in points if not p['label']])
            predictions, scores, _, _ = model.mask_decoder(features, prompts, position)
            masks = (torch.nn.functional.interpolate(predictions, size=(h, w), mode='bilinear', align_corners=False) > 0)[0].cpu().numpy()
            web_masks = [np.fromfile(folder / f"{case['name']}-web-{i}.bin", dtype=np.uint8).reshape(h, w) > 0 for i in range(len(case['scores']))]
            for i, mask in enumerate(masks):
                cv2.imwrite(str(folder / f"{case['name']}-sam2-{i}.png"), mask.astype(np.uint8)*255)
            overlaps = [[iou(a, b) for b in masks] for a in web_masks]
            result = {'name': case['name'], 'sam2Scores': scores[0].tolist(),
                      'webSelected': case['selected'], 'pairwiseIoU_webRows_sam2Columns': overlaps,
                      'webSelected_vs_sagoLevel0': overlaps[case['selected']][0],
                      'sam2Area': [int(m.sum()) for m in masks], 'webArea': [int(m.sum()) for m in web_masks]}
            if (folder / f"{case['name']}-web-logits.f32").exists():
                web_logits = np.fromfile(folder / f"{case['name']}-web-logits.f32", dtype=np.float32).reshape(1, 4, 256, 256)
                error = np.abs(predictions.cpu().numpy() - web_logits)
                result['webVsPytorchLogitMAE'] = float(error.mean())
                result['webVsPytorchLogitMaxError'] = float(error.max())
                result['matchingLevelIoU'] = [overlaps[i][i] for i in range(4)]
            if box and report.get('model') != 'sago/sam2-hiera-large':
                # Isolate crop+center semantics from model capacity, using SAM2 on both paths.
                x0, y0 = int(np.floor(box['x0']*w)), int(np.floor(box['y0']*h))
                x1, y1 = int(np.ceil(box['x1']*w)), int(np.ceil(box['y1']*h))
                crop_features, crop_hw, _ = model.encode_image(image[y0:y1, x0:x1], max_side_length=1024, use_square_sizing=True)
                crop_prompt = model.encode_prompts([], [(0.5, 0.5)], [])
                cp, _, _, _ = model.mask_decoder(crop_features, crop_prompt, model.coordinate_encoder.get_grid_position_encoding(crop_hw))
                cm = (torch.nn.functional.interpolate(cp, size=(y1-y0, x1-x0), mode='bilinear', align_corners=False) > 0)[0, 0].cpu().numpy()
                crop_mask = np.zeros((h, w), dtype=bool)
                crop_mask[y0:y1, x0:x1] = cm
                result['sam2NativeBox_vs_sam2CropCenterLevel0_IoU'] = iou(masks[0], crop_mask)
                cv2.imwrite(str(folder / f"{case['name']}-sam2-crop.png"), crop_mask.astype(np.uint8)*255)
                # Restore full-image positional encoding after crop encoding.
                position = model.coordinate_encoder.get_grid_position_encoding(token_hw)
            rows.append([overlay(image, web_masks[case['selected']], f"{case['name']} | Web level {case['selected']}", points, box)] +
                        [overlay(image, mask, f'SAM2 Large | level {i}', points, box) for i, mask in enumerate(masks)])
            results.append(result)
            print(case['name'], 'Web auto vs SAM2 level0 IoU:', round(result['webSelected_vs_sagoLevel0'], 4), flush=True)
    torch.cuda.synchronize()
    summary = {'checkpoint': Path(args.checkpoint).name, 'sam2Parameters': parameter_count,
               'elapsedSeconds': time.perf_counter()-started, 'peakCudaMiB': torch.cuda.max_memory_allocated()/1024**2,
               'pixelMaxError': float(pixel_error.max()), 'pixelMAE': float(pixel_error.mean()),
               'note': 'IoU measures agreement, not accuracy: no ground truth annotations.', 'cases': results}
    (folder / 'sam2-report.json').write_text(json.dumps(summary, indent=2))
    sheet = Image.new('RGB', (360 * 5, 388 * len(rows)), '#111b23')
    for y, row in enumerate(rows):
        for x, tile in enumerate(row): sheet.paste(tile, (x*360, y*388))
    sheet.save(folder / 'comparison.png')
    print(json.dumps({k:v for k,v in summary.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
