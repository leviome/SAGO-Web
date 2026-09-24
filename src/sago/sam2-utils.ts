import { PromptBox, PromptPoint } from './mask-utils';

// Match torch interpolate(bilinear, align_corners=False, antialias=True).
const weights = (input: number, output: number) => {
    const scale = input / output;
    const support = Math.max(1, scale);
    return Array.from({ length: output }, (_, i) => {
        const center = (i + 0.5) * scale;
        const start = Math.max(0, Math.floor(center - support + 0.5));
        const end = Math.min(input, Math.floor(center + support + 0.5));
        const values = Array.from({ length: end - start }, (_, j) => Math.max(0, 1 - Math.abs((start + j + 0.5 - center) / support)));
        const sum = values.reduce((a, b) => a + b, 0);
        return { start, values: values.map(v => v / sum) };
    });
};

export const sam2Pixels = (rgba: Uint8ClampedArray, width: number, height: number, size = 1024) => {
    const wx = weights(width, size);
    const wy = weights(height, size);
    const horizontal = new Float32Array(height * size * 3);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < size; x++) {
            const { start, values } = wx[x];
            for (let c = 0; c < 3; c++) {
                let value = 0;
                for (let j = 0; j < values.length; j++) value += rgba[(y * width + start + j) * 4 + c] * values[j];
                horizontal[(y * size + x) * 3 + c] = value;
            }
        }
    }
    const result = new Float32Array(3 * size * size);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    for (let y = 0; y < size; y++) {
        const { start, values } = wy[y];
        for (let x = 0; x < size; x++) {
            for (let c = 0; c < 3; c++) {
                let value = 0;
                for (let j = 0; j < values.length; j++) value += horizontal[((start + j) * size + x) * 3 + c] * values[j];
                result[c * size * size + y * size + x] = (Math.fround(value) - 255 * mean[c]) / (255 * std[c]);
            }
        }
    }
    return result;
};

export const sam2Prompts = (points: PromptPoint[], box: PromptBox | null) => {
    const coords: number[] = [];
    const labels: number[] = [];
    if (box) {
        coords.push(Math.min(box.x0, box.x1), Math.min(box.y0, box.y1), Math.max(box.x0, box.x1), Math.max(box.y0, box.y1));
        labels.push(2, 3);
    }
    // SAGO concatenates boxes, foreground, background, then one padding token.
    for (const label of [1, 0]) {
        for (const p of points.filter(p => p.label === label)) {
            coords.push(p.x, p.y); labels.push(label);
        }
    }
    if (!labels.length) throw new Error('请添加点或框提示。');
    coords.push(0, 0); labels.push(4);
    return { coords: new Float32Array(coords), labels: BigInt64Array.from(labels, BigInt) };
};

// Resize logits before thresholding, matching the desktop decoder postprocessing.
export const sam2Masks = (data: Float32Array, count: number, width: number, height: number, lowWidth = 256, lowHeight = 256) => {
    return Array.from({ length: count }, (_, level) => {
        const mask = new Uint8Array(width * height);
        const offset = level * lowWidth * lowHeight;
        for (let y = 0; y < height; y++) {
            const sy = Math.max(0, (y + 0.5) * lowHeight / height - 0.5);
            const y0 = Math.floor(sy), y1 = Math.min(y0 + 1, lowHeight - 1), fy = sy - y0;
            for (let x = 0; x < width; x++) {
                const sx = Math.max(0, (x + 0.5) * lowWidth / width - 0.5);
                const x0 = Math.floor(sx), x1 = Math.min(x0 + 1, lowWidth - 1), fx = sx - x0;
                const a = data[offset + y0 * lowWidth + x0] * (1 - fx) + data[offset + y0 * lowWidth + x1] * fx;
                const b = data[offset + y1 * lowWidth + x0] * (1 - fx) + data[offset + y1 * lowWidth + x1] * fx;
                mask[y * width + x] = a * (1 - fy) + b * fy > 0 ? 1 : 0;
            }
        }
        return mask;
    });
};
