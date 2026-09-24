export type PromptPoint = { x: number; y: number; label: 0 | 1 };
export type PromptBox = { x0: number; y0: number; x1: number; y1: number };

// Coordinates are normalized to the captured image, never to the modal or window.
export const imagePoint = (x: number, y: number, rect: { left: number; top: number; width: number; height: number }) => ({
    x: Math.max(0, Math.min(1, (x - rect.left) / rect.width)),
    y: Math.max(0, Math.min(1, (y - rect.top) / rect.height))
});

export const promptArrays = (points: PromptPoint[], width: number, height: number) => {
    const coords = points.flatMap(p => [p.x * width, p.y * height]);
    const labels: number[] = points.map(p => p.label);
    // The published SlimSAM decoder appends its own not-a-point padding.
    return { coords, labels };
};

// A separable max filter preserves small details while expanding the mask in pixels.
export const dilateMask = (mask: Uint8Array, width: number, height: number, radius: number) => {
    if (mask.length !== width * height) throw new Error('Mask dimensions do not match');
    const r = Math.max(0, Math.min(12, Math.floor(radius)));
    if (!r) return mask.slice();
    const horizontal = new Uint8Array(mask.length);
    const result = new Uint8Array(mask.length);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            for (let dx = Math.max(0, x - r); dx <= Math.min(width - 1, x + r); dx++) {
                if (mask[y * width + dx]) {
                    horizontal[y * width + x] = 1; break;
                }
            }
        }
    }
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            for (let dy = Math.max(0, y - r); dy <= Math.min(height - 1, y + r); dy++) {
                if (horizontal[dy * width + x]) {
                    result[y * width + x] = 1; break;
                }
            }
        }
    }
    return result;
};

export const maskRgba = (mask: Uint8Array, preview = false) => {
    const data = new Uint8ClampedArray(mask.length * 4);
    for (let i = 0; i < mask.length; i++) {
        if (mask[i]) data.set(preview ? [63, 232, 175, 110] : [255, 255, 255, 255], i * 4);
    }
    return data;
};
