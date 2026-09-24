export const droneYaws = [45, 90, 135];

export const touchesBorder = (mask: Uint8Array, width: number, height: number) => {
    for (let x = 0; x < width; x++) if (mask[x] || mask[(height - 1) * width + x]) return true;
    for (let y = 0; y < height; y++) if (mask[y * width] || mask[(y + 1) * width - 1]) return true;
    return false;
};

// Original SAGO's ioi=True: intersection / initial-mask area, not union IoU.
export const initialCoverage = (initial: Uint8Array, rgba: Uint8Array) => {
    let area = 0, intersection = 0;
    for (let i = 0; i < initial.length; i++) {
        if (initial[i]) {
            area++;
            if (rgba[i * 4 + 3] > 0.3 * 255) intersection++;
        }
    }
    return area ? intersection / area : 0;
};

export const refineCandidates = (previous: Uint8Array, hits: Uint8Array, onScreen?: Uint8Array) => previous.map((value, i) => (value && (hits[i] || (onScreen && !onScreen[i])) ? 255 : 0));

export const meanCenter = (positions: Float32Array, mask: Uint8Array) => {
    const center = [0, 0, 0];
    let count = 0;
    for (let i = 0; i < mask.length; i++) {
        if (!mask[i]) continue;
        count++;
        for (let axis = 0; axis < 3; axis++) center[axis] += positions[i * 3 + axis];
    }
    if (!count) throw new Error('三维候选为空，请重新选择目标。');
    return center.map(value => value / count);
};
