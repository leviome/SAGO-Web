import assert from 'node:assert/strict';
import { test } from 'node:test';

import { dilateMask, imagePoint, maskRgba, promptArrays } from '../src/sago/mask-utils.ts';

test('pointer coordinates use image bounds and clamp captured drags', () => {
    const rect = { left: 200, top: 100, width: 600, height: 300 };
    assert.deepEqual(imagePoint(500, 250, rect), { x: 0.5, y: 0.5 });
    assert.deepEqual(imagePoint(-20, 500, rect), { x: 0, y: 1 });
});

test('positive and negative prompts scale independently for non-square images', () => {
    assert.deepEqual(promptArrays([{ x: 0.25, y: 0.5, label: 1 }, { x: 1, y: 0, label: 0 }], 1024, 512), {
        coords: [256, 256, 1024, 0], labels: [1, 0]
    });
});

test('dilation clips at image edges without wrapping across rows', () => {
    const mask = new Uint8Array(15);
    mask[4] = 1;
    const result = dilateMask(mask, 5, 3, 1);
    assert.deepEqual([...result], [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]);
    assert.equal(mask.reduce((a, b) => a + b), 1);
    assert.throws(() => dilateMask(mask, 5, 4, 2));
});

test('selection masks use exact alpha 255; visual overlay alpha is separate', () => {
    assert.deepEqual([...maskRgba(new Uint8Array([0, 1]))], [0, 0, 0, 0, 255, 255, 255, 255]);
    assert.equal(maskRgba(new Uint8Array([1]), true)[3], 110);
});
