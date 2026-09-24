import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';
const source = await readFile(new URL('../src/sago/sam2-utils.ts', import.meta.url), 'utf8');
const js = ts.transpile(source, { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 });
const { sam2Prompts, sam2Masks, sam2Pixels } = await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`);

test('native box corners, fg/bg ordering and padding match SAGO', () => {
    const { coords, labels } = sam2Prompts([{ x: .75, y: .25, label: 0 }, { x: .5, y: .5, label: 1 }], { x0: 1, y0: 1, x1: 0, y1: 0 });
    assert.deepEqual(Array.from(labels, Number), [2, 3, 1, 0, 4]);
    assert.deepEqual(Array.from(coords), [0, 0, 1, 1, .5, .5, .75, .25, 0, 0]);
    assert.throws(() => sam2Prompts([], null));
});
test('mask interpolation thresholds logits, not a resized binary mask', () => {
    const [mask] = sam2Masks(Float32Array.of(-10, 1, -10, 1), 1, 4, 2, 2, 2);
    assert.deepEqual(Array.from(mask), [0, 0, 0, 1, 0, 0, 0, 1]);
});
test('non-square RGB input retains channels and antialiases when shrinking', () => {
    const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255, 255, 0, 0, 255, 0, 255, 0, 255]);
    const data = sam2Pixels(rgba, 4, 1, 1);
    assert.ok(Math.abs(data[0] - (.5 - .485) / .229) < 1e-6);
    assert.ok(Math.abs(data[1] - (.5 - .456) / .224) < 1e-6);
    assert.ok(Math.abs(data[2] - (0 - .406) / .225) < 1e-6);
});
