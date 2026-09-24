import { test } from 'node:test';
import assert from 'node:assert/strict';
import { initialCoverage, meanCenter, refineCandidates, touchesBorder } from '../src/sago/drone-utils.ts';

test('drone silhouette validation uses initial area and the original 0.3 alpha threshold', () => {
    const initial = Uint8Array.of(1, 1, 1, 1, 1, 0);
    const pixels = new Uint8Array(24);
    [255, 255, 255, 77, 76, 255].forEach((alpha, i) => pixels[i * 4 + 3] = alpha);
    assert.equal(initialCoverage(initial, pixels), 0.8);
    assert.equal(initialCoverage(new Uint8Array(6), pixels), 0);
});

test('multi-view filtering only shrinks candidates; a border mask preserves offscreen candidates', () => {
    const previous = Uint8Array.of(255, 255, 255, 0);
    const hits = Uint8Array.of(255, 0, 0, 255);
    assert.deepEqual([...refineCandidates(previous, hits)], [255, 0, 0, 0]);
    assert.deepEqual([...refineCandidates(previous, hits, Uint8Array.of(255, 255, 0, 0))], [255, 0, 255, 0]);
    assert.equal(touchesBorder(Uint8Array.of(0, 0, 0, 0, 1, 0, 0, 0, 0), 3, 3), false);
    assert.equal(touchesBorder(Uint8Array.of(0, 1, 0, 0, 1, 0, 0, 0, 0), 3, 3), true);
});

test('drone recentering uses only surviving world-space centers', () => {
    assert.deepEqual(meanCenter(Float32Array.of(1, 2, 3, 100, 200, 300, 3, 4, 5), Uint8Array.of(255, 0, 255)), [2, 3, 4]);
    assert.throws(() => meanCenter(new Float32Array(3), Uint8Array.of(0)));
});
