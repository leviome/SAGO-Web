import { loadDemo } from './helpers.mjs';
import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';

const openSago = async (page) => {
    await page.locator('#sago-open').click();
    await expect.poll(() => page.evaluate(() =>
        document.querySelector('.sago-orientation-dialog').open || document.querySelector('.sago-dialog').open
    )).toBe(true);
    if (await page.locator('.sago-orientation-dialog').isVisible()) {
        await page.locator('#sago-orientation-confirm').check();
        await page.locator('#sago-orientation-enter').click();
    }
};

test('local WebGPU segmentation, 3D selection, undo, and stale-view protection', async ({ page }) => {
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('console', m => { if (m.type() === 'error') console.log('browser:', m.text()); });
    // Every required model/runtime asset must be available on the static site.
    await page.route('**/*', route => {
        const url = route.request().url();
        return url.startsWith('http://127.0.0.1:3000/') || url.startsWith('blob:') || url.startsWith('data:') ? route.continue() : route.abort();
    });
    await page.goto('/');
    await expect(page.locator('#sago-open')).toBeVisible({ timeout: 60000 });
    await expect(page.locator('#sago-demo')).toHaveCount(0);
    await expect(page.locator('#scene-panel #sago-open')).toBeVisible();
    console.log('GPU:', await page.evaluate(() => window.scene.events.invoke('scene.gpu')));
    await page.evaluate(async () => {
        await window.scene.events.invoke('import', [{ filename: 'two-objects.ply', url: new URL('static/examples/two-objects.ply', document.baseURI).href }]);
        window.scene.camera.focus();
        window.scene.camera.setAzimElev(0, 0, 0);
        window.scene.camera.setDistance(1.4, 0);
    });
    await expect.poll(() => page.evaluate(() => window.scene.events.invoke('selection')?.instances.count)).toBe(6000);
    await expect.poll(() => page.evaluate(() => {
        const c = window.scene.camera;
        return c.distanceTween.timer >= c.distanceTween.transitionTime && c.focalPointTween.timer >= c.focalPointTween.transitionTime;
    })).toBe(true);
    await page.screenshot({ path: 'test-results/scene.png' });
    await openSago(page);
    await expect(page.locator('#sago-status')).toContainText('已固定', { timeout: 30000 });
    await page.locator('#sago-load').click();
    await expect(page.locator('.sago-download-dialog')).toBeVisible();
    await page.locator('#sago-download-confirm').click();
    await expect(page.locator('#sago-download-progress')).toBeVisible();
    await expect(page.locator('#sago-status')).toContainText('模型已就绪', { timeout: 300000 });
    const bounds = await page.locator('#sago-image').boundingBox();
    await page.mouse.click(bounds.x + bounds.width * 0.32, bounds.y + bounds.height * 0.5);
    await page.locator('#sago-predict').click();
    await expect(page.locator('#sago-status')).toContainText('分割完成', { timeout: 60000 });
    await expect(page.locator('#sago-candidate option')).toHaveCount(4);
    await expect(page.locator('#sago-candidates button')).toHaveCount(4);
    const autoCandidate = await page.locator('#sago-candidate').inputValue();
    await page.locator('#sago-candidates button').first().click();
    await expect(page.locator('#sago-candidate')).toHaveValue('0');
    await expect(page.locator('#sago-candidates button').first()).toHaveAttribute('aria-pressed', 'true');
    await page.locator('#sago-candidate').selectOption(autoCandidate);
    const pendingDownload = page.waitForEvent('download');
    await page.locator('#sago-debug').click();
    const download = await pendingDownload;
    const bundle = JSON.parse(await readFile(await download.path(), 'utf8'));
    expect(bundle.schema).toBe('sago-segmentation-debug-v1');
    expect(bundle.image.png).toMatch(/^data:image\/png;base64,/);
    expect(bundle.prompts.points).toHaveLength(1);
    expect(bundle.result.candidates).toHaveLength(4);
    expect(bundle.result.selected).toBe(Number(autoCandidate));
    expect(bundle.result.masksIncludePadding).toBe(false);
    expect(bundle.model.id).toBe('sago/sam2-hiera-large');
    expect(bundle.prompts.boxMode).toBe('native-sam2');
    expect(bundle.result.selected).toBe(0);
    // Isolated previews must be complementary and retain original RGB, not green mask tint.
    const originalPixels = await page.evaluate(async (png) => {
        const bitmap = await createImageBitmap(await (await fetch(png)).blob());
        const canvas = document.createElement('canvas');
        canvas.width = bitmap.width; canvas.height = bitmap.height;
        canvas.getContext('2d').drawImage(bitmap, 0, 0);
        window.previewOriginal = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
        return canvas.width * canvas.height;
    }, bundle.image.png);
    for (const mode of ['foreground', 'background']) {
        await page.locator('#sago-preview').selectOption(mode);
        await page.evaluate((mode) => {
            const canvas = document.querySelector('#sago-image');
            window[mode] = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
        }, mode);
        await page.screenshot({ path: `test-results/preview-${mode}.png` });
    }
    const complement = await page.evaluate(() => {
        let foreground = 0, background = 0, mismatch = 0;
        const canvas = document.querySelector('#sago-image');
        for (let i = 0; i < window.previewOriginal.length; i += 4) {
            const same = data => [0, 1, 2].every(c => data[i + c] === window.previewOriginal[i + c]);
            const shade = ((Math.floor(((i / 4) % canvas.width) / 16) + Math.floor(i / 4 / canvas.width / 16)) % 2) ? 28 : 38;
            const hidden = data => [0, 1, 2].every(c => data[i + c] === shade);
            if (same(window.foreground) && hidden(window.background)) foreground++;
            else if (same(window.background) && hidden(window.foreground)) background++;
            else mismatch++;
        }
        return { foreground, background, mismatch };
    });
    expect(complement.foreground).toBeGreaterThan(100);
    expect(complement.background).toBeGreaterThan(100);
    // Prompt marker remains visible on both previews.
    expect(complement.mismatch / originalPixels).toBeLessThan(0.002);
    await page.locator('#sago-preview').selectOption('all');
    await page.screenshot({ path: 'test-results/segmentation.png' });
    await page.locator('#sago-apply').click();
    await expect(page.locator('.sago-dialog')).not.toBeVisible({ timeout: 30000 });
    const selected = await page.evaluate(() => window.scene.events.invoke('selection').numSelected);
    console.log('selected gaussians:', selected);
    expect(selected).toBeGreaterThan(0);
    expect(selected).toBeLessThan(6000);
    const counts = [];
    const captures = [];
    for (const mode of ['0', '1', '2', '0']) {
        await page.locator('#sago-scene-view').selectOption(mode);
        counts.push(await page.evaluate(async () => {
            await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
            const data = await window.scene.projectedSplatRenderer.splatCounter.read(0, 8, null, true);
            return new Uint32Array(data.buffer, data.byteOffset, 2)[0];
        }));
        captures.push(await page.evaluate(async () => {
            const pixels = await window.scene.events.invoke('render.offscreen', 320, 225);
            return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', pixels))).join(',');
        }));
        expect(await page.evaluate(() => window.scene.events.invoke('selection').numSelected)).toBe(selected);
        await page.screenshot({ path: `test-results/scene-view-${mode}.png` });
    }
    console.log('Preview counts (all / foreground / background / restored):', counts);
    expect(counts[1]).toBeGreaterThan(0);
    expect(counts[2]).toBeGreaterThan(0);
    expect(counts[1] + counts[2]).toBe(counts[0]);
    expect(counts[3]).toBe(counts[0]);
    expect(new Set(captures).size).toBe(1); // Inference/export inputs ignore display isolation.
    await page.locator('#sago-scene-view').selectOption('1');

    await page.evaluate(async () => {
        window.scene.events.fire('edit.undo');
        await window.scene.events.invoke('queue', () => {});
    });
    await expect.poll(() => page.evaluate(() => window.scene.events.invoke('selection').numSelected)).toBe(0);
    await expect(page.locator('#sago-scene-view')).toHaveValue('0');
    await expect(page.locator('#sago-scene-view')).toBeDisabled();
    await page.evaluate(async () => {
        window.scene.events.fire('edit.redo');
        await window.scene.events.invoke('queue', () => {});
    });
    await expect.poll(() => page.evaluate(() => window.scene.events.invoke('selection').numSelected)).toBe(selected);
    await openSago(page);
    await expect(page.locator('#sago-status')).toContainText('已固定');
    await page.locator('#sago-load').click();
    await expect(page.locator('#sago-status')).toContainText('模型已就绪', { timeout: 60000 });
    await page.locator('#sago-mode').selectOption('box');
    const rect = await page.locator('#sago-image').boundingBox();
    await page.mouse.move(rect.x + rect.width * 0.12, rect.y + rect.height * 0.2);
    await page.mouse.down();
    await page.mouse.move(rect.x + rect.width * 0.48, rect.y + rect.height * 0.8, { steps: 6 });
    await page.mouse.up();
    await page.locator('#sago-predict').click();
    await expect(page.locator('#sago-status')).toContainText('分割完成', { timeout: 60000 });
    await page.locator('#sago-op').selectOption('intersect');
    await page.locator('#sago-apply').click();
    await expect(page.locator('.sago-dialog')).not.toBeVisible({ timeout: 30000 });
    const refined = await page.evaluate(() => window.scene.events.invoke('selection').numSelected);
    expect(refined).toBeGreaterThan(0);
    expect(refined).toBeLessThanOrEqual(selected);
    await openSago(page);
    await expect(page.locator('#sago-status')).toContainText('已固定');
    await page.locator('#sago-load').click();
    await expect(page.locator('#sago-status')).toContainText('模型已就绪', { timeout: 60000 });
    await page.locator('#sago-mode').selectOption('positive');
    const last = await page.locator('#sago-image').boundingBox();
    await page.mouse.click(last.x + last.width * 0.32, last.y + last.height * 0.5);
    await page.locator('#sago-predict').click();
    await expect(page.locator('#sago-status')).toContainText('分割完成', { timeout: 60000 });
    await page.evaluate(() => window.scene.events.fire('selection.changed'));
    await page.locator('#sago-apply').click();
    await expect(page.locator('#sago-status')).toContainText('场景已变化');
    await page.locator('#sago-close').click();
    expect(await page.evaluate(() => window.scene.camera.interactionLocked)).toBe(false);
    expect(errors).toEqual([]);
});

test('optional real local PLY smoke test', async ({ page }) => {
    test.skip(!process.env.SAGO_TEST_SCENE, 'Set SAGO_TEST_SCENE to a local binary PLY.');
    await page.goto('/');
    await expect(page.locator('#sago-open')).toBeVisible({ timeout: 60000 });
    await page.evaluate(() => {
        const input = document.createElement('input');
        input.type = 'file';
        input.id = 'test-local-file';
        input.hidden = true;
        document.body.append(input);
    });
    await page.locator('#test-local-file').setInputFiles(process.env.SAGO_TEST_SCENE);
    await page.evaluate(async () => {
        const file = document.querySelector('#test-local-file').files[0];
        await window.scene.events.invoke('import', [{ filename: file.name, contents: file }]);
        window.scene.camera.focus();
        window.scene.camera.setDistance(1.3, 0);
    });
    await expect.poll(() => page.evaluate(() => window.scene.events.invoke('selection')?.instances.count), { timeout: 60000 }).toBeGreaterThan(0);
    await openSago(page);
    await expect(page.locator('#sago-status')).toContainText('已固定', { timeout: 30000 });
    await page.locator('#sago-load').click();
    await expect(page.locator('.sago-download-dialog')).toBeVisible();
    await page.locator('#sago-download-confirm').click();
    await expect(page.locator('#sago-download-progress')).toBeVisible();
    await expect(page.locator('#sago-status')).toContainText('模型已就绪', { timeout: 300000 });
    const rect = await page.locator('#sago-image').boundingBox();
    await page.mouse.click(rect.x + rect.width * 0.5, rect.y + rect.height * 0.5);
    await page.locator('#sago-predict').click();
    await expect(page.locator('#sago-status')).toContainText('分割完成', { timeout: 60000 });
    await page.screenshot({ path: 'test-results/real-scene-segmentation.png' });
    await page.locator('#sago-apply').click();
    await expect(page.locator('.sago-dialog')).not.toBeVisible({ timeout: 30000 });
    console.log('Real scene:', await page.evaluate(() => {
        const s = window.scene.events.invoke('selection');
        return { total: s.instances.count, selected: s.numSelected };
    }));
    expect(await page.evaluate(() => window.scene.events.invoke('selection').numSelected)).toBeGreaterThan(0);
});

test('download consent and progress, confirmed cross-view memory and reset', async ({ page }) => {
    const modelRequests = [];
    page.on('pageerror', error => console.log('memory / storage page error:', error.message));
    page.on('console', message => { if (message.type() === 'error') console.log('memory / storage:', message.text()); });
    page.on('request', request => {
        if (/\/sam2-large\/(?!manifest\.json)/.test(request.url())) modelRequests.push(request.url());
    });
    await page.goto('/');
    await loadDemo(page);
    await openSago(page);
    await expect(page.locator('#sago-status')).toContainText('已固定');
    await page.locator('#sago-load').click();
    await expect(page.locator('.sago-download-dialog')).toBeVisible();
    expect(modelRequests).toHaveLength(0);
    await expect(page.locator('#sago-download-description')).toContainText('本次需要下载');
    await page.locator('#sago-download-cancel').click();
    await expect(page.locator('#sago-status')).toContainText('已取消下载');
    expect(modelRequests).toHaveLength(0);
    await page.locator('#sago-load').click();
    await page.locator('#sago-download-confirm').click();
    await expect(page.locator('#sago-download-progress')).toBeVisible();
    await expect.poll(() => page.locator('#sago-download-progress').evaluate(p => p.value), { timeout: 30000 }).toBeGreaterThan(0);
    await expect(page.locator('#sago-status')).toContainText('模型已就绪', { timeout: 300000 });
    const downloads = modelRequests.length;
    const rect = await page.locator('#sago-image').boundingBox();
    await page.mouse.click(rect.x + rect.width * .32, rect.y + rect.height * .5);
    await page.locator('#sago-predict').click();
    await expect(page.locator('#sago-status')).toContainText('分割完成');
    await page.locator('#sago-remember').click();
    await expect(page.locator('#sago-memory-status')).toContainText('已确认 1 个视角');
    await expect(page.locator('#sago-remember')).toBeDisabled();
    await page.locator('#sago-apply').click();
    await expect(page.locator('.sago-dialog')).not.toBeVisible();
    const initialSelected = await page.evaluate(() => window.scene.events.invoke('selection').numSelected);
    expect(initialSelected).toBeGreaterThan(0);
    for (const [step, angle] of [8, 16].entries()) {
        await page.evaluate(angle => window.scene.camera.setAzimElev(angle, 0, 0), angle);
        await openSago(page);
        await expect(page.locator('#sago-status')).toContainText('已固定');
        await page.locator('#sago-load').click();
        await expect(page.locator('#sago-status')).toContainText('模型已就绪', { timeout: 60000 });
        await expect(page.locator('#sago-memory-status')).toContainText(`已确认 ${step + 1} 个视角`);
        await page.locator('#sago-track').click();
        await expect(page.locator('#sago-status')).toContainText('记忆跟踪完成', { timeout: 120000 });
        await expect(page.locator('#sago-memory-status')).toContainText(`已确认 ${step + 1} 个视角`);
        // A second preview never inserts unconfirmed data into the temporal buffer.
        if (step === 0) {
            await page.locator('#sago-track').click();
            await expect(page.locator('#sago-status')).toContainText('记忆跟踪完成', { timeout: 120000 });
            await expect(page.locator('#sago-memory-status')).toContainText('已确认 1 个视角');
        }
        await page.locator('#sago-remember').click();
        await expect(page.locator('#sago-memory-status')).toContainText(`已确认 ${step + 2} 个视角`);
        const downloadEvent = page.waitForEvent('download');
        await page.locator('#sago-debug').click();
        const result = JSON.parse(await readFile(await (await downloadEvent).path(), 'utf8'));
        expect(result.prompts.points).toEqual([]);
        expect(result.prompts.box).toBeNull();
        expect(result.memory.usedForPrediction).toBe(true);
        expect(result.memory.recentMemories).toBe(step + 1);
        expect(result.result.selected).toBeGreaterThan(0);
        if (step === 1) await page.screenshot({ path: 'test-results/memory-tracking.png' });
        await page.locator('#sago-op').selectOption('intersect');
        await page.locator('#sago-apply').click();
        await expect(page.locator('.sago-dialog')).not.toBeVisible();
        expect(await page.evaluate(() => window.scene.events.invoke('selection').numSelected)).toBeGreaterThan(0);
    }
    expect(modelRequests).toHaveLength(downloads);
    await openSago(page);
    await expect(page.locator('#sago-status')).toContainText('已固定');
    await page.locator('#sago-forget').click();
    await expect(page.locator('#sago-memory-status')).toContainText('尚未建立');
    await expect(page.locator('#sago-track')).toBeDisabled();

    // Exercise actual browser FileSystemDirectoryHandle reads/writes. The native
    // OS chooser needs a human; use OPFS as the test directory behind the chooser.
    // Cross-origin portability is covered separately with independent cache maps.
    await page.evaluate(() => { window.showDirectoryPicker = () => navigator.storage.getDirectory(); });
    await page.locator('#sago-directory').click();
    await expect(page.locator('#sago-directory-status')).toContainText('sago-sam2-large-fp32');
    await page.locator('#sago-load').click();
    await expect(page.locator('#sago-status')).toContainText('模型已就绪', { timeout: 120000 });
    expect(modelRequests).toHaveLength(downloads);
    const saved = await page.evaluate(async () => {
        const root = await navigator.storage.getDirectory();
        const directory = await root.getDirectoryHandle('sago-sam2-large-fp32');
        const manifest = JSON.parse(await (await (await directory.getFileHandle('manifest.json')).getFile()).text());
        for (const [name, info] of Object.entries(manifest.files)) {
            if ((await (await directory.getFileHandle(name)).getFile()).size !== info.bytes) throw new Error(`Missing local model file: ${name}`);
        }
        await caches.delete('sago-sam2-large-v1');
        return Object.keys(manifest.files).length;
    });
    expect(saved).toBe(39);
    console.log('Saved all 39 model files; browser model cache cleared.');
    await page.locator('#sago-close').click();
    page.once('dialog', dialog => dialog.accept());
    await page.reload();
    await loadDemo(page);
    await openSago(page);
    await expect(page.locator('#sago-status')).toContainText('已固定');
    await page.evaluate(() => { window.showDirectoryPicker = () => navigator.storage.getDirectory(); });
    await page.locator('#sago-directory').click();
    await expect(page.locator('#sago-directory-status')).toContainText('sago-sam2-large-fp32');
    await page.locator('#sago-load').click();
    await expect(page.locator('#sago-status')).toContainText('模型已就绪', { timeout: 120000 });
    expect(modelRequests).toHaveLength(downloads);
});

test('Virtual Drone traverses six views, restores camera and memory, cancels and commits one undoable edit', async ({ page }) => {
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', m => { if (m.type() === 'error') console.log('drone browser:', m.text()); });
    await page.goto('/');
    await loadDemo(page);
    await openSago(page);
    await expect(page.locator('#sago-status')).toContainText('已固定');
    await page.locator('#sago-load').click();
    await page.locator('#sago-download-confirm').click();
    await expect(page.locator('#sago-status')).toContainText('模型已就绪', { timeout: 300000 });
    const rect = await page.locator('#sago-image').boundingBox();
    await page.mouse.click(rect.x + rect.width * .32, rect.y + rect.height * .5);
    await page.locator('#sago-predict').click();
    await expect(page.locator('#sago-status')).toContainText('分割完成', { timeout: 60000 });
    await page.locator('#sago-remember').click();
    await expect(page.locator('#sago-memory-status')).toContainText('已确认 1 个视角');
    const snapshot = () => page.evaluate(() => ({
        pose: window.scene.camera.docSerialize(),
        position: window.scene.camera.position.toString(),
        flags: [...window.scene.events.invoke('selection').instances.flags]
    }));
    const before = await snapshot();
    await page.locator('#sago-drone').click();
    await expect(page.locator('#sago-drone-status')).toContainText('视角 1/6', { timeout: 60000 });
    await page.locator('#sago-drone-cancel').click();
    await expect(page.locator('#sago-status')).toContainText('巡航已取消', { timeout: 120000 });
    expect(await snapshot()).toEqual(before);
    expect(await page.evaluate(() => window.scene.camera.poseOverride)).toBeNull();
    expect(await page.evaluate(() => window.scene.projectedSplatRenderer.captureFilter)).toBeNull();
    await expect(page.locator('#sago-memory-status')).toContainText('已确认 1 个视角');
    await page.locator('#sago-load').click();
    await expect(page.locator('#sago-status')).toContainText('模型已就绪', { timeout: 60000 });
    await page.locator('#sago-predict').click();
    await expect(page.locator('#sago-status')).toContainText('分割完成', { timeout: 60000 });
    await page.locator('#sago-drone').click();
    await expect(page.locator('#sago-summary')).toContainText('Virtual Drone 完成', { timeout: 300000 });
    await expect(page.locator('.sago-dialog')).not.toBeVisible();
    const summary = await page.locator('#sago-summary').textContent();
    console.log(summary);
    expect(summary).toMatch(/[1-6]\/6 视角通过/);
    expect(await page.locator('#sago-drone-progress').evaluate(p => p.value)).toBe(6);
    const after = await snapshot();
    expect(after.pose).toEqual(before.pose);
    const selected = after.flags.filter(flag => flag === 1).length;
    expect(selected).toBeGreaterThan(2000);
    expect(selected).toBeLessThanOrEqual(3300);
    await expect(page.locator('#sago-memory-status')).toContainText('已确认 1 个视角');
    await page.screenshot({ path: 'test-results/virtual-drone.png' });
    await page.evaluate(async () => {
        window.scene.events.fire('edit.undo');
        await window.scene.events.invoke('queue', () => {});
    });
    expect((await snapshot()).flags).toEqual(before.flags);
    await page.evaluate(async () => {
        window.scene.events.fire('edit.redo');
        await window.scene.events.invoke('queue', () => {});
    });
    expect((await snapshot()).flags).toEqual(after.flags);

    // Looking along x puts the other ellipsoid behind the prompted object.
    // The initial silhouette selects both; orbiting must actually prune leakage.
    await page.evaluate(() => window.scene.camera.setAzimElev(90, 0, 0));
    await openSago(page);
    await expect(page.locator('#sago-status')).toContainText('已固定');
    await page.locator('#sago-load').click();
    await expect(page.locator('#sago-status')).toContainText('模型已就绪', { timeout: 60000 });
    const overlap = await page.locator('#sago-image').boundingBox();
    await page.mouse.click(overlap.x + overlap.width * .5, overlap.y + overlap.height * .5);
    await page.locator('#sago-predict').click();
    await expect(page.locator('#sago-status')).toContainText('分割完成', { timeout: 60000 });
    await page.locator('#sago-drone').click();
    await expect(page.locator('.sago-dialog')).not.toBeVisible({ timeout: 300000 });
    const overlapSummary = await page.locator('#sago-summary').textContent();
    console.log('overlap:', overlapSummary);
    const counts = overlapSummary.match(/候选 ([\d,]+) → ([\d,]+)/).slice(1).map(n => Number(n.replaceAll(',', '')));
    expect(counts[0]).toBeGreaterThan(4500);
    expect(counts[1]).toBeGreaterThan(2000);
    expect(counts[1]).toBeLessThan(3500);
    await openSago(page);
    await expect(page.locator('#sago-status')).toContainText('已固定');
    const reportDownload = page.waitForEvent('download');
    await page.locator('#sago-debug').click();
    const diagnostic = JSON.parse(await readFile(await (await reportDownload).path(), 'utf8'));
    console.log('drone views:', diagnostic.virtualDrone);
    expect(diagnostic.virtualDrone.map(view => view.yaw)).toEqual([45, -45, 90, -90, 135, -135]);
    expect(diagnostic.virtualDrone.every(view => !view.accepted || view.coverage >= .8)).toBe(true);
    expect(errors).toEqual([]);
});
