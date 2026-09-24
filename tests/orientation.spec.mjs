import { loadDemo } from './helpers.mjs';
import { test, expect } from '@playwright/test';

test('SAGO requires world-up confirmation, guides ground alignment, and invalidates after geometry edits', async ({ page }) => {
    page.on('console', message => { if (message.type() === 'error') console.log('orientation:', message.text()); });
    const requests = [];
    page.on('request', request => { if (request.url().includes('/sam2-large/')) requests.push(request.url()); });
    await page.goto('/');
    await loadDemo(page);
    await page.locator('#sago-open').click();
    await expect(page.locator('.sago-orientation-dialog')).toBeVisible();
    await expect(page.locator('.sago-dialog')).not.toBeVisible();
    await expect(page.locator('#sago-orientation-enter')).toBeDisabled();
    expect(await page.evaluate(() => window.scene.camera.interactionLocked)).toBe(false);
    await page.screenshot({ path: 'test-results/orientation-required.png' });
    await page.locator('#sago-orientation-cancel').click();
    await expect(page.locator('.sago-dialog')).not.toBeVisible();
    expect(requests).toEqual([]);

    // Already-oriented data can be explicitly confirmed without manufacturing
    // ground points (e.g. standalone object models with no floor).
    await page.locator('#sago-open').click();
    await page.locator('#sago-orientation-confirm').check();
    await page.locator('#sago-orientation-enter').click();
    await expect(page.locator('#sago-status')).toContainText('已固定');
    await page.locator('#sago-close').click();
    await page.evaluate(() => window.scene.camera.setAzimElev(10, -30, 0));
    await page.locator('#sago-open').click();
    await expect(page.locator('#sago-status')).toContainText('已固定');
    await expect(page.locator('.sago-orientation-dialog')).not.toBeVisible();
    await page.locator('#sago-close').click();

    // Use a controlled tilted plane to verify the real editor alignment action.
    // Points are in the layer frame, just like points picked on its surface.
    const before = await page.evaluate(() => {
        const splat = window.scene.events.invoke('selection');
        const point = (x, y, z) => window.scene.camera.position.clone().set(x, y, z);
        splat.orientPoints = [point(0, 0, 0), point(1, 1, 0), point(0, 0, 1)];
        return [...splat.worldTransform.data];
    });
    await page.locator('#sago-open').click();
    await expect(page.locator('#sago-orientation-status')).toContainText('45.0°');
    await expect(page.locator('#sago-orientation-confirm')).toBeDisabled();
    await page.locator('#sago-orientation-adjust').click();
    await expect(page.locator('.sago-orientation-guide')).toBeVisible();
    expect(await page.evaluate(() => window.scene.events.invoke('tool.active'))).toBe('orient');
    expect(await page.evaluate(() => window.scene.events.invoke('grid.planes'))).toEqual(['xz']);
    await page.locator('#orient-align-to-grid').click();
    await page.evaluate(() => window.scene.events.invoke('queue', () => {}));
    expect(await page.evaluate(() => [...window.scene.events.invoke('selection').worldTransform.data])).not.toEqual(before);
    await page.locator('#sago-orientation-review').click();
    await expect(page.locator('#sago-orientation-status')).toContainText('0.0°');
    await page.locator('#sago-orientation-confirm').check();
    await page.locator('#sago-orientation-enter').click();
    await expect(page.locator('#sago-status')).toContainText('已固定');
    await page.locator('#sago-close').click();
    // Undo restores the actual geometry and must require direction review again.
    await page.evaluate(async () => {
        window.scene.events.fire('edit.undo');
        await window.scene.events.invoke('queue', () => {});
    });
    await page.locator('#sago-open').click();
    await expect(page.locator('#sago-orientation-status')).toContainText('45.0°');
    await expect(page.locator('#sago-orientation-enter')).toBeDisabled();
    expect(requests).toEqual([]);
});

test('scene foreground and background previews render pixels and reset when changing layers', async ({ page }) => {
    test.setTimeout(45000);
    page.on('console', message => { if (message.type() === 'error') console.log('preview:', message.text()); });
    page.on('pageerror', error => console.log('preview:', error.message));
    await page.goto('/');
    await loadDemo(page);
    await page.evaluate(async () => {
        const mask = new Uint8Array(6000);
        mask.fill(255, 0, 3000);
        window.scene.events.fire('select.mask', 'set', mask);
        await window.scene.events.invoke('queue', () => {});
    });
    console.log('Preview fixture ready');
    const samples = [];
    for (const mode of ['0', '1', '2', '0']) {
        await page.locator('#sago-scene-view').selectOption(mode);
        samples.push(await page.evaluate(async () => {
            await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
            const scene = window.scene;
            const { mainTarget, workTarget } = scene.camera;
            const { width, height } = mainTarget;
            scene.dataProcessor.copyRt(mainTarget, workTarget);
            const pixels = new Uint8Array(width * height * 4);
            await workTarget.colorBuffer.read(0, 0, width, height, { renderTarget: workTarget, data: pixels, immediate: true });
            let colored = 0;
            for (let i = 0; i < pixels.length; i += 4) {
                if (Math.max(pixels[i], pixels[i + 1], pixels[i + 2]) - Math.min(pixels[i], pixels[i + 1], pixels[i + 2]) > 30) colored++;
            }
            return colored;
        }));
    }
    console.log('Colored viewport pixels:', samples);
    expect(samples[1]).toBeGreaterThan(100);
    expect(samples[2]).toBeGreaterThan(100);
    expect(samples[0]).toBeGreaterThan(samples[1]);
    expect(samples[0]).toBeGreaterThan(samples[2]);
    expect(samples[3]).toBe(samples[0]);
    await page.locator('#sago-scene-view').selectOption('1');
    await loadDemo(page);
    await expect(page.locator('#sago-scene-view')).toHaveValue('0');
    await expect(page.locator('#sago-scene-view')).toBeDisabled();
});
