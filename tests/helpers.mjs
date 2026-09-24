import { expect } from '@playwright/test';

export const loadDemo = async (page) => {
    await expect(page.locator('#sago-open')).toBeVisible({ timeout: 60000 });
    await page.evaluate(async () => {
        await window.scene.events.invoke('import', [{ filename: 'two-objects.ply', url: new URL('static/examples/two-objects.ply', document.baseURI).href }]);
        window.scene.camera.focus();
        window.scene.camera.setAzimElev(0, 0, 0);
        window.scene.camera.setDistance(1.4, 0);
    });
    await expect.poll(() => page.evaluate(() => window.scene.events.invoke('selection')?.instances.count)).toBe(6000);
};
