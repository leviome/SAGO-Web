import { loadDemo } from '../tests/helpers.mjs';
import { chromium } from '@playwright/test';
import { mkdir } from 'node:fs/promises';

const url = process.argv[2];
if (!url || !/^https?:\/\//.test(url)) throw new Error('Usage: node scripts/check-public.mjs https://example.com');
const browser = await chromium.launch({
    executablePath: process.env.SAGO_CHROME || '/usr/bin/google-chrome',
    proxy: process.env.SAGO_CHECK_PROXY ? { server: process.env.SAGO_CHECK_PROXY } : undefined,
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface']
});
try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    page.on('pageerror', error => console.log('pageerror:', error.message));
    page.on('requestfailed', request => console.log('requestfailed:', request.url(), request.failure()?.errorText));
    page.on('response', response => {
        if (response.status() >= 400) console.log('http-error:', response.status(), response.url());
    });
    const response = await page.goto(url, { timeout: 45000, waitUntil: 'domcontentloaded' });
    console.log('document:', response.status());
    await page.locator('#sago-open').waitFor({ state: 'visible', timeout: 60000 });
    await loadDemo(page);
    await page.locator('#sago-open').click();
    await page.locator('#sago-orientation-confirm').check();
    await page.locator('#sago-orientation-enter').click();
    await page.waitForFunction(() => document.querySelector('#sago-status')?.textContent.includes('已固定'), undefined, { timeout: 30000 });
    await mkdir('test-results', { recursive: true });
    await page.screenshot({ path: 'test-results/public-site.png' });
    console.log('success:', await page.locator('#sago-status').textContent());
} finally {
    await browser.close();
}
