import { defineConfig } from '@playwright/test';

export default defineConfig({
    testDir: './tests',
    testMatch: ['**/browser.spec.mjs', '**/model-storage.spec.mjs', '**/orientation.spec.mjs'],
    timeout: 600000,
    workers: 1,
    use: {
        baseURL: 'http://127.0.0.1:3000',
        viewport: { width: 1280, height: 900 },
        screenshot: 'only-on-failure',
        launchOptions: {
            executablePath: process.env.SAGO_CHROME || '/usr/bin/google-chrome',
            args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface']
        }
    }
});
