import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';

test('model directory survives reload and cache deletion in the real browser filesystem', async ({ page }) => {
    test.setTimeout(90000);
    page.on('console', message => console.log('storage:', message.text()));
    const manifest = JSON.parse(await readFile('static/models/sam2-large/manifest.json', 'utf8'));
    const names = ['decoder.onnx', 'memoryEncoder.onnx'];
    manifest.files = Object.fromEntries(names.map(name => [name, manifest.files[name]]));
    const transpile = async file => ts.transpile(await readFile(file, 'utf8'), { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 });
    const assetsCode = await transpile('src/sago/model-assets.ts');
    const directoryCode = await transpile('src/sago/model-directory.ts');
    await page.route('**/storage-test', route => route.fulfill({ contentType: 'text/html', body: '<html><body>Model storage test</body></html>' }));
    await page.route('**/asset-module.js', route => route.fulfill({ contentType: 'text/javascript', body: assetsCode }));
    await page.route('**/directory-module.js', route => route.fulfill({ contentType: 'text/javascript', body: directoryCode }));
    await page.route('**/static/models/sam2-large/manifest.json', route => route.fulfill({ json: manifest }));
    const requests = [];
    page.on('request', request => { if (/\.onnx\?sha256=/.test(request.url())) requests.push(request.url()); });
    await page.goto('/storage-test');
    const first = await page.evaluate(async (names) => {
        const { ModelAssets } = await import('/asset-module.js');
        const { pickModelDirectory } = await import('/directory-module.js');
        window.showDirectoryPicker = () => navigator.storage.getDirectory();
        const directory = await pickModelDirectory();
        let confirmations = 0;
        const assets = await ModelAssets.open({ directory, confirm: async () => { confirmations++; return true; } });
        await assets.fetchMany(names);
        await assets.writeManifest();
        await caches.delete('sago-sam2-large-v1');
        return confirmations;
    }, names);
    expect(first).toBe(1);
    expect(requests).toHaveLength(2);
    await page.reload();
    const second = await page.evaluate(async (names) => {
        const { ModelAssets } = await import('/asset-module.js');
        const { pickModelDirectory } = await import('/directory-module.js');
        window.showDirectoryPicker = () => navigator.storage.getDirectory();
        const directory = await pickModelDirectory();
        const assets = await ModelAssets.open({ directory, confirm: async () => { throw new Error('Unexpected download'); } });
        const files = await assets.fetchMany(names);
        return files.map(file => file.length);
    }, names);
    expect(second).toEqual(names.map(name => manifest.files[name].bytes));
    expect(requests).toHaveLength(2);
});
