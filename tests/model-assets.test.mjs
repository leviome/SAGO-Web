import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';
const code = ts.transpile(await readFile(new URL('../src/sago/model-assets.ts', import.meta.url), 'utf8'), { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 });
const { ModelAssets } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

test('download consent, streamed progress, cancellation, resume and cache integrity', async () => {
    const previous = { fetch: globalThis.fetch, caches: globalThis.caches, document: globalThis.document };
    const files = { 'one.bin': Uint8Array.of(1,2,3), 'two.bin': Uint8Array.of(4,5,6,7) };
    const info = {};
    for (const [file, data] of Object.entries(files)) info[file] = { bytes: data.length, sha256: Buffer.from(await crypto.subtle.digest('SHA-256', data)).toString('hex') };
    const manifest = { id: 'sago/sam2-hiera-large', dtype: 'fp32', memory: { version: 1 }, files: info };
    const cache = new Map(), requests = [];
    globalThis.document = { baseURI: 'https://example.test/' };
    globalThis.caches = { open: async () => ({ keys: async () => [...cache.keys()].map(url => ({ url })), match: async url => cache.get(String(url))?.clone(), put: async (url, response) => cache.set(String(url), response.clone()), delete: async url => cache.delete(String(url)) }) };
    globalThis.fetch = async (url) => {
        const name = new URL(url).pathname.split('/').pop();
        if (name === 'manifest.json') return Response.json(manifest);
        requests.push(name);
        return new Response(new ReadableStream({ start(controller) { for (const byte of files[name]) controller.enqueue(Uint8Array.of(byte)); controller.close(); } }));
    };
    try {
        await assert.rejects(ModelAssets.open({ confirm: async plan => { assert.equal(plan.downloadBytes, 7); return false; } }), { name: 'AbortError' });
        assert.deepEqual(requests, []);
        const controller = new AbortController();
        const progress = [];
        const first = await ModelAssets.open({ confirm: async () => true, signal: controller.signal,
            progress: p => { progress.push(p); if (p.file === 'two.bin' && p.phase === 'download') controller.abort(); } });
        assert.deepEqual(await first.fetch('one.bin'), files['one.bin']);
        await assert.rejects(first.fetch('two.bin'), { name: 'AbortError' });
        assert.equal(cache.size, 1);
        assert.ok(progress.some(p => p.downloadedBytes === 1 && p.phase === 'download'));
        const resumed = await ModelAssets.open({ confirm: async plan => { assert.deepEqual(plan, { totalBytes: 7, cachedBytes: 3, downloadBytes: 4 }); return true; } });
        await resumed.fetch('two.bin');
        assert.equal(cache.size, 2);
        const cached = await ModelAssets.open({ confirm: async () => { assert.fail('Cached assets must not ask again'); } });
        await cached.fetch('one.bin');
        assert.equal(requests.filter(n => n === 'one.bin').length, 1);
        const key = [...cache.keys()].find(key => key.includes('one.bin'));
        cache.set(key, new Response('bad'));
        await assert.rejects(cached.fetch('one.bin'), /校验失败/);
        assert.equal(cache.has(key), false);
    } finally {
        Object.assign(globalThis, previous);
    }
});

const fixture = async (files, work) => {
    const previous = { fetch: globalThis.fetch, caches: globalThis.caches, document: globalThis.document };
    const cache = new Map(), requests = [];
    const info = {};
    for (const [file, bytes] of Object.entries(files)) info[file] = { bytes: bytes.length, sha256: Buffer.from(await crypto.subtle.digest('SHA-256', bytes)).toString('hex') };
    const manifest = { id: 'sago/sam2-hiera-large', dtype: 'fp32', memory: { version: 1 }, files: info };
    globalThis.document = { baseURI: 'https://first.test/' };
    globalThis.caches = { open: async () => ({ keys: async () => [...cache.keys()].map(url => ({ url })), match: async url => cache.get(String(url))?.clone(), put: async (url, response) => cache.set(String(url), response.clone()), delete: async url => cache.delete(String(url)) }) };
    let transport = async (name, options) => {
        const range = options.headers?.Range?.match(/^bytes=(\d+)-(\d+)$/);
        if (!range) return new Response(files[name]);
        const [start, end] = range.slice(1).map(Number);
        return new Response(files[name].slice(start, end + 1), { status: 206, headers: { 'Content-Range': `bytes ${start}-${end}/${files[name].length}` } });
    };
    globalThis.fetch = async (url, options = {}) => {
        const name = new URL(url).pathname.split('/').pop();
        if (name === 'manifest.json') return Response.json(manifest);
        requests.push({ name, range: options.headers?.Range });
        return transport(name, options);
    };
    try { await work({ cache, requests, setTransport: fn => { transport = fn; } }); }
    finally { Object.assign(globalThis, previous); }
};

test('model downloads use at most four workers and preserve graph/weight order and total progress', async () => {
    const files = Object.fromEntries(Array.from({ length: 9 }, (_, i) => [`${i}.bin`, Uint8Array.of(i)]));
    await fixture(files, async ({ setTransport }) => {
        let active = 0, peak = 0;
        const progress = [];
        setTransport(async name => {
            active++; peak = Math.max(peak, active);
            await new Promise(resolve => setTimeout(resolve, 15));
            active--;
            return new Response(files[name]);
        });
        const assets = await ModelAssets.open({ confirm: async () => true, progress: p => progress.push(p) });
        assert.deepEqual(await assets.fetchMany(Object.keys(files)), Object.values(files));
        assert.equal(peak, 4);
        assert.equal(progress.at(-1).downloadedBytes, 9);
        assert.ok(progress.every(p => p.downloadedBytes <= p.downloadBytes));
        assert.ok(progress.some(p => p.bytesPerSecond > 0));
    });
});

test('completed ranges survive cancellation and only missing ranges download on the next visit', async () => {
    const size = 8 * 1024 * 1024;
    const data = new Uint8Array(size * 2 + 3).fill(7);
    await fixture({ 'large.bin': data }, async ({ requests }) => {
        const abort = new AbortController();
        const first = await ModelAssets.open({ confirm: async () => true, signal: abort.signal, progress: p => {
            if (p.downloadedBytes > size) abort.abort();
        } });
        await assert.rejects(first.fetch('large.bin'), { name: 'AbortError' });
        const before = requests.length;
        const next = await ModelAssets.open({ confirm: async plan => {
            assert.equal(plan.cachedBytes, size);
            assert.equal(plan.downloadBytes, size + 3);
            return true;
        } });
        assert.deepEqual(await next.fetch('large.bin'), data);
        assert.deepEqual(requests.slice(before).map(r => r.range), [`bytes=${size}-${size * 2 - 1}`, `bytes=${size * 2}-${size * 2 + 2}`]);
    });
});

test('Range-ignoring hosts fall back to a full response; transient failures retry without double progress', async () => {
    const data = new Uint8Array(16 * 1024 * 1024).fill(8);
    await fixture({ 'large.bin': data }, async ({ requests, setTransport }) => {
        setTransport(async () => requests.length === 1 ? new Response('temporarily unavailable', { status: 503 }) : new Response(data));
        const progress = [];
        const assets = await ModelAssets.open({ confirm: async () => true, progress: p => progress.push(p) });
        assert.deepEqual(await assets.fetch('large.bin'), data);
        assert.equal(requests.length, 2);
        assert.ok(progress.some(p => p.phase === 'retry'));
        assert.equal(progress.at(-1).downloadedBytes, data.length);
    });
});

test('permanent HTTP errors stop a batch and no extra files start', async () => {
    const files = Object.fromEntries(Array.from({ length: 10 }, (_, i) => [`${i}.bin`, Uint8Array.of(i)]));
    await fixture(files, async ({ requests, setTransport }) => {
        setTransport(async () => new Response('missing', { status: 404 }));
        const assets = await ModelAssets.open({ confirm: async () => true });
        await assert.rejects(assets.fetchMany(Object.keys(files)), /404/);
        assert.equal(requests.length, 4);
    });
});

test('a local model directory can reuse checkpoint bytes across origins without network or consent', async () => {
    const files = { 'one.bin': Uint8Array.of(1, 2), 'two.bin': Uint8Array.of(3, 4) };
    await fixture(files, async ({ requests, cache }) => {
        const disk = new Map();
        const directory = { getFileHandle: async (name, options) => {
            if (!disk.has(name) && !options?.create) throw new DOMException('not found', 'NotFoundError');
            return {
                getFile: async () => new Blob([disk.get(name)]),
                createWritable: async () => {
                    let pending;
                    return { write: async data => { pending = data; }, close: async () => disk.set(name, pending), abort: async () => {} };
                }
            };
        } };
        const first = await ModelAssets.open({ directory, confirm: async () => true });
        await first.fetchMany(Object.keys(files));
        await first.writeManifest();
        assert.ok(disk.has('manifest.json'));
        cache.clear();
        globalThis.document.baseURI = 'https://new-domain.test/';
        const second = await ModelAssets.open({ directory, confirm: async () => assert.fail('local files require no download') });
        assert.deepEqual(await second.fetchMany(Object.keys(files)), Object.values(files));
        assert.equal(requests.length, 2);
        // A damaged local file is excluded from the plan; repair requires consent.
        disk.set('one.bin', Uint8Array.of(0, 0));
        const repair = await ModelAssets.open({ directory, confirm: async plan => {
            assert.equal(plan.downloadBytes, 2); return true;
        } });
        await repair.saveToDirectory();
        assert.deepEqual(disk.get('one.bin'), files['one.bin']);
    });
});
