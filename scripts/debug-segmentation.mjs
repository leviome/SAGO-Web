// Offline diagnostic: replay the production browser inference class on a fixed image,
// then replay exactly the same input tensors through ONNX Runtime CPU.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createRequire } from 'node:module';

import { chromium } from '@playwright/test';
import sharp from 'sharp';
import ts from 'typescript';

const require = createRequire(import.meta.url);
const ort = require('onnxruntime-node');
const out = resolve(process.env.SAGO_DEBUG_OUT || 'artifacts/sam2-large-debug');
const bundle = process.env.SAGO_DEBUG_BUNDLE ? JSON.parse(await readFile(process.env.SAGO_DEBUG_BUNDLE, 'utf8')) : null;
if (bundle?.memory?.usedForPrediction) throw new Error('This diagnostic uses cross-view memory; it contains no history tensors. Use validate-memory.py/.mjs to compare the memory path.');
if (bundle && bundle.schema !== 'sago-segmentation-debug-v1') throw new Error('Unsupported diagnostic schema');
const imagePath = bundle ? Buffer.from(bundle.image.png.split(',')[1], 'base64') : resolve(process.env.SAGO_DEBUG_IMAGE || '../debug/first_view.png');
await mkdir(out, { recursive: true });
const cases = bundle ? [{ name: 'user-sample', points: bundle.prompts.points, box: bundle.prompts.box }] : process.env.SAGO_DEBUG_CASES ? JSON.parse(await readFile(process.env.SAGO_DEBUG_CASES, 'utf8')) : [
    { name: 'green-apple', points: [{ x: 0.292, y: 0.424, label: 1 }] },
    { name: 'camera', points: [{ x: 0.390, y: 0.310, label: 1 }] },
    { name: 'yellow-legs', points: [{ x: 0.500, y: 0.495, label: 1 }] },
    { name: 'apple-negative', points: [{ x: 0.292, y: 0.424, label: 1 }, { x: 0.321, y: 0.534, label: 0 }] },
    { name: 'camera-box', points: [], box: { x0: 0.202, y0: 0.175, x1: 0.585, y1: 0.375 } },
    { name: 'banana-box', points: [], box: { x0: 0.380, y0: 0.642, x1: 0.556, y1: 0.782 } }
];
for (const item of cases) {
    if (!/^[a-z0-9-]+$/i.test(item.name)) throw new Error('Case names must be alphanumeric with hyphens');
}
const original = await sharp(imagePath).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
const { width, height } = original.info;
await sharp(original.data, { raw: { width, height, channels: 4 } }).png().toFile(resolve(out, 'input.png'));
const browser = await chromium.launch({ executablePath: process.env.SAGO_CHROME || '/usr/bin/google-chrome',
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface'] });
const page = await browser.newPage();
page.on('pageerror', e => console.error(e));
page.on('console', m => { if (m.type() === 'error' || m.text().startsWith('[SAM2]')) console.log(m.text()); });
await page.route('**/__sago_debug/*', async route => {
    const name = new URL(route.request().url()).pathname.split('/').pop();
    const source = await readFile(`src/sago/${name.replace('.js', '.ts')}`, 'utf8');
    const js = ts.transpile(source, { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 })
    .replaceAll("from './mask-utils'", "from './mask-utils.js'")
    .replaceAll("from './sam2-utils'", "from './sam2-utils.js'")
    .replaceAll("from './model-assets'", "from './model-assets.js'");
    await route.fulfill({ body: js, contentType: 'text/javascript' });
});
await page.exposeFunction('saveDiagnostic', async (name, base64) => writeFile(resolve(out, name), Buffer.from(base64, 'base64')));
let report;
try {
    await page.goto('http://127.0.0.1:3000');
    report = await page.evaluate(async ({ rgba, width, height, cases }) => {
        const save = (name, values) => {
            const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
            let s = '';
            for (let i = 0; i < bytes.length; i += 32768) s += String.fromCharCode(...bytes.subarray(i, i + 32768));
            return window.saveDiagnostic(name, btoa(s));
        };
        const { BrowserSegmenter } = await import('/__sago_debug/inference.js');
        const engine = new BrowserSegmenter();
        await engine.load('local', s => console.log('[SAM2]', s), { confirm: async () => true });
        const raw = Uint8ClampedArray.from(atob(rgba), c => c.charCodeAt(0));
        const input = new ImageData(raw, width, height);
        await engine.prepare(input, s => console.log('[SAM2]', s));
        for (const [name, tensor] of Object.entries(engine.embeddings)) await save(`${name}-web.f32`, tensor.data);
        const lib = engine.lib;
        const { sam2Pixels, sam2Prompts } = await import('/__sago_debug/sam2-utils.js');
        await save('pixel-values.f32', sam2Pixels(raw, width, height));
        const results = [];
        for (const item of cases) {
            const result = await engine.predict(item.points, item.box || null);
            for (const [i, mask] of result.masks.entries()) await save(`${item.name}-web-${i}.bin`, mask);
            const record = { ...item, scores: result.scores, selected: 0 };
            const { coords, labels } = sam2Prompts(item.points, item.box || null);
            const pointCoords = new lib.Tensor('float32', coords, [1, labels.length, 2]);
            const pointLabels = new lib.Tensor('int64', labels, [1, labels.length]);
            const outputs = await engine.decoder.run({ ...engine.embeddings, point_coords: pointCoords, point_labels: pointLabels });
            await save(`${item.name}-web-logits.f32`, outputs.masks.data);
            record.logitDims = outputs.masks.dims;
            record.coords = Array.from(coords); record.labels = Array.from(labels, Number);
            Object.values(outputs).forEach(t => t.dispose()); pointCoords.dispose(); pointLabels.dispose();
            results.push(record);
            console.log('[SAM2]', item.name, result.scores);
        }
        const result = { width, height, model: engine.metadata.id, pixelDims: [1, 3, 1024, 1024], cases: results };
        engine.clearView(); await engine.encoder.release(); await engine.decoder.release();
        return result;
    }, { rgba: original.data.toString('base64'), width, height, cases });
} finally {
    await browser.close();
}
const floatData = async name => {
    const buf = await readFile(resolve(out, name));
    return new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
};
await writeFile(resolve(out, 'web-report.json'), JSON.stringify(report, null, 2));
const encoder = await ort.InferenceSession.create('static/models/sam2-large/encoder.onnx', { executionProviders: ['cpu'], intraOpNumThreads: 8 });
const decoder = await ort.InferenceSession.create('static/models/sam2-large/decoder.onnx', { executionProviders: ['cpu'], intraOpNumThreads: 8 });
const embeddings = await encoder.run({ pixel_values: new ort.Tensor('float32', await floatData('pixel-values.f32'), report.pixelDims) });
const gpuEmbeddings = {};
for (const [name, tensor] of Object.entries(embeddings)) {
    const data = await floatData(`${name}-web.f32`);
    let max = 0, sum = 0;
    for (let i = 0; i < data.length; i++) { const d = Math.abs(data[i] - tensor.data[i]); max = Math.max(max, d); sum += d; }
    console.log('FEATURE', name, { max, mae: sum / data.length });
    gpuEmbeddings[name] = new ort.Tensor('float32', data, tensor.dims);
}
for (const item of report.cases) {
    for (let i = 0; i < item.scores.length; i++) {
        const bin = await readFile(resolve(out, `${item.name}-web-${i}.bin`));
        await sharp(bin.map(v => v * 255), { raw: { width, height, channels: 1 } }).png().toFile(resolve(out, `${item.name}-web-${i}.png`));
    }
    const prediction = await decoder.run({ ...embeddings,
        point_coords: new ort.Tensor('float32', Float32Array.from(item.coords), [1, item.labels.length, 2]),
        point_labels: new ort.Tensor('int64', BigInt64Array.from(item.labels, BigInt), [1, item.labels.length]) });
    const fromGpu = await decoder.run({ ...gpuEmbeddings,
        point_coords: new ort.Tensor('float32', Float32Array.from(item.coords), [1, item.labels.length, 2]),
        point_labels: new ort.Tensor('int64', BigInt64Array.from(item.labels, BigInt), [1, item.labels.length]) });
    const cpu = prediction.masks.data;
    const gpu = await floatData(`${item.name}-web-logits.f32`);
    console.log('DECODER CPU WITH GPU FEATURES', item.name, Array.from(fromGpu.scores.data));
    let max = 0; let sum = 0;
    const intersections = [0, 0, 0, 0]; const unions = [0, 0, 0, 0];
    for (let i = 0; i < cpu.length; i++) {
        const delta = Math.abs(cpu[i] - gpu[i]); max = Math.max(max, delta); sum += delta;
        const k = Math.floor(i / (cpu.length / 4));
        if (cpu[i] > 0 && gpu[i] > 0) intersections[k]++;
        if (cpu[i] > 0 || gpu[i] > 0) unions[k]++;
    }
    item.cpuVsWebgpu = { logitMAE: sum / cpu.length, logitMaxError: max,
        lowResMaskIoU: unions.map((v, i) => v ? intersections[i] / v : 1), cpuScores: Array.from(prediction.scores.data) };
    console.log(item.name, item.cpuVsWebgpu);
}
await encoder.release(); await decoder.release();
await writeFile(resolve(out, 'web-report.json'), JSON.stringify(report, null, 2));
console.log(`Diagnostic saved to ${out}`);
