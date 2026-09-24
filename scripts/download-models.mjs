// Prepare the exact local SAGO checkpoint; no model substitution or remote inference.
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { readFile, access } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = resolve('static/models/sam2-large');
let ready = false;
try {
    const manifest = JSON.parse(await readFile(resolve(root, 'manifest.json'), 'utf8'));
    if (manifest.id !== 'sago/sam2-hiera-large' || manifest.dtype !== 'fp32' || manifest.memory?.version !== 1) throw new Error('Wrong model');
    for (const [name, info] of Object.entries(manifest.files)) {
        const bytes = await readFile(resolve(root, name));
        if (bytes.length !== info.bytes || createHash('sha256').update(bytes).digest('hex') !== info.sha256) throw new Error(`Corrupt ${name}`);
    }
    ready = true;
} catch { /* Missing or interrupted export: regenerate from checkpoint. */ }
if (ready) {
    console.log('Original SAM2 Large FP32 assets verified. Run npm run build.');
} else {
    const checkpoint = resolve(process.env.SAGO_SAM2_CHECKPOINT || '../sago_core/sam2/checkpoints/sam2_hiera_large.pt');
    await access(checkpoint).catch(() => { throw new Error(`Missing original checkpoint: ${checkpoint}. Set SAGO_SAM2_CHECKPOINT.`); });
    const python = process.env.SAGO_PYTHON || 'python3';
    for (const script of ['scripts/export-sam2.py', 'scripts/export-sam2-memory.py']) {
        const child = spawn(python, [script, '--checkpoint', checkpoint], { stdio: 'inherit' });
        const code = await new Promise((resolve, reject) => { child.on('error', reject); child.on('exit', resolve); });
        if (code !== 0) throw new Error('SAM2 export failed. The export environment needs torch, numpy, opencv-python and onnx (see README-SAGO.md).');
    }
}
