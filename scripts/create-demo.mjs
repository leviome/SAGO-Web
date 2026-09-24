import { mkdir, writeFile } from 'node:fs/promises';

// Two asymmetric colored ellipsoids, no external scene data or downloads.
const count = 6000;
const fields = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
const header = `ply\nformat binary_little_endian 1.0\nelement vertex ${count}\n${fields.map(f => `property float ${f}`).join('\n')}\nend_header\n`;
const data = Buffer.alloc(count * fields.length * 4);
for (let i = 0; i < count; i++) {
    const object = i < count / 2 ? 0 : 1;
    const j = i % (count / 2);
    const y = 1 - 2 * (j + 0.5) / (count / 2);
    const r = Math.sqrt(1 - y * y);
    const angle = j * Math.PI * (3 - Math.sqrt(5));
    const color = object ? [0.16, 0.48, 0.94] : [0.95, 0.32, 0.12];
    const row = [
        (object ? 1.3 : -1.3) + r * Math.cos(angle) * 0.85,
        y * (object ? 0.8 : 1.1), r * Math.sin(angle) * 0.8,
        ...color.map(c => (c - 0.5) / 0.2820947918), 4,
        -3.35, -3.35, -3.35, 1, 0, 0, 0
    ];
    row.forEach((value, j) => data.writeFloatLE(value, (i * fields.length + j) * 4));
}
await mkdir('static/examples', { recursive: true });
await writeFile('static/examples/two-objects.ply', Buffer.concat([Buffer.from(header), data]));
