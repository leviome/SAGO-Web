import { BUFFERUSAGE_COPY_DST, Mat4, Quat, StorageBuffer, Texture, Vec3 } from 'playcanvas';

import { groupInstancesByChunk } from '../gaussian-instances';
import { Scene } from '../scene';
import { Splat } from '../splat';
import { State } from '../splat-state';
import { droneYaws, initialCoverage, meanCenter, refineCandidates, touchesBorder } from './drone-utils';
import { BrowserSegmenter } from './inference';
import { dilateMask, maskRgba } from './mask-utils';

export type DroneView = { yaw: number; accepted: boolean; coverage: number; count: number; reason: string };

// Read world centers once, respecting duplicate/separated instance mappings and
// both layer and per-Gaussian transforms. SAM and mask projection stay on WebGPU.
const worldCenters = async (splat: Splat, check: () => void) => {
    const { source, sourcePool } = splat.resource;
    const { instances } = splat;
    const { starts, ordered } = groupInstancesByChunk(instances, source.meta.chunkSize, source.meta.numChunks[0]);
    const result = new Float32Array(instances.count * 3);
    const transforms = new Map<number, Mat4>();
    const p = new Vec3();
    for (let chunkIndex = 0; chunkIndex < source.meta.numChunks[0]; chunkIndex++) {
        check();
        if (starts[chunkIndex] === starts[chunkIndex + 1]) continue;
        const count = Math.min(source.meta.chunkSize, source.meta.numGaussians - chunkIndex * source.meta.chunkSize);
        const position = sourcePool.acquire('position', source.meta.layouts.position, count);
        try {
            await source.read({ chunkIndex, position });
            const xyz = new Float32Array(position.data);
            for (let slot = starts[chunkIndex]; slot < starts[chunkIndex + 1]; slot++) {
                const i = ordered[slot];
                const index = instances.transformIndex(i);
                if (!transforms.has(index)) {
                    const transform = new Mat4();
                    splat.transformPalette.getTransform(index, transform);
                    transforms.set(index, new Mat4().mul2(splat.entity.getWorldTransform(), transform));
                }
                const row = (instances.sourceRow[i] - chunkIndex * source.meta.chunkSize) * 3;
                p.set(xyz[row], xyz[row + 1], xyz[row + 2]);
                transforms.get(index).transformPoint(p, p);
                result.set([p.x, p.y, p.z], i * 3);
            }
        } finally {
            position.release();
        }
    }
    return result;
};

export const runVirtualDrone = async (options: {
    scene: Scene; splat: Splat; engine: BrowserSegmenter; mask: Uint8Array;
    width: number; height: number; level: number; padding: number;
    check: () => void;
    progress: (done: number, text: string) => void;
    preview: (image: ImageData, mask: Uint8Array) => void;
}) => {
    const { scene, splat, engine, mask, width, height, check, progress, preview } = options;
    const { camera, events, projectedSplatRenderer: renderer } = scene;
    if (camera.ortho) throw new Error('Virtual Drone 需要透视相机，请先关闭正交视图。');
    if (!mask.some(Boolean)) throw new Error('初始 mask 为空。');
    const previousPose = camera.poseOverride;
    const pose = { position: camera.position.clone(), rotation: camera.mainCamera.getRotation().clone(), fov: camera.fov, near: camera.near, far: camera.far };
    const originalMemory = engine.snapshotMemory();
    const previousFilter = renderer.captureFilter;
    const flags = new Uint8Array(Math.ceil(splat.instances.count / 4) * 4);
    const buffer = new StorageBuffer(scene.graphicsDevice, Math.max(4, flags.length), BUFFERUSAGE_COPY_DST);
    const reports: DroneView[] = [];
    const restorePose = () => {
        camera.setPoseOverride(pose);
    };
    const capture = async (subset: Uint8Array) => {
        check();
        for (let i = 0; i < splat.instances.count; i++) flags[i] = subset[i] ? 0 : 128;
        buffer.write(0, flags, 0, flags.length);
        renderer.captureFilter = { splat, flags: buffer };
        try {
            const pixels = await events.invoke('render.offscreen', width, height) as Uint8Array;
            check();
            return pixels;
        } finally {
            renderer.captureFilter = previousFilter;
        }
    };
    const project = (binary: Uint8Array, previous: Uint8Array, first = false) => {
        const expanded = dilateMask(binary, width, height, options.padding);
        const canvas = document.createElement('canvas');
        canvas.width = width; canvas.height = height;
        canvas.getContext('2d')!.putImageData(new ImageData(maskRgba(expanded), width, height), 0, 0);
        const texture = new Texture(scene.graphicsDevice);
        texture.setSource(canvas);
        return events.invoke('queue', async () => {
            let hits: Uint8Array, screen: Uint8Array;
            try {
                check();
                camera.onUpdate(0);
                hits = await scene.dataProcessor.intersect({ mask: texture }, splat);
                if (!first && touchesBorder(expanded, width, height)) {
                    screen = await scene.dataProcessor.intersect({ rect: { x1: 0, y1: 0, x2: 1, y2: 1 } }, splat);
                }
                check();
                return refineCandidates(previous, hits, screen);
            } finally {
                if (hits) scene.dataProcessor.releaseMask(hits);
                if (screen) scene.dataProcessor.releaseMask(screen);
                texture.destroy();
            }
        }) as Promise<Uint8Array>;
    };
    try {
        check();
        progress(0, '建立初始视锥与两路独立 memory…');
        let candidates: Uint8Array = Uint8Array.from(splat.instances.flags.subarray(0, splat.instances.count), flag => (flag & State.locked ? 0 : 255));
        candidates = await project(mask, candidates, true);
        if (!candidates.some(Boolean)) throw new Error('初始 mask 没有投影到可编辑高斯。');
        const initialCount = candidates.reduce((n, value) => n + Number(!!value), 0);
        const positions = await worldCenters(splat, check);
        const center = new Vec3(meanCenter(positions, candidates));
        // Estimate the seed center from rendered depth, as SAGO does. Sampling
        // bounds readback cost; subsequent rounds use the surviving 3D mean.
        const samples: { x: number; y: number }[] = [];
        const step = Math.max(1, Math.ceil(Math.sqrt(width * height / 256)));
        for (let y = step / 2 | 0; y < height; y += step) {
            for (let x = step / 2 | 0; x < width; x += step) if (mask[y * width + x]) samples.push({ x: (x + 0.5) / width, y: (y + 0.5) / height });
        }
        if (samples.length) {
            const hits = await events.invoke('queue', () => camera.intersectMany(samples, [splat])) as Awaited<ReturnType<typeof camera.intersectMany>>;
            const valid = hits.filter(hit => hit && Number.isFinite(hit.position.length()));
            if (valid.length) {
                center.set(0, 0, 0);
                valid.forEach(hit => center.add(hit.position));
                center.mulScalar(1 / valid.length);
            }
        }
        const offset = camera.position.clone().sub(center);
        const radius = offset.length();
        const initialYaw = Math.atan2(offset.x, offset.z) * 180 / Math.PI;
        if (!Number.isFinite(radius) || radius < 1e-6) throw new Error('目标中心距离相机太近，请调整视角。');
        await engine.acceptMemory(options.level, true);
        check();
        const seed = engine.snapshotMemory();
        // The native drone starts its previous-frame deque with the seed too.
        seed.history = [{ ...seed.seed }];
        const banks = [seed, seed];
        for (const yaw of droneYaws) {
            const roundCandidates = candidates.slice();
            const lookAt = new Vec3(meanCenter(positions, roundCandidates));
            for (let direction = 0; direction < 2; direction++) {
                const angle = direction === 0 ? yaw : -yaw;
                const done = reports.length;
                check();
                progress(done, `视角 ${done + 1}/6 · ${angle > 0 ? '+' : ''}${angle}° · 编码与跟踪…`);
                const radians = (initialYaw + angle) * Math.PI / 180;
                const position = new Vec3(Math.sin(radians) * radius * Math.cos(Math.PI / 6), radius * 0.5, Math.cos(radians) * radius * Math.cos(Math.PI / 6)).add(lookAt);
                const rotation = new Quat().setFromMat4(new Mat4().setLookAt(position, lookAt, Vec3.UP));
                camera.fitClippingPlanes(position, new Vec3().sub2(lookAt, position).normalize());
                camera.setPoseOverride({ position, rotation, fov: pose.fov, near: camera.near, far: camera.far });
                const pixels = await capture(roundCandidates);
                const background = events.invoke('bgClr');
                const rgba = new Uint8ClampedArray(pixels.length);
                for (let i = 0; i < pixels.length; i += 4) {
                    const alpha = pixels[i + 3] / 255;
                    rgba[i] = pixels[i] + (1 - alpha) * background.r * 255;
                    rgba[i + 1] = pixels[i + 1] + (1 - alpha) * background.g * 255;
                    rgba[i + 2] = pixels[i + 2] + (1 - alpha) * background.b * 255;
                    rgba[i + 3] = 255;
                }
                const image = new ImageData(rgba, width, height);
                engine.restoreMemory(banks[direction]);
                await engine.prepare(image, () => {});
                check();
                const result = await engine.track();
                check();
                const tracked = result.masks[result.selected];
                preview(image, tracked);
                let coverage = 0, accepted = false, reason = '目标丢失或 mask 为空';
                if (result.objectScore > 0 && tracked.some(Boolean)) {
                    const proposed = await project(tracked, candidates);
                    progress(done, `视角 ${done + 1}/6 · 回初始视角自验证…`);
                    restorePose();
                    coverage = initialCoverage(mask, await capture(proposed));
                    if (coverage >= 0.8 && proposed.some(Boolean)) {
                        await engine.acceptMemory(result.selected);
                        check();
                        banks[direction] = engine.snapshotMemory();
                        candidates = proposed;
                        accepted = true; reason = '通过';
                    } else {
                        reason = '初始 mask 覆盖不足 80%，保留原候选与记忆';
                    }
                }
                reports.push({ yaw: angle, accepted, coverage, count: candidates.reduce((n, value) => n + Number(!!value), 0), reason });
                progress(reports.length, `${reports.length}/6 · ${accepted ? '通过' : '拒绝'} · 初始覆盖 ${(coverage * 100).toFixed(1)}% · 剩余 ${reports.at(-1).count.toLocaleString()} 个高斯`);
            }
        }
        check();
        return { mask: candidates, views: reports, initialCount };
    } finally {
        renderer.captureFilter = previousFilter;
        camera.setPoseOverride(previousPose);
        engine.clearView();
        engine.restoreMemory(originalMemory);
        buffer.destroy();
        scene.forceRender = true;
    }
};
