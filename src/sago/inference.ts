import type { InferenceSession, Tensor } from 'onnxruntime-common';

import { PromptBox, PromptPoint } from './mask-utils';
import { DownloadOptions, ModelAssets, ModelManifest, ModelPart } from './model-assets';
import { sam2Masks, sam2Pixels, sam2Prompts } from './sam2-utils';

type Ort = typeof import('onnxruntime-common');
type MemoryFrame = { memory: Float32Array; pointer: Float32Array };
export type MemorySnapshot = { seed: MemoryFrame; history: MemoryFrame[]; acceptedViews: number };
export class BrowserSegmenter {
    private lib: Ort;
    private encoder: InferenceSession;
    private decoder: InferenceSession;
    private embeddings: Record<string, Tensor>;
    private image: ImageData;
    private device: GPUDevice;
    private deviceError = '';
    metadata: ModelManifest;
    private memoryEncoder: InferenceSession;
    private memoryFusion: InferenceSession;
    private seed: { memory: Float32Array; pointer: Float32Array };
    private history: Array<{ memory: Float32Array; pointer: Float32Array }> = [];
    private acceptedViews = 0;
    private pending: { outputs: Record<string, Tensor>; tracked: boolean; accepted: boolean };

    get memoryState() {
        return { active: !!this.seed, acceptedViews: this.acceptedViews, recentMemories: Math.min(6, this.history.length), recentPointers: this.history.length };
    }
    get resultIsTracked() {
        return this.pending?.tracked ?? false;
    }
    get resultAccepted() {
        return this.pending?.accepted ?? false;
    }
    get objectScore() {
        return this.pending ? Number(this.pending.outputs.object_score.data[0]) : null;
    }

    async load(_source: 'local', progress: (message: string) => void, downloads: DownloadOptions) {
        const loaded = this.encoder && this.decoder && this.memoryEncoder && this.memoryFusion;
        if (loaded && !downloads.directory) return;
        progress('正在检查模型缓存和本机目录…');
        const assets = await ModelAssets.open(downloads);
        this.metadata = assets.manifest;
        if (loaded) {
            await assets.saveToDirectory();
            await assets.writeManifest();
            return;
        }
        if (!navigator.gpu) throw new Error('此浏览器没有 WebGPU，请使用支持 WebGPU 的 Chrome / Edge。');
        if (!this.lib) {
            const url = new URL('static/lib/ort/ort.webgpu.bundle.min.mjs', document.baseURI);
            this.lib = await import(/** @vite-ignore */ url.href);
            this.lib.env.wasm.wasmPaths = new URL('static/lib/ort/', document.baseURI).href;
            this.lib.env.wasm.numThreads = 1;
            this.lib.env.webgpu.powerPreference = 'high-performance';
        }
        this.clearView();
        const create = async (part: ModelPart, title: string) => {
            progress(`加载 SAM2 Large ${title}：读取本机文件，缺失部分 4 路并发下载…`);
            const [graph, ...weights] = await assets.fetchMany([part.file, ...part.externalData]);
            const externalData = part.externalData.map((path, i) => ({ path, data: weights[i] }));
            downloads.signal?.throwIfAborted();
            progress(`正在初始化 SAM2 Large ${title} WebGPU…`);
            return this.lib.InferenceSession.create(graph, { executionProviders: ['webgpu'], externalData });
        };
        try {
            this.encoder = await create(this.metadata.encoder, '图像编码器');
            this.device = await this.lib.env.webgpu.device;
            this.device.addEventListener('uncapturederror', (event) => {
                this.deviceError = event.error.message;
            });
            this.device.lost.then((info) => {
                this.deviceError = `GPU 连接已断开：${info.message}。请刷新页面重试。`;
            });
            this.decoder = await create(this.metadata.decoder, '掩码解码器');
            this.memoryEncoder = await create(this.metadata.memoryEncoder, '记忆编码器');
            this.memoryFusion = await create(this.metadata.memoryFusion, '记忆融合器');
            downloads.signal?.throwIfAborted();
            await assets.writeManifest();
        } catch (error) {
            await this.encoder?.release();
            await this.decoder?.release();
            await this.memoryEncoder?.release();
            await this.memoryFusion?.release();
            this.memoryEncoder = null;
            this.memoryFusion = null;
            this.encoder = null;
            this.decoder = null;
            throw error;
        }
    }

    async prepare(image: ImageData, progress: (message: string) => void) {
        if (!this.encoder) throw new Error('请先加载 SAM2 Large。');
        this.clearView();
        progress('SAM2 Large 正在编码当前画面，首次运行需要编译 WebGPU 着色器…');
        const tensor = new this.lib.Tensor('float32', sam2Pixels(image.data, image.width, image.height), [1, 3, 1024, 1024]);
        try {
            this.embeddings = await this.checkedRun(this.encoder, { pixel_values: tensor });
            this.image = image;
        } finally {
            tensor.dispose();
        }
    }

    predict(points: PromptPoint[], box: PromptBox | null) {
        return this.decode(sam2Prompts(points, box), false);
    }

    async track() {
        if (!this.seed || !this.embeddings) throw new Error('请先在一个视角分割目标并建立记忆。');
        this.discardResult();
        const frames = [this.seed, ...this.history.slice(0, 6)];
        const memories = new Float32Array(frames.length * 64 * 64 * 64);
        frames.forEach((frame, i) => memories.set(frame.memory, i * 64 * 64 * 64));
        const pointers = new Float32Array((1 + this.history.length) * 256);
        [this.seed, ...this.history].forEach((frame, i) => pointers.set(frame.pointer, i * 256));
        const feeds = {
            memories: new this.lib.Tensor('float32', memories, [frames.length, 64, 64, 64]),
            memory_indices: new this.lib.Tensor('int64', BigInt64Array.from([6, ...this.history.slice(0, 6).map((_, i) => i)], BigInt), [frames.length]),
            pointers: new this.lib.Tensor('float32', pointers, [1, 1 + this.history.length, 256])
        };
        let fused: Record<string, Tensor>;
        try {
            fused = await this.checkedRun(this.memoryFusion, { features: this.embeddings.features, ...feeds });
            // Original video decoder uses two padding tokens and selects levels 1–3.
            return await this.decode({ coords: new Float32Array(4), labels: BigInt64Array.of(4n, 4n) }, true, fused.fused_features);
        } finally {
            Object.values(feeds).forEach(t => t.dispose());
            if (fused) Object.values(fused).forEach(t => t.dispose());
        }
    }

    private async decode(prompts: { coords: Float32Array; labels: BigInt64Array }, tracked: boolean, features?: Tensor) {
        if (!this.image || !this.embeddings) throw new Error('请先加载模型并编码画面。');
        this.discardResult();
        const { coords, labels } = prompts;
        const pointCoords = new this.lib.Tensor('float32', coords, [1, labels.length, 2]);
        const pointLabels = new this.lib.Tensor('int64', labels, [1, labels.length]);
        let outputs: Record<string, Tensor>;
        try {
            outputs = await this.checkedRun(this.decoder, { ...this.embeddings,
                features: features ?? this.embeddings.features,
                point_coords: pointCoords,
                point_labels: pointLabels });
            const scores = Array.from(outputs.scores.data, Number);
            const logits = outputs.masks.data as Float32Array;
            if (scores.length !== 4 || !scores.every(Number.isFinite) || !logits.every(Number.isFinite) ||
                !Number.isFinite(Number(outputs.object_score.data[0]))) throw new Error('SAM2 返回了无效结果。');
            const masks = sam2Masks(logits, scores.length, this.image.width, this.image.height);
            const selected = tracked ? 1 + scores.slice(1).indexOf(Math.max(...scores.slice(1))) : 0;
            this.pending = { outputs, tracked, accepted: false };
            outputs = null;
            return { masks, scores, selected, objectScore: this.objectScore, tracked };
        } finally {
            pointCoords.dispose(); pointLabels.dispose();
            if (outputs) Object.values(outputs).forEach(t => t.dispose());
        }
    }

    async acceptMemory(level: number, asSeed = false) {
        if (!this.pending || !this.embeddings || (this.pending.accepted && !asSeed)) throw new Error('请先生成新的分割结果。');
        if (!Number.isInteger(level) || level < 0 || level > 3) throw new Error('无效候选层级。');
        if (this.objectScore <= 0) throw new Error('未找到目标，记忆保持不变；请补点重新分割。');
        const pending = this.pending;
        const logits = (pending.outputs.masks.data as Float32Array).slice(level * 256 * 256, (level + 1) * 256 * 256);
        if (!logits.some(v => v > 0)) throw new Error('空掩码不能写入记忆。');
        const mask = new this.lib.Tensor('float32', logits, [1, 1, 256, 256]);
        const isPrompt = new this.lib.Tensor('float32', Float32Array.of(pending.tracked && !asSeed ? 0 : 1), [1]);
        let encoded: Record<string, Tensor>;
        try {
            encoded = await this.checkedRun(this.memoryEncoder, { features: this.embeddings.features, mask, is_prompt: isPrompt });
            const memory = Float32Array.from(encoded.memory.data as Float32Array);
            const pointer = (pending.outputs.pointers.data as Float32Array).slice(level * 256, (level + 1) * 256);
            if (!memory.every(Number.isFinite) || !pointer.every(Number.isFinite)) throw new Error('记忆编码无效。');
            if (!pending.tracked || asSeed) {
                this.clearMemory(); this.seed = { memory, pointer }; this.acceptedViews = 1;
            } else {
                this.history.unshift({ memory, pointer }); this.history.splice(15);
                this.history.slice(6).forEach((frame) => {
                    frame.memory = null;
                });
                this.acceptedViews++;
            }
            pending.accepted = true;
        } finally {
            mask.dispose(); isPrompt.dispose();
            if (encoded) Object.values(encoded).forEach(t => t.dispose());
        }
    }

    clearMemory() {
        this.seed = null; this.history = []; this.acceptedViews = 0;
    }

    // Tensor data is immutable; copy frame records because eviction clears their
    // memory field. Each drone can advance without modifying another history.
    snapshotMemory(): MemorySnapshot {
        return { seed: this.seed && { ...this.seed }, history: this.history.map(frame => ({ ...frame })), acceptedViews: this.acceptedViews };
    }

    restoreMemory(snapshot: MemorySnapshot) {
        this.seed = snapshot.seed && { ...snapshot.seed };
        this.history = snapshot.history.map(frame => ({ ...frame }));
        this.acceptedViews = snapshot.acceptedViews;
    }

    discardResult() {
        if (this.pending) Object.values(this.pending.outputs).forEach(t => t.dispose());
        this.pending = null;
    }

    private async checkedRun(session: InferenceSession, inputs: Record<string, Tensor>) {
        if (this.deviceError) throw new Error(this.deviceError);
        this.device.pushErrorScope('out-of-memory');
        this.device.pushErrorScope('validation');
        let outputs: Record<string, Tensor>;
        let failure: unknown;
        try {
            outputs = await session.run(inputs);
        } catch (error) {
            failure = error;
        }
        const validation = await this.device.popErrorScope();
        const memory = await this.device.popErrorScope();
        if (failure || validation || memory || this.deviceError) {
            if (outputs) Object.values(outputs).forEach(t => t.dispose());
            throw new Error(String(validation?.message || memory?.message || this.deviceError || failure));
        }
        return outputs;
    }

    clearView() {
        this.discardResult();
        if (this.embeddings) Object.values(this.embeddings).forEach(t => t.dispose());
        this.embeddings = null;
        this.image = null;
    }
}
