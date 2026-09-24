export type ModelPart = { file: string; externalData: string[] };
export type ModelManifest = {
    id: string; checkpointSha256: string; dtype: string;
    encoder: ModelPart; decoder: ModelPart; memoryEncoder: ModelPart; memoryFusion: ModelPart;
    memory: { version: number };
    files: Record<string, { bytes: number; sha256: string }>;
};
export type DownloadPlan = { totalBytes: number; cachedBytes: number; downloadBytes: number };
export type DownloadProgress = DownloadPlan & {
    downloadedBytes: number; file: string; phase: 'download' | 'verify' | 'cached' | 'retry';
    activeFiles: number; bytesPerSecond: number; remainingSeconds: number;
};
export type DownloadOptions = {
    confirm: (plan: DownloadPlan) => Promise<boolean>;
    progress?: (progress: DownloadProgress) => void;
    signal?: AbortSignal;
    directory?: FileSystemDirectoryHandle;
};

export class ModelAssets {
    // Bounded parallelism avoids loading multiple model sessions at once. Large
    // files use resumable 8 MiB ranges; small graphs use a single normal request.
    private static readonly chunkBytes = 8 * 1024 * 1024;
    private cache: Cache;
    private plan: DownloadPlan;
    private downloadedBytes = 0;
    private transferredBytes = 0;
    private started = 0;
    private active = new Set<string>();
    private available = new Set<string>();
    private partial = new Map<string, Set<number>>();
    private local = new Set<string>();
    private stop = new AbortController();
    private signal: AbortSignal;
    private constructor(readonly manifest: ModelManifest, private root: URL, private options: DownloadOptions) {
        this.signal = options.signal ? AbortSignal.any([options.signal, this.stop.signal]) : this.stop.signal;
    }

    static async open(options: DownloadOptions) {
        const root = new URL('static/models/sam2-large/', document.baseURI);
        const response = await fetch(new URL('manifest.json', root), { cache: 'no-cache', signal: options.signal });
        if (!response.ok) throw new Error('模型清单不可用，请检查网络后重试。');
        const manifest: ModelManifest = await response.json();
        if (manifest.id !== 'sago/sam2-hiera-large' || manifest.dtype !== 'fp32' || manifest.memory?.version !== 1) {
            throw new Error('请部署包含跨视角 memory 的 SAM2 Large 模型。');
        }
        const assets = new ModelAssets(manifest, root, options);
        try {
            assets.cache = await caches.open('sago-sam2-large-v1');
        } catch { /* Storage is optional; consent is still required for network downloads. */ }
        let totalBytes = 0, cachedBytes = 0;
        // Enumerate keys without opening hundreds of cached response bodies.
        let keys = new Set<string>();
        try {
            keys = new Set((await assets.cache?.keys() ?? []).map(request => request.url));
        } catch {
            assets.cache = null;
        }
        for (const [file, info] of Object.entries(manifest.files)) {
            totalBytes += info.bytes;
            // A selected local folder is portable across origins. Verify it
            // before planning downloads; a stale/corrupt file is not a cache hit.
            if (options.directory) {
                options.signal?.throwIfAborted();
                try {
                    const handle = await options.directory.getFileHandle(file);
                    const diskFile = await handle.getFile();
                    if (diskFile.size === info.bytes && await assets.hash(await diskFile.arrayBuffer()) === info.sha256) assets.local.add(file);
                } catch (error) {
                    if (error.name !== 'NotFoundError') throw new Error('无法读取模型目录，请重新点击“选择本机模型文件夹”授权。');
                }
            }
            if (assets.local.has(file)) {
                cachedBytes += info.bytes;
            } else if (keys.has(assets.url(file).href)) {
                assets.available.add(file);
                cachedBytes += info.bytes;
            } else if (info.bytes >= ModelAssets.chunkBytes * 2) {
                const chunks = new Set<number>();
                for (let start = 0; start < info.bytes; start += ModelAssets.chunkBytes) {
                    if (keys.has(assets.chunkUrl(file, start).href)) {
                        chunks.add(start);
                        cachedBytes += Math.min(ModelAssets.chunkBytes, info.bytes - start);
                    }
                }
                assets.partial.set(file, chunks);
            }
        }
        assets.plan = { totalBytes, cachedBytes, downloadBytes: totalBytes - cachedBytes };
        if (assets.plan.downloadBytes && !await options.confirm(assets.plan)) throw new DOMException('已取消模型下载', 'AbortError');
        options.signal?.throwIfAborted();
        return assets;
    }

    private async hash(data: ArrayBuffer) {
        return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', data)), b => b.toString(16).padStart(2, '0')).join('');
    }

    // Also works after the GPU sessions are already loaded: export existing
    // browser cache into a portable directory without allocating all 957 MB.
    async saveToDirectory() {
        const files = Object.keys(this.manifest.files).filter(file => !this.local.has(file));
        for (let i = 0; i < files.length; i += 4) await this.fetchMany(files.slice(i, i + 4));
    }

    async writeManifest() {
        if (!this.options.directory) return;
        const handle = await this.options.directory.getFileHandle('manifest.json', { create: true });
        const writable = await handle.createWritable();
        try {
            await writable.write(JSON.stringify(this.manifest, null, 2));
            await writable.close();
        } catch (error) {
            await writable.abort().catch(() => {});
            throw error;
        }
    }

    private notify(file: string, phase: DownloadProgress['phase']) {
        const seconds = this.started ? (performance.now() - this.started) / 1000 : 0;
        const bytesPerSecond = seconds > 0 ? this.transferredBytes / seconds : 0;
        this.options.progress?.({
            ...this.plan,
            downloadedBytes: this.downloadedBytes,
            file,
            phase,
            activeFiles: this.active.size,
            bytesPerSecond,
            remainingSeconds: bytesPerSecond ? Math.max(0, this.plan.downloadBytes - this.downloadedBytes) / bytesPerSecond : 0
        });
    }

    private url(file: string) {
        const url = new URL(file, this.root);
        url.searchParams.set('sha256', this.manifest.files[file].sha256);
        return url;
    }

    private chunkUrl(file: string, start: number) {
        const url = this.url(file);
        url.searchParams.set('chunk', String(start));
        return url;
    }

    private async clearChunks(file: string) {
        const bytes = this.manifest.files[file].bytes;
        if (bytes < ModelAssets.chunkBytes * 2) return;
        for (let start = 0; start < bytes; start += ModelAssets.chunkBytes) await this.cache?.delete(this.chunkUrl(file, start));
    }

    async fetchMany(files: string[]) {
        const result: Uint8Array<ArrayBuffer>[] = new Array(files.length);
        let next = 0, failure: unknown;
        const worker = async () => {
            try {
                while (next < files.length) {
                    this.signal.throwIfAborted();
                    const index = next++;
                    result[index] = await this.fetch(files[index]);
                }
            } catch (error) {
                if (!failure) failure = error;
                this.stop.abort();
            }
        };
        // Drain every worker before returning or failing, so cancellation never
        // leaves background downloads racing the next model load.
        await Promise.all(Array.from({ length: Math.min(4, files.length) }, worker));
        if (failure) throw failure;
        return result;
    }

    private async download(file: string, start: number, end: number, ranged: boolean) {
        const entry = this.manifest.files[file];
        for (let attempt = 0; ; attempt++) {
            this.signal.throwIfAborted();
            let credited = 0;
            let reader: ReadableStreamDefaultReader<Uint8Array>;
            let retryable = true;
            // Bound a stalled read as well as connect time. Retrying requests a
            // fresh range; already completed ranges remain cached.
            const timeout = AbortSignal.timeout(120000);
            try {
                const response = await fetch(this.url(file), {
                    signal: AbortSignal.any([this.signal, timeout]),
                    headers: ranged ? { Range: `bytes=${start}-${end}` } : undefined
                });
                if (!response.ok) {
                    retryable = response.status === 408 || response.status === 429 || response.status >= 500;
                    await response.body?.cancel();
                    throw new Error(`模型下载失败：${file} (${response.status})`);
                }
                const full = response.status === 200;
                if (!full && (!ranged || response.status !== 206 || response.headers.get('Content-Range') !== `bytes ${start}-${end}/${entry.bytes}`)) {
                    retryable = false;
                    await response.body?.cancel();
                    throw new Error(`模型分块范围异常：${file}`);
                }
                const data = new Uint8Array(full ? entry.bytes : end - start + 1);
                reader = response.body.getReader();
                let offset = 0;
                for (;;) {
                    this.signal.throwIfAborted();
                    const { value, done } = await reader.read();
                    if (done) break;
                    if (offset + value.length > data.length) throw new Error(`模型文件长度异常：${file}`);
                    data.set(value, offset); offset += value.length;
                    credited += value.length;
                    this.downloadedBytes += value.length;
                    this.transferredBytes += value.length;
                    this.notify(file, 'download');
                }
                if (offset !== data.length) throw new Error(`模型下载不完整：${file}`);
                return { data, full };
            } catch (error) {
                this.downloadedBytes -= credited;
                await reader?.cancel().catch(() => {});
                this.signal.throwIfAborted();
                if (!retryable || attempt >= 2) throw error;
                this.notify(file, 'retry');
                await new Promise<void>((resolve, reject) => {
                    const pending: { timer?: ReturnType<typeof setTimeout> } = {};
                    const abort = () => {
                        clearTimeout(pending.timer); reject(this.signal.reason);
                    };
                    pending.timer = setTimeout(() => {
                        this.signal.removeEventListener('abort', abort); resolve();
                    }, 500 * 2 ** attempt);
                    this.signal.addEventListener('abort', abort, { once: true });
                });
            } finally {
                reader?.releaseLock();
            }
        }
    }

    async fetch(file: string) {
        this.signal.throwIfAborted();
        const entry = this.manifest.files[file];
        const url = this.url(file);
        const disk = this.local.has(file) ? await (await this.options.directory.getFileHandle(file)).getFile() : null;
        const response = disk ? new Response(disk) : await this.cache?.match(url);
        const cached = !!response;
        if (!cached && this.available.has(file)) throw new Error('模型缓存已被浏览器清理，请重新加载并确认下载。');
        let data: Uint8Array<ArrayBuffer>;
        if (cached) {
            this.notify(file, 'cached');
            data = new Uint8Array(await response.arrayBuffer());
        } else {
            data = new Uint8Array(entry.bytes);
            const ranged = entry.bytes >= ModelAssets.chunkBytes * 2;
            const step = ranged ? ModelAssets.chunkBytes : entry.bytes;
            let fetched = 0;
            if (!this.started) this.started = performance.now();
            this.active.add(file);
            try {
                for (let start = 0; start < entry.bytes; start += step) {
                    this.signal.throwIfAborted();
                    const end = Math.min(start + step, entry.bytes) - 1;
                    if (this.partial.get(file)?.has(start)) {
                        const chunk = await this.cache?.match(this.chunkUrl(file, start));
                        if (!chunk) throw new Error('模型分块缓存已被清理，请重新加载并确认下载。');
                        const bytes = new Uint8Array(await chunk.arrayBuffer());
                        if (bytes.length !== end - start + 1) {
                            await this.clearChunks(file);
                            throw new Error('模型分块缓存损坏，已清理，请重新加载。');
                        }
                        data.set(bytes, start);
                        this.notify(file, 'cached');
                        continue;
                    }
                    const received = await this.download(file, start, end, ranged);
                    if (received.full) {
                        // Static hosts may ignore Range. Use their complete 200
                        // response, never append it to a partial model.
                        data = received.data;
                        this.downloadedBytes -= fetched;
                        const reused = [...this.partial.get(file) ?? []].reduce((sum, offset) => sum + Math.min(step, entry.bytes - offset), 0);
                        this.plan.cachedBytes -= reused;
                        this.plan.downloadBytes += reused;
                        break;
                    }
                    data.set(received.data, start);
                    fetched += received.data.length;
                    if (this.cache) {
                        try {
                            await this.cache.put(this.chunkUrl(file, start), new Response(received.data));
                        } catch { /* Storage is optional. */ }
                    }
                }
            } finally {
                this.active.delete(file);
            }
        }
        this.notify(file, 'verify');
        const hash = await this.hash(data.buffer);
        if (data.length !== entry.bytes || hash !== entry.sha256) {
            await this.cache?.delete(url);
            await this.clearChunks(file);
            throw new Error(`模型文件校验失败：${file}。已清除浏览器中的对应缓存，请重试加载。`);
        }
        this.signal.throwIfAborted();
        if (this.options.directory && !disk) {
            const handle = await this.options.directory.getFileHandle(file, { create: true });
            const writable = await handle.createWritable();
            try {
                await writable.write(data);
                this.signal.throwIfAborted();
                await writable.close();
                this.local.add(file);
            } catch (error) {
                await writable.abort().catch(() => {});
                this.signal.throwIfAborted();
                throw new Error(`模型保存到本机目录失败：${error.message}`);
            }
        }
        if (!cached && this.cache) {
            try {
                await this.cache.put(url, new Response(data));
                await this.clearChunks(file);
            } catch { /* A full cache never prevents inference. */ }
        }
        return data;
    }
}
