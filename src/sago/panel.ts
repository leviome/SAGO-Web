import { SelectOp } from '../edit-ops';
import { Events } from '../events';
import { Scene } from '../scene';
import { Splat } from '../splat';
import { modelDownloadDialog } from './download-dialog';
import { BrowserSegmenter } from './inference';
import { dilateMask, imagePoint, maskRgba, PromptBox, PromptPoint } from './mask-utils';
import { pickModelDirectory } from './model-directory';
import { createOrientationGate } from './orientation-gate';
import { DroneView, runVirtualDrone } from './virtual-drone';

import './style.scss';

export const registerSago = (scene: Scene, events: Events) => {
    const engine = new BrowserSegmenter();
    const launcher = document.createElement('div');
    launcher.className = 'sago-launcher';
    launcher.innerHTML = `<button id="sago-open">SAGO · 智能分割</button>
        <label>三维场景显示<select id="sago-scene-view" title="前景为当前图层的已选高斯；背景为其余可见高斯。仅影响显示。">
            <option value="0">完整场景</option><option value="1">仅前景（当前选区）</option><option value="2">仅背景</option>
        </select></label><span id="sago-summary">本地 WebGPU · 先导入场景，再开始分割</span>`;
    document.querySelector('#scene-panel .splat-list-container')!.after(launcher);
    // Native controls must not trigger editor keyboard shortcuts.
    launcher.addEventListener('keydown', e => e.stopPropagation());
    launcher.addEventListener('keyup', e => e.stopPropagation());
    const sceneView = launcher.querySelector<HTMLSelectElement>('#sago-scene-view')!;
    const updateSceneView = (reset = false) => {
        const splat = events.invoke('selection') as Splat;
        sceneView.disabled = !splat?.numSelected;
        if (reset || sceneView.disabled) sceneView.value = '0';
        scene.projectedSplatRenderer.selectionPreview = Number(sceneView.value);
        scene.forceRender = true;
    };
    sceneView.onchange = () => updateSceneView();
    events.on('selection.changed', () => updateSceneView(true));
    events.on('splat.stateChanged', () => updateSceneView());
    updateSceneView();

    const dialog = document.createElement('dialog');
    dialog.className = 'sago-dialog';
    dialog.innerHTML = `
        <header><div><strong>SAGO</strong><span>本地智能分割 · SAM2 Large</span></div><button id="sago-close">返回场景 / 取消</button></header>
        <div class="sago-content">
            <div class="sago-stage">
                <label class="sago-preview-toolbar">分割画面显示 <select id="sago-preview">
                    <option value="all">完整画面 + mask</option><option value="foreground">仅前景</option><option value="background">仅背景</option>
                </select></label>
                <canvas id="sago-image" aria-label="分割提示画面"></canvas><p>左键添加提示 · 右键排除背景 · 框选模式拖动绘制</p>
            </div>
            <aside>
                <div class="sago-badge">场景与推理均在本机</div>
                <h2>从画面选中三维对象</h2>
                <p>使用原版 SAM2 Large（FP32）生成 2D mask，再投影为 Gaussian 选区。可换视角重复分割、取交集细化。</p>
                <label>模型来源<select id="sago-source"><option value="local">SAM2 Large · 本站下载，在本机运行</option></select></label>
                <button id="sago-directory">选择本机模型文件夹（跨域名复用）</button>
                <button id="sago-browser-storage" hidden>改用浏览器缓存</button>
                <output id="sago-directory-status">当前使用浏览器缓存，仅限此域名。选择本机文件夹后，加载模型时会保存或复用文件。</output>
                <button id="sago-load">1. 加载模型并编码画面</button>
                <label>提示方式<select id="sago-mode"><option value="positive">前景点 ＋</option><option value="negative">背景点 −</option><option value="box">框提示（SAM2 原生）</option></select></label>
                <div class="sago-row"><button id="sago-pop">撤回提示</button><button id="sago-reset">清空提示</button></div>
                <button id="sago-predict" class="primary">2. 预览 AI 分割</button>
                <div class="sago-memory">
                    <output id="sago-memory-status">尚未建立对象记忆</output>
                    <button id="sago-track">用记忆跟踪此视角（无需打点）</button>
                    <button id="sago-remember">记住当前目标</button>
                    <button id="sago-forget">清除对象记忆</button>
                </div>
                <label>候选 mask<select id="sago-candidate"><option>尚无结果</option></select></label>
                <div id="sago-candidates" class="sago-candidates" aria-label="候选 mask 对比"></div>
                <p class="sago-note">模型评分不是准确率，也不表示对象完整程度。点 / 框分割默认 level 0；记忆跟踪在 level 1–3 中推荐。请对比面积和缩略图，确认目标完整后再保存记忆。</p>
                <label>边缘扩展（像素）<input id="sago-padding" type="number" min="0" max="12" value="2"></label>
                <label>应用方式<select id="sago-op"><option value="set">替换当前选区</option><option value="intersect">与已有选区取交集（多视角细化）</option><option value="add">加入当前选区</option><option value="remove">从选区减去</option></select></label>
                <button id="sago-apply" class="primary">3. 应用为三维选区</button>
                <div class="sago-drone">
                    <button id="sago-drone" class="primary">Virtual Drone · 自动多视角细化并应用</button>
                    <p class="sago-note">以当前 mask 为目标，双向巡航 6 个视角，独立记忆与 80% 初始覆盖检查。完成后按上方应用方式写入选区，可一次撤销。</p>
                    <progress id="sago-drone-progress" max="6" value="0" hidden></progress>
                    <button id="sago-drone-cancel" hidden>取消巡航（保留原选区）</button>
                    <output id="sago-drone-status" aria-live="polite"></output>
                </div>
                <button id="sago-save-mask">下载 2D mask</button>
                <button id="sago-debug">导出诊断样本（本地 JSON）</button>
                <output id="sago-status" role="status" aria-live="polite"></output>
                <p class="sago-note">支持 SAM2 原生框提示，可搭配前景 / 背景点。首次加载会询问是否下载模型并显示进度。先分割并记住目标，再返回场景小幅旋转，加载新画面后用记忆跟踪；确认结果后更新记忆。目标丢失时补点重新建立记忆。建议使用独立显卡。投影会包含视线后方的高斯；建议完整框住对象，并换视角取交集。应用后可撤销、修正、分离和导出。</p>
            </aside>
        </div>`;
    document.body.append(dialog);
    const element = <T extends HTMLElement>(id: string) => dialog.querySelector<T>(`#sago-${id}`)!;
    const openButton = launcher.querySelector<HTMLButtonElement>('button')!;
    const summary = launcher.querySelector<HTMLSpanElement>('span')!;
    const orientation = createOrientationGate(scene, events, () => openButton.click());
    const canvas = element<HTMLCanvasElement>('image');
    const ctx = canvas.getContext('2d')!;
    const status = element<HTMLOutputElement>('status');
    const source = element<HTMLSelectElement>('source');
    const candidate = element<HTMLSelectElement>('candidate');
    const padding = element<HTMLInputElement>('padding');
    const previewMode = element<HTMLSelectElement>('preview');
    const op = element<HTMLSelectElement>('op');
    const base = document.createElement('canvas');
    const overlay = document.createElement('canvas');
    const maskCanvas = document.createElement('canvas');
    let image: ImageData;
    let points: PromptPoint[] = [];
    let box: PromptBox = null;
    let history: Array<{ points: PromptPoint[]; box: PromptBox }> = [];
    let masks: Uint8Array[] = [];
    let scores: number[] = [];
    let ready = false;
    let busy = false;
    let committing = false;
    let generation = 0;
    let revision = 0;
    let capturedRevision = 0;
    let memorySplat: Splat;
    let memoryEpoch = 0;
    let droneAbort: AbortController;
    let droneReport: DroneView[] = [];
    let modelDirectory: FileSystemDirectoryHandle;
    let capturedSplat: Splat;
    let capturedSize: { width: number; height: number };
    let drag: { x: number; y: number; id: number } = null;

    const message = (text: string, error = false) => {
        status.textContent = text;
        status.classList.toggle('error', error);
    };
    const selectedMask = () => masks[Number(candidate.value)];
    const expandedMask = () => dilateMask(selectedMask(), canvas.width, canvas.height, Number(padding.value) || 0);
    const refresh = () => {
        openButton.disabled = busy;
        for (const id of ['load', 'pop', 'reset', 'mode', 'source', 'padding', 'op', 'directory', 'browser-storage']) {
            (element(id) as HTMLButtonElement).disabled = busy || !image;
        }
        element<HTMLButtonElement>('predict').disabled = busy || !ready || (!points.some(p => p.label === 1) && !box);
        element<HTMLButtonElement>('apply').disabled = busy || !masks.length || (engine.resultIsTracked && engine.objectScore <= 0);
        element<HTMLButtonElement>('track').disabled = busy || !ready || !engine.memoryState.active;
        element<HTMLButtonElement>('remember').disabled = busy || !ready || !masks.length || engine.resultAccepted || engine.objectScore <= 0;
        element<HTMLButtonElement>('drone').disabled = busy || !ready || !masks.length || engine.objectScore <= 0;
        element<HTMLButtonElement>('forget').disabled = busy || !engine.memoryState.active;
        element('remember').textContent = engine.resultIsTracked ? '确认并更新此视角记忆' : '以当前结果建立 / 重建记忆';
        const memory = engine.memoryState;
        element('memory-status').textContent = memory.active ? `已记住目标 · 已确认 ${memory.acceptedViews} 个视角 · 近期记忆 ${memory.recentMemories}/6` : '尚未建立对象记忆';
        element<HTMLButtonElement>('save-mask').disabled = busy || !masks.length;
        element<HTMLButtonElement>('debug').disabled = busy || !image;
        element('candidates').querySelectorAll('button').forEach((button) => {
            button.disabled = busy;
        });
        candidate.disabled = busy || !masks.length;
        previewMode.disabled = busy || !masks.length;
        if (!masks.length) previewMode.value = 'all';
        canvas.style.cursor = busy ? 'wait' : 'crosshair';
    };
    const dropMemory = () => {
        engine.clearMemory(); memorySplat = null; memoryEpoch++;
        queueMicrotask(() => refresh());
    };
    const bumpRevision = () => revision++;
    events.on('edit.apply', () => {
        if (!committing) dropMemory();
    });
    events.on('scene.elementRemoved', (splat: Splat) => {
        if (splat === memorySplat) dropMemory();
    });
    events.on('selection.changed', () => {
        if (memorySplat && events.invoke('selection') !== memorySplat) dropMemory();
    });
    for (const event of ['edit.apply', 'scene.elementAdded', 'scene.elementRemoved', 'selection.changed']) {
        events.on(event, bumpRevision);
    }

    // Work on a display-only copy: inference, memory and export retain the original RGB and mask.
    const paintPreview = (frame: ImageData, mask?: Uint8Array) => {
        if (mask && previewMode.value !== 'all') {
            const pixels = new Uint8ClampedArray(frame.data);
            const foreground = previewMode.value === 'foreground';
            for (let i = 0; i < mask.length; i++) {
                if (Boolean(mask[i]) === foreground) continue;
                const shade = ((Math.floor((i % frame.width) / 16) + Math.floor(i / frame.width / 16)) % 2) ? 28 : 38;
                pixels.set([shade, shade, shade, 255], i * 4);
            }
            ctx.putImageData(new ImageData(pixels, frame.width, frame.height), 0, 0);
        } else {
            ctx.putImageData(frame, 0, 0);
            if (mask) {
                overlay.width = frame.width;
                overlay.height = frame.height;
                overlay.getContext('2d')!.putImageData(new ImageData(maskRgba(mask, true), frame.width, frame.height), 0, 0);
                ctx.drawImage(overlay, 0, 0);
            }
        }
    };
    const draw = () => {
        if (!image) return;
        element('candidates').querySelectorAll('button').forEach((button, i) => {
            button.classList.toggle('active', i === Number(candidate.value));
            button.setAttribute('aria-pressed', String(i === Number(candidate.value)));
        });
        paintPreview(image, masks.length ? expandedMask() : undefined);
        const scale = canvas.width / Math.max(canvas.getBoundingClientRect().width, 1);
        ctx.lineWidth = 2 * scale;
        if (box) {
            ctx.strokeStyle = '#ffce66';
            ctx.strokeRect(box.x0 * canvas.width, box.y0 * canvas.height, (box.x1 - box.x0) * canvas.width, (box.y1 - box.y0) * canvas.height);
        }
        for (const p of points) {
            ctx.beginPath();
            ctx.arc(p.x * canvas.width, p.y * canvas.height, 5 * scale, 0, Math.PI * 2);
            ctx.fillStyle = p.label ? '#43e8af' : '#ff647d';
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.stroke();
        }
    };
    const invalidateMask = () => {
        engine.discardResult();
        masks = [];
        scores = [];
        element('candidates').replaceChildren();
        candidate.replaceChildren(new Option('尚无结果', '0'));
        refresh();
        draw();
    };
    const remember = () => history.push({ points: structuredClone(points), box: box && { ...box } });
    const close = () => {
        generation++;
        dialog.close();
        scene.camera.interactionLocked = false;
        drag = null;
        if (!busy) engine.clearView();
        scene.forceRender = true;
    };
    element('close').onclick = () => {
        if (droneAbort) {
            droneAbort.abort(); return;
        }
        if (!committing) close();
    };
    dialog.addEventListener('cancel', (e) => {
        e.preventDefault();
        if (droneAbort) droneAbort.abort();
        else if (!committing) close();
    });
    // Do not let prompt-session keys reach the editor's shortcut handlers.
    dialog.addEventListener('keydown', e => e.stopPropagation());
    dialog.addEventListener('keyup', e => e.stopPropagation());

    const run = async (work: (token: number) => Promise<void>) => {
        if (busy) return;
        busy = true;
        const token = generation;
        refresh();
        try {
            await work(token);
        } catch (err) {
            if (err.name !== 'AbortError') console.error('[SAGO]', err);
            if (token === generation && err.name === 'AbortError') message('已取消下载，已完成的模型文件可在下次继续使用。');
            else if (token === generation) message(`${err.message ?? err}。请检查模型下载和 WebGPU 状态；Large 模型建议使用独立显卡。`, true);
        } finally {
            busy = false;
            if (token !== generation) engine.clearView();
            refresh();
        }
    };

    openButton.onclick = async () => {
        if (busy) return;
        const splat = events.invoke('selection') as Splat;
        if (!splat?.visible) {
            summary.textContent = '请先打开 PLY，并选中一个可见图层。';
            return;
        }
        events.fire('timeline.setPlaying', false);
        if (!await orientation.require(splat)) return;
        if (events.invoke('selection') !== splat || !splat.scene || !splat.visible) return;
        if (memorySplat && memorySplat !== splat) dropMemory();
        events.fire('tool.deactivate');
        events.fire('timeline.setPlaying', false);
        scene.camera.interactionLocked = true;
        capturedSplat = splat;
        capturedRevision = revision;
        capturedSize = { ...scene.targetSize };
        image = null;
        ready = false;
        points = [];
        box = null;
        history = [];
        masks = [];
        scores = [];
        element('drone-progress').hidden = true;
        element('candidates').replaceChildren();
        candidate.replaceChildren(new Option('尚无结果', '0'));
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        dialog.showModal();
        message('正在捕获不含编辑器覆盖层的画面…');
        run(async (token) => {
            await events.invoke('queue', () => {});
            if (token !== generation) return;
            if (!orientation.isConfirmed(splat)) {
                summary.textContent = '场景方向已变化，请重新检查坐标系。';
                close();
                return;
            }
            capturedRevision = revision;
            const { width, height } = capturedSize;
            const pixels = await events.invoke('render.offscreen', width, height) as Uint8Array;
            if (token !== generation) return;
            // Composite the transparent capture over the editor background before SAM sees it.
            const background = events.invoke('bgClr');
            const rgba = new Uint8ClampedArray(pixels.length);
            for (let i = 0; i < pixels.length; i += 4) {
                const alpha = pixels[i + 3] / 255;
                for (let c = 0; c < 3; c++) rgba[i + c] = pixels[i + c] + (1 - alpha) * background[['r', 'g', 'b'][c]] * 255;
                rgba[i + 3] = 255;
            }
            canvas.width = base.width = width;
            canvas.height = base.height = height;
            image = new ImageData(rgba, width, height);
            base.getContext('2d')!.putImageData(image, 0, 0);
            draw();
            message(`已固定 ${width} × ${height} 画面。加载模型后，点击前景或拖框，再预览分割。`);
        });
    };
    const showDirectory = () => {
        element('directory-status').textContent = modelDirectory ? `本机目录：${modelDirectory.name}。点击“加载模型”保存 / 复用。刷新或换域名后重新选择同一目录即可，无需重复下载。` : '当前使用浏览器缓存，仅限此域名。选择本机文件夹后，加载模型时会保存或复用文件。';
        element('browser-storage').hidden = !modelDirectory;
    };
    element('directory').onclick = () => run(async () => {
        try {
            modelDirectory = await pickModelDirectory();
            showDirectory();
            message('模型目录已选择。点击“加载模型”会复用已有文件，并把浏览器中已缓存的权重保存到此目录。');
        } catch (error) {
            if (error.name !== 'AbortError') throw error;
        }
    });
    element('browser-storage').onclick = () => {
        modelDirectory = null;
        showDirectory();
    };
    element('load').onclick = () => run(async (token) => {
        ready = false;
        invalidateMask();
        const progress = (text: string) => {
            if (token === generation) message(text);
        };
        const download = modelDownloadDialog(modelDirectory);
        try {
            await engine.load('local', (text) => {
                progress(text); download.initializing(text);
            }, download.options);
        } finally {
            download.close();
        }
        if (token !== generation) return;
        await engine.prepare(image, progress);
        if (token !== generation) return;
        ready = true;
        message('模型已就绪。添加前景 / 背景点或框提示，点击“预览 AI 分割”。');
    });
    source.onchange = () => {
        ready = false; engine.clearView(); invalidateMask();
    };
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    canvas.addEventListener('pointerdown', (e) => {
        if (busy || !image || (e.button !== 0 && e.button !== 2)) return;
        e.preventDefault();
        const p = imagePoint(e.clientX, e.clientY, canvas.getBoundingClientRect());
        remember();
        if (element<HTMLSelectElement>('mode').value === 'box' && e.button === 0) {
            drag = { ...p, id: e.pointerId };
            canvas.setPointerCapture(e.pointerId);
            box = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
        } else {
            points.push({ ...p, label: e.button === 2 || element<HTMLSelectElement>('mode').value === 'negative' ? 0 : 1 });
        }
        invalidateMask();
    });
    canvas.addEventListener('pointermove', (e) => {
        if (!drag || drag.id !== e.pointerId) return;
        const p = imagePoint(e.clientX, e.clientY, canvas.getBoundingClientRect());
        box = { x0: drag.x, y0: drag.y, x1: p.x, y1: p.y };
        draw();
    });
    const finishDrag = (e: PointerEvent) => {
        if (!drag || drag.id !== e.pointerId) return;
        if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
        if (e.type === 'pointercancel' || Math.abs(box.x1 - box.x0) * canvas.width < 3 || Math.abs(box.y1 - box.y0) * canvas.height < 3) {
            const prev = history.pop();
            points = prev.points;
            box = prev.box;
        }
        drag = null;
        invalidateMask();
    };
    canvas.addEventListener('pointerup', finishDrag);
    canvas.addEventListener('pointercancel', finishDrag);
    element('pop').onclick = () => {
        const prev = history.pop();
        if (prev) {
            points = prev.points; box = prev.box; invalidateMask();
        }
    };
    element('reset').onclick = () => {
        remember(); points = []; box = null; invalidateMask();
    };
    const showResult = (result: Awaited<ReturnType<BrowserSegmenter['predict']>>, start: number) => {
        masks = result.masks;
        scores = result.scores;
        const areas = masks.map(mask => mask.reduce((a, b) => a + Number(b > 0), 0));
        candidate.replaceChildren(...scores.map((score, i) => new Option(`level ${i} · 评分 ${score.toFixed(3)} · 面积 ${(areas[i] / (canvas.width * canvas.height) * 100).toFixed(1)}%`, String(i))));
        candidate.value = String(result.selected);
        element('candidates').replaceChildren(...masks.map((mask, i) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.title = `查看level ${i}，面积 ${areas[i].toLocaleString()} 像素`;
            const thumbnail = document.createElement('canvas');
            thumbnail.width = 150;
            thumbnail.height = Math.max(1, Math.round(150 * canvas.height / canvas.width));
            const preview = document.createElement('canvas');
            preview.width = canvas.width;
            preview.height = canvas.height;
            preview.getContext('2d')!.putImageData(new ImageData(maskRgba(mask, true), canvas.width, canvas.height), 0, 0);
            const context = thumbnail.getContext('2d')!;
            context.drawImage(base, 0, 0, thumbnail.width, thumbnail.height);
            context.drawImage(preview, 0, 0, thumbnail.width, thumbnail.height);
            const label = document.createElement('span');
            label.textContent = `${i} · ${(areas[i] / (canvas.width * canvas.height) * 100).toFixed(1)}%`;
            button.append(thumbnail, label);
            button.onclick = () => {
                candidate.value = String(i); draw();
            };
            return button;
        }));
        draw();
        message(result.tracked && result.objectScore <= 0 ? '当前视角未找到目标，记忆未更新。请返回较近视角，或补点重新分割并建立记忆。' :
            `${result.tracked ? '记忆跟踪完成' : '分割完成'}，用时 ${((performance.now() - start) / 1000).toFixed(2)} 秒。已选择 level ${result.selected}。${result.tracked ? '确认目标正确后再更新记忆。' : '可将此结果建立为对象记忆。'}`);
    };
    element('predict').onclick = () => run(async (token) => {
        invalidateMask();
        message('WebGPU 正在生成 mask…');
        const start = performance.now();
        const result = await engine.predict(points, box);
        if (token === generation) showResult(result, start);
    });
    element('track').onclick = () => run(async (token) => {
        invalidateMask();
        const epoch = memoryEpoch;
        const start = performance.now();
        message('正在融合跨视角记忆并跟踪目标…');
        const result = await engine.track();
        if (token !== generation) return;
        if (epoch !== memoryEpoch || events.invoke('selection') !== memorySplat) throw new Error('对象或场景已变化，请重新建立记忆。');
        points = []; box = null; history = [];
        showResult(result, start);
    });
    element('remember').onclick = () => run(async (token) => {
        if (revision !== capturedRevision || events.invoke('selection') !== capturedSplat || !capturedSplat.scene) throw new Error('场景已变化，请重新捕获画面。');
        const epoch = memoryEpoch;
        message('正在编码并保存对象记忆…');
        await engine.acceptMemory(Number(candidate.value));
        if (token !== generation || epoch !== memoryEpoch || revision !== capturedRevision) {
            dropMemory();
            if (token === generation) throw new Error('场景已变化，已丢弃未确认记忆。');
            return;
        }
        memorySplat = capturedSplat;
        message('对象记忆已保存。可以应用选区，或返回场景小幅旋转，再加载画面并点击“用记忆跟踪此视角”。');
    });
    element('forget').onclick = () => {
        dropMemory();
        invalidateMask();
        message('已清除对象记忆。可重新点选或框选另一个目标。');
    };
    candidate.onchange = draw;
    previewMode.onchange = draw;
    padding.oninput = draw;

    element('drone-cancel').onclick = () => droneAbort?.abort();
    element('drone').onclick = () => run(async () => {
        const epoch = memoryEpoch;
        const controller = new AbortController();
        const check = () => {
            controller.signal.throwIfAborted();
            if (revision !== capturedRevision || epoch !== memoryEpoch || events.invoke('selection') !== capturedSplat || !capturedSplat.scene || !capturedSplat.visible) {
                throw new Error('场景或对象已变化，巡航结果已丢弃。');
            }
            if (capturedSize.width !== scene.targetSize.width || capturedSize.height !== scene.targetSize.height) throw new Error('视口尺寸已变化，巡航结果已丢弃。');
        };
        check();
        if (op.value === 'intersect' && !capturedSplat.numSelected) throw new Error('取交集需要已有选区。');
        droneAbort = controller;
        droneReport = [];
        const progress = element<HTMLProgressElement>('drone-progress');
        progress.value = 0; progress.hidden = false;
        element('drone-cancel').hidden = false;
        try {
            const result = await runVirtualDrone({
                scene,
                splat: capturedSplat,
                engine,
                mask: selectedMask().slice(),
                width: canvas.width,
                height: canvas.height,
                level: Number(candidate.value),
                padding: Number(padding.value) || 0,
                check,
                progress: (done, text) => {
                    progress.value = done;
                    element('drone-status').textContent = text;
                    message(text);
                },
                preview: (frame, mask) => {
                    paintPreview(frame, mask);
                }
            });
            check();
            droneReport = result.views;
            committing = true;
            element<HTMLButtonElement>('close').disabled = true;
            // Exactly one edit, after all views and restoration have succeeded.
            await events.invoke('queue', () => {
                check();
                events.fire('edit.add', new SelectOp(capturedSplat, op.value as 'set' | 'add' | 'remove' | 'intersect', result.mask));
            });
            await events.invoke('queue', () => {});
            const accepted = result.views.filter(view => view.accepted).length;
            summary.textContent = `Virtual Drone 完成 · ${accepted}/6 视角通过 · 候选 ${result.initialCount.toLocaleString()} → ${result.views.at(-1).count.toLocaleString()} · 已选 ${capturedSplat.numSelected.toLocaleString()} 个高斯 · 可一次撤销`;
            element('drone-status').textContent = summary.textContent;
            close();
        } catch (error) {
            if (error.name !== 'AbortError') throw error;
            message('巡航已取消，相机、对象记忆和原选区已恢复。重新加载画面后可再次运行。');
            element('drone-status').textContent = '已取消，未应用选区';
        } finally {
            if (epoch !== memoryEpoch) engine.clearMemory();
            droneAbort = null;
            committing = false;
            ready = false;
            invalidateMask();
            element('drone-cancel').hidden = true;
            element<HTMLButtonElement>('close').disabled = false;
        }
    });

    const paintMask = () => {
        const data = expandedMask();
        if (!data.some(Boolean)) throw new Error('当前 mask 为空，请调整提示或切换候选。');
        maskCanvas.width = canvas.width;
        maskCanvas.height = canvas.height;
        maskCanvas.getContext('2d')!.putImageData(new ImageData(maskRgba(data), canvas.width, canvas.height), 0, 0);
    };
    element('apply').onclick = () => run(async () => {
        if (revision !== capturedRevision || events.invoke('selection') !== capturedSplat || !capturedSplat.scene || !capturedSplat.visible) {
            throw new Error('场景已变化，请返回并重新捕获画面。');
        }
        if (capturedSize.width !== scene.targetSize.width || capturedSize.height !== scene.targetSize.height) {
            throw new Error('视口尺寸已变化，请返回并重新捕获画面。');
        }
        if (op.value === 'intersect' && !capturedSplat.numSelected) throw new Error('取交集需要已有选区，请先使用“替换当前选区”。');
        paintMask();
        message('正在通过 WebGPU 将 mask 投影为 Gaussian 选区…');
        const depth = events.invoke('selection.useDepth');
        const footprint = events.invoke('selection.footprint');
        committing = true;
        element<HTMLButtonElement>('close').disabled = true;
        try {
            events.fire('selection.setUseDepth', false);
            events.fire('selection.setFootprint', 0);
            await events.invoke('select.byMask', op.value, maskCanvas, maskCanvas.getContext('2d'));
            await events.invoke('queue', () => {}); // also wait for the queued edit history operation
            summary.textContent = `已选 ${capturedSplat.numSelected.toLocaleString()} 个高斯 · 可换视角继续取交集`;
            close();
        } finally {
            committing = false;
            element<HTMLButtonElement>('close').disabled = false;
            events.fire('selection.setUseDepth', depth);
            events.fire('selection.setFootprint', footprint);
        }
    });
    element('save-mask').onclick = () => {
        try {
            paintMask();
            maskCanvas.toBlob((blob) => {
                if (!blob) return;
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = 'sago-mask.png';
                a.click();
                setTimeout(() => URL.revokeObjectURL(a.href), 1000);
            });
        } catch (err) {
            message(err.message, true);
        }
    };
    element('debug').onclick = () => {
        const maskImages = masks.map((mask, i) => {
            const output = document.createElement('canvas');
            output.width = canvas.width;
            output.height = canvas.height;
            output.getContext('2d')!.putImageData(new ImageData(maskRgba(mask), canvas.width, canvas.height), 0, 0);
            return { index: i, score: scores[i], areaPixels: mask.reduce((a, b) => a + Number(b > 0), 0), png: output.toDataURL('image/png') };
        });
        const diagnostic = {
            schema: 'sago-segmentation-debug-v1',
            createdAt: new Date().toISOString(),
            model: { id: 'sago/sam2-hiera-large', checkpointSha256: engine.metadata?.checkpointSha256, dtype: 'fp32', source: source.value, executionProvider: 'webgpu' },
            image: { width: canvas.width, height: canvas.height, png: base.toDataURL('image/png') },
            prompts: { points, box, coordinateSpace: 'normalized-original-image', boxMode: 'native-sam2' },
            memory: { ...engine.memoryState, usedForPrediction: engine.resultIsTracked, objectScore: engine.objectScore },
            virtualDrone: droneReport,
            result: { candidates: maskImages, selected: masks.length ? Number(candidate.value) : null, paddingPixels: Number(padding.value) || 0, masksIncludePadding: false },
            environment: { userAgent: navigator.userAgent, gpu: events.invoke('scene.gpu') },
            note: 'Contains the current image, prompts and masks; no PLY or memory-history tensors. Tracking diagnostics cannot be replayed as independent point prompts. Exported locally, never uploaded automatically.'
        };
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob([JSON.stringify(diagnostic)], { type: 'application/json' }));
        a.download = `sago-debug-${Date.now()}.json`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 1000);
        message('已导出原始画面、提示和全部候选 mask。文件仅下载到本机；可以用作同图同提示的离线对照。');
    };
};
