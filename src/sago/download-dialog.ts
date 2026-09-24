import { DownloadOptions } from './model-assets';

const mb = (bytes: number) => `${(bytes / 1e6).toFixed(1)} MB`;

export const modelDownloadDialog = (directory?: FileSystemDirectoryHandle) => {
    const dialog = document.createElement('dialog');
    dialog.className = 'sago-download-dialog';
    dialog.setAttribute('aria-labelledby', 'sago-download-title');
    dialog.innerHTML = `<h2 id="sago-download-title">下载 SAM2 Large 模型？</h2>
        <p id="sago-download-description"></p>
        <p>模型在本机运行。使用 4 路并发下载，大文件按 8 MiB 分块续传，网络波动时自动重试。已完成文件及完整分块会尽可能缓存。</p>
        <progress id="sago-download-progress" max="1" value="0" aria-label="模型下载进度" hidden></progress>
        <output id="sago-download-status" role="status" aria-live="polite"></output>
        <div><button id="sago-download-cancel">暂不下载</button><button id="sago-download-confirm">下载模型</button></div>`;
    document.body.append(dialog);
    const get = <T extends HTMLElement>(id: string) => dialog.querySelector<T>(`#sago-download-${id}`)!;
    const bar = get<HTMLProgressElement>('progress');
    const confirm = get<HTMLButtonElement>('confirm');
    const cancel = get<HTMLButtonElement>('cancel');
    const controller = new AbortController();
    let resolveConfirmation: (value: boolean) => void;
    const stop = () => {
        controller.abort();
        resolveConfirmation?.(false);
        resolveConfirmation = null;
        dialog.close();
    };
    cancel.onclick = stop;
    dialog.addEventListener('cancel', (event) => {
        event.preventDefault(); stop();
    });
    dialog.addEventListener('keydown', e => e.stopPropagation());
    dialog.addEventListener('keyup', e => e.stopPropagation());
    const options: DownloadOptions = {
        directory,
        signal: controller.signal,
        confirm: plan => new Promise((resolve) => {
            resolveConfirmation = resolve;
            get('description').textContent = `模型总大小 ${mb(plan.totalBytes)}；已缓存 / 本机已有 ${mb(plan.cachedBytes)}；本次需要下载 ${mb(plan.downloadBytes)}。${directory ? `校验后保存到所选目录 ${directory.name}，换域名后可重新选择该目录复用。` : ''}`;
            dialog.showModal();
            confirm.onclick = () => {
                confirm.hidden = true;
                cancel.textContent = '取消下载';
                bar.hidden = false;
                get('title').textContent = '正在下载模型';
                resolveConfirmation = null;
                resolve(true);
            };
        }),
        progress: (state) => {
            if (controller.signal.aborted) return;
            if (!dialog.open) {
                confirm.hidden = true;
                cancel.textContent = '取消加载';
                bar.hidden = false;
                get('title').textContent = '正在读取本机模型';
                dialog.showModal();
            }
            bar.value = state.downloadBytes ? Math.min(1, state.downloadedBytes / state.downloadBytes) : 1;
            const percent = Math.floor(bar.value * 100);
            const eta = state.bytesPerSecond ? `预计剩余 ${state.remainingSeconds < 60 ? `${Math.ceil(state.remainingSeconds)} 秒` : `${Math.ceil(state.remainingSeconds / 60)} 分钟`}` : '正在测速…';
            const detail = `${percent}% · ${mb(Math.min(state.downloadedBytes, state.downloadBytes))} / ${mb(state.downloadBytes)}\n${mb(state.bytesPerSecond)}/s · ${eta} · ${state.activeFiles} 路下载中`;
            const phase = state.phase === 'retry' ? '网络波动，正在重试' : state.phase === 'verify' ? '正在校验' : state.phase === 'cached' ? '正在读取缓存' : '正在下载';
            get('status').textContent = `${detail}\n${phase}：${state.file}`;
        }
    };
    return {
        options,
        initializing: (text: string) => {
            get('status').textContent = text;
        },
        close: () => {
            resolveConfirmation?.(false); dialog.close(); dialog.remove();
        }
    };
};
