import { Vec3 } from 'playcanvas';

import { Events } from '../events';
import { Scene } from '../scene';
import { Splat } from '../splat';

// Three sampled points define an unoriented plane. Its tilt can be checked,
// but which side is physically "up" still requires the user's confirmation.
const groundPlane = (splat: Splat) => {
    if (!splat.orientPoints.length) return { valid: true, text: '尚未采样地面。请观察网格与场景，确认世界 +Y 朝上、地面与 XZ 平面平行。' };
    if (splat.orientPoints.length !== 3) return { valid: false, text: `已选 ${splat.orientPoints.length}/3 个地面点，请继续取点；没有地面的模型可在方向工具中清除取点，再人工确认。` };
    const [a, b, c] = splat.orientPoints.map(point => splat.worldTransform.transformPoint(point, new Vec3()));
    const u = b.sub(a), v = c.sub(a);
    const normal = new Vec3().cross(u, v);
    const scale = u.length() * v.length();
    if (!Number.isFinite(scale) || !scale || normal.length() <= scale * 1e-6) {
        return { valid: false, text: '这三个点无法确定地面，请清除后重新选择三个不共线的地面点。' };
    }
    normal.normalize();
    const tilt = Math.acos(Math.min(1, Math.abs(normal.y))) * 180 / Math.PI;
    return { valid: tilt <= 5, text: `所选地面与 XZ 平面的夹角：${tilt.toFixed(1)}°（需 ≤ 5°）。${tilt <= 5 ? '请继续确认场景没有上下颠倒。' : '请先使用“对齐到网格”校正。'}` };
};

export const createOrientationGate = (scene: Scene, events: Events, review: () => void) => {
    const approved = new WeakMap<Splat, string>();
    const signature = (splat: Splat) => `${Array.from(splat.worldTransform.data).join(',')}|${splat.orientPoints.map(point => point.toString()).join(';')}`;
    const isConfirmed = (splat: Splat) => !!splat?.scene && approved.get(splat) === signature(splat);
    const dialog = document.createElement('dialog');
    dialog.className = 'sago-orientation-dialog';
    dialog.setAttribute('aria-labelledby', 'sago-orientation-title');
    dialog.innerHTML = `
        <h2 id="sago-orientation-title">进入分割前，先确认场景坐标系</h2>
        <p>当前图层：<strong id="sago-orientation-layer"></strong></p>
        <p>请将<strong>地面法向对齐世界 +Y</strong>，让地面平行于 <strong>XZ 网格</strong>。Virtual Drone 将围绕 +Y 巡航；方向错误会影响自动视角和分割结果。</p>
        <p>需要校正时，从地面上方观察，在真实地面选三个不共线的点，再点击“对齐到网格”。该操作会旋转、平移当前图层，可撤销。</p>
        <output id="sago-orientation-status" role="status"></output>
        <label><input id="sago-orientation-confirm" type="checkbox">我已检查：世界 +Y 朝上，地面与 XZ 平面平行，场景没有上下颠倒。</label>
        <p class="sago-note">仅旋转相机或设置局部枢轴，不会校正场景坐标系。无地面的模型请按其实际上方向人工确认。</p>
        <div class="sago-orientation-actions">
            <button id="sago-orientation-adjust">选三点校正地面</button>
            <button id="sago-orientation-cancel">返回场景</button>
            <button id="sago-orientation-enter" disabled>确认并进入分割</button>
        </div>`;
    const guide = document.createElement('section');
    guide.className = 'sago-orientation-guide';
    guide.hidden = true;
    guide.innerHTML = `<strong>SAGO · 校正地面方向</strong>
        <p>从地面上方观察 → 选择三个不共线的地面点 → 点击下方工具栏“对齐到网格”。请确认场景正立，再进入分割。</p>
        <p>本次以 XZ 网格为地面、+Y 为上方向；校正当前图层，可撤销。</p>
        <button id="sago-orientation-review">完成调整，检查坐标系</button>
        <button id="sago-orientation-hide-guide">收起指引</button>`;
    document.body.append(dialog, guide);
    const checkbox = dialog.querySelector<HTMLInputElement>('#sago-orientation-confirm')!;
    const enter = dialog.querySelector<HTMLButtonElement>('#sago-orientation-enter')!;
    const status = dialog.querySelector<HTMLOutputElement>('#sago-orientation-status')!;
    let target: Splat;
    let resolver: (value: boolean) => void;
    let shownSignature: string;
    const finish = (value: boolean) => {
        dialog.close();
        const resolve = resolver;
        resolver = null;
        resolve?.(value);
    };
    const refresh = () => {
        if (!dialog.open) return;
        if (!target?.scene || !target.visible || events.invoke('selection') !== target) {
            finish(false); return;
        }
        const current = signature(target);
        if (shownSignature !== current) {
            checkbox.checked = false; shownSignature = current;
        }
        const plane = groundPlane(target);
        status.textContent = plane.text;
        status.classList.toggle('error', !plane.valid);
        checkbox.disabled = !plane.valid;
        enter.disabled = !plane.valid || !checkbox.checked;
    };
    checkbox.onchange = refresh;
    enter.onclick = () => {
        refresh();
        if (enter.disabled || !dialog.open) return;
        approved.set(target, signature(target));
        guide.hidden = true;
        finish(true);
    };
    dialog.querySelector<HTMLButtonElement>('#sago-orientation-cancel')!.onclick = () => finish(false);
    dialog.querySelector<HTMLButtonElement>('#sago-orientation-adjust')!.onclick = () => {
        const splat = target;
        finish(false);
        if (!splat?.scene || events.invoke('selection') !== splat) return;
        approved.delete(splat);
        events.fire('grid.setPlanes', ['xz']);
        events.fire('grid.setVisible', true);
        events.fire('tool.setCoordSpace', 'world');
        if (events.invoke('tool.active') !== 'orient') events.fire('tool.orient');
        guide.hidden = false;
    };
    guide.querySelector<HTMLButtonElement>('#sago-orientation-review')!.onclick = review;
    guide.querySelector<HTMLButtonElement>('#sago-orientation-hide-guide')!.onclick = () => {
        guide.hidden = true;
    };
    dialog.addEventListener('cancel', (event) => {
        event.preventDefault(); finish(false);
    });
    dialog.addEventListener('keydown', event => event.stopPropagation());
    dialog.addEventListener('keyup', event => event.stopPropagation());
    events.on('selection.changed', () => {
        guide.hidden = true; refresh();
    });
    const invalidate = (splat: Splat) => {
        approved.delete(splat);
        if (splat === target) {
            checkbox.checked = false; refresh();
        }
    };
    for (const event of ['splat.moved', 'splat.positionsChanged', 'splat.replaced', 'scene.elementRemoved']) events.on(event, invalidate);
    // Picking/moving orientation points doesn't emit a geometry edit event.
    events.on('postrender', refresh);
    return {
        isConfirmed,
        require: (splat: Splat): Promise<boolean> => {
            if (isConfirmed(splat)) return Promise.resolve(true);
            if (resolver) return Promise.resolve(false);
            target = splat;
            checkbox.checked = false;
            shownSignature = signature(splat);
            dialog.querySelector('#sago-orientation-layer')!.textContent = splat.name;
            const result = new Promise<boolean>((resolve) => {
                resolver = resolve;
            });
            dialog.showModal();
            refresh();
            return result;
        }
    };
};
