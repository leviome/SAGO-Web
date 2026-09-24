// Keep handles in the active page only. Restoring persisted filesystem handles
// crashes some Chromium builds; ordinary model files remain portable and durable.
export const pickModelDirectory = async () => {
    if (!window.showDirectoryPicker) throw new Error('此浏览器不支持模型目录保存，请使用桌面版 Chrome / Edge。');
    const parent = await window.showDirectoryPicker({ id: 'sago-models', mode: 'readwrite' });
    const name = 'sago-sam2-large-fp32';
    return parent.name === name ? parent : parent.getDirectoryHandle(name, { create: true });
};
