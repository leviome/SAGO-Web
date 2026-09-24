# SAGO Web v1.0 — 纯前端分割工具

基于 SuperSplat 的浏览器分割工具。PLY、图像和推理均留在本机，无需 Python、CUDA 服务或业务后端；网页及模型是普通静态文件。

当前版本使用项目原版 **SAM2 Hiera Large FP32**，支持原生框提示、level 0–3 和跨视角 memory。首次模型下载会弹框询问并显示进度。memory 使用说明见 [SAM2-MEMORY.md](docs/SAM2-MEMORY.md)。迁移说明见 [SAM2-LARGE.md](docs/SAM2-LARGE.md)。旧版 SlimSAM 分割效果差异的实测、原因和复现方法见 [SEGMENTATION-DEBUG.md](docs/SEGMENTATION-DEBUG.md)。现在可对比候选缩略图，并导出原图、提示及全部 mask 作为诊断样本。

## 立即测试

首次安装请先按 [README.md](README.md) 准备依赖、模型并构建，然后打开 **http://127.0.0.1:3000**。

使用支持 WebGPU 的 Chrome / Edge，开启硬件加速。需要 localhost 或 HTTPS，不能直接双击 HTML。Linux 上若 WebGPU 不可用，先检查 `chrome://gpu`。

1. 拖入标准二进制 3DGS PLY，或使用场景面板的导入按钮。
2. 调整视角，使目标完整出现在画面内，并选中对应场景图层。
3. 点击左侧场景面板内的 **SAGO · 智能分割**，确认坐标系后固定当前画面及相机。
4. 保持“SAM2 Large”，点击 **加载模型并编码画面**，按弹框确认下载；已有缓存自动复用。
5. 左键添加前景点，右键添加背景点，点击 **预览 AI 分割**。补点后重新预览；同一画面复用图像特征。
6. 也可 **框选区域**：使用原生框提示，也可以组合前景 / 背景点；不会裁剪图像或自动加中心点。
7. 检查候选 mask 和边缘扩展；画面上方可切换 **完整画面 + mask / 仅前景 / 仅背景**，隔离区域显示为棋盘格。需跨视角跟踪时，先点击 **以当前结果建立 / 重建记忆**，再点击 **应用为三维选区**。
8. 返回场景后，入口下方的 **三维场景显示** 可切换完整场景、仅前景或仅背景：前景为当前图层已选高斯，背景为其余可见高斯。切换只影响显示，不更改选区、隐藏状态或导出内容；切换图层或清空选区会恢复完整场景。可旋转检查结果。新视角加载画面后点击 **用记忆跟踪此视角**，确认正确后更新记忆，选择 **与已有选区取交集** 可以细化。
9. 通过 SuperSplat 原有工具修正、撤销/重做、分离或导出选区。也可下载 2D mask。

如需换域名后继续复用模型，加载前点击 **选择本机模型文件夹（跨域名复用）**。程序会在所选目录的 `sago-sam2-large-fp32/` 中保存模型；已有浏览器缓存可直接复制。刷新或换域名后重新选择该目录，再加载即可，不用再次下载。此入口需要桌面 Chrome / Edge。

## 功能范围

进入 SAGO 前会先要求检查坐标系：世界 **+Y 朝上**，地面平行于 **XZ**。可点击“选三点校正地面”，从地面上方观察，在地面取三个不共线的点并点击“对齐到网格”；此操作旋转 / 平移当前图层，支持撤销。有地面采样时，夹角必须 ≤ 5° 才能确认；没有地面的对象可人工确认上方向。确认保留在本次页面的当前图层上，修改图层变换、几何或重新导入后需再次检查；仅旋转相机不需要重复确认。不会自动判断语义上的地面或场景是否颠倒。

- 原版 SAM2 Large FP32 本地推理，使用 ONNX Runtime Web 的 WebGPU execution provider；部分形状运算由 CPU/WASM 执行。
- 正负点、原生框提示、四种候选 mask（level 0–3）、边缘扩展。
- 原版 memory encoder / fusion 与对象指针；单对象跨视角跟踪、确认写入、6 份近期记忆 / 15 个近期指针。
- 模型下载前确认、字节进度、取消、缓存复用和完整性校验。
- 4 路并发、8 MiB 分块续传、自动重试、下载速度及剩余时间；本机模型文件夹可跨域名复用（新域名需重新选择目录授权）。
- 复用 SuperSplat 的 WebGPU mask 投影与实例映射，支持替换、增加、减去、取交集。
- 原生撤销/重做、画面冻结、场景变化及视口尺寸变化校验。
- 推理期间可返回场景并丢弃结果。已提交 GPU 运算不会立即中断，结束后才能开启下次会话；提交选区时短暂禁止关闭。
- 示例场景、本地模型和推理运行时静态打包。

**Virtual Drone 已实现：** 预览 mask 后点击“Virtual Drone · 自动多视角细化并应用”，自动双向扫描六个视角、使用独立 memory 跟踪并进行初始视角 80% 覆盖自验证，最后提交一次可撤销选区。支持进度、取消和相机恢复，详见 [VIRTUAL-DRONE.md](docs/VIRTUAL-DRONE.md)。文字提示、动态 TAG 和命名对象列表尚未实现。

单视角投影会包含遮挡后的高斯，也会丢弃画面外的高斯，应完整显示目标后操作。手动多视角交集不等价于完整 SAGO，可能过度裁剪。已对齐图像分割模型；跨视角流程仍与完整桌面 SAGO 不同。

图像按桌面版进行 RGB、抗锯齿双线性缩放到 1024×1024、ImageNet 归一化；掩码先缩放 logits 再按 0 阈值二值化。

首版优先测试 `binary_little_endian` 3DGS PLY。上游此版本读取 ASCII PLY 有兼容问题，请先转换为二进制。其他格式沿用 SuperSplat 支持，尚未全面验证。

## 从源码运行

需要 Node.js 20.19+（工具测试推荐 Node.js 24）和 npm。仅重新导出模型时需要 SAGO 的 Python 环境，以及 `onnx==1.16.2`；已导出的静态站点无需 Python。

```bash
# 跳过 Node 端依赖附带的 CUDA 安装；网页不使用该组件。
ONNXRUNTIME_NODE_INSTALL_CUDA=skip npm ci
# 将 Release 的 sago-sam2-large-fp32-v1.0.tar.gz 解压到仓库根目录
# tar -xzf /path/to/sago-sam2-large-fp32-v1.0.tar.gz
npm run models:download
npm run build
npm run serve -- --listen tcp://127.0.0.1:3000
```

开发用 `npm run develop`。先从 Release 解压模型或自行完成导出，再启动构建/监听。

模型在 `static/models/sam2-large/`，约 913 MiB，不纳入 Git。整个 `dist/` 包含网页、模型和 ONNX WASM，可直接部署到 HTTPS 静态主机。

`npm run models:download` 验证已有文件；缺失时从本地 `sam2_hiera_large.pt` 导出。可用 `SAGO_PYTHON`、`SAGO_SAM2_CHECKPOINT` 指定环境与权重路径；独立仓库需额外设置 `SAGO_ROOT` 指向含 `sago_core/` 的桌面版 SAGO 源码目录。也可直接复制导出的目录到其他静态主机。浏览器只从本站下载模型，按 SHA-256 校验和缓存；缓存不可用时仍可运行。首次完整下载约 957 MB，建议独立显卡。

## 验证

```bash
npm run typecheck
npm run lint
npm run test:sago
# 先启动静态服务；默认使用 /usr/bin/google-chrome
npm run test:browser
# 可选真实场景测试，文件不会发送到外部服务
SAGO_TEST_SCENE=/absolute/path/scene.ply npm run test:browser
```

浏览器测试阻断示例测试的外部网络请求，覆盖本地模型加载、mask、三维选区、撤销/重做、交集和过期结果保护。截图在 `test-results/`。测试使用 Chrome WebGPU/Vulkan 参数，可用 `SAGO_CHROME` 指定浏览器路径。

## 实现与来源

- `src/sago/panel.ts`：提示界面、会话状态、选区适配。
- `src/sago/inference.ts`：模型分块加载、完整性校验、图像特征缓存与推理。
- `src/sago/mask-utils.ts`：坐标映射和 mask 处理。
- `src/camera.ts`：提示会话期间冻结相机。
- `src/sago/sam2-utils.ts`：对齐桌面版的预处理、原生提示和 logits 后处理。
- `src/sago/model-assets.ts` / `download-dialog.ts`：下载确认、实时进度、取消与缓存。
- `scripts/export-sam2.py` / `export-sam2-memory.py`：从原始权重离线导出单图和记忆模型。
- `scripts/download-models.mjs`：验证模型文件，必要时调用导出。
- `scripts/create-demo.mjs`：生成二进制 PLY 示例。

SuperSplat 3.3.1，commit `c9e29913f3c86f2a4ae80f71b331116ac7eb6815`，MIT；原说明见 [SUPERSPLAT-UPSTREAM.md](docs/SUPERSPLAT-UPSTREAM.md)。

SAM2 Large：项目原有 `sago_core/sam2/checkpoints/sam2_hiera_large.pt`，通过 SAGO 自身 `make_sam_from_state_dict` 加载和导出，保留 FP32；模型为 Apache-2.0。ONNX Runtime 原有许可保留在依赖中。模型和场景共享设备内存，建议先用示例确认可运行，再测试较大场景。
