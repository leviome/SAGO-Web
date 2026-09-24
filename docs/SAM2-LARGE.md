> 本文记录单图 SAM2 Large 迁移。当前版本已加入 memory 和下载确认，见 [SAM2-MEMORY.md](SAM2-MEMORY.md)。下文 869 MiB 为原单图模型大小。

# 原版 SAM2 Large 的纯前端迁移

Web 端使用项目原有 `sam2_hiera_large.pt`，经 SAGO 自身的 `make_sam_from_state_dict` 加载并导出。保留 Hiera Large 架构及 FP32 权重，没有蒸馏、量化或换用 SAM2.1。模型加载、图像编码和提示推理均在访问者浏览器完成；Python 只用于开发阶段导出与验证。

## 模型与行为

- 原始模型参数量：224,430,113。网页导出图像编码器、提示编码和掩码解码器；该阶段尚未包含 memory 模块；现已在后续版本接入。
- 权重 SHA-256：`7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b`。
- RGB 图像采用抗锯齿双线性缩放至 1024×1024，再按 ImageNet 均值/标准差归一化，与 SAGO 图像路径一致。
- 使用归一化的原图坐标。框提示编码为原生左上角/右下角标签，可与前景、背景点组合；不裁图、不补框中心点。
- 返回 level 0–3 四个掩码，默认 level 0，与桌面 `seg_level=0` 对齐。模型评分仅供参考。
- 先将 256×256 logits 双线性插值到原图尺寸，再按 `> 0` 得到掩码。边缘扩展为独立 UI 选项，诊断数据保存扩展前结果。

模型资产约 **869 MiB / 911 MB**，位于 `static/models/sam2-large/`，不提交 Git。权重分块存储，每个文件均校验 SHA-256，浏览器尽可能使用 Cache API 缓存。缓存被禁用或空间不足时仍能下载运行。首次加载需要时间，建议使用开启硬件加速的 Chrome / Edge 和独立显卡。

## WebGPU 兼容性

运行时固定为 **ONNX Runtime Web 1.23.2**，使用配套 `ort.webgpu.bundle.min.mjs` 和 `ort-wasm-simd-threaded.asyncify.*`。旧版本 `1.22.0-dev.20250409-89f8206ba4` 在此模型的解码器交叉注意力计算中存在显著数值偏差，不能沿用旧运行时文件。

导出时将固定输入尺寸的位置编码提前计算为常量，避免浏览器不支持的位置编码初始化分支。大规模全局自注意力按 head 计算，并将多输入合并拆成每次最多四个输入，避免 WebGPU 缓冲区大小与绑定数量限制。交叉注意力保留原生计算图。这些处理保留权重与数学语义。

WebGPU 负责模型计算，少量形状运算由浏览器 CPU/WASM 执行。前端检查 GPU validation / out-of-memory / device lost 错误和非有限输出，发生异常时拒绝应用结果。

## 数值验证

环境：Chrome、NVIDIA RTX 4090，浏览器 WebGPU 对照 ONNX Runtime Node CPU，以及项目原版 PyTorch FP32/CUDA。

使用项目已有桌面场景图 `debug/first_view.png`（1024×1024），另将同图缩放为 1280×867 验证非正方形视口。每种尺寸测试苹果点选、相机点选、细腿物体、正负点、相机框和香蕉框，共 12 组提示、48 个候选。

| 检查 | 结果 |
| --- | --- |
| 浏览器与桌面预处理最大误差 | 1024 方图：2.38×10⁻⁷；1280×867：7.15×10⁻⁷ |
| 相同 ONNX 的 CPU/WebGPU 低分辨率二值掩码 | 48/48 完全一致 |
| 浏览器与原版 PyTorch 的原图尺寸 level 0 IoU | 0.999518–1.000000 |
| 浏览器与原版 PyTorch 的全部同层级 IoU | 0.998906–1.000000 |

这些指标衡量实现之间的一致性，不是有标注真值下的准确率。两种尺寸来自同一张图，不能视为独立质量评测。少量边界像素差异来自浮点计算与插值。

原始报告、逐候选掩码和对比图保存在 `artifacts/sam2-large-debug/` 与 `artifacts/sam2-large-wide/`，不发布到静态站点。

## 重建与复现

已导出的模型可直接复制到其他静态站点，不需要安装 Python。重新导出时使用已有 SAGO Python 环境，并安装 `onnx==1.16.2`：

```bash
# 按实际环境调整路径；仅开发阶段需要。
SAGO_PYTHON=/path/to/sago/python npm run models:export
npm run build
npm run serve -- --listen tcp://127.0.0.1:3000
```

`models:export` / `models:download` 会先验证现有资产，缺失或损坏时导出。需要强制重新导出可直接运行：

```bash
/path/to/sago/python scripts/export-sam2.py \
  --checkpoint ../sago_core/sam2/checkpoints/sam2_hiera_large.pt
```

`manifest.json` 记录原始权重指纹、精度、运行时版本和所有 ONNX / 权重块的哈希。构建后只需发布 `dist/` 到 HTTPS 静态主机。

诊断复现（先启动静态站点）：

```bash
node scripts/debug-segmentation.mjs
/path/to/sago/python scripts/debug-sam2.py --directory artifacts/sam2-large-debug
# 也可重放界面导出的本地 JSON：
SAGO_DEBUG_BUNDLE=/path/to/sago-debug.json SAGO_DEBUG_OUT=artifacts/user-sample \
  node scripts/debug-segmentation.mjs
/path/to/sago/python scripts/debug-sam2.py --directory artifacts/user-sample
```

其他验证：`npm run typecheck`、`npm run lint`、`npm run test:sago`、`npm run test:browser`。浏览器回归阻断外部网络，覆盖静态模型加载、四候选及诊断导出、原生框、三维选区、撤销/重做、交集和过期视图拒绝。

## 范围

本次对齐的是单图 SAM2 分割。后续版本已加入用户控制视角的 memory 跟踪；自动虚拟相机与自验证仍未移植。三维选区继续采用 SuperSplat 的 mask 投影，因此单图掩码一致不代表完整三维管线已经一致。
