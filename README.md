# SAGO Web v1.0

**SAGO（Segment Any Gaussians Online，ECCV 2026）的 Web 工具与浏览器移植。** 本工具将 SAGO 的交互式 3D Gaussian 分割、跨视角记忆与 Virtual Drone 空间细化移植到浏览器，以 [SuperSplat](https://github.com/playcanvas/supersplat) 为编辑器和渲染基础。场景加载与 SAM2 Hiera Large FP32 推理均在本机完成，无需业务服务器或 Python 推理后端。

> **Online Segment 3D Gaussians via Launching Virtual Drones**
>
> Liwei Liao, Rongjie Wang, Ronggang Wang · **ECCV 2026**

[论文 / arXiv](https://arxiv.org/abs/2607.01628) · [PDF](https://arxiv.org/pdf/2607.01628) · [SAGO 原始代码](https://github.com/leviome/SAGO) · [Web 工具发布版](https://github.com/leviome/SAGO-Web/releases/tag/v1.0) · [引用](#citation)

关于论文方法与本浏览器版本的对应关系，见 [论文与 Web 实现](docs/PAPER.md)。论文中的速度和精度以原始实验设置为准；Web 版本的固定六视角巡航、渲染和运行环境存在差异。

## 功能

- 原版 SAM2 Large：前景点、背景点、原生框提示、四个候选 mask、边缘扩展。
- 单对象跨视角 memory，以及双向六视角 Virtual Drone 自动细化。
- 进入分割前确认世界 +Y 朝上；支持地面三点校正。
- 二维 mask 和三维选区均支持完整画面、仅前景、仅背景显示。
- 下载模型前询问，显示进度、速率、剩余时间；支持重试、分块续传和 SHA-256 校验。
- 浏览器缓存，以及跨域名复用本机模型目录（新域名或刷新后重新选择目录）。
- 三维选区替换、增加、减去、取交集，撤销/重做，以及 SuperSplat 原有编辑、导出功能。

完整操作说明见 [README-SAGO.md](README-SAGO.md)，版本记录见 [CHANGELOG.md](CHANGELOG.md)。

## 从源码运行

需要 Node.js **24+**、npm，以及支持 WebGPU 的桌面 Chrome / Edge。开启硬件加速，建议使用独立显卡。网页必须运行在 localhost 或 HTTPS 下。

在源码目录执行：

```bash
ONNXRUNTIME_NODE_INSTALL_CUDA=skip npm ci
```

从此仓库 **v1.0 Release** 下载 `sago-sam2-large-fp32-v1.0.tar.gz` 和 `SHA256SUMS`，校验后将模型解压到源码根目录：

```bash
# 在附件下载目录校验
sha256sum -c SHA256SUMS
# 回到源码根目录，路径替换为实际下载位置
tar -xzf /path/to/sago-sam2-large-fp32-v1.0.tar.gz
npm run models:download
npm run build
npm run serve -- --listen tcp://127.0.0.1:3000
```

打开 **http://127.0.0.1:3000**，拖入 3DGS PLY，选中图层，再点击左侧场景面板中的 **SAGO · 智能分割**。

模型附件约 913 MiB（未压缩），解压后位于 `static/models/sam2-large/`，包含四个 ONNX 图和权重分块。模型、`node_modules/`、`dist/`、本机场景和诊断输出不纳入 Git。`npm run models:download` 会验证已解压模型；若缺失，则尝试从桌面版 SAGO 原始 checkpoint 导出，并非从公网自动下载。

### 自行导出模型（可选）

运行网页不需要 Python。仅重新导出模型时需要桌面版 SAGO 源码及其 Python 环境（torch、numpy、opencv-python、onnx==1.16.2）：

```bash
SAGO_ROOT=/path/to/SAGO \
SAGO_PYTHON=/path/to/python \
SAGO_SAM2_CHECKPOINT=/path/to/sam2_hiera_large.pt \
npm run models:export
```

### 静态部署

准备模型后运行 `npm run build`，将整个 `dist/` 部署到 HTTPS 静态主机。静态主机需能托管约 1 GiB 文件，推荐支持 HTTP Range。临时隧道域名不属于固定服务地址。开发模式使用 `npm run develop`。

## 验证

```bash
npm run typecheck
npm run lint
npm run test:sago
npm run build
# 需要已准备模型、启动 localhost:3000，以及支持 WebGPU 的 Chrome
npm run test:browser
```

CI 验证源码类型、lint、单元测试和无模型的构建；GPU 推理回归需在支持 WebGPU 的机器上运行。

## 已知范围

- 首版以二进制 3DGS PLY 为主要验证格式。
- Virtual Drone 和手动多视角交集仍可能漏选、过度裁剪；请检查候选结果。
- 文字提示、动态 TAG、命名对象列表尚未实现。
- 模型缓存遵循浏览器同源规则；跨域名复用需使用本机模型目录。

## 来源与许可

SAGO Web v1.0 基于 **SuperSplat 3.3.1**，上游 commit：`c9e29913f3c86f2a4ae80f71b331116ac7eb6815`。保留上游 [MIT 许可](LICENSE) 与 [原项目说明](docs/SUPERSPLAT-UPSTREAM.md)。SAGO Web 的版本号独立于 SuperSplat。

SAM2 模型采用 [Apache-2.0](docs/LICENSE-SAM2)；ONNX Runtime 和其他依赖保留各自许可。模型从 SAGO 原始 `sam2_hiera_large.pt` 导出为 FP32，详情见 [SAM2-LARGE.md](docs/SAM2-LARGE.md) 与 [SAM2-MEMORY.md](docs/SAM2-MEMORY.md)。

<a id="citation"></a>
## 引用 / Citation

如果本工具或 SAGO 方法对你的研究有帮助，请引用 SAGO 论文：

```bibtex
@inproceedings{liao2026sago,
  title     = {Online Segment 3D Gaussians via Launching Virtual Drones},
  author    = {Liao, Liwei and Wang, Rongjie and Wang, Ronggang},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2607.01628}
}
```

可下载 [BibTeX](docs/SAGO.bib)。[CITATION.cff](CITATION.cff) 也提供论文引用信息，供 GitHub 的 “Cite this repository” 使用。
