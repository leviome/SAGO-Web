# SAGO 论文与 Web 实现

**SAGO — Segment Any Gaussians Online**，ECCV 2026。

**Online Segment 3D Gaussians via Launching Virtual Drones**

Liwei Liao, Rongjie Wang, Ronggang Wang

- [论文摘要与版本记录](https://arxiv.org/abs/2607.01628)
- [论文 PDF](https://arxiv.org/pdf/2607.01628)
- [原始 SAGO 实现](https://github.com/leviome/SAGO)
- [SAGO Web 仓库](https://github.com/leviome/SAGO-Web)
- [BibTeX](SAGO.bib) · [GitHub 引用元数据](../CITATION.cff)

## 方法与工具的关系

论文提出 SAGO，通过 Virtual Drone 将交互式 3D Gaussian 分割表述为马尔可夫过程中的在线 Next-Best-View（NBV）规划，省去面向单场景的预处理阶段。

SAGO Web 将原始项目中的交互式分割、跨视角记忆和 Virtual Drone 空间细化流程迁移到浏览器，提供可直接操作的 Web 工具。SuperSplat 提供场景编辑、渲染、选区管理和导出能力；SAM2 Hiera Large FP32 通过 ONNX Runtime Web 在本机运行。

| 能力 | 当前 Web 实现 |
| --- | --- |
| 交互式分割 | 前景点、背景点、原生框提示、候选 mask，投影为三维 Gaussian 选区 |
| 跨视角记忆 | 移植 SAM2 memory encoder / fusion、对象指针与单对象跟踪 |
| Virtual Drone | 按原始 `spatial_track` 的双向六视角流程细化，带覆盖检查和一次可撤销提交；详见 [实现说明](VIRTUAL-DRONE.md) |
| 浏览器交互 | 坐标系确认、二维和三维前景/背景显示、模型下载及本机目录复用 |

## 实现范围

当前 Web 版本使用固定的 ±45°、±90°、±135° 六视角巡航，并对记忆写入、覆盖检查和渲染作了适配。完整差异见 [Virtual Drone](VIRTUAL-DRONE.md)、[跨视角记忆](SAM2-MEMORY.md) 和 [SAM2 迁移](SAM2-LARGE.md)。该版本未覆盖原始项目的全部路径，如旧 `tag_widget` 的预热、BEV 补分割和动态 TAG。

复现论文实验时应使用论文所述设置与原始 SAGO 实现。论文报告的亚秒级延迟和加速比不能直接用作 Web 版本的性能指标；浏览器模型加载、GPU 编译、显卡和场景规模都会影响运行时间。当前仓库的自动化测试验证功能行为，不构成论文基准的复现。

## 引用

使用 SAGO 方法或本 Web 工具进行研究时，请引用论文。可直接使用 [SAGO.bib](SAGO.bib) 或仓库首页的 BibTeX。

标题、作者和 arXiv 标识依据 [arXiv:2607.01628](https://arxiv.org/abs/2607.01628)；会议标注为 ECCV 2026。论文入口指向 arXiv，未填写未核实的会议卷号、页码或出版 DOI。
