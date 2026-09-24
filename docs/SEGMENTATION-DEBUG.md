> 历史报告：本文记录 SlimSAM 首版与桌面 SAM2 的差异。当前 Web 已迁移到原版 SAM2 Large，见 [SAM2-LARGE.md](SAM2-LARGE.md)。

# Web 分割与 SAGO 桌面版差异调查

调查日期：2026-09-23。结论针对本地受控样本，不将模型间一致性当成准确率，也不代表已经复现用户所有失败场景。

## 结论

当前 Web 版并未移植桌面版的同一个 SAM：它使用 SlimSAM，桌面 `TagWidget` 使用 SAM2 Hiera Large。差距包含三个相互独立的来源：模型能力、框提示语义、默认候选选择；最终三维结果还缺少 SAGO 的自动多视角筛选。

同模型同输入的 CPU / WebGPU 对照未发现可解释明显质量退化的数值或提示坐标错误。不能据此断言所有浏览器和 GPU 都没有问题。

## 实验设计

- 图像：项目 `debug/first_view.png`，1024 × 1024 的桌面渲染，桌上含苹果、相机、细腿玩偶和香蕉。
- 对照输入 PNG SHA-256：`29b2c971b17d6cf95e5057c888330c9f8785ff304a31f18997481489fe0b880d`。
- 所有模型使用完全相同的原始 RGB 画面，排除 PlayCanvas / CUDA 渲染器不同造成的干扰。
- Web：实际 `src/sago/inference.ts`，SlimSAM FP32，Transformers.js 3.8.1，ONNX Runtime Web 1.22.0-dev.20250409-89f8206ba4。
- 数值基线：同一套 ONNX 权重、从浏览器导出的同一输入张量，ONNX Runtime Node 1.21.0 CPU，重新运行 encoder + decoder。
- SAGO：项目原始 `make_sam_from_state_dict` 和 `sam2_hiera_large.pt`，PyTorch FP32/CUDA；使用原版图像编码、提示编码和 mask decoder。
- 硬件：RTX 4090。SAM2 参数数为 224,430,113；模型文件约 857 MiB，Web 两个 SlimSAM ONNX 文件合计约 38 MiB。
- 提示：4 组点提示（含一组正负点）、2 组框；另外把同图拉伸到 1280 × 867，仅用于检查非正方形坐标和 CPU/GPU 数值一致性，不作为独立准确率样本。

原图、全部 mask、精确坐标和报告保存在 `artifacts/segmentation-debug/`；非正方形检查在 `artifacts/segmentation-debug-wide/`。这些诊断图片不放进公开站点。

## 1. WebGPU 没有在这些样本上算错

两种画面尺寸 × 4 组点提示 × 3 个候选，共 **24 张 256 × 256 mask**，CPU 和 WebGPU 在 logit > 0 阈值下完全一致，IoU 全为 1。

- logits 平均绝对误差：约 0.000038–0.000184。
- logits 最大绝对误差：不超过 0.00216。
- 当前实现的提示坐标与 Transformers.js 官方 processor 的坐标，最大误差约 0.000021 像素（浮点表示误差）。

RGB 顺序检查：Web processor 将 RGBA 转成 RGB；SAGO 的 `load_view()` 将 RGB 转成 BGR，而 `prepare_image()` 再转回 RGB。这条桌面路径本身没有漏转或双重颠倒。

Web SlimSAM 使用最长边缩放并补零，SAM2 使用 1024 方形缩放，这是不同模型的预处理约定。不能把两者的 resize 方式盲目统一；本次 1024 方形对照已经排除了宽高比影响。

## 2. 自动评分会选中对象的不同部分

Web 用三个候选中最高模型评分；桌面默认使用 `seg_level = 0`，并允许 0–3 切换。两模型候选编号也不能一一对应。

相机点选样本：

| Web 候选（从 1 编号） | 像素面积 | 模型评分 | 表现 |
|---|---:|---:|---|
| 1 | 13,493 | 0.803 | 较完整的相机主体 |
| 2 | 4,881 | 0.850 | 相机的一个区域 |
| 3（自动推荐） | 1,060 | 0.905 | 小零件 / 局部 |

Web 候选 1 与 SAM2 level 2 的 IoU 达 0.955，但自动选择没有选它。这个样本不能简单归因于“轻量模型完全不会分割相机”。评分估计的是候选 mask 的质量，不是用户想要的语义范围，也不是对象完整程度；评分超过 1 也不是 100% 以上准确率。

## 3. 轻量模型的细节能力存在差距

黄色玩偶样本中，SAM2 level 0/2 保留了细长腿部，而 SlimSAM 自动候选主要覆盖身体。切换候选能改变对象范围，但不能保证恢复所有细节。

下表的 IoU 表示 Web 自动结果与 SAGO 默认 level 0 的一致性，**不是带真值的分割准确率**：

| 场景与提示 | IoU |
|---|---:|
| 绿苹果，单点 | 0.957 |
| 相机，单点 | 0.200 |
| 黄色细腿玩偶，单点 | 0.453 |
| 绿苹果，正负点 | 0.952 |
| 相机，框 | 0.057 |
| 香蕉，框 | 0.813 |

全部候选对比图：`artifacts/segmentation-debug/comparison.png`。

## 4. Web 框选不是 SAGO 原生框提示

当前 SlimSAM ONNX 只导出了点输入。Web 使用“裁剪到框 + 默认中心前景点”，桌面直接在完整图像上编码两个框角。

这会改变上下文、目标尺度和提示含义；框中心也可能落在背景或某个零件上，框外负点则不参与裁剪图推理。

为了分离模型因素，又用 **同一个 SAM2 Large** 分别执行原生框和裁剪中心点，比较 level 0：

- 相机：IoU **0.347**，说明仅提示语义变化就能造成大幅差异。
- 香蕉：IoU **0.944**，说明影响取决于对象和框位置，不能用单一修正系数解决。

把不支持的 2/3 框角标签塞给当前 ONNX 不构成修复。真正对齐需要支持原生框提示的模型导出。

## 本次改动

- 增加三种候选的缩略图和面积，点击即可切换。
- 明确区分模型评分、准确率和对象范围；自动推荐保留，避免未经验证地改成“永远取最大 mask”。
- 框内没有显式前景点时，提示当前在使用中心点，引导用户补点。
- 增加“导出诊断样本”：下载包含原始输入 PNG、归一化提示、全部未膨胀 mask、评分、当前候选、padding 和浏览器信息的 JSON。不会自动上传，也不含 PLY。
- 新增可复现的 CPU/WebGPU 和原版 SAM2 对照脚本。

本次没有替换模型，没有将这几个样本的观察宣称为普遍质量提升。

## 复现

先启动已构建的静态站点，然后：

```bash
node scripts/debug-segmentation.mjs
/home/pc/anaconda3/envs/TrackAnyGaussian/bin/python scripts/debug-sam2.py
```

回放 UI 导出的失败样本：

```bash
SAGO_DEBUG_BUNDLE=/path/to/sago-debug.json \
SAGO_DEBUG_OUT=artifacts/user-sample \
node scripts/debug-segmentation.mjs

/home/pc/anaconda3/envs/TrackAnyGaussian/bin/python scripts/debug-sam2.py \
  --directory artifacts/user-sample
```

对照脚本使用 Python 仅限开发期离线实验，Web 应用仍为纯前端。框提示回放生成所有候选，用户选中的候选仍以导出 JSON 为准；报告中的 Web selected 是重新运行的自动推荐。

## 后续真正对齐的顺序

1. 导出并接入同一套 SAM2 的 image encoder、prompt encoder、mask decoder，提供原生框提示和原版 level 0–3。
2. 用本次同图同提示流程验收数值与候选对应，再评估 Tiny/Small 等模型的速度和质量取舍。
3. 如差距发生在最终三维选区，再移植 memory/object pointer、双向虚拟视角和首视角自验证；这些不会仅靠替换 2D 模型自动补齐。

目前仍未对用户具体失败场景、所有渲染设置、所有显卡或完整多视角结果做验证。

## 本次验证记录

`npm run build`、`npm run typecheck`、`npm run lint`、4 项 mask 单元测试及浏览器集成测试通过。浏览器测试包括候选缩略图切换和诊断 JSON 下载；可选真实 PLY 测试本轮未重新运行。
