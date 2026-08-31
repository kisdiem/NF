# Neural Field / BioNeuron / Local Electrical NF 实验总览

本目录整理 `kisdiem/NF` 项目从简单基线到局部电信号神经场的完整实验过程。目标不是选择性证明 NF 优于 MLP，而是逐步回答：

> 非线性是否可以由神经元状态、局部传播和多步演化产生，并且这种计算是否在真实任务中带来可重复的优势？

## 阅读顺序

1. `01_实验路线与研究问题.md`：实验为什么逐步演化。
2. `02_设计说明.md`：各版本的数据流、状态和公式。
3. `03_结果总表.md`：MNIST、合成数据、困难任务和速度结果。
4. `04_复现命令.md`：在本机重新运行的命令。
5. `05_文件索引.md`：代码、结果和图的位置。

## 版本路线

```text
Linear / ReLU / GELU
        ↓
原始 NF：能量、阈值、固定邻域、多步传播
        ↓
Dynamic NF：根据 H_t 生成动态关系 G_t
        ↓
BioNeuron：树突、兴奋/抑制、膜电位、动态阈值
        ↓
Local Electrical NF v1：固定 8×8 网格、R=1、局部 conv2d
        ↓
v2：活动依赖抑制与 refractory
        ↓
v3/v4：能量归一化、泄漏、中心化、软复位、bounded/unbounded
        ↓
困难任务、因果冻结、推理性能和跨数据集比较
```

## 最重要的结论

* Local NF 能解决 XOR，说明它不是纯线性变换。
* 在 circles、moons 等简单任务上，ReLU/GELU 已经很强，Local NF 没有明显优势。
* 在 checkerboard 和高维 noisy spiral 任务上，Local NF 曾取得高于 ReLU 的 seed=0 结果，但还需要多 seed 验证。
* 固定 inhibitory sign 会造成膜电位偏移；活动依赖抑制可以降低激活率，但不足以单独抵消正向累积。
* `raw_bounded` 是目前动力学与准确率最平衡的版本；最高 93.38%，但仍略低于 no-inhibition 的 93.47%。
* 因果实验表明，空间 field 输出确实被使用，但冻结 field 参数与可训练 field 的准确率几乎相同，说明当前动态参数还没有转化为明显的泛化收益。
* Local NF 推理比普通 MLP 慢；batch=128 时融合局部卷积有效，batch=1 时融合反而可能因 kernel 调度开销变慢。

## 数据说明

MNIST 原始数据没有放入压缩包。运行 MNIST 命令时，`torchvision` 会下载到 `data/mnist`；所有 JSON、PNG、Markdown 和源代码均已包含。

## 来源

源项目目录：`C:\Users\sixth\Downloads\neural-field-mlp\neural-field-mlp`  
GitHub：`https://github.com/kisdiem/NF`
