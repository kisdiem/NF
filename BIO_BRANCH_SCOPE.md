# BioNeuron / BioNF 专题分支

本分支用于集中维护 BioNeuron 研究，不把它与空间 Local NF 混为同一模型。

## 当前结构

```text
输入投影
→ 多树突分支局部非线性组合
→ 兴奋/抑制
→ 胞体膜电位
→ 动态阈值与 adaptation
→ activation
→ 分类器
```

核心代码为 `models/bio_neuron.py`，实验入口为 `experiments/bio_experiments.py`。

## 主要历史结果

- XOR、同心圆和 Two Moons：BioNeuron 可形成线性模型没有的非线性表示。
- MNIST 5000 样本、20 轮：seed 0 最高约 93.3%，seed 1 最高约 93.7%。
- MNIST 30 轮：seed 0 约 93.39%，seed 1 约 93.66%。

## 关键消融

树突分支消融影响最大；在 XOR 上，去掉时间积累、抑制或动态阈值暂未产生明显影响。

详细设计和结果见 `README_BioNeuron.md` 与 `bio_results/`。
