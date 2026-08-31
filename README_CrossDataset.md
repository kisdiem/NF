# Local Electrical NF 跨数据集比较

运行脚本：`benchmark_cross_dataset.py`。每个任务使用 hidden=64、Adam、lr=0.003、weight decay=1e-4、seed=0；普通模型和 Local NF 使用相同训练轮数与 batch。合成数据只用训练划分的均值/标准差做标准化，避免测试集泄漏。

```powershell
py -3 benchmark_cross_dataset.py --datasets xor,circles,moons,classification10 --n 4000 --epochs 50 --batch 128 --result-tag synthetic_seed0
py -3 benchmark_cross_dataset.py --datasets digits8x8 --n 1797 --epochs 100 --batch 128 --result-tag digits_seed0
```

Local NF 使用当前 `raw_bounded`：输入投影到 8×8 场，4 步局部传播，保留原始兴奋尺度并用 tanh 限制净输入；同时保留 v1 `no_inhibition` 作为对照。

## 结果

| 数据集 | Linear | ReLU | GELU | Local raw_bounded | Local no-inhibition |
|---|---:|---:|---:|---:|---:|
| XOR（50轮） | 62.50% | 99.75% | **100.00%** | 99.75% | 99.62% |
| Circles（50轮） | 62.38% | **99.25%** | **99.25%** | **99.25%** | 99.25% |
| Moons（50轮） | 88.12% | 99.12% | 99.00% | **99.25%** | 99.12% |
| Classification-10（50轮） | 83.37% | 94.37% | 94.13% | **95.37%** | 94.75% |
| Digits 8×8（100轮） | 97.50% | 98.89% | **99.17%** | 97.22% | 97.22% |

表中是每个 seed 的最高测试准确率，不是多 seed 平均。合成任务每个包含 4000 样本、80/20 train/test；Digits 使用 sklearn 内置的 1797 个 8×8 手写数字样本、80/20 split。

## 训练耗时与参数

以 4000 样本合成任务为例，单个任务的参数量为：二维输入时普通 MLP 322、Local raw_bounded 517、no-inhibition 578；10 维输入时分别为 834、1029、1090。Local raw_bounded 的训练耗时约为普通 MLP 的 3.5～4.5 倍，因为每个 batch 要执行 4 个内部传播步；这不是推理延迟基准，推理请使用 `benchmark_local_nf_inference.py`。

## 解释

1. Local NF 能解决 XOR，说明它确实包含有效的非线性计算，不是纯线性模型。
2. 在 circles/moons 上与 ReLU/GELU 基本持平，说明局部传播没有阻止它学习简单非线性边界。
3. 在 Classification-10 上 raw_bounded 最高值高于 ReLU 1.00 个百分点，但第 50 轮回落到 93.87%，说明这个优势不稳定，不能仅凭一次最高值宣称普遍优越。
4. 在 Digits 8×8 上 Local NF 比 GELU 低 1.95 个百分点。即使数据本身是 8×8 网格，固定局部传播也没有自动形成更好的图像表征。
5. 因此当前 Local NF 更像一个能够表达局部组合的非线性算子，但没有证据表明它在一般视觉识别任务上优于 ReLU/GELU；它的额外动力学成本目前没有转化为稳定的泛化优势。

完整逐轮结果位于 `cross_dataset_results/results_synthetic_seed0.json` 和 `cross_dataset_results/results_digits_seed0.json`。
