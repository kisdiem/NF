# Neural Field 参数审计（全仓库与早期设计）

生成日期：2026-09-01。审计范围包括 `main`、`new`、完整 Git 历史、当前工作树中全部 NF/BioNeuron 实现，以及两份早期设计文档：`定向矩形神经场MLP完整架构设计_v4.docx`、`通用离散神经场MLP完整架构设计_v3.docx`。

## 分类原则

- **连接参数（synaptic）**：决定谁影响谁以及影响多少，包括输入/输出 Linear、卷积核、关系生成投影、树突输入权重。
- **神经元内在属性（intrinsic）**：决定单个或共享神经元如何响应，包括阈值、衰减、增益、释放强度、不应期和内在偏置。
- **动力学状态（state）**：由 forward 计算产生且随样本/时间变化，不能交给 optimizer，例如膜电位、场状态、放电和 refractory trace。
- **固定超参数（hyperparameter）**：定义实验或离散结构，例如网格、步数、半径、最大延迟。默认不自动转为 `Parameter`。
- **离散内在属性（discrete intrinsic）**：属于神经元但不适合 Adam 的整数/类别属性，单独登记。

代码中的权威分类由 `training_strategies/registry.py` 提供。Phase 1 不强迫不同模型拥有同一组属性。

## 1. Minimal Local NF（A/B/循环/步数系列）

**文件/版本**：`train_simple_field_ab.py`；独立可复用实现 `models/minimal_local_nf.py`；`new` 分支。

- 连接参数：`input.weight/bias`、`output.weight/bias`、3×3 `field.kernel`。kernel 表示局部邻接作用，因此属于连接而非内在属性。
- 内在属性：共享 `decay_raw`；B 版本另有每节点 `theta`。
- 动力学状态：每一步的二维场/膜电位 `V_t`、局部消息和激活。
- 固定超参数：网格边长、steps、普通边界或 circular boundary、A/B 模式。
- 当前可训练性：Linear、kernel、decay 均可训练；B 的 threshold 可训练。
- 原优化器/损失：AdamW，cross entropy，统一 task BP。
- 适用实验：E0、E1、E2；E3/E4/E5/E6/E7 很适合；E8 可用于 kernel；仅 B 的硬门版本适合 E9/E10。

## 2. Local Electrical NF v1

**文件/版本**：`local_electrical_nf.py`、`train_local_electrical_nf.py`；最初局部电传播版本。

- 连接参数：输入/输出 Linear。3×3 local kernel 是固定连接 buffer，不可训练。
- 内在属性：每节点 `theta_raw`、`strength_raw`、`decay_raw`、`sign_raw`（固定兴奋/抑制身份的连续参数化）。
- 动力学状态：`V_t`、release/release gate、incoming message。
- 固定超参数：8×8 网格、radius=1 对应 kernel、steps、surrogate temperature、是否 threshold/persistence/inhibition。
- 当前可训练性：四类内在属性和 Linear 均可训练，kernel 固定。
- 原优化器/损失：Adam，cross entropy，统一 task BP。
- 适用实验：E0–E7；若 kernel 解冻可做 E8；当前 sigmoid surrogate 不属于真正 hard firing，E9/E10 不适用。

## 3. Local Electrical NF v2

**文件/版本**：`local_electrical_nf_v2.py`、`train_local_electrical_nf_v2.py`；活动依赖抑制与可选 refractory。

- 连接参数：输入/输出 Linear；`exc_kernel`、`inh_kernel` 为固定 buffers。
- 内在属性：每节点 threshold/strength/decay；共享 `rho_raw`（局部活动启动抑制的水平）、`beta_raw`（抑制强度）、`gamma_raw`（不应期抬高阈值的强度）。
- 动力学状态：`V_t`、release、excitation、inhibition、inhibition gate、refractory `R_t`、effective threshold。
- 固定超参数：steps、tau、tau_I、lambda_R、动态抑制/refractory 开关、网格与固定局部核。
- 当前可训练性：上述 raw intrinsic 和 Linear 可训练，局部核固定。
- 原优化器/损失：Adam，cross entropy，统一 task BP。
- 适用实验：E0–E7；kernel 解冻后可 E8；无硬阈值所以 E9/E10 不适用。

## 4. Local Electrical NF v3/v4 与 field 属性实验

**文件/版本**：`local_electrical_nf_v3.py`、`train_local_electrical_nf_v3.py`、`train_local_electrical_nf_v4.py`、`experiment_neuron_attribute_training.py`、`ablate_field_all.py`、`experiment_field_expressive_variants.py`、`search_field_hyperparams.py`、`validate_field_hyperparams_seeds.py`。

- 基础分类同 v2；v3 增加 `fixed`、`raw_bounded`、`direct_projected` 等属性训练方式和融合卷积实现。
- expressive variants 额外连接参数：可学习局部 kernel；额外内在/调制参数：`interaction_gain`、`gate_slope`（依具体公式归为内在响应调制）。
- `field frozen`、逐属性训练、固定超参数搜索已经显示：固定动力学变换常有用，但 threshold/strength/decay 的普通 BP 改变量对准确率贡献很弱。这正是 E1/E2 的首要目标。
- 适用实验：E0–E8；不存在真实离散 firing 时不做 E9/E10。

## 5. Dynamic NF（16 个平级节点）

**文件/版本**：`dynamic_nf.py`、`train_dynamic_nf.py`。

- 连接参数：Q/K 关系投影、self/message/update/gate 映射、LayerNorm、branch mixing；固定关系消融中的 `static_relation` 也是连接参数。
- 内在属性：共享 `relation_gain_raw`；branch-local bias 视作局部内在偏置。
- 动力学状态：`H_t`、样本依赖关系 `G_t`、message、gate、candidate。
- 固定超参数：N=16、node_dim=4、branches、steps、temperature、dynamic/static relation、bidirectional/upper-triangular、state persistence 等开关。
- 当前可训练性：连接与 intrinsic 都由 Adam + task loss 联合训练。
- 原优化器/损失：Adam，cross entropy。
- 适用实验：E0/E1/E2；E3/E4 △（活动定义需谨慎）；E5/E6 ✓；E7 △；E8/E9/E10 ✗。

## 6. Hierarchical Dynamic NF（8+8）

**文件/版本**：`hierarchical_nf.py`、`train_hierarchical_nf.py`。

- 连接参数：Layer1 内部、L1→L2、L2 内部和可选 L2→L1 的 q/k/self/message/gate/norm/mix。
- 内在属性：`gain1_raw`、`gain12_raw`、`gain2_raw`、可选 `gain21_raw`；`bias1/bias2` 是节点局部响应偏置。
- 动力学状态：`H1_t/H2_t` 与 `r11/r12/r22/r21`。
- 固定超参数：8+8 分层、steps、temperature、反馈和 L2 内部作用开关。
- 当前优化器/损失：Adam，cross entropy，统一 task BP。
- 适用实验：同 Dynamic NF。审计期间修复了关系诊断张量取样维度错误；前向状态更新未改变。

## 7. Temporal NF / sequence wrapper

**文件/版本**：`temporal_nf.py`、`train_temporal_nf.py`、`train_seq.py`。

- 连接参数：Dynamic NF 内部连接参数、输入/输出投影；LSTM/GRU 对照不属于 NF。
- 内在属性：Dynamic NF 的 relation gain、branch bias；sequence persistence 若可学习则应归 intrinsic。
- 动力学状态：序列隐藏场状态、每帧 Dynamic NF 状态及关系。
- 固定超参数：序列切分、permutation、pixelwise/rowwise 模式、sequence steps。
- 当前优化器/损失：Adam，cross entropy，统一 BP。
- 适用实验：E0/E1/E2/E5/E6；E3/E4/E7 △；其余通常不适用。

## 8. BioNeuron / BioNF

**文件/版本**：`models/bio_neuron.py`、`experiments/bio_experiments.py`。

- 连接参数：树突 excitatory/inhibitory 输入权重（dense 或 low-rank factors）、输入 projection、classifier。
- 内在属性：每神经元/树突 `branch_bias`、`branch_gain_raw`、`soma_gain_raw`、`theta_raw`、`adaptation_raw`。
- 动力学状态：input trace、dendritic state `D_t`、membrane `V_t`、adaptation `A_t`、activation/spike `S_t`。
- 固定超参数：branches、steps、trace/dendritic/membrane/adaptation decay（当前实现为 float）、continuous/hard output、temporal/inhibition/adaptation 开关、weight rank。
- 当前优化器/损失：Adam，task cross entropy/BCE；hard spike 使用 surrogate。
- 适用实验：E0–E7；局部树突连接可探索 E8 △；hard 模式适合 E9/E10。

## 9. Directional Rectangular NF v4

**文件/版本**：`nf_field.py::DirectionalRectNeuralField`、`RectNFMLPBlock`；设计源 `定向矩形神经场MLP完整架构设计_v4.docx`。

- 连接参数：输入/输出 Linear；路由 `Q`；不同能量模式下的 `kernel_raw`、`column_attr`、`full_raw`、energy-score 参数。
- 内在属性：每列/神经元 threshold `theta`、gain `g_raw`。
- 动力学状态：逐列能量、active gate、传播后的下一列状态。
- 固定超参数：宽度 W、tau_a/tau_p、route mode、energy mode、active_W、residual、input scale。列序本身承担传播时间，因此没有单独 delay/duration。
- 当前训练：硬路由 forward + softmax STE backward；硬阈值 forward + surrogate derivative；连续 task BP 联合更新 Q/T/G 与 Linear。
- 适用实验：E0–E7；Q 的离散前向适合 E9/E10；路由类别还可单独探索离散搜索，但不应和 Phase 1 混合。

## 10. General Discrete NF v3

**文件/版本**：`nf_field.py::DiscreteNeuralField`、`NFMLPBlock`/`NFGridMLPBlock`/`NFCNNBlock`/`NFPoolCNNBlock`；设计源 `通用离散神经场MLP完整架构设计_v3.docx`。

- 连接参数：输入 seed generator 的 A/c/w/b、输入/输出 Linear、readout 时间权重 eta。
- 连续内在属性：每节点 threshold T、release strength `s_param`。
- **离散内在属性**：整数 delay L、duration D。文档明确规定它们应由局部候选 `-1/0/+1` 搜索更新，不能当连续实数交给 Adam。
- 动力学状态：当前场/膜电位、spike/release、remaining duration、future-signal delay buffer、时间读出历史。
- 固定超参数：H/W、spatial/temporal steps、L_max、D_max、readout horizon R、surrogate tau、read mode、注入尺度。
- 当前训练：task BP 更新连续参数；L/D 为 buffers，当前不由 optimizer 更新。
- 适用实验：E0–E7；局部邻接可 E8 △；hard threshold 适合 E9/E10；L/D 需要独立的离散坐标搜索，列为后续 Phase 而非 E0–E2。

## 11. 远程 Habitat/full-image 包装与几何读出实验

**文件/版本**：`remote_habitat_nf_compare.py`、`remote_habitat_nf_full_replace.py`、`remote_full_image_nf_from_scratch.py`、`Geo10RectNFBlock`、teacher/geometry 实验。

- 这些主要是上述 field 的任务包装、规模变体或读出变体，不是新的神经动力学家族。
- 参数按被包装的 Local Electrical/Rect/Minimal field 规则分类；Habitat encoder/decoder/MLP 是普通连接参数。
- Phase 1 不把它们作为独立物理模型重复跑；当核心策略在 MNIST 小规模验证有效后，再做迁移检验。

## Applicability matrix

| Model family | E0 | E1 | E2 | E3 | E4 | E5 | E6 | E7 | E8 | E9 | E10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Minimal Local A | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| Minimal Local B | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | △ |
| Local Electrical v1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✗ | ✗ |
| Local Electrical v2/v3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✗ | ✗ |
| Dynamic NF | ✓ | ✓ | ✓ | △ | △ | ✓ | ✓ | △ | ✗ | ✗ | ✗ |
| Hierarchical NF | ✓ | ✓ | ✓ | △ | △ | ✓ | ✓ | △ | ✗ | ✗ | ✗ |
| Temporal NF | ✓ | ✓ | ✓ | △ | △ | ✓ | ✓ | △ | ✗ | ✗ | ✗ |
| BioNeuron continuous | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✗ | ✗ |
| BioNeuron hard | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✓ | ✓ |
| Directional Rect v4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✓ | ✓ |
| General Discrete v3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✓ | ✓ |

`✓`：定义清晰且值得测试；`△`：可实现但需模型专用活动/局部规则；`✗`：机制不存在或会人为改变模型。

## Phase 1 的优先级

1. **Minimal Local NF**：参数最少，最适合判定 E1/E2 本身是否有效，且已有 field-frozen ≈ field-trained 的关键观察。
2. **Local Electrical v1/v3**：每节点 intrinsic 属性最明确，可直接测 threshold/strength/decay 的慢学习和交替学习。
3. **Directional Rect v4 与 General Discrete v3**：分别代表路由离散性和 L/D 离散属性；Phase 1 先只验证连续 intrinsic，保留离散更新为后续独立实验。
4. **BioNeuron**：属性丰富、参数量较大，用于检查结论是否跨结构成立。
5. **Dynamic/Hierarchical NF**：可分出的 intrinsic 参数较少，作为负对照和泛化检查，而非最先耗费预算。

Phase 1 只比较 E0、E1、E2，先证明参数分组、冻结和记录可靠；E3–E10 不会在框架未经验证时批量启动。
