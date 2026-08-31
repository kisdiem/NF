# 两层分层 Dynamic NF

本实验在当前 Dynamic NF 旁增加 8+8 分层版本，不修改原始 flat Dynamic NF。

## 数据流

```text
Linear(784,64)
→ reshape [B,16,4]
→ split：Layer1 [B,8,4] + Layer2 [B,8,4]
→ Layer1 内部关系 R11 [B,8,8]
→ Layer1→Layer2 跨层关系 R12 [B,8,8]
→ Layer2 内部关系 R22 [B,8,8]
→ 两层同步消息、局部 branch 和 gated update
→ concat [B,16,4] → flatten [B,64] → Linear(64,10)
```

默认不启用 Layer2→Layer1 反馈；`hier_feedback` 使用额外的弱 `R21 [B,8,8]`，初始增益为 0.02。

每个时间步都从旧状态同时计算关系和消息，然后同步更新两层，没有层内节点顺序依赖。

## 运行

```powershell
py -3 train_hierarchical_nf.py --epochs 5 --subset 5000 --batch 128
py -3 train_hierarchical_nf.py --epochs 10 --subset 5000 --batch 128
```

默认比较 Linear、ReLU、GELU、flat Dynamic NF、hier、hier_no_l2、hier_feedback。

## 参数与理论关系计算量

| 模型 | 节点关系 | 每步关系矩阵元素 | 每步 QK 乘法量 |
|---|---|---:|---:|
| Flat Dynamic NF | 1×16×16 | 256 | 1024 |
| Hierarchical NF | R11+R12+R22 | 3×8×8=192 | 768 |
| Hierarchical + feedback | 再加 R21 | 4×8×8=256 | 1024 |

无反馈分层版理论关系分数乘法比 flat 少 25%。但实际 wall/CPU 时间不一定更快，因为分层版有两套局部更新和额外的关系组织开销。

## MNIST 结果

配置：子集 5000、batch=128、hidden=64、10 轮、seed=0。

| 模型 | 最高准确率 | 最终准确率 | 参数量 | CPU时间 |
|---|---:|---:|---:|---:|
| Linear | 90.61% | 90.43% | 50,890 | 94.1s |
| ReLU | 92.40% | 92.40% | 50,890 | 99.2s |
| GELU | 92.13% | 92.13% | 50,890 | 82.9s |
| Flat Dynamic NF | 92.53% | 92.53% | 51,183 | 127.9s |
| Hierarchical | 91.92% | 91.26% | 51,405 | 153.1s |
| Hierarchical，无 Layer2 内部关系 | 92.67% | 92.34% | 51,405 | 142.1s |
| Hierarchical，弱反馈 | 92.49% | 92.06% | 51,406 | 153.6s |

## 当前结论

分层版本确实减少了无反馈关系矩阵的理论规模，但没有减少实际训练时间；默认分层结构也没有提高准确率。关闭 Layer2 内部作用后结果略好，提示 Layer2 内部关系可能与跨层组合发生冗余或干扰。弱反馈没有带来收益，第一版不应继续扩大反馈机制。

关系矩阵和 state-change 图保存在 `hierarchical_nf_results/`。
