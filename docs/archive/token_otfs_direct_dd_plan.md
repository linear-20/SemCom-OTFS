# 面向 Token 直接传输的 OTFS 系统规划书

## 1. 项目目标

本项目目标是构建一种不经过传统 bit 化流程的 OTFS token 传输系统，将离散 token 直接映射到 OTFS 的 delay-Doppler, DD, 域资源中，并利用已有的可微无线信道模型进行端到端训练。

传统链路通常为：

```text
token -> bits -> channel coding -> QAM -> OTFS DD grid -> channel
```

目标链路为：

```text
token / token sequence -> learned DD-domain representation -> OTFS modulation
-> differentiable wireless channel -> OTFS demodulation
-> neural receiver -> recovered token / token sequence
```

核心目标：

- 避免显式 bit-level source/channel separation
- 学习 token 到 DD 域复数符号或复数图样的直接映射
- 利用 OTFS 对高速移动信道的鲁棒性
- 通过端到端训练优化 token 恢复性能
- 为语义通信或 LLM token 物理层传输提供新框架

## 2. 系统总体框架

推荐系统结构如下：

```text
Token sequence
    |
    v
Token embedding
    |
    v
Neural DD mapper
    |
    v
OTFS DD-domain grid X_DD
    |
    v
OTFS modulation
    |
    v
Differentiable wireless channel
    |
    v
OTFS demodulation
    |
    v
Neural DD receiver
    |
    v
Token classifier / decoder
    |
    v
Recovered token sequence
```

其中关键模块包括：

- Token embedding 模块
- DD 域映射模块
- OTFS 调制与解调模块
- 可微信道模块
- DD 域神经接收机
- Token 级损失函数

## 3. 阶段性实施路线

### 阶段一：最小可行系统

先从单 token 或短 token block 开始。

输入为：

```text
token id: t in {1, 2, ..., V}
```

其中 `V` 是词表大小。

发送端：

```text
t -> embedding e_t -> MLP -> K 个复数 DD symbols
```

也就是说，一个 token 不只占用一个 DD bin，而是映射为一个小的 DD patch。

例如：

```text
token -> 2 x 2 DD patch
token -> 4 x 4 DD patch
token -> K 个 scattered DD bins
```

接收端：

```text
received DD patch -> CNN / MLP -> token logits -> token prediction
```

损失函数：

```text
L = CE(t_hat, t)
  + lambda_power * L_power
  + lambda_papr * L_papr
```

优先验证：

- token accuracy vs SNR
- token error rate vs SNR
- 不同 Doppler 下的鲁棒性
- 不同 patch size 下的性能

### 阶段二：短序列 token 传输

将单 token 扩展为 token sequence。

```text
[t_1, t_2, ..., t_L]
    -> Transformer / GRU / CNN encoder
    -> full DD grid X_DD
    -> OTFS transmission
    -> neural receiver
    -> [t_hat_1, t_hat_2, ..., t_hat_L]
```

这一阶段重点不是单个 token 的孤立分类，而是利用 token 之间的上下文相关性。

可选结构：

- Transformer encoder 作为发送端 token encoder
- CNN / U-Net / Transformer 作为 DD 域接收机
- Transformer decoder 输出 token 序列

损失函数：

```text
L = sum_i CE(t_hat_i, t_i)
  + lambda_power * L_power
  + lambda_papr * L_papr
  + lambda_reg * L_regularization
```

### 阶段三：Token-aware DD 域资源分配

进一步引入 token 重要性。不同 token 可以拥有不同保护强度：

- 高频 token：可使用更紧凑映射
- 低频 token：可使用更强保护
- 语义关键 token：分配更多 DD 资源
- 控制 token / 特殊 token：提高恢复可靠性
- 低置信度 token：增加冗余

可以设计 importance weight：

```text
L = sum_i w_i * CE(t_hat_i, t_i)
```

其中 `w_i` 表示第 `i` 个 token 的重要性。

`w_i` 可以来自：

- token frequency
- language model perplexity
- attention score
- semantic importance score
- downstream task sensitivity

### 阶段四：端到端联合优化

最终系统可以联合训练：

```text
token encoder
+ DD-domain mapper
+ OTFS transmitter
+ differentiable channel
+ OTFS receiver
+ neural detector
+ token decoder
```

训练信道条件应随机化：

- SNR
- delay spread
- Doppler spread
- path number
- fractional Doppler
- channel estimation error
- mobility speed
- power constraint

推荐训练策略：

1. 低噪声、简单信道预训练
2. 加入多径 delay-Doppler 信道
3. 加入高速移动 Doppler
4. 加入随机 SNR
5. 加入信道失配和估计误差
6. 联合微调整个系统

## 4. 推荐实验设置

### 4.1 Token 设置

可以从小词表开始：

| 实验阶段 | Vocabulary size | Token sequence length |
| --- | ---: | ---: |
| 初始验证 | 16 / 64 / 256 | 1 |
| 小规模实验 | 256 / 1024 | 4-16 |
| 论文实验 | 1024 / 4096 / 8192 | 16-128 |

不建议一开始直接使用 32000 或 50000 词表，否则分类难度和训练开销都会很高。

### 4.2 OTFS 参数

建议初始参数：

| 参数 | 建议值 |
| --- | ---: |
| delay bins | 16 / 32 |
| Doppler bins | 16 / 32 |
| modulation domain | delay-Doppler |
| channel paths | 3-8 |
| SNR range | 0-30 dB |
| Doppler condition | low / medium / high mobility |

### 4.3 对比方法

至少需要以下 baseline：

1. OFDM + QAM + token bitization
2. OTFS + QAM + token bitization
3. OTFS + learned receiver
4. Proposed direct token-to-DD OTFS
5. Proposed token-aware direct token-to-DD OTFS

如果论文强调“非 bit 化 token 传输”，传统 bit 化 baseline 必须保留，否则说服力不够。

### 4.4 评价指标

基础指标：

- Token accuracy
- Token error rate, TER
- Sequence accuracy
- Cross entropy loss
- Robustness vs SNR
- Robustness vs Doppler

通信指标：

- Spectral efficiency
- Power efficiency
- Latency
- PAPR
- Complexity
- Resource usage per token

语义指标，可选：

- BLEU
- ROUGE
- BERTScore
- Perplexity after recovery
- Downstream task accuracy

## 5. 关键创新点设计

### 创新点一：Direct Token-to-DD Mapping

提出一种 token 到 OTFS delay-Doppler 域的直接映射机制，避免显式 bit 化和传统 QAM 调制。

表述建议：

```text
We propose a direct token-to-delay-Doppler mapping framework for OTFS systems,
where discrete tokens are embedded into learnable DD-domain complex patterns
and transmitted through a differentiable high-mobility wireless channel.
```

### 创新点二：Token-aware Resource Allocation

根据 token 重要性、频率或语义贡献度，自适应分配 DD 域资源。

可以设计：

```text
important token -> larger DD patch / stronger redundancy
less important token -> compact DD representation
```

### 创新点三：End-to-end Differentiable OTFS Token Transmission

利用可微 OTFS 调制、信道、解调和神经接收机，联合优化发送端与接收端。

训练目标直接面向 token 恢复：

```text
minimize token-level reconstruction loss
```

而不是传统 BER 最小化。

## 6. 重要注意事项

### 6.1 不能忽略信息容量限制

直接传 token 不代表可以绕开通信容量限制。

一个 token 至少包含：

```text
log2(V) bits
```

例如：

| Vocabulary size | Information per token |
| ---: | ---: |
| 256 | 8 bits |
| 1024 | 10 bits |
| 8192 | 13 bits |
| 32768 | 15 bits |

因此，DD 域资源数量、发射功率和信道条件必须足够支撑对应的信息量。

### 6.2 不建议一个 token 只占一个 DD bin

如果词表很大，一个 DD bin 要区分几千甚至几万个 token，本质上相当于超高阶调制，抗噪声能力会很差。

更合理的方式是：

```text
one token -> multiple DD bins
```

或者：

```text
token block -> whole DD grid
```

推荐优先采用 block-level mapping。

### 6.3 注意 DD 域干扰

OTFS 在 DD 域中通常表现为二维卷积或近似卷积关系：

```text
Y_DD = H_DD * X_DD + N
```

因此不同 token 的 DD 图样会互相干扰。

需要考虑：

- token patch 间隔
- guard region
- learned interference-aware mapping
- neural equalization
- channel-aware resource placement

### 6.4 发送功率必须归一化

神经网络可能通过无限增大发射功率来降低 loss，因此必须加入功率约束。

例如：

```text
X_DD = sqrt(P) * X_DD / ||X_DD||
```

或者加入损失项：

```text
L_power = max(0, average_power - P_limit)
```

### 6.5 需要控制 PAPR

OTFS/OFDM 类系统可能存在较高 PAPR。神经发送端如果不约束，可能学习出实际射频系统难以发送的波形。

建议加入：

```text
L_papr = PAPR(X_TF or x_time)
```

并在实验中报告 PAPR。

### 6.6 需要防止模型只记住 token ID

如果训练数据过小，模型可能变成查表器，只在固定 token 集上表现好。

建议：

- 增大 token 组合
- 使用随机 token sequence
- 使用真实语料 token 序列
- 训练集和测试集分离
- 测试 unseen token combination
- 测试不同信道分布

### 6.7 注意公平对比

和传统 bit/QAM 方法对比时，必须公平控制：

- 相同平均发射功率
- 相同 DD 资源数量
- 相同带宽
- 相同 SNR
- 相同 token rate
- 相同信道条件

否则 direct token 方法可能因为使用了更多资源而显得更好。

## 7. 推荐论文题目方向

可选题目：

1. Direct Token-to-Delay-Doppler Mapping for OTFS-Based Semantic Communications
2. Token-Oriented OTFS Transmission over High-Mobility Wireless Channels
3. End-to-End Differentiable OTFS for Direct Token-Level Wireless Transmission
4. Semantic Token Transmission via Learnable Delay-Doppler Representations in OTFS Systems
5. Token-Aware Resource Allocation for Direct Delay-Doppler Domain OTFS Transmission

## 8. 推荐下一步工作

建议马上做一个最小闭环：

```text
token id
-> embedding
-> MLP
-> K complex DD symbols
-> OTFS modulation
-> differentiable channel
-> OTFS demodulation
-> CNN / MLP receiver
-> token classifier
```

第一版实验只需要回答三个问题：

1. 不经过 bit 化，token 能不能被可靠恢复？
2. 增加 DD patch size 是否能降低 token error rate？
3. 在高 Doppler 信道下，OTFS direct-token 方法是否优于 OFDM 或普通 learned mapper？

如果这三点成立，就可以继续扩展为 token sequence 和 token-aware 资源分配。

