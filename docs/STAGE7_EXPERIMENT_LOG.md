# Stage 7 实验日志：DD Token Receiver

本文档是实验日志。早期过程只保留关键结论，重点记录 Stage 7A/7B 的有效结果、失败诊断和当前最可靠的对比数据。

## 1. 实验目标

Stage 7 的目标是训练一个学习型 DD 域 token receiver，用它替代复杂信道下的最近邻码本判决：

```text
received Y_DD
-> DDTokenPerceiverReceiver
-> token logits
-> recovered token IDs
```

当前训练仍使用随机 token maps，暂不使用真实图片 token dataset。

## 2. 环境与入口

本地环境：

```text
Windows
RTX 3060 Laptop
Python: E:\pytorch\python.exe
```

云端环境：

```text
AutoDL
Python 3.12
PyTorch 2.8.0+cu128
CUDA 12.8
Project: /root/autodl-tmp/SemCom-OTFS
```

GitHub：

```text
https://github.com/linear-20/SemCom-OTFS.git
```

云端训练入口：

```bash
bash scripts/run_stage7b_train.sh
```

稳定重评估入口：

```bash
python eval_dd_token_receiver_checkpoint.py
```

## 3. 已实现能力

训练脚本 `train_dd_token_perceiver_receiver.py` 当前支持：

```text
训练日志：
    train/eval loss
    train/eval token accuracy
    grad_norm
    param_delta
    sec_per_step

checkpoint:
    receiver_checkpoint.pt
    receiver_final.pt
    receiver_best.pt
    checkpoint_step_*.pt

训练配置：
    --resume-checkpoint
    --channel-mode identity / channel
    --fixed-channel
    --no-awgn
    --use-packed-local
    --use-channel-features
    --use-delay-window-local
    --delay-window-radius
```

## 4. Stage 7A：本地 Smoke Test

配置：

```text
codebook_size = 256
batch_size = 2
num_steps = 50
embed_dim = 128
num_heads = 4
self_attn_layers = 2
SNR = 20-30 dB
num_paths = 3
max_delay_samples = 3
max_doppler_hz = 500
```

结果：

```text
loss finite
grad_norm finite
param_delta > 0
training_log.pt saved
receiver_checkpoint.pt saved
receiver_final.pt saved
checkpoint_step_25.pt / checkpoint_step_50.pt saved
```

最后一条 smoke 记录：

```text
step = 50
train_loss = 5.557186
train_token_accuracy = 0.003906
eval_loss = 5.557599
eval_token_accuracy = 0.001953
grad_norm = 0.581195
param_delta = 0.00033822
sec_per_step = 0.0475
```

结论：

```text
Stage 7A 通过。训练链路、梯度、参数更新、日志和 checkpoint 都正常。
短训练 accuracy 接近随机水平是预期现象。
```

## 5. 原始 Blind Perceiver 诊断

### 5.1 full random channel pilot

配置：

```text
codebook_size = 256
batch_size = 4
num_steps = 1000
num_paths = 3
max_delay_samples = 3
max_doppler_hz = 500
SNR = 20-30 dB
receiver = 原始 blind Perceiver
```

结果：

```text
step 1000 eval_loss = 5.549743
step 1000 eval_token_accuracy = 0.002930
ln(256) = 5.545177
random accuracy = 1/256 = 0.003906
```

结论：

```text
原始 blind Perceiver 在 full random channel 下没有明显学起来。
```

### 5.2 identity sanity

配置：

```text
channel-mode = identity
X_DD -> OTFS modulate -> OTFS demodulate -> Y_DD
```

结果：

```text
cb256 identity 3k:
    eval_loss = 5.543744
    eval_acc = 0.0048

cb32 identity 3k:
    eval_loss = 3.462734
    eval_acc = 0.0348
```

对比随机：

```text
ln(256) = 5.545177, random acc = 0.003906
ln(32) = 3.465736, random acc = 0.03125
```

结论：

```text
原始 blind Perceiver 有很弱学习信号，但对当前固定 packing 缺少归纳偏置。
```

## 6. Packed-Local Receiver

设计：

```text
当前 TokenDDMapper 使用固定 packing：
每个 token -> 连续 symbols_per_token 个 DD bins

packed-local receiver：
每个 token 额外读取对应的 4 个复数 DD bins，
用 MLP 编码后融合到 token hidden。
```

结果：

```text
cb32 identity + packed-local:
    eval_acc = 1.0000
    eval_loss = 0.000093

cb256 identity + packed-local:
    eval_acc = 1.0000
    eval_loss = 0.000058

cb256 random flat Rayleigh + AWGN + packed-local:
    eval_acc = 1.0000
    eval_loss = 0.000097
```

结论：

```text
packed-local 解决了固定 mapper packing 的归纳偏置问题。
identity 和 flat fading 下 receiver 容量、训练机制都正常。
```

## 7. Random Multipath Delay 瓶颈

主要结果：

```text
fixed multipath delay / no AWGN:
    eval_acc ≈ 0.9367
    eval_loss ≈ 0.160507

fixed multipath delay / AWGN:
    eval_acc ≈ 0.9351
    eval_loss ≈ 0.173224

random multipath delay / no AWGN:
    eval_acc ≈ 0.4554
    eval_loss ≈ 2.950565

random multipath delay / AWGN:
    5k eval_acc ≈ 0.5391
    10k eval_acc ≈ 0.4187
```

结论：

```text
AWGN 不是主要瓶颈。
固定 delay 扩散可学到约 0.94。
随机 multipath delay 使 blind packed-local 降到约 0.45-0.54。
主要问题是 receiver 对随机多径信道变化不够稳健。
```

## 8. Channel-Aware Packed-Local

设计：

```text
DDTokenPerceiverReceiver.forward(y_dd, channel_features=None)

channel_features:
    [delay / max_delay_samples, path_gain.real, path_gain.imag]

channel_features -> MLP -> channel embedding
注入 token queries / token hidden
```

参数：

```text
--use-channel-features
--max-channel-paths 3
```

小 eval batch 训练日志曾出现较高峰值：

```text
seed0 best train eval ≈ 0.6837
seed1 final train eval ≈ 0.5588
```

但 128-batch 稳定重评估后：

```text
channel-aware checkpoint 2500:
    eval_acc = 0.474892

channel-aware checkpoint 5000:
    eval_acc = 0.482162
```

对比：

```text
blind packed-local baseline ≈ 0.455
channel-aware global only ≈ 0.475-0.482
```

结论：

```text
全局 CSI embedding 有小幅稳定提升，但不足以解决 random delay。
瓶颈不只是“有没有 CSI”，还包括 token 局部观测范围太窄。
```

## 9. Delay-Window Local Receiver

设计动机：

```text
random multipath delay 会让能量沿 delay 方向扩散。
packed-local 只读每个 token 原始 4 个 DD bins，容易漏掉邻域能量。
```

新增局部观察：

```text
--use-delay-window-local
--delay-window-radius r
```

每个 token 的局部输入：

```text
symbols_per_token * (2r + 1) complex samples
```

例如：

```text
r = 1: 4 * 3 = 12 complex = 24 real features
r = 2: 4 * 5 = 20 complex = 40 real features
r = 3: 4 * 7 = 28 complex = 56 real features
```

## 10. 当前最可靠结果表

统一条件：

```text
codebook_size = 256
batch_size = 8
num_steps = 5000
num_paths = 3
max_delay_samples = 3
max_doppler_hz = 0
no_awgn = true
eval_batches = 128
seed = 0
```

| Receiver | CSI | Delay Window | Stable Eval Acc |
|---|---:|---:|---:|
| blind packed-local baseline | no | no | ≈ 0.455 |
| channel-aware packed-local | yes | no | 0.482162 |
| delay-window local r=1 | yes | r=1 | 0.540382 |
| delay-window local r=2 | yes | r=2 | 0.574120 |
| delay-window local r=3 | yes | r=3 | 0.564617 |
| delay-window local r=2 | no | r=2 | 0.484348 |

关键结论：

```text
1. delay-window local 明显有效。
2. r=2 当前最好。
3. 只加 CSI 的提升很小。
4. 只加 delay-window 但不加 CSI 也只有约 0.484。
5. 最有效的是 delay-window local + CSI 的组合。
```

当前阶段最重要结论：

```text
在 random multipath delay + no AWGN 下，
token-level receiver 需要同时获得：
1. delay 方向局部邻域观测
2. 信道状态信息

仅全局 CSI embedding 或仅扩大局部窗口都不够强，
二者组合能把稳定 token accuracy 从约 0.455 提升到约 0.574。
```

## 11. 推荐复现实验命令

当前最佳配置：

```bash
bash scripts/run_stage7b_train.sh \
  --mode train \
  --output-tag delaywin_r2_ca_cb256_random_delay_noawgn_5k_seed0 \
  --codebook-size 256 \
  --batch-size 8 \
  --steps 5000 \
  --use-delay-window-local \
  --delay-window-radius 2 \
  --use-channel-features \
  --max-channel-paths 3 \
  --max-delay-samples 3 \
  --max-doppler-hz 0 \
  --no-awgn \
  --eval-every 250 \
  --eval-batches 16 \
  --save-every 250 \
  --seed 0
```

稳定重评估：

```bash
python eval_dd_token_receiver_checkpoint.py \
  --checkpoint outputs/stage7b_receiver_train/delaywin_r2_ca_cb256_random_delay_noawgn_5k_seed0/receiver_best.pt \
  --eval-batches 128 \
  --batch-size 8 \
  --device cuda
```

无 CSI 对照：

```bash
bash scripts/run_stage7b_train.sh \
  --mode train \
  --output-tag delaywin_r2_nocsi_cb256_random_delay_noawgn_5k_seed0 \
  --codebook-size 256 \
  --batch-size 8 \
  --steps 5000 \
  --use-delay-window-local \
  --delay-window-radius 2 \
  --max-delay-samples 3 \
  --max-doppler-hz 0 \
  --no-awgn \
  --eval-every 250 \
  --eval-batches 16 \
  --save-every 250 \
  --seed 0
```

## 12. 暂时不要做

```text
不要直接加 Doppler。
不要上真实图片 token dataset。
不要训练 learnable mapper。
不要修改 channel_model.py。
不要只看训练中的小 eval batch 峰值，必须做 128-batch re-eval。
```
