# 下一步规划书

本文档只写从当前状态出发的后续路线。实验细节与历史结果看 `STAGE7_EXPERIMENT_LOG.md`，程序使用方法看 `PROJECT_MANUAL.md`。

## 1. 当前判断

项目已经完成基础链路和 Stage 7B 的关键诊断：

```text
图片/token
-> DD grid
-> OTFS time waveform
-> random multipath delay channel
-> received DD grid
-> learned token receiver
-> recovered tokens
```

当前最可靠结论：

```text
blind packed-local baseline:
    random multipath delay + no AWGN 下约 0.455 token accuracy

channel-aware packed-local:
    稳定重评估约 0.482

delay-window r=2 + channel-aware:
    稳定重评估约 0.574

delay-window r=2 no CSI:
    稳定重评估约 0.484
```

因此当前瓶颈可以表述为：

```text
随机多径 delay 会造成 DD 域局部扩散。
receiver 既需要 delay 方向邻域观测，也需要 CSI。
只给全局 CSI 或只扩大局部窗口都不够强。
```

## 2. 最近目标

近期不要跳到真实图片、learnable mapper 或 Doppler。先把 Stage 7B 的结论做扎实：

```text
目标 1：确认 delay-window + CSI 在多个 seed 下稳定优于 baseline。
目标 2：完成 radius 消融和 CSI 消融。
目标 3：确认加入 AWGN 后性能是否保持。
目标 4：整理为 Stage 7B 可写进论文/报告的实验表。
```

## 3. 近期实验矩阵

统一基础配置：

```text
codebook_size = 256
batch_size = 8
num_steps = 5000
num_paths = 3
max_delay_samples = 3
max_doppler_hz = 0
cp_len = 4
eval_batches for train = 16
eval_batches for re-eval = 128
```

### 3.1 已完成对照

| Experiment | Stable Eval Acc | 结论 |
|---|---:|---|
| blind packed-local | ≈0.455 | baseline |
| channel-aware packed-local | 0.482 | CSI alone helps little |
| delay-window r=1 + CSI | 0.540 | window helps |
| delay-window r=2 + CSI | 0.574 | current best |
| delay-window r=3 + CSI | 0.565 | larger not always better |
| delay-window r=2 no CSI | 0.484 | window alone not enough |

### 3.2 下一组优先实验：多 seed 验证

目的：

```text
确认 r=2 + CSI 的提升不是 seed0 偶然。
```

建议跑：

```text
seed = 1
seed = 2
```

命令模板，seed1：

```bash
bash scripts/run_stage7b_train.sh \
  --mode train \
  --output-tag delaywin_r2_ca_cb256_random_delay_noawgn_5k_seed1 \
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
  --seed 1
```

seed2 只改：

```text
--output-tag delaywin_r2_ca_cb256_random_delay_noawgn_5k_seed2
--seed 2
```

每个 seed 跑完后都做：

```bash
python eval_dd_token_receiver_checkpoint.py \
  --checkpoint outputs/stage7b_receiver_train/delaywin_r2_ca_cb256_random_delay_noawgn_5k_seed1/receiver_best.pt \
  --eval-batches 128 \
  --batch-size 8 \
  --device cuda
```

验收标准：

```text
如果 seed0/1/2 平均 token accuracy 稳定高于 0.53，
则可确认 delay-window r=2 + CSI 是当前 Stage 7B 最佳方案。
```

## 4. 第二优先实验：加入 AWGN

前提：

```text
多 seed 验证通过后再做。
```

目的：

```text
验证 delay-window r=2 + CSI 在 random multipath delay + AWGN 下是否仍优于 baseline。
```

建议配置：

```text
SNR = 20-30 dB
max_doppler_hz = 0
no_awgn = false
```

命令：

```bash
bash scripts/run_stage7b_train.sh \
  --mode train \
  --output-tag delaywin_r2_ca_cb256_random_delay_awgn_5k_seed0 \
  --codebook-size 256 \
  --batch-size 8 \
  --steps 5000 \
  --use-delay-window-local \
  --delay-window-radius 2 \
  --use-channel-features \
  --max-channel-paths 3 \
  --max-delay-samples 3 \
  --max-doppler-hz 0 \
  --snr-min 20 \
  --snr-max 30 \
  --eval-every 250 \
  --eval-batches 16 \
  --save-every 250 \
  --seed 0
```

重评估：

```bash
python eval_dd_token_receiver_checkpoint.py \
  --checkpoint outputs/stage7b_receiver_train/delaywin_r2_ca_cb256_random_delay_awgn_5k_seed0/receiver_best.pt \
  --eval-batches 128 \
  --batch-size 8 \
  --device cuda
```

预期：

```text
如果 no AWGN 为 0.57 左右，
20-30 dB AWGN 下应不低于 0.50。
```

## 5. 第三优先实验：模型结构小改

如果多 seed 或 AWGN 后发现性能不稳定，再考虑结构小改，不要马上加大系统复杂度。

可选改动：

```text
1. delay-window local 使用 channel-conditioned gate
2. 将 delay-window radius 与 CSI delay 动态关联
3. 对 local window 加 attention pooling，而不是简单 MLP flatten
```

优先级建议：

```text
先做 attention pooling。
再做 channel-conditioned gate。
最后再考虑动态 window。
```

原因：

```text
r=3 不如 r=2，说明窗口变大后会引入无关信息。
attention pooling 可能让 receiver 在窗口内自动挑有用 bins。
```

## 6. 暂缓事项

### 6.1 暂缓 Doppler

原因：

```text
当前只处理 random delay 就还没有完全稳定。
Doppler 会引入另一个维度的扩散和时变，过早加入会让诊断变混。
```

进入 Doppler 前的条件：

```text
delay-window r=2 + CSI 在 random delay + AWGN 下稳定优于 baseline。
```

### 6.2 暂缓真实图片 token dataset

原因：

```text
当前 receiver 仍在通信层诊断阶段。
真实 token 分布会引入数据分布偏置，不利于定位信道/receiver 问题。
```

进入真实图片 token 前的条件：

```text
random token 下 receiver 架构和评估脚本稳定。
```

### 6.3 暂缓 learnable mapper

原因：

```text
当前固定 mapper + receiver 的瓶颈还没完全处理清楚。
过早训练 mapper 会混合 mapper 和 receiver 两类变量。
```

进入 learnable mapper 前的条件：

```text
固定 mapper 下有明确、稳定、可复现的 receiver baseline。
```

## 7. 中期路线

### Stage 8：系统评估脚本

目标：

```text
写一个统一评估脚本，系统比较：
raw nearest-neighbor
oracle scalar equalization
blind packed-local
channel-aware packed-local
delay-window local + CSI
```

建议脚本名：

```text
eval_stage8_receiver_suite.py
```

输出：

```text
outputs/stage8_receiver_suite/*.pt
outputs/stage8_receiver_suite/*.csv
```

指标：

```text
token accuracy
token error rate
sequence accuracy
eval loss
channel config
model config
checkpoint path
```

### Stage 9：真实图片 token dataset

目标：

```text
批量图片 -> tokenizer -> token maps dataset
```

建议脚本：

```text
build_token_dataset.py
```

输出：

```text
datasets/visual_token_maps.pt
```

### Stage 10：真实 token receiver 训练

目标：

```text
比较 random tokens 与 real image tokens 上的 receiver 表现。
```

训练脚本新增：

```text
--token-dataset
--token-source random / real / mixed
```

### Stage 11：Doppler-aware receiver

前提：

```text
random delay + AWGN 已稳定。
```

逐步加入：

```text
max_doppler_hz = 100
max_doppler_hz = 500
max_doppler_hz = 1000
```

CSI 扩展：

```text
[delay, doppler, gain.real, gain.imag]
```

### Stage 12：Learnable token-to-DD mapper

目标：

```text
固定 receiver 诊断清楚后，再训练 mapper。
```

基本形式：

```text
nn.Embedding(codebook_size, 2 * symbols_per_token)
-> complex codewords
-> average power normalization
-> DD grid packing
```

### Stage 13：完整图片级评估

链路：

```text
image
-> visual tokens
-> communication chain
-> recovered tokens
-> reconstructed image
```

指标：

```text
token accuracy
token error rate
PSNR
SSIM
LPIPS, optional
CLIP similarity, optional
```

## 8. 当前下一步一句话

```text
先跑 delay-window r=2 + CSI 的 seed1 / seed2，并用 128-batch re-eval 验证均值。
如果稳定，再加入 AWGN；暂时不要加 Doppler、真实图片 token 或 learnable mapper。
```
