# Stage 7 实验日志：DD Token Receiver

## 1. 当前代码状态

本阶段围绕 `DDTokenPerceiverReceiver` 和训练入口完成了以下更新：

```text
train_dd_token_perceiver_receiver.py
scripts/run_stage7a_train.ps1
scripts/run_stage7b_train.sh
dd_token_perceiver_receiver.py
```

主要能力：

```text
训练日志：train/eval loss、token accuracy、grad_norm、param_delta、耗时
checkpoint：receiver_checkpoint.pt、receiver_final.pt、checkpoint_step_*.pt
断点续训：--resume-checkpoint
云端入口：scripts/run_stage7b_train.sh
信道诊断：--channel-mode identity、--fixed-channel、--no-awgn
receiver 结构：可选 --use-packed-local
```

GitHub 仓库：

```text
https://github.com/linear-20/SemCom-OTFS.git
```

关键提交：

```text
b172822 Prepare Stage 7B receiver training
b14da56 Add deterministic channel diagnostics
6798ffe Add packed local receiver path
```

## 2. Stage 7A：本机 GPU Smoke Test

环境：

```text
本机 GPU：RTX 3060 Laptop
device：cuda
codebook_size：256
batch_size：2
num_steps：50
embed_dim：128
self_attn_layers：2
```

结果：

```text
precheck 10 steps：通过
smoke 50 steps：通过
loss：finite，无 NaN/Inf
grad_norm：finite
param_delta：> 0
training_log.pt：已保存
receiver_checkpoint.pt：已保存
receiver_final.pt：已保存
checkpoint_step_25.pt / checkpoint_step_50.pt：已保存
```

smoke 最后一次记录：

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
Stage 7A 通过。
训练链路、梯度、参数更新、日志和 checkpoint 均正常。
accuracy 接近随机水平是短训练 smoke test 的预期现象。
```

## 3. Stage 7B：云端训练环境

云端环境：

```text
AutoDL
Python 3.12
PyTorch 2.8.0+cu128
CUDA 12.8
项目路径：/root/autodl-tmp/SemCom-OTFS
```

GitHub 直连曾超时，使用代理成功 clone：

```bash
git clone https://gh-proxy.com/https://github.com/linear-20/SemCom-OTFS.git
```

2-step 云端验证已通过：

```text
device = cuda
loss finite
param_delta > 0
training_log.pt / receiver_checkpoint.pt / receiver_final.pt 正常保存
```

## 4. 原始 Blind Perceiver 诊断

### 4.1 full random channel pilot

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

### 4.2 identity sanity

新增 `--channel-mode identity` 后，绕过信道，只保留：

```text
X_DD -> OTFS modulate -> OTFS demodulate -> Y_DD
```

原始 blind Perceiver 结果：

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
原始 blind Perceiver 有学习信号，但非常弱。
问题不只来自信道，receiver 对当前固定 packing 的 token-DD mapper 缺少足够归纳偏置。
```

## 5. Packed-Local Receiver

为 `DDTokenPerceiverReceiver` 增加可选：

```text
--use-packed-local
```

设计：

```text
当前 TokenDDMapper 使用固定 packing：
每个 token -> 连续 symbols_per_token 个 DD bins

packed-local receiver 额外读取每个 token 对应的 4 个复数 DD bins，
用 MLP 编码成 token-local embedding，
再与 Perceiver token hidden 融合。
```

注意：

```text
这仍然是 token 级接收机。
没有 bit/QAM。
没有修改 channel_model.py。
```

## 6. Packed-Local 结果

### 6.1 identity

cb32 identity + packed-local：

```text
step 3000 eval_loss = 0.000093
step 3000 eval_acc = 1.0000
```

cb256 identity + packed-local：

```text
step 5000 eval_loss = 0.000058
step 5000 eval_acc = 1.0000
```

结论：

```text
packed-local receiver 可以完全学会固定 mapper + OTFS mod/demod 的 token 判决。
训练机制和模型容量在无信道条件下没有问题。
```

### 6.2 random flat Rayleigh + AWGN

配置：

```text
codebook_size = 256
num_paths = 1
max_delay_samples = 0
max_doppler_hz = 0
SNR = 20-30 dB
random Rayleigh flat fading
packed-local
```

结果：

```text
step 5000 eval_loss = 0.000097
step 5000 eval_acc = 1.0000
```

结论：

```text
packed-local receiver 可以处理随机复增益/相位旋转和 AWGN。
```

### 6.3 random multipath delay + AWGN

配置：

```text
codebook_size = 256
num_paths = 3
max_delay_samples = 3
max_doppler_hz = 0
SNR = 20-30 dB
random multipath delay
packed-local
```

5k 结果：

```text
step 5000 eval_loss = 2.246095
step 5000 eval_acc = 0.5391
```

续训到 10k 后：

```text
step 10000 eval_loss = 2.954157
step 10000 eval_acc = 0.4187
```

结论：

```text
random multipath delay 可学，但不稳定。
继续盲训没有稳定提升。
```

### 6.4 fixed multipath delay

fixed multipath delay / no AWGN：

```text
step 5000 eval_loss = 0.160507
step 5000 eval_acc = 0.9367
```

fixed multipath delay / AWGN：

```text
step 5000 eval_loss = 0.173224
step 5000 eval_acc = 0.9351
```

random multipath delay / no AWGN：

```text
step 5000 eval_loss = 2.950565
step 5000 eval_acc = 0.4554
```

结论：

```text
噪声不是主要瓶颈。
固定多径 delay 可以学到约 0.94 token accuracy。
随机多径 delay 即使无 AWGN 也只有约 0.45 token accuracy。
主要瓶颈是 blind receiver 对随机多径信道变化不够稳健。
```

## 7. 当前总结合理表述

可以作为论文/报告中的阶段性结论：

```text
原始 blind Perceiver receiver 在固定 random mapper 的 identity 任务上学习信号很弱。
加入 mapper-packing 对应的 packed-local 观测路径后，receiver 在 identity 和 flat fading + AWGN 下均达到 100% token accuracy。
在固定 multipath delay 下，packed-local receiver 可以达到约 0.94 token accuracy。
但在随机 multipath delay 下，blind packed-local receiver 下降到约 0.45-0.54，说明瓶颈主要来自未知随机信道变化，而不是 AWGN 或固定 delay 扩散本身。
```

## 8. 下一步建议

下一步进入简化版 Stage 11：

```text
Channel-aware packed-local receiver
```

优先目标：

```text
在 random multipath delay + no AWGN 条件下，
利用 channel_model.py 输出的 delays 和 path_gains，
让 receiver 明显优于 blind packed-local baseline。
```

建议实现：

```text
DDTokenPerceiverReceiver.forward(y_dd, channel_features=None)

channel_features:
    delays
    path_gains.real
    path_gains.imag

channel MLP:
    channel vector -> channel embedding
    加到 token queries 或 token hidden
```

训练脚本新增参数：

```text
--use-channel-features
--max-channel-paths 3
```

第一组对照：

```text
blind packed-local:
    random multipath delay + no AWGN
    eval_acc ≈ 0.455

channel-aware packed-local:
    random multipath delay + no AWGN
    目标：明显高于 0.455
```

暂时不要做：

```text
不要加 Doppler。
不要上真实图片 token dataset。
不要训练 learnable mapper。
不要改 channel_model.py。
```

## 9. Channel-Aware Packed-Local Receiver 已接入

本地代码已完成最小实现：

```text
DDTokenPerceiverReceiver.forward(y_dd, channel_features=None)
train_dd_token_perceiver_receiver.py:
    --use-channel-features
    --max-channel-paths 3
scripts/run_stage7b_train.sh:
    --use-channel-features
    --max-channel-paths 3
```

当前 channel feature 定义为每条路径 3 个实数：

```text
[delay / max_delay_samples, path_gain.real, path_gain.imag]
```

第一组云端建议命令：

```bash
bash scripts/run_stage7b_train.sh \
  --mode train \
  --output-tag ca_packed_cb256_random_delay_noawgn_5k_seed0 \
  --codebook-size 256 \
  --batch-size 8 \
  --steps 5000 \
  --use-packed-local \
  --use-channel-features \
  --max-channel-paths 3 \
  --max-delay-samples 3 \
  --max-doppler-hz 0 \
  --no-awgn \
  --eval-every 250 \
  --eval-batches 4 \
  --save-every 500 \
  --seed 0
```

本地验证已通过：

```text
channel-aware dry-run: channel_features shape = [2, 3, 3]
blind packed-local compatibility dry-run: passed
channel-aware tiny CPU train 2 steps: loss finite, grad_norm finite, param_delta > 0
```
