# 项目上下文压缩摘要

## 1. 基本信息

项目路径：

```text
F:\OTFS下的token传输
```

Python / PyTorch 环境：

```text
E:\pytorch\python.exe
```

这是本项目使用的 PyTorch conda 环境。后续运行脚本时优先使用该解释器。

## 2. 项目目标

研究一种不经过传统 bit / QAM 映射的 OTFS token 传输链路：

```text
image
-> visual tokens
-> DD-domain complex representation
-> OTFS modulation
-> wireless channel
-> OTFS demodulation
-> DD-domain receiver
-> recovered tokens
-> reconstructed image
```

核心思想是：

```text
token 直接映射到 OTFS delay-Doppler 域复数码字
```

而不是：

```text
token -> bits -> channel coding -> QAM -> OTFS
```

## 3. 外部源码和模型

```text
LlamaGen\
pretrained_models\vq_ds16_c2i.pt
```

`LlamaGen` 是图像 tokenizer 的外部源码，通常不改。

## 4. 当前目录结构

根目录保留源码：

```text
image_tokenizer.py
token_dd_mapper.py
otfs_modem.py
channel_model.py
run_token_otfs_roundtrip.py
run_token_otfs_awgn_sweep.py
run_token_otfs_channel_sweep.py
run_token_otfs_channel_equalized.py
dd_token_perceiver_receiver.py
train_dd_token_perceiver_receiver.py
```

文档：

```text
docs\PROJECT_MANUAL.md
docs\ROADMAP_NEXT.md
docs\token_dd_mapper_task.md
docs\token_otfs_direct_dd_plan.md
docs\CONVERSATION_SUMMARY.md
```

输入：

```text
inputs\demo.png
```

输出目录：

```text
outputs\tokenizer\
outputs\stage1_token_dd\
outputs\stage2_otfs\
outputs\stage3_roundtrip\
outputs\stage4_awgn\
outputs\stage5_channel\
outputs\stage6_equalized\
outputs\stage7_perceiver_receiver\
```

## 5. 核心模块说明

### 5.1 image_tokenizer.py

作用：

```text
图片 <-> visual tokens
```

主要类：

```text
ImageTokenizer
```

主要方法：

```text
encode(image) -> token_ids [B, 16, 16]
decode(token_ids) -> reconstructed image [B, 3, 256, 256]
```

已有 `tokens.pt` 信息：

```text
token_ids shape: [1, 16, 16]
codebook_size: 16384
grid_size: (16, 16)
image_size: 256
downsample_ratio: 16
```

### 5.2 token_dd_mapper.py

作用：

使用固定随机复数码本，把 token IDs 直接映射到 DD 域复数码字。该模块不做 bit / QAM。

主要类：

```text
TokenDDMapper
```

核心数学：

```text
token t -> codeword x_t in C^K
```

当前默认：

```text
codebook_size = 16384
symbols_per_token = 4
dd_shape = (32, 32)
```

因为：

```text
16 * 16 tokens * 4 symbols/token = 32 * 32 DD bins
```

主要方法：

```text
encode(token_ids) -> dd_grid [B, 32, 32] complex
decode(dd_grid, token_shape=(16,16)) -> recovered token_ids [B, 16, 16]
roundtrip
token_accuracy
token_error_rate
```

阶段 1 已验证：

```text
tokens.pt -> dd_grid.pt -> tokens_recovered.pt
token accuracy = 1.0
token error rate = 0.0
```

### 5.3 otfs_modem.py

作用：

```text
DD grid -> OTFS time waveform -> DD grid
```

主要类：

```text
OTFSModem
```

主要方法：

```text
isfft
sfft
modulate(x_dd) -> time_signal
demodulate(time_signal) -> y_dd
roundtrip
normalized_mse
max_abs_error
```

采用：

```text
torch.fft, norm="ortho"
```

默认：

```text
dd_shape = (32, 32)
cp_len = 0 或 4
```

时域长度：

```text
cp_len=0 -> time shape [B, 1024]
cp_len=4 -> time shape [B, 1152]
```

阶段 2 已验证：

```text
DD -> OTFS -> DD
normalized MSE ≈ 7.78e-14
max abs error ≈ 8.4e-7
```

### 5.4 run_token_otfs_roundtrip.py

作用：

串联 `token_dd_mapper.py` 和 `otfs_modem.py`。

链路：

```text
tokens.pt
-> X_DD
-> OTFS modulate
-> OTFS demodulate
-> Y_DD
-> recovered tokens
```

无信道、无噪声。

阶段 3 已验证：

```text
cp_len=0 和 cp_len=4 均 token accuracy = 1.0
```

### 5.5 run_token_otfs_awgn_sweep.py

作用：

在 OTFS 时域加入复 AWGN，扫描 SNR。

链路：

```text
tokens -> DD -> OTFS -> AWGN -> OTFS -> DD -> token
```

阶段 4 结果：

无 CP：

```text
SNR 0 dB:  token accuracy mean ≈ 0.0127
SNR 10 dB: token accuracy mean ≈ 0.6576
SNR 20 dB: token accuracy mean = 1.0
SNR 30 dB: token accuracy mean = 1.0
```

带 CP：

```text
SNR 10 dB: token accuracy mean ≈ 0.6656
SNR 20/30 dB: token accuracy mean = 1.0
```

### 5.6 channel_model.py

已有信道模型，不要改。

作用：

```text
时域复基带时变多径多普勒信道
```

主要类：

```text
ChannelConfig
TimeVaryingMultipathChannel
```

模型：

```text
y[n] = sum_p h_p exp(j 2*pi*f_p*n/fs) x[n - tau_p] + w[n]
```

输入 shape：

```text
[time] 或 [batch, time] complex
```

因此必须接在 `OTFSModem.modulate()` 之后，不能直接接 DD grid。

主要参数：

```text
num_paths
sample_rate
snr_db
max_delay_samples
max_doppler_hz
fading: rayleigh / rician / fixed
rician_k_db
doppler_distribution: jakes / uniform
randomize_each_forward
fractional_delays
seed
```

`return_info=True` 可得到：

```text
out.y
out.clean
out.noise
out.path_gains
out.delays
out.dopplers_hz
out.conditioning
```

### 5.7 run_token_otfs_channel_sweep.py

作用：

复用 `channel_model.py`，做多径多普勒 baseline。

链路：

```text
tokens -> DD -> OTFS -> TimeVaryingMultipathChannel -> OTFS demod -> DD -> token
```

阶段 5 已验证：

近似无信道 sanity check：

```text
num_paths=1
max_delay=0
max_doppler=0
SNR=60
accuracy=1.0
DD NMSE≈1e-6
```

多径多普勒 baseline：

```text
cp_len=4
num_paths=3
max_delay_samples=3
max_doppler_hz=500
```

结果：

```text
accuracy 很低且不稳定
```

这是无均衡器条件下的正常现象。

### 5.8 run_token_otfs_channel_equalized.py

作用：

做诊断和 oracle scalar equalization baseline。

核心：

```text
alpha = sum(Y_DD * conj(X_DD)) / sum(abs(X_DD)^2)
Y_eq = Y_DD / alpha
```

注意：

```text
这是 oracle scalar equalizer，使用了发送端 X_DD，只用于诊断，不是实际接收机。
```

阶段 6 结果：

Sanity 60 dB：

```text
raw accuracy = 1.0
equalized accuracy = 1.0
alpha abs ≈ 1
```

多径多普勒 30 dB：

```text
raw accuracy mean ≈ 0.0129
scalar equalized accuracy mean ≈ 0.2959
```

结论：

```text
问题不只是全局幅度/相位旋转，仍有明显 DD 扩散/卷积失真。
后续需要学习型接收机或更完整的均衡器。
```

### 5.9 dd_token_perceiver_receiver.py

作用：

DD 域 Perceiver receiver 模型，不使用 CNN。

主要类：

```text
DDTokenPerceiverReceiver
```

输入：

```text
y_dd complex [B, M, N]
```

处理：

```text
real/imag + normalized DD position
-> DD bin embeddings [B, M*N, D]
-> learned token queries [Ht*Wt, D]
-> cross-attention: token queries read DD tokens
-> token self-attention
-> classifier
```

输出：

```text
logits [B, Ht*Wt, codebook_size]
```

另有：

```text
count_parameters
```

dry-run 已验证配置：

```text
codebook_size = 256
token_shape = (16,16)
dd_shape = (32,32)
batch_size = 2
embed_dim = 128
num_heads = 4
self_attn_layers = 2
```

dry-run 输出：

```text
token_ids [2,16,16]
x_dd [2,32,32]
x_time [2,1152]
y_time [2,1152]
y_dd [2,32,32]
logits [2,256,256]
loss ≈ 5.7113
params ≈ 529536
```

### 5.10 train_dd_token_perceiver_receiver.py

作用：

Perceiver receiver 训练脚本骨架，目前已支持 dry-run。

链路：

```text
random tokens
-> TokenDDMapper.encode
-> X_DD
-> OTFSModem.modulate
-> TimeVaryingMultipathChannel
-> OTFSModem.demodulate
-> DDTokenPerceiverReceiver
-> logits
-> CE loss
```

默认：

```text
num_steps = 0
```

不会正式训练。

dry-run 已跑通并保存：

```text
outputs\stage7_perceiver_receiver\perceiver_dry_run.pt
```

## 6. 三类 decode 的区别

项目里有三类“decode”：

### 6.1 ImageTokenizer.decode

```text
visual token IDs -> reconstructed image
```

属于图像 tokenizer 层。

### 6.2 TokenDDMapper.decode

```text
DD grid -> nearest-neighbor codebook matching -> token IDs
```

属于通信层传统最近邻 token 判决。

### 6.3 DDTokenPerceiverReceiver

```text
received DD grid -> token logits -> argmax -> token IDs
```

属于学习型通信接收机。

未来复杂信道下，Perceiver receiver 用于替代 `TokenDDMapper.decode`。恢复出的 tokens 再交给 `ImageTokenizer.decode` 还原图片。

完整链路：

```text
received DD
-> Perceiver receiver
-> recovered tokens
-> ImageTokenizer.decode
-> reconstructed image
```

## 7. 理论依据摘要

因为发送端直接定义了：

```text
token -> X_DD
```

所以接收端直接学习：

```text
p(T | Y_DD)
```

是合理的。

OTFS 解调后的 `Y_DD` 是接收时域信号的 DD 域表示，保留主要信息，并且多径多普勒失真在 DD 域结构更清晰。

直接 token 级接收机优化：

```text
CE(T, T_hat)
```

比先恢复 `X_DD` 的 MSE 更贴合最终目标。

Perceiver 比普通 Transformer 更适合当前任务：

```text
输入：1024 DD bins
输出：256 token predictions
```

普通 Transformer：

```text
DD bins 之间 self-attention，复杂度约 1024 x 1024
```

Perceiver：

```text
256 token queries cross-attend to 1024 DD bins
然后做 token self-attention
```

更贴合：

```text
每个 token query 从整个 DD grid 读取信息
```

## 8. 后续规划

### Stage 7A：短训练 smoke test

目标：

```text
确认 Perceiver receiver 能训练
```

建议配置：

```text
codebook_size = 256
batch_size = 2
num_steps = 50
embed_dim = 128
num_heads = 4
self_attn_layers = 2
SNR = 20-30 dB
num_paths = 3
max_delay = 3
max_doppler = 500
```

验收：

```text
loss finite
参数更新
checkpoint/log 保存
accuracy 有波动
```

### Stage 7B：小规模正式 receiver 训练

目标：

```text
Perceiver 优于 raw nearest-neighbor 和 oracle scalar equalization
```

建议：

```text
codebook_size = 256 -> 1024
num_steps = 1000-5000
```

### Stage 8：独立评估脚本

新增：

```text
eval_dd_token_perceiver_receiver.py
```

功能：

```text
加载 checkpoint
独立评估
对比 raw nearest-neighbor / oracle scalar equalization / Perceiver receiver
扫 SNR / delay / Doppler / num_paths
```

### Stage 9：真实图片 token 数据集

新增：

```text
build_token_dataset.py
```

功能：

```text
批量图片 -> visual token maps
```

输出：

```text
datasets\visual_token_maps.pt
```

### Stage 10：真实 token 训练

扩展：

```text
train_dd_token_perceiver_receiver.py
```

加入：

```text
--token-dataset datasets\visual_token_maps.pt
```

比较：

```text
random tokens vs real image tokens
```

### Stage 11：Channel-aware Perceiver

输入：

```text
Y_DD + CSI
```

CSI 可来自：

```text
delays
dopplers_hz
path_gains
conditioning
```

比较：

```text
blind receiver vs channel-aware receiver
```

### Stage 12：可学习 token-to-DD mapper

新增：

```text
learnable_token_dd_mapper.py
```

形式：

```text
nn.Embedding(codebook_size, 2 * symbols_per_token)
-> complex codewords
-> power normalization
-> DD grid packing
```

必须加入：

```text
average power normalization
```

可选：

```text
PAPR regularization
codeword diversity regularization
```

链路：

```text
tokens -> learnable mapper -> OTFS -> channel -> Perceiver -> token CE
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
sequence accuracy
PSNR / SSIM / LPIPS
可选 CLIP similarity
可选 downstream task accuracy
```

## 9. 近期推荐执行顺序

```text
1. 跑 Stage 7A 短训练 smoke test
2. 修训练脚本可能的问题
3. 做 codebook_size=256 的正式小实验
4. 写 eval_dd_token_perceiver_receiver.py
5. 扩展到 codebook_size=1024
6. 构建真实图片 token dataset
```

## 10. 注意事项

```text
不要改 channel_model.py，只复用。
channel_model 是时域信道，只能接 OTFS time waveform，不能直接接 DD grid。
cp_len 与 max_delay_samples 要配合。例如 cp_len=4 时，先用 max_delay_samples<=3。
当前 Perceiver 训练先用 random token maps，不要只用一张图片 tokens，否则容易过拟合。
先不要训练 learnable mapper，先把 receiver 训练和评估做扎实。
```

## 11. 新对话启动提示

下次新对话可以直接说：

```text
请基于 docs\CONVERSATION_SUMMARY.md 继续 Stage 7A 短训练 smoke test。
```

## 12. 最新进展：Stage 7A / 7B

### 12.1 Stage 7A 已通过

本机 RTX 3060 Laptop 上已完成 GPU smoke test：

```text
scripts\run_stage7a_train.ps1 -Mode precheck
scripts\run_stage7a_train.ps1 -Mode smoke
```

结果：

```text
loss finite，无 NaN/Inf
grad_norm finite
param_delta > 0
training_log.pt / receiver_checkpoint.pt / receiver_final.pt 均已保存
checkpoint_step_25.pt / checkpoint_step_50.pt 已保存
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

### 12.2 Stage 7B 云端环境

云服务器：

```text
AutoDL
Python 3.12
PyTorch 2.8.0+cu128
CUDA 12.8
项目路径：/root/autodl-tmp/SemCom-OTFS
```

GitHub 仓库：

```text
https://github.com/linear-20/SemCom-OTFS.git
```

GitHub 直连曾超时，云端使用代理 clone 成功：

```bash
git clone https://gh-proxy.com/https://github.com/linear-20/SemCom-OTFS.git
```

### 12.3 训练脚本新增能力

```text
train_dd_token_perceiver_receiver.py:
  --resume-checkpoint
  --channel-mode identity / channel
  --fixed-channel
  --no-awgn
  --use-packed-local

scripts/run_stage7b_train.sh:
  Linux 云端训练入口
  支持 pilot / train / custom
  支持 resume、信道诊断和 packed-local receiver
```

关键提交：

```text
b172822 Prepare Stage 7B receiver training
b14da56 Add deterministic channel diagnostics
6798ffe Add packed local receiver path
```

### 12.4 原始 blind Perceiver 诊断

原始 blind Perceiver 在 identity 任务上学习信号很弱：

```text
cb256 identity 3k:
eval_loss = 5.543744
eval_acc = 0.0048

cb32 identity 3k:
eval_loss = 3.462734
eval_acc = 0.0348
```

对比：

```text
ln(256)=5.545177, random acc=0.003906
ln(32)=3.465736, random acc=0.03125
```

结论：

```text
原始 blind Perceiver 不是完全不能学，但对当前固定 packing 的 mapper 缺少归纳偏置。
```

### 12.5 Packed-Local Receiver 结论

新增 `--use-packed-local`：

```text
每个 token 额外读取当前 mapper packing 下对应的 4 个复数 DD bins，
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

### 12.6 多径 delay 诊断

random multipath delay + AWGN：

```text
5k:
eval_acc = 0.5391
eval_loss = 2.246095

resume 到 10k:
eval_acc = 0.4187
eval_loss = 2.954157
```

fixed multipath delay / no AWGN：

```text
eval_acc = 0.9367
eval_loss = 0.160507
```

fixed multipath delay / AWGN：

```text
eval_acc = 0.9351
eval_loss = 0.173224
```

random multipath delay / no AWGN：

```text
eval_acc = 0.4554
eval_loss = 2.950565
```

结论：

```text
AWGN 不是主要瓶颈。
固定 delay 扩散可学到约 0.94 token accuracy。
随机多径信道变化使 blind packed-local receiver 下降到约 0.45-0.54。
下一步应做 channel-aware receiver，而不是继续盲训或直接加 Doppler。
```

## 13. 新的下一步

下一步建议直接做简化版 Stage 11：

```text
Channel-aware packed-local receiver
```

目标：

```text
在 random multipath delay + no AWGN 条件下，
利用 channel_model.py 输出的 delays 和 path_gains，
明显优于 blind packed-local baseline eval_acc≈0.455。
```

建议改动：

```text
DDTokenPerceiverReceiver.forward(y_dd, channel_features=None)
train_dd_token_perceiver_receiver.py 从 return_info=True 的 channel 输出中提取 delays/path_gains
新增参数：
  --use-channel-features
  --max-channel-paths 3
```

暂时不要做：

```text
不要加 Doppler。
不要上真实图片 token dataset。
不要训练 learnable mapper。
不要改 channel_model.py。
```

新对话启动提示可改为：

```text
请基于 docs\CONVERSATION_SUMMARY.md 和 docs\STAGE7_EXPERIMENT_LOG.md，继续实现 channel-aware packed-local receiver。
```
