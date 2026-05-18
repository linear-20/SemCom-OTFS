# 项目文件使用说明书

## 1. 项目定位

本项目研究的是：

```text
tokens 不经过 bit / QAM 映射，直接映射到 OTFS 的 delay-Doppler, DD, 域进行传输。
```

当前主链路为：

```text
图片
-> visual tokens
-> DD 域复数码字
-> OTFS 时域波形
-> AWGN / 多径多普勒信道
-> OTFS 解调
-> 接收 DD grid
-> 恢复 tokens
-> 重建图片
```

运行 Python 时优先使用：

```text
E:\pytorch\python.exe
```

项目根目录：

```text
F:\OTFS下的token传输
```

## 2. 目录结构

```text
F:\OTFS下的token传输
|-- image_tokenizer.py
|-- token_dd_mapper.py
|-- otfs_modem.py
|-- channel_model.py
|-- run_token_otfs_roundtrip.py
|-- run_token_otfs_awgn_sweep.py
|-- run_token_otfs_channel_sweep.py
|-- run_token_otfs_channel_equalized.py
|-- dd_token_perceiver_receiver.py
|-- train_dd_token_perceiver_receiver.py
|-- inputs\
|-- outputs\
|-- docs\
|-- pretrained_models\
`-- LlamaGen\
```

建议只把源码放在根目录，把图片、实验产物和文档分别放进对应目录。

## 3. 文档目录

```text
docs\README.md
```

文档索引，说明每个 Markdown 文件是什么。

```text
docs\PROJECT_MANUAL.md
```

本文件。说明项目文件、脚本作用、参数怎么改、图片怎么传。

```text
docs\ROADMAP_NEXT.md
```

后续规划书。说明后面每一阶段要做什么、写什么脚本、怎么验收。

```text
docs\CONVERSATION_SUMMARY.md
```

对话压缩摘要。新对话时可以直接作为上下文给模型，防止遗忘和幻觉。

```text
docs\archive\
```

历史阶段文档和旧任务说明，通常不用改。

## 4. 输入和输出目录

输入图片放这里：

```text
inputs\
```

实验输出按阶段放这里：

```text
outputs\tokenizer\              图片 tokenizer 输出
outputs\stage1_token_dd\        token -> DD -> token 输出
outputs\stage2_otfs\            DD -> OTFS -> DD 输出
outputs\stage3_roundtrip\       token -> DD -> OTFS -> DD -> token 输出
outputs\stage4_awgn\            AWGN 扫描输出
outputs\stage5_channel\         多径多普勒信道 baseline 输出
outputs\stage6_equalized\       scalar 均衡诊断输出
outputs\stage7_perceiver_receiver\  Perceiver 接收机 dry-run / 训练输出
```

## 5. 源码文件说明

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
encode(image) -> token_ids
decode(token_ids) -> reconstructed image
```

常用参数：

```text
--input              输入图片路径
--checkpoint         LlamaGen VQ tokenizer 权重
--llamagen-root      LlamaGen 源码目录
--image-size         图片尺寸，默认 256
--output             重建图片输出路径
--save-tokens        token 文件保存路径
--device             cpu / cuda / cuda:0
```

典型输出：

```text
token_ids shape: [1, 16, 16]
codebook_size: 16384
grid_size: (16, 16)
image_size: 256
downsample_ratio: 16
```

### 5.2 token_dd_mapper.py

作用：

```text
token IDs <-> DD 域复数 grid
```

它使用固定随机复数码本：

```text
token t -> codeword x_t in C^K
```

不做 bit 转换，也不做 QAM。

主要类：

```text
TokenDDMapper
```

主要方法：

```text
encode(token_ids) -> dd_grid
decode(dd_grid) -> recovered token_ids
roundtrip
token_accuracy
token_error_rate
```

常用参数：

```text
--tokens              输入 token .pt 文件
--output-dd           输出 DD grid 文件
--output-tokens       输出恢复 token 文件
--symbols-per-token   每个 token 占用多少个 DD 复符号
--dd-shape M N        DD grid 大小
--seed                固定随机码本种子
--device              cpu / cuda / cuda:0
```

重要约束：

```text
token 数量 * symbols_per_token <= M * N
```

当前默认设置：

```text
16 * 16 tokens * 4 symbols/token = 32 * 32 DD bins
```

所以：

```text
symbols_per_token = 4
dd_shape = 32 32
```

刚好铺满 DD grid。

如果把 `symbols_per_token` 改成 8，那么 `32 x 32` 不够，需要改成例如：

```text
--dd-shape 32 64
```

或者：

```text
--dd-shape 64 32
```

### 5.3 otfs_modem.py

作用：

```text
DD grid <-> OTFS 时域波形
```

主要类：

```text
OTFSModem
```

主要方法：

```text
isfft
sfft
modulate(x_dd)
demodulate(time_signal)
roundtrip
normalized_mse
max_abs_error
```

常用参数：

```text
--input-dd       输入 DD grid 文件
--output-time    输出 OTFS 时域波形
--output-dd      输出恢复 DD grid
--cp-len         循环前缀长度
--device         cpu / cuda / cuda:0
```

时域长度：

```text
time_len = N * (M + cp_len)
```

例如：

```text
M = 32
N = 32
cp_len = 4
time_len = 32 * (32 + 4) = 1152
```

### 5.4 channel_model.py

作用：

```text
时域复基带时变多径多普勒信道
```

不要改这个文件，后续只复用。

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
[time] 或 [batch, time]
```

注意：它是时域信道，所以必须接在：

```text
OTFSModem.modulate()
```

之后，不能直接接 DD grid。

常改参数：

```text
num_paths              路径数
sample_rate            采样率
snr_db                 信噪比
max_delay_samples      最大延迟，单位 samples
max_doppler_hz         最大多普勒，单位 Hz
fading                 rayleigh / rician / fixed
rician_k_db            Rician K 因子
doppler_distribution   jakes / uniform
randomize_each_forward 每次 forward 是否随机信道
fractional_delays      是否使用分数延迟
seed                   随机种子
```

当 `return_info=True` 时可得到：

```text
out.y
out.clean
out.noise
out.path_gains
out.delays
out.dopplers_hz
out.conditioning
```

### 5.5 run_token_otfs_roundtrip.py

作用：

```text
无信道 token-OTFS-token 闭环
```

链路：

```text
tokens -> DD grid -> OTFS time -> DD grid -> recovered tokens
```

用于确认无信道情况下链路没有 shape、FFT、码本问题。

### 5.6 run_token_otfs_awgn_sweep.py

作用：

```text
OTFS 时域加入 AWGN，并扫描 SNR
```

常用参数：

```text
--snr-db-list          SNR 列表，例如 0 5 10 15 20 25 30
--num-trials           每个 SNR 的独立噪声 trial 数
--save-last            保存最后一次 trial 细节
--symbols-per-token    每个 token 占用 DD 符号数
--dd-shape M N         DD grid 大小
--cp-len               循环前缀长度
--seed                 随机种子
--device               cpu / cuda / cuda:0
```

### 5.7 run_token_otfs_channel_sweep.py

作用：

```text
接入 channel_model.py，做多径多普勒 baseline
```

链路：

```text
tokens -> DD -> OTFS -> 多径多普勒信道 -> OTFS 解调 -> DD -> tokens
```

常用参数：

```text
--num-paths
--sample-rate
--max-delay-samples
--max-doppler-hz
--fading
--rician-k-db
--doppler-distribution
--randomize-each-forward
--integer-delays
```

建议 sanity check：

```text
num_paths = 1
max_delay_samples = 0
max_doppler_hz = 0
snr_db = 60
```

理论上 token accuracy 应接近 1.0。

### 5.8 run_token_otfs_channel_equalized.py

作用：

```text
比较 raw decode 和 oracle scalar equalization
```

scalar 均衡：

```text
alpha = sum(Y_DD * conj(X_DD)) / sum(abs(X_DD)^2)
Y_eq = Y_DD / alpha
```

注意：

```text
这是 oracle 诊断方法，因为它使用了发送端 X_DD。
它不是实际可部署接收机。
```

### 5.9 dd_token_perceiver_receiver.py

作用：

```text
DD 域 Perceiver 学习型接收机模型
```

不使用 CNN。

输入：

```text
y_dd complex [B, M, N]
```

输出：

```text
logits [B, Ht*Wt, codebook_size]
```

结构：

```text
real/imag + DD 位置
-> DD bin embedding
-> learned token queries
-> cross-attention
-> token self-attention
-> classifier
```

### 5.10 train_dd_token_perceiver_receiver.py

作用：

```text
Perceiver receiver 训练脚本骨架
```

目前已支持：

```text
--dry-run
```

dry-run 只检查前向链路：

```text
random tokens
-> DD
-> OTFS
-> channel
-> DD
-> Perceiver
-> logits
-> CE loss
```

不正式训练。

常改参数：

```text
--codebook-size
--token-shape H W
--symbols-per-token
--dd-shape M N
--cp-len
--batch-size
--num-steps
--lr
--embed-dim
--num-heads
--self-attn-layers
--dropout
--snr-db-min
--snr-db-max
--num-paths
--max-delay-samples
--max-doppler-hz
--device
--dry-run
```

## 6. 图片怎么传

### 第一步：把图片放进 inputs

例如：

```text
inputs\your_image.png
```

### 第二步：图片转 tokens

```powershell
& 'E:\pytorch\python.exe' image_tokenizer.py `
  --input "F:\OTFS下的token传输\inputs\your_image.png" `
  --checkpoint "F:\OTFS下的token传输\pretrained_models\vq_ds16_c2i.pt" `
  --llamagen-root "F:\OTFS下的token传输\LlamaGen" `
  --image-size 256 `
  --output "F:\OTFS下的token传输\outputs\tokenizer\recon_your_image.png" `
  --save-tokens "F:\OTFS下的token传输\outputs\tokenizer\tokens_your_image.pt" `
  --device cuda
```

得到：

```text
outputs\tokenizer\tokens_your_image.pt
outputs\tokenizer\recon_your_image.png
```

### 第三步：跑无信道 token-OTFS-token 闭环

```powershell
& 'E:\pytorch\python.exe' run_token_otfs_roundtrip.py `
  --tokens "F:\OTFS下的token传输\outputs\tokenizer\tokens_your_image.pt" `
  --output-dd-input "F:\OTFS下的token传输\outputs\stage3_roundtrip\dd_input.pt" `
  --output-time "F:\OTFS下的token传输\outputs\stage3_roundtrip\time.pt" `
  --output-dd-output "F:\OTFS下的token传输\outputs\stage3_roundtrip\dd_output.pt" `
  --output-tokens "F:\OTFS下的token传输\outputs\stage3_roundtrip\tokens_recovered.pt" `
  --symbols-per-token 4 `
  --dd-shape 32 32 `
  --cp-len 0 `
  --seed 0 `
  --device cuda
```

### 第四步：跑 AWGN 实验

```powershell
& 'E:\pytorch\python.exe' run_token_otfs_awgn_sweep.py `
  --tokens "F:\OTFS下的token传输\outputs\tokenizer\tokens_your_image.pt" `
  --output "F:\OTFS下的token传输\outputs\stage4_awgn\awgn_sweep_your_image.pt" `
  --snr-db-list 0 5 10 15 20 25 30 `
  --num-trials 100 `
  --symbols-per-token 4 `
  --dd-shape 32 32 `
  --cp-len 0 `
  --seed 0 `
  --device cuda `
  --save-last
```

## 7. 参数怎么改

### 7.1 symbols_per_token

含义：

```text
每个 token 占用多少个 DD 复符号
```

影响：

```text
越大，单个 token 冗余越强，抗噪声可能更好，但 token rate 更低。
```

改它时通常也要改 `dd_shape`。

### 7.2 dd_shape

含义：

```text
DD grid 大小，写作 M N
```

约束：

```text
token_shape[0] * token_shape[1] * symbols_per_token <= M * N
```

### 7.3 cp_len

含义：

```text
OTFS / OFDM 符号循环前缀长度
```

影响时域长度：

```text
time_len = N * (M + cp_len)
```

多径信道下建议：

```text
cp_len >= max_delay_samples
```

至少初期实验中建议这样配。

### 7.4 seed

含义：

```text
固定随机码本和随机实验的种子
```

当前 `TokenDDMapper` 使用确定性随机码本。换 seed 就等于换一套 token-DD 码本。

### 7.5 snr-db-list

含义：

```text
要扫描的 SNR 点
```

例如：

```text
--snr-db-list 0 5 10 15 20 25 30
```

### 7.6 num-trials

含义：

```text
每个 SNR / 信道设置下重复多少次随机 trial
```

越大统计越稳定，但运行越慢。

### 7.7 max_delay_samples

含义：

```text
信道最大延迟扩展，单位 samples
```

建议：

```text
cp_len = 4 时，先用 max_delay_samples <= 3
```

### 7.8 max_doppler_hz

含义：

```text
最大多普勒频移，单位 Hz
```

越大，时变越强，接收越难。

### 7.9 codebook_size

含义：

```text
token 词表大小
```

训练 Perceiver receiver 时，建议先小后大：

```text
256 -> 1024 -> 4096 -> 16384
```

不要一开始直接训练 16384 类 softmax。

## 8. 当前已知 baseline 结果

固定码本：

```text
symbols_per_token = 4
dd_shape = 32 32
```

结果：

```text
无信道：token accuracy = 1.0
AWGN 0 dB：约 1% token accuracy
AWGN 10 dB：约 66% token accuracy
AWGN 20 / 30 dB：token accuracy = 1.0
多径多普勒无均衡：很低且不稳定
oracle scalar 均衡：有改善，但有限
```

结论：

```text
复杂信道下主要问题不是整体幅度/相位旋转，而是 DD 域扩散/卷积失真。
下一步重点是学习型 DD 域接收机。
```

