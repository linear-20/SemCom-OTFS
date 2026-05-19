# 项目使用说明书

本文档是项目的稳定使用手册，一般不频繁大改。它只说明项目定位、文件作用、常用命令和参数含义；具体实验结果看 `STAGE7_EXPERIMENT_LOG.md`，下一步安排看 `ROADMAP_NEXT.md`。

## 1. 项目目标

本项目研究一条不经过传统 bit / QAM 映射的 OTFS token 传输链路：

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

核心思想：

```text
token 直接映射到 OTFS delay-Doppler 域复数码字
```

而不是：

```text
token -> bits -> channel coding -> QAM -> OTFS
```

本地项目路径：

```text
F:\OTFS下的token传输
```

本地 Python / PyTorch 环境优先使用：

```text
E:\pytorch\python.exe
```

云端 AutoDL 项目路径：

```text
/root/autodl-tmp/SemCom-OTFS
```

GitHub 仓库：

```text
https://github.com/linear-20/SemCom-OTFS.git
```

## 2. 当前主文件

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
eval_dd_token_receiver_checkpoint.py
scripts/run_stage7b_train.sh
```

文档目录当前只保留三份主文档：

```text
docs/PROJECT_MANUAL.md          使用说明书
docs/STAGE7_EXPERIMENT_LOG.md   实验日志
docs/ROADMAP_NEXT.md            下一步规划书
```

历史材料放在：

```text
docs/archive/
```

## 3. 模块说明

### 3.1 image_tokenizer.py

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

典型 tokenizer 信息：

```text
token_ids shape: [1, 16, 16]
codebook_size: 16384
grid_size: (16, 16)
image_size: 256
downsample_ratio: 16
```

### 3.2 token_dd_mapper.py

作用：

```text
token IDs <-> DD 域复数 grid
```

它使用固定随机复数码本：

```text
token t -> codeword x_t in C^K
```

不做 bit 转换，也不做 QAM。

默认设置：

```text
codebook_size = 16384
symbols_per_token = 4
dd_shape = (32, 32)
```

原因：

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

### 3.3 otfs_modem.py

作用：

```text
DD grid <-> OTFS time waveform
```

主要类：

```text
OTFSModem
```

主要方法：

```text
modulate(x_dd) -> time_signal
demodulate(time_signal) -> y_dd
roundtrip
normalized_mse
max_abs_error
```

默认采用：

```text
torch.fft, norm="ortho"
```

时域长度：

```text
time_len = N * (M + cp_len)
```

例如：

```text
dd_shape=(32,32), cp_len=0 -> [B, 1024]
dd_shape=(32,32), cp_len=4 -> [B, 1152]
```

### 3.4 channel_model.py

作用：

```text
时域复基带时变多径多普勒信道
```

不要改这个文件，后续实验只复用。

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

注意：它是时域信道，必须接在 `OTFSModem.modulate()` 之后，不能直接接 DD grid。

常用参数：

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

### 3.5 dd_token_perceiver_receiver.py

作用：

```text
DD 域学习型 token receiver
```

输入：

```text
y_dd complex [B, M, N]
```

输出：

```text
logits [B, Ht*Wt, codebook_size]
```

当前支持的接收机分支：

```text
blind Perceiver:
    只使用 Y_DD

packed-local:
    每个 token 额外读取 mapper packing 对应的 4 个 DD bins

channel-aware:
    使用 delays 和 path_gains 形成 CSI embedding

delay-window local:
    每个 token 读取原始 4 个 bins 及 delay 方向邻域
```

常用结构参数：

```text
--embed-dim
--num-heads
--self-attn-layers
--dropout
--use-packed-local
--use-channel-features
--max-channel-paths
--use-delay-window-local
--delay-window-radius
```

### 3.6 train_dd_token_perceiver_receiver.py

作用：

```text
训练 DD-token receiver
```

训练链路：

```text
random tokens
-> TokenDDMapper.encode
-> X_DD
-> OTFSModem.modulate
-> TimeVaryingMultipathChannel
-> OTFSModem.demodulate
-> DDTokenPerceiverReceiver
-> token CE loss
```

重要能力：

```text
--dry-run
--resume-checkpoint
--channel-mode identity / channel
--fixed-channel
--no-awgn
--use-packed-local
--use-channel-features
--use-delay-window-local
```

训练输出：

```text
training_log.pt
receiver_checkpoint.pt
receiver_final.pt
receiver_best.pt
checkpoint_step_*.pt
```

`receiver_best.pt` 保存规则：

```text
每次 eval 后，如果 eval_token_accuracy 刷新历史最高值，就保存 best checkpoint。
```

### 3.7 eval_dd_token_receiver_checkpoint.py

作用：

```text
加载已有 receiver checkpoint，用更多 eval_batches 做稳定重评估。
```

常用命令：

```bash
python eval_dd_token_receiver_checkpoint.py \
  --checkpoint outputs/stage7b_receiver_train/EXPERIMENT_NAME/receiver_best.pt \
  --eval-batches 128 \
  --batch-size 8 \
  --device cuda
```

输出：

```text
*_reeval.pt
```

## 4. 常用本地命令

### 4.1 图片转 token

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

### 4.2 无信道 token-OTFS-token 闭环

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

### 4.3 Stage 7B 云端训练入口

```bash
bash scripts/run_stage7b_train.sh --help
```

推荐从脚本入口跑，不直接手写长命令：

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

### 4.4 稳定重评估

```bash
python eval_dd_token_receiver_checkpoint.py \
  --checkpoint outputs/stage7b_receiver_train/delaywin_r2_ca_cb256_random_delay_noawgn_5k_seed0/receiver_best.pt \
  --eval-batches 128 \
  --batch-size 8 \
  --device cuda
```

## 5. 关键参数说明

```text
codebook_size:
    token 词表大小。Stage 7B 先用 256。

symbols_per_token:
    每个 token 占用的 DD 复符号数。当前默认 4。

dd_shape:
    DD grid 大小。当前默认 32 32。

cp_len:
    循环前缀长度。多径 delay 实验中通常 cp_len=4。

max_delay_samples:
    最大路径延迟。当前 random delay 实验用 3。

max_doppler_hz:
    最大多普勒。当前先用 0，暂不加 Doppler。

no_awgn:
    关闭 AWGN，用于先隔离 random delay 问题。

eval_batches:
    训练中可用 16，正式重评估建议 128 或更高。
```

## 6. 注意事项

```text
不要修改 channel_model.py。
channel_model.py 是时域信道，只能接 OTFS time waveform。
cp_len 应与 max_delay_samples 匹配，当前 cp_len=4, max_delay_samples=3。
Stage 7B 先用 random token maps，不要只用单张图片 tokens 训练。
目前不要训练 learnable mapper，也不要直接加 Doppler。
```
