# 后续规划：Token 直接到 OTFS DD 域传输

## 1. 当前状态

项目已经打通了基础链路：

```text
图片 -> visual tokens -> DD grid -> OTFS 时域波形
     -> AWGN / 多径多普勒信道
     -> 接收 DD grid -> 恢复 tokens
```

已完成模块：

```text
image_tokenizer.py                    图片 <-> visual tokens
token_dd_mapper.py                    固定码本 token <-> DD grid 映射
otfs_modem.py                         DD grid <-> OTFS 时域波形
channel_model.py                      可复用的时变多径信道
run_token_otfs_roundtrip.py           无信道 token-OTFS-token 闭环
run_token_otfs_awgn_sweep.py          AWGN baseline
run_token_otfs_channel_sweep.py       多径多普勒 baseline
run_token_otfs_channel_equalized.py   oracle scalar 均衡 baseline
dd_token_perceiver_receiver.py        DD 域 Perceiver 接收机模型
train_dd_token_perceiver_receiver.py  接收机训练骨架和 dry-run
```

当前关键实验结论：

```text
无信道：token accuracy = 1.0
AWGN 高 SNR：baseline 测试中 token accuracy = 1.0
多径多普勒无均衡：token accuracy 很低且不稳定
oracle scalar 均衡：有改善，但无法消除 DD 扩散
Perceiver receiver dry-run：前向链路和 loss 正常
```

因此，下一步核心问题是：

```text
Y_DD -> token logits -> recovered token IDs
```

也就是训练学习型 DD 域接收机，而不是恢复 bit、QAM 符号，或者马上做图像重建。

## 2. 三类 token decode 的区别

项目里现在有三种不同层面的 decode：

```text
ImageTokenizer.decode:
    visual token IDs -> reconstructed image

TokenDDMapper.decode:
    DD grid -> 最近邻码本匹配 -> token IDs

DDTokenPerceiverReceiver:
    received DD grid -> token logits -> argmax -> token IDs
```

Perceiver receiver 的目标是在复杂信道下替代 `TokenDDMapper.decode`。

完整关系是：

```text
received DD
-> Perceiver receiver
-> recovered tokens
-> ImageTokenizer.decode
-> reconstructed image
```

也就是说，学习型接收机负责通信层 token 判决；图像 tokenizer 的 decoder 仍然负责 token 到图片的还原。

## 3. 阶段 7A：短训练 Smoke Test

### 目标

先验证 Perceiver receiver 真的能训练。

链路：

```text
random tokens
-> fixed TokenDDMapper
-> fixed OTFSModem
-> random channel_model
-> received DD
-> DDTokenPerceiverReceiver
-> CE loss
```

这一阶段不追求最终性能，只检查：

```text
loss 不是 NaN / Inf
梯度能回传
参数会更新
checkpoint 和 log 能保存
train accuracy 至少有波动
```

### 建议命令

```powershell
& 'E:\pytorch\python.exe' train_dd_token_perceiver_receiver.py `
  --output-dir "F:\OTFS下的token传输\outputs\stage7_perceiver_receiver_smoke_train" `
  --codebook-size 256 `
  --token-shape 16 16 `
  --symbols-per-token 4 `
  --dd-shape 32 32 `
  --cp-len 4 `
  --batch-size 2 `
  --num-steps 50 `
  --embed-dim 128 `
  --num-heads 4 `
  --self-attn-layers 2 `
  --snr-db-min 20 `
  --snr-db-max 30 `
  --num-paths 3 `
  --max-delay-samples 3 `
  --max-doppler-hz 500 `
  --device cuda `
  --eval-every 10 `
  --eval-batches 1 `
  --save-every 25
```

### 验收标准

```text
训练正常结束
loss 无 NaN / Inf
receiver_checkpoint.pt 已保存
receiver_final.pt 已保存
training_log.pt 已保存
loss 和 accuracy 有记录
```

如果这一步失败，不要扩大模型，先修训练脚本。

## 4. 阶段 7B：小规模正式 Receiver 训练

### 目标

证明 Perceiver receiver 相比下面两个 baseline 有提升：

```text
raw nearest-neighbor decode
oracle scalar equalization
```

建议先从小规模开始：

```text
codebook_size = 256
batch_size = 4 或 8
num_steps = 1000 到 5000
embed_dim = 128 或 256
self_attn_layers = 2 到 4
```

### 记录指标

```text
train loss
train token accuracy
eval loss
eval token accuracy
SNR range
channel configuration
```

### 验收标准

```text
eval accuracy 高于随机猜测
eval accuracy 高于同信道条件下 raw nearest-neighbor
多次运行训练稳定
```

## 5. 阶段 8：Perceiver 独立评估脚本

### 新增脚本

```text
eval_dd_token_perceiver_receiver.py
```

### 目标

加载训练好的 Perceiver checkpoint，在独立脚本里评估。

需要比较：

```text
raw nearest-neighbor decode
oracle scalar equalization
Perceiver receiver
```

并且三者必须使用相同信道条件。

### 建议扫描参数

```text
SNR: 0, 5, 10, 15, 20, 25, 30 dB
max_delay_samples: 0, 1, 2, 3
max_doppler_hz: 0, 100, 500, 1000
num_paths: 1, 3, 6
```

### 输出

建议保存到：

```text
outputs/stage8_perceiver_eval/perceiver_eval.pt
```

内容包括：

```text
raw accuracy / TER
scalar-equalized accuracy / TER
Perceiver accuracy / TER
channel config
model config
```

### 验收标准

Perceiver 在至少一个有意义的多径多普勒设置下，明显优于 raw nearest-neighbor。

## 6. 阶段 9：真实图片 Token 数据集

### 新增脚本

```text
build_token_dataset.py
```

### 目标

从随机 token maps 过渡到真实 visual token maps。

流程：

```text
image folder
-> ImageTokenizer.encode
-> token maps [N, 16, 16]
-> dataset .pt file
```

建议输出：

```text
datasets/visual_token_maps.pt
```

内容包括：

```text
token_ids
grid_size
codebook_size
image_size
downsample_ratio
source_image_paths
```

### 为什么要做

随机 token maps 适合验证通信链路是否可学习，但真实图片 tokens 的分布不是均匀随机的，而是有结构、有频率偏置的。后续图像重建和语义通信必须用真实 token 分布。

### 验收标准

```text
dataset 中包含多张图片的 token maps
ImageTokenizer.decode 能从 dataset 中的样本重建图片
训练脚本支持从该 dataset 采样
```

## 7. 阶段 10：基于真实 Token 的 Receiver 训练

### 修改脚本

扩展：

```text
train_dd_token_perceiver_receiver.py
```

加入参数：

```text
--token-dataset datasets/visual_token_maps.pt
```

支持三种模式：

```text
random-token training
real-token training
mixed random + real-token training
```

### 实验对比

```text
train random / test random
train random / test real
train real / test real
train mixed / test real
```

### 验收标准

真实图片 tokens 上的性能要被实际测量，而不是只根据随机 tokens 的结果推断。

## 8. 阶段 11：Channel-Aware Perceiver Receiver

### 目标

利用 `channel_model.py` 输出的信道信息。

可用信息包括：

```text
delays
dopplers_hz
path_gains
conditioning
```

### 可能设计

把信道参数编码成一个向量：

```text
channel_embedding = MLP([delay, doppler, gain_real, gain_imag])
```

然后注入 token queries：

```text
queries = learned_queries + channel_embedding
```

或者把信道信息作为额外 tokens 拼到 DD token sequence 中，让 Perceiver cross-attention 自己读取。

### 对比对象

```text
blind Perceiver:       g(Y_DD)
channel-aware version: g(Y_DD, CSI)
```

### 验收标准

Channel-aware receiver 在变化的 delay / Doppler / path 条件下，比 blind receiver 更稳健。

## 9. 阶段 12：可学习 Token-to-DD Mapper

### 目标

把固定随机 `TokenDDMapper` 替换成可训练 mapper。

建议新增：

```text
learnable_token_dd_mapper.py
```

基本形式：

```text
nn.Embedding(codebook_size, 2 * symbols_per_token)
-> complex codewords
-> power normalization
-> DD grid packing
```

训练链路：

```text
tokens
-> learnable mapper
-> OTFS
-> channel
-> Perceiver receiver
-> token CE
```

### 必须加入约束

```text
平均发射功率归一化
可选 PAPR regularization
可选 codeword diversity regularization
```

### 验收标准

在相同 DD 资源数和平均功率下：

```text
learned mapper + Perceiver
```

优于：

```text
fixed random mapper + Perceiver
```

## 10. 阶段 13：完整图片级评估

### 目标

闭合完整图片链路：

```text
image
-> visual tokens
-> communication chain
-> recovered tokens
-> reconstructed image
```

### 指标

Token 指标：

```text
token accuracy
token error rate
sequence accuracy
```

图片指标：

```text
MSE / PSNR
SSIM
LPIPS，可选
可视化对比
```

语义或任务指标，可选：

```text
CLIP similarity
downstream classification accuracy
```

### 验收标准

随着信道恶化，重建图片应该平滑退化；学习型 receiver / mapper 应该优于 raw baseline。

## 11. 近期推荐执行顺序

建议按这个顺序走：

```text
1. Stage 7A：短训练 smoke test
2. Stage 7B：codebook_size=256 的小规模正式 receiver 训练
3. Stage 8：写独立评估脚本，对比 raw / scalar / Perceiver
4. 扩展 receiver 到 codebook_size=1024
5. 构建真实图片 token dataset
```

暂时不要直接跳到端到端图片训练。先把 receiver 的独立评估做扎实，再训练 learnable mapper 和图片级系统。

