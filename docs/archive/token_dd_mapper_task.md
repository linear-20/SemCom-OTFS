# 下一步任务说明书：Token-to-DD 映射模块 v0

## 1. 任务目标

在当前项目中实现一个不训练的 `token-to-DD` 映射模块，完成：

```text
visual token IDs -> complex DD grid -> recovered visual token IDs
```

当前已有 token 文件：

```text
F:\OTFS下的token传输\tokens.pt
```

其内容为：

```text
token_ids: shape [1, 16, 16]
codebook_size: 16384
grid_size: [16, 16]
image_size: 256
downsample_ratio: 16
```

本阶段不接 OTFS 调制、不接无线信道、不训练神经网络，只验证 token 能否被稳定映射到 DD 域并恢复。

## 2. 新增文件

请新增：

```text
token_dd_mapper.py
```

可选新增：

```text
run_token_dd_mapper.py
```

如果逻辑简单，也可以把 CLI 测试直接放在 `token_dd_mapper.py` 里。

## 3. 推荐设计

### 3.1 基本配置

第一版使用以下参数：

```text
codebook_size = 16384
token_grid_size = 16 x 16
num_image_tokens = 256
dd_grid_size = 32 x 32
dd_resources = 1024
symbols_per_token = K = 4
```

关系为：

```text
256 visual tokens x 4 DD symbols/token = 1024 DD symbols
32 x 32 DD grid = 1024 DD bins
```

刚好铺满整个 DD grid。

## 4. 核心类设计

实现一个类：

```python
class TokenDDMapper:
    def __init__(
        self,
        codebook_size: int = 16384,
        symbols_per_token: int = 4,
        dd_shape: tuple[int, int] = (32, 32),
        seed: int = 0,
        device: str | None = None,
    ):
        ...
```

需要维护：

```python
self.codebook_size
self.symbols_per_token
self.dd_shape
self.num_dd_bins
self.seed
self.device
self.codebook
```

其中 `self.codebook` 是固定复数码本：

```text
shape: [codebook_size, symbols_per_token]
dtype: complex64 or complex128
```

## 5. 码本设计

第一版使用确定性随机复数码本，不训练。

推荐生成方式：

```python
real = torch.randn(codebook_size, symbols_per_token)
imag = torch.randn(codebook_size, symbols_per_token)
codebook = torch.complex(real, imag)
codebook = codebook / sqrt(mean(abs(codebook)^2 per codeword))
```

要求：

- 同一个 seed 下码本完全可复现
- 每个 token 对应一个长度为 `K` 的复数 codeword
- 每个 codeword 做平均功率归一化
- 最好整体平均功率约为 1

## 6. 必须实现的方法

### 6.1 encode

```python
def encode(self, token_ids: torch.LongTensor) -> torch.Tensor:
    """
    token_ids:
        shape [B, Ht, Wt] or [B, N]

    return:
        dd_grid:
            shape [B, M, N]
            complex tensor
    """
```

功能：

1. 检查 token id 是否在 `[0, codebook_size - 1]`
2. 将 token 展平成 `[B, num_tokens]`
3. 查表得到 codewords：

```text
[B, num_tokens, K]
```

4. 展平成：

```text
[B, num_tokens * K]
```

5. 填入 DD grid：

```text
[B, dd_M, dd_N]
```

要求：

```text
num_tokens * K <= dd_M * dd_N
```

如果不足，剩余 DD bins 填 0。当前第一版正好铺满。

### 6.2 decode

```python
def decode(
    self,
    dd_grid: torch.Tensor,
    token_shape: tuple[int, int] | None = None,
) -> torch.LongTensor:
    """
    dd_grid:
        shape [B, M, N]

    return:
        recovered token_ids:
            shape [B, Ht, Wt] if token_shape is given
            otherwise [B, num_tokens]
    """
```

功能：

1. 将 DD grid 展平成 `[B, dd_M * dd_N]`
2. 每 `K` 个复数符号切成一个 received codeword：

```text
[B, num_tokens, K]
```

3. 对每个 received codeword 做最近邻检测：

```text
argmin_v ||received - codebook[v]||^2
```

4. 输出恢复 token IDs

注意：

- 第一版可以直接和全 codebook 做距离计算
- 但 `16384 x 4` 对 `256` 个 token 做最近邻，计算量可以接受
- 如果显存不足，可以实现 chunked nearest neighbor

### 6.3 roundtrip

```python
def roundtrip(self, token_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.LongTensor]:
    """
    token_ids -> dd_grid -> recovered_token_ids
    """
```

### 6.4 metrics

提供简单函数：

```python
def token_accuracy(original: torch.Tensor, recovered: torch.Tensor) -> float:
    ...

def token_error_rate(original: torch.Tensor, recovered: torch.Tensor) -> float:
    ...
```

## 7. 命令行测试

支持运行：

```powershell
& 'E:\pytorch\python.exe' token_dd_mapper.py `
  --tokens "F:\OTFS下的token传输\tokens.pt" `
  --output-dd "F:\OTFS下的token传输\dd_grid.pt" `
  --output-tokens "F:\OTFS下的token传输\tokens_recovered.pt" `
  --symbols-per-token 4 `
  --dd-shape 32 32 `
  --seed 0
```

运行后打印：

```text
original token shape
dd grid shape
recovered token shape
token accuracy
token error rate
codebook size
symbols per token
DD grid shape
average DD power
```

并保存：

```python
dd_grid.pt:
{
    "dd_grid": dd_grid.cpu(),
    "dd_shape": (32, 32),
    "symbols_per_token": 4,
    "codebook_size": 16384,
    "token_shape": (16, 16),
}

tokens_recovered.pt:
{
    "token_ids": recovered.cpu(),
    "grid_size": (16, 16),
    "codebook_size": 16384,
    "image_size": 256,
    "downsample_ratio": 16,
}
```

## 8. 预期结果

在无信道、无噪声条件下，必须达到：

```text
token accuracy = 1.0
token error rate = 0.0
```

DD grid 形状应为：

```text
[1, 32, 32]
```

恢复 token 形状应为：

```text
[1, 16, 16]
```

## 9. 注意事项

1. 不要修改 `image_tokenizer.py`
2. 不要修改 `channel_model.py`
3. 不要加入 OTFS 调制
4. 不要加入无线信道
5. 不要训练神经网络
6. 不要改动 LlamaGen 源码
7. 当前阶段只做 token 与 DD grid 之间的确定性映射和恢复
8. 保证 seed 固定时结果可复现
9. 保证所有输出 tensor shape 明确
10. 保证复数 dtype 和 device 处理清楚

## 10. 完成标准

本任务完成后，项目应该新增：

```text
token_dd_mapper.py
dd_grid.pt
tokens_recovered.pt
```

并能证明：

```text
tokens.pt -> dd_grid.pt -> tokens_recovered.pt
```

无误差恢复。

完成后，下一阶段才进入：

```text
DD grid -> OTFS modulation -> OTFS demodulation -> DD grid
```

