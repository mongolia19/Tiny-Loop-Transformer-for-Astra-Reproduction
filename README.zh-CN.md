# Tiny Loop Transformer for Astra Reproduction

[English README](README.md)

这是一个受 Astra 相关讨论启发的极小循环 Transformer 复现项目。模型在推理时重复使用同一个共享核心，通过增加循环次数提高有效计算深度，但不增加物理参数量。

> 重要边界：Astra 的名称和内部架构并未被 OpenAI 公开确认。本项目复现的是公开的 Huginn / recurrent-depth / looped Transformer 研究思路，不声称获得了 OpenAI 内部实现。

## 五分钟离线运行

在全新 checkout 中执行：

```bash
cd /Users/a58/prjs/recurrent-transformer-100m  # 替换成你的目录
python3 -m venv .venv
source .venv/bin/activate                       # Windows：.venv\\Scripts\\activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
python -m pytest -q
python -m recurrent_transformer.cli tiny-pipeline \\
  --english tests/fixtures/en.txt \\
  --chinese tests/fixtures/zh.txt \\
  --output artifacts/tiny \\
  --steps 1
```

成功标准：测试显示 `19 passed`，并生成 `artifacts/tiny/checkpoint-000001.pt` 与非空的 `generated:` 输出。该流程不需要联网、Hugging Face 账号或远程语料。

## 模型结构

- 2 层独立 prelude
- 8 层共享 recurrent core
- 2 层独立 coda
- 16,000 词元、768 hidden、12 heads、SwiGLU、RMSNorm、RoPE
- 实体参数：`97,241,856`
- 循环次数 1/2/3/4 对应有效深度 12/20/28/36

循环次数只改变计算量，不创建新的参数。smoke checkpoint 只用于验证工程链路，不代表模型已经学会有用语言能力。

## 安装与测试

需要 Python 3.11 或更高版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

如果命令行脚本不在 `PATH`，始终可以使用：

```bash
python -m recurrent_transformer.cli --help
```

## 语料

配置中的远程语料每种语言最多下载约 20 MiB：

- 英文：`HuggingFaceFW/fineweb-edu` / `sample-10BT`，ODC-By 1.0
- 中文：`0xDing/wikipedia-cn-20230720-filtered`，CC-BY-SA-3.0

联网后执行：

```bash
python -m recurrent_transformer.cli prepare-data --config configs/smoke.yaml
```

如果 Hugging Face、`datasets` 或 DNS 不可用，直接使用本地 UTF-8 文本：

```bash
python -m recurrent_transformer.cli train \\
  --config configs/smoke.yaml \\
  --english path/to/english.txt \\
  --chinese path/to/chinese.txt \\
  --output artifacts/smoke \\
  --device cpu \\
  --sequence-length 32 \\
  --gradient-accumulation 1
```

程序会执行 NFKC 规范化、空白合并、精确去重和 UTF-8 字节上限检查。

## 100M smoke 训练

### Apple Silicon MPS（已验证）

建议在项目目录创建带系统 PyTorch 的虚拟环境，避免修改系统 Python：

```bash
cd /Users/a58/prjs/recurrent-transformer-100m
python3 -m venv --system-site-packages .venv-mps
.venv-mps/bin/python -m pip install -U pip
.venv-mps/bin/python -m pip install sentencepiece pyyaml
```

先用短序列确认 MPS 可以完成反向传播：

```bash
PYTHONPATH=src .venv-mps/bin/python -m recurrent_transformer.cli train \
  --config configs/smoke.yaml \
  --english tests/fixtures/en.txt --chinese tests/fixtures/zh.txt \
  --output artifacts/mps-probe --memory-probe \
  --sequence-length 32 --gradient-accumulation 1 --device mps
```

探针成功后运行 smoke 训练（Apple Silicon 16GB 建议从此配置开始）：

```bash
PYTHONPATH=src .venv-mps/bin/python -m recurrent_transformer.cli train \
  --config configs/smoke.yaml \
  --english tests/fixtures/en.txt --chinese tests/fixtures/zh.txt \
  --output artifacts/mps-smoke-20 --steps 20 \
  --sequence-length 32 --gradient-accumulation 1 --device mps
```

成功标志是输出 `selected device: mps`、`steps: 20` 和 checkpoint 路径。若进程在 MPS 初始化阶段被系统终止，先关闭其他占用统一内存的程序，并保持 `--sequence-length 32 --gradient-accumulation 1`；确认探针通过后再逐步增加序列长度。

默认配置是上下文 512、micro-batch 1、梯度累积 4、循环次数随机采样 1–4，并对 recurrent core 启用 activation checkpointing。建议先跑一次 memory probe：

```bash
python -m recurrent_transformer.cli train \\
  --config configs/smoke.yaml \\
  --english data/smoke/english.txt \\
  --chinese data/smoke/chinese.txt \\
  --output artifacts/probe \\
  --memory-probe
```

如果 16GB Mac 的 MPS 在搬运模型时被系统终止，使用 CPU 或降低序列长度：

```bash
python -m recurrent_transformer.cli train \\
  --config configs/smoke.yaml \\
  --english data/smoke/english.txt \\
  --chinese data/smoke/chinese.txt \\
  --output artifacts/smoke \\
  --memory-probe --sequence-length 32 \\
  --gradient-accumulation 1 --device cpu
```

## 循环深度生成

```bash
python -m recurrent_transformer.cli generate \\
  --checkpoint artifacts/smoke/checkpoint-000020.pt \\
  --prompt '循环 Transformer can' \\
  --recurrences 1,2,4 \\
  --max-new-tokens 24
```

输出会分别标记 `[recurrences=1]`、`[recurrences=2]` 和 `[recurrences=4]`。几步 smoke 训练产生的文本不能作为质量提升证据。

## 研究依据

- [Scaling up Test-Time Compute with Latent Reasoning](https://arxiv.org/abs/2502.05171) 及 [seal-rg/recurrent-pretraining](https://github.com/seal-rg/recurrent-pretraining)
- [Reasoning with Latent Thoughts: On the Power of Looped Transformers](https://arxiv.org/abs/2502.17416)
- 参考视频：B 站 `BV1gBto6QEqa`

本项目与 Huginn 的主要差异是规模缩小到约 100M、单机训练、固定 2/8/2 结构、小型双语语料和无自适应停止机制。

## 输出与验证

```text
artifacts/<run>/
├── tokenizer/tokenizer.model
├── tokenizer/tokenizer.vocab
├── metrics.jsonl
└── checkpoint-000001.pt
```

本地 Apple M1 / 16GB 记录：19 个测试通过；97,241,856 参数；CPU 完成 1 step，loss `9.89547061920166`；checkpoint 约 1.1 GiB；循环 1/2/4 重载生成成功。远程数据下载曾因 DNS 不可用而失败，使用本地文本完成链路验证。

运行测试：

```bash
python -m pytest -q
python -m compileall -q src
git diff --check
```
