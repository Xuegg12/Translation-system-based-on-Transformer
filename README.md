# 从零实现基于 Transformer 的中英翻译系统

本仓库实现了一个 **手写 Transformer** 中英翻译系统，包括：

- 手写缩放点积注意力 + 多头注意力（`src/transformer.py`）；
- 自定义 Encoder / Decoder / Transformer 封装（接口对齐 `nn.Transformer`）；
- 中英平行语料的预处理、分词与加载（`src/process.py`、`src/tokenizer.py`、`src/dataset.py`）；
- 训练过程可视化（Loss / PPL / BLEU / 学习率 / 梯度范数，`src/train_visualization.py` + `src/visualize.py`）；
- 多种消融实验（位置编码、模型维度、多头注意力、标签平滑）及结果对比（`results/plots/*.png`）。

报告中的所有曲线和表格均可通过本仓库代码复现。

---

## 1. 仓库结构

```text
.
├── src/
│   ├── config.py               # 模型与训练超参数、实验名、消融配置
│   ├── tokenizer.py            # 中/英文分词与词表构建
│   ├── process.py              # 原始数据 -> jsonl + vocab
│   ├── dataset.py              # TranslationDataset & DataLoader
│   ├── transformer.py          # 手写 Multi-Head Attention / Transformer
│   ├── model.py                # TranslationModel（嵌入 + PE + Transformer）
│   ├── train_visualization.py  # 训练+记录 loss/bleu/ppl 等
│   ├── evaluate.py             # 评估 BLEU
│   ├── predict.py              # 推理 demo：中文句子 -> 英文翻译
│   └── visualize.py            # 读取 metrics_*.csv，画曲线 & 消融对比
├── data/
│   ├── raw/                    # 原始数据（cmn.txt）
│   └── processed/              # 预处理后的 jsonl 与词表
├── models/                     # 训练好的权重 model_*.pt
├── results/
│   ├── metrics_*.csv           # 各实验的训练指标
│   └── plots/                  # 训练曲线 & 消融对比图
├── scripts/
│   └── run.sh                  # 一键运行脚本
├── requirements.txt
└── README.md
```

## 2. 环境与依赖

### 2.1 参考运行环境

- OS：Ubuntu 20.04 / Windows 10 / macOS 均可
- Python：3.9
- GPU：NVIDIA RTX 3060 12GB（推荐，有 CPU 也能跑，只是较慢）

### 2.2 创建环境并安装依赖

```
# 1) 创建并激活虚拟环境（可选）
conda create -n transformer-zh-en python=3.9 -y
conda activate transformer-zh-en

# 2) 安装依赖
pip install -r requirements.txt

# 3) 下载 nltk 需要的资源
python -m nltk.downloader punkt
python -m nltk.downloader wordnet
```

------

## 3. 数据准备

本实验使用阿里云天池公开的中英机器翻译语料（AI Challenger 英中翻译任务子集）。

1. 访问数据集：
    https://tianchi.aliyun.com/dataset/174937
2. 将包含中英平行句子的文件（如 `cmn.txt`）放到：

```
data/raw/cmn.txt
```

1. 运行预处理脚本，对原始数据进行清洗、分词、截断/填充，并生成 jsonl：

```
python -m src.process
```

预处理完成后，`data/processed/` 下会包含：

- `train.jsonl` / `val.jsonl`：中英索引序列；
- `zh_vocab.txt` / `en_vocab.txt`：中英文词表。

------

## 4. 实验配置与 exact 命令

所有实验通过 `src/config.py` 中的参数控制，主要字段如下：

```
EXPERIMENT_NAME = 'baseline'   # 当前实验名，对应 metrics_*.csv
USE_POSITION_ENCODING = True   # 是否使用位置编码
LABEL_SMOOTHING = 0.1          # 标签平滑系数

DIM_MODEL = 128                # 模型维度（Small Model 时可改小，如 64）
NUM_HEADS = 4                  # 注意力头数（single_head 实验设为 1）
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2

SEQ_LEN = 32
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 30
```

训练脚本内部统一设置随机种子为 `42`（Python / NumPy / PyTorch / CuDNN），命令完全相同的情况下结果在统计意义上可复现。

### 4.1 基线实验（baseline）

`src/config.py` 设置：

```
EXPERIMENT_NAME = 'baseline'
USE_POSITION_ENCODING = True
LABEL_SMOOTHING = 0.1
DIM_MODEL = 128
NUM_HEADS = 4
```

运行命令：

```
bash scripts/run.sh baseline
```

关键输出：

- `models/model_baseline.pt`
- `results/metrics_baseline.csv`
- `results/plots/loss_curve_baseline.png` 等

### 4.2 消融实验

以下为报告中用到的四组消融实验及 **exact 命令**：

#### (1) 去掉位置编码（No Positional Encoding）

```
EXPERIMENT_NAME = 'no_pe'
USE_POSITION_ENCODING = False
LABEL_SMOOTHING = 0.1
DIM_MODEL = 128
NUM_HEADS = 4
bash scripts/run.sh no_pe
```

#### (2) 减小模型维度（Small Model Size）

```
EXPERIMENT_NAME = 'small_dim'
USE_POSITION_ENCODING = True
LABEL_SMOOTHING = 0.1
DIM_MODEL = 64    # 缩小模型
NUM_HEADS = 4
bash scripts/run.sh small_dim
```

#### (3) 去掉多头注意力（单头 attention）

```
EXPERIMENT_NAME = 'single_head'
USE_POSITION_ENCODING = True
LABEL_SMOOTHING = 0.1
DIM_MODEL = 128
NUM_HEADS = 1     # 使用单头注意力
bash scripts/run.sh single_head
```

#### (4) 去掉标签平滑（No Label Smoothing）

```
EXPERIMENT_NAME = 'no_ls'
USE_POSITION_ENCODING = True
LABEL_SMOOTHING = 0.0
DIM_MODEL = 128
NUM_HEADS = 4
bash scripts/run.sh no_ls
```

每次运行会生成对应的：

- `results/metrics_<exp_name>.csv`
- `models/model_<exp_name>.pt`
- 若干单实验曲线图（`loss_curve_<exp_name>.png` 等）

------

## 5. 可视化与消融对比

当 baseline + 若干消融实验都训练完后，执行：

```
python -m src.visualize
```

该脚本会自动扫描 `results/metrics_*.csv`，生成：

- 单实验曲线：`loss_curve_*.png`、`ppl_curve_*.png`、`bleu_curve_*.png`
- 消融对比图：
  - `results/plots/ablation_val_loss.png`
  - `results/plots/ablation_val_ppl.png`
  - `results/plots/ablation_bleu.png`

这些图可以直接放入实验报告的「训练曲线」和「消融分析」部分。

------

## 6. 推理 Demo

使用训练好的基线模型做中英翻译：

```
python -m src.predict \
  --ckpt models/model_baseline.pt \
  --sentence "我特别喜欢这门大模型课程。"
```

示例输出：

```
> I really like this large model course .
```

`predict.py` 中实现了自回归贪心解码：每一步将当前已生成的序列喂入解码器，通过 masked self-attention 预测下一个 token，直到生成 `<eos>` 或达到最大长度。

------

## 7. 许可说明

本代码仅用于课程作业与教学演示，暂未指定开源协议。如需在其他项目中使用，请联系作者。