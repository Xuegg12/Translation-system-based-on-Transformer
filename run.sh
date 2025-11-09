#!/usr/bin/env bash
# 一键运行训练 + 可视化流水线
# 用法示例：
#   bash run.sh baseline
#   bash run.sh no_pe
#   bash run.sh small_dim
#   bash run.sh single_head
#   bash run.sh no_ls

set -e

EXP_NAME=${1:-baseline}  # 不传参数时默认 baseline

echo "[INFO] 当前实验名: ${EXP_NAME}"
echo "[INFO] 请在 src/config.py 中将 EXPERIMENT_NAME 设为同样的名字。"

# 固定 Python hash 的随机种子（其余种子在 train_visualization.py 中设置）
export PYTHONHASHSEED=42

# 1) 数据预处理（如果已经处理过，可以手动注释掉）
echo "[STEP] 数据预处理..."
python -m src.process

# 2) 训练 + 记录指标 (metrics_*.csv)
echo "[STEP] 训练模型并记录指标..."
python -m src.train_visualization

# 3) 根据 metrics_*.csv 生成训练曲线 & 消融对比图
echo "[STEP] 生成可视化曲线..."
python -m src.visualize

echo "[DONE] 实验 ${EXP_NAME} 运行完成，请查看 results/ 和 results/plots/。"
