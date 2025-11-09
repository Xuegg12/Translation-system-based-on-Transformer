# visualize.py
from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt
import config

RESULTS = getattr(config, 'RESULTS_DIR', config.ROOT_DIR / 'results')
PLOTS_DIR = RESULTS / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ===== 1) 单次实验：读取 metrics.csv，画 Loss / PPL / BLEU / LR / GradNorm =====
csv_path = RESULTS / 'metrics.csv'
df = pd.read_csv(csv_path)

# 若没有 ppl 列，就根据 loss 现算一遍（兼容旧日志）
if 'train_ppl' not in df.columns:
    df['train_ppl'] = df['train_loss'].apply(math.exp)
if 'val_ppl' not in df.columns:
    df['val_ppl'] = df['val_loss'].apply(math.exp)

# 1) Train vs Val Loss
plt.figure()
plt.plot(df['epoch'], df['train_loss'], label='train_loss')
plt.plot(df['epoch'], df['val_loss'], label='val_loss')
plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.title('Loss Curve')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'loss_curve.png'); plt.close()

# 2) Train vs Val PPL
plt.figure()
plt.plot(df['epoch'], df['train_ppl'], label='train_ppl')
plt.plot(df['epoch'], df['val_ppl'], label='val_ppl')
plt.xlabel('epoch'); plt.ylabel('Perplexity'); plt.legend(); plt.title('Perplexity Curve')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'ppl_curve.png'); plt.close()

# 3) BLEU
plt.figure()
plt.plot(df['epoch'], df['val_bleu'], label='val_bleu')
plt.xlabel('epoch'); plt.ylabel('BLEU'); plt.legend(); plt.title('Validation BLEU')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'bleu_curve.png'); plt.close()

# 4) LR & Grad Norm（两个独立图）
plt.figure()
plt.plot(df['epoch'], df['lr'], label='lr')
plt.xlabel('epoch'); plt.ylabel('learning rate'); plt.legend(); plt.title('Learning Rate')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'lr_curve.png'); plt.close()

plt.figure()
plt.plot(df['epoch'], df['train_grad_norm'], label='train_grad_norm')
plt.xlabel('epoch'); plt.ylabel('grad norm'); plt.legend(); plt.title('Gradient Norm')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'grad_norm_curve.png'); plt.close()

# ===== 2) 消融实验可视化：对比多个 metrics_*.csv =====
# 约定：RESULTS 目录下：
#   metrics.csv              -> baseline
#   metrics_no_pe.csv        -> 去掉位置编码
#   metrics_small_dim.csv    -> 降低模型维度
#   ... 以此类推

ablation_files = sorted(RESULTS.glob('metrics_*.csv'))

def label_from_name(path: Path) -> str:
    """从文件名中提取一个简短的 label."""
    name = path.stem  # e.g. metrics_no_pe
    if name == 'metrics':
        return 'baseline'
    return name.replace('metrics_', '')

if ablation_files:
    # 只画验证指标的对比（更直观）
    plt.figure()
    for path in ablation_files:
        df_a = pd.read_csv(path)
        lbl = label_from_name(path)
        plt.plot(df_a['epoch'], df_a['val_loss'], label=f'{lbl}')
    plt.xlabel('epoch'); plt.ylabel('val_loss')
    plt.legend(); plt.title('Val Loss (Ablation)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'ablation_val_loss.png'); plt.close()

    plt.figure()
    for path in ablation_files:
        df_a = pd.read_csv(path)
        lbl = label_from_name(path)
        if 'val_ppl' not in df_a.columns:
            df_a['val_ppl'] = df_a['val_loss'].apply(math.exp)
        plt.plot(df_a['epoch'], df_a['val_ppl'], label=f'{lbl}')
    plt.xlabel('epoch'); plt.ylabel('val_ppl')
    plt.legend(); plt.title('Val Perplexity (Ablation)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'ablation_val_ppl.png'); plt.close()

    plt.figure()
    for path in ablation_files:
        df_a = pd.read_csv(path)
        lbl = label_from_name(path)
        plt.plot(df_a['epoch'], df_a['val_bleu'], label=f'{lbl}')
    plt.xlabel('epoch'); plt.ylabel('BLEU')
    plt.legend(); plt.title('Validation BLEU (Ablation)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'ablation_bleu.png'); plt.close()

print(f"Saved plots to: {PLOTS_DIR}")
