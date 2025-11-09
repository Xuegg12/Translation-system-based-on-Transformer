# visualize.py
from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt
import config

RESULTS = getattr(config, 'RESULTS_DIR', config.ROOT_DIR / 'results')
PLOTS_DIR = RESULTS / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ================== 1) 单个实验曲线 ==================
exp_name = getattr(config, 'EXPERIMENT_NAME', 'baseline')
csv_path = RESULTS / f'metrics_{exp_name}.csv'

if csv_path.exists():
    df = pd.read_csv(csv_path)

    if 'train_ppl' not in df.columns:
        df['train_ppl'] = df['train_loss'].apply(math.exp)
    if 'val_ppl' not in df.columns:
        df['val_ppl'] = df['val_loss'].apply(math.exp)

    # Loss
    plt.figure()
    plt.plot(df['epoch'], df['train_loss'], label='train_loss')
    plt.plot(df['epoch'], df['val_loss'], label='val_loss')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.title(f'Loss Curve ({exp_name})')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'loss_curve_{exp_name}.png'); plt.close()

    # PPL
    plt.figure()
    plt.plot(df['epoch'], df['train_ppl'], label='train_ppl')
    plt.plot(df['epoch'], df['val_ppl'], label='val_ppl')
    plt.xlabel('epoch'); plt.ylabel('Perplexity'); plt.legend(); plt.title(f'Perplexity Curve ({exp_name})')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'ppl_curve_{exp_name}.png'); plt.close()

    # BLEU
    plt.figure()
    plt.plot(df['epoch'], df['val_bleu'], label='val_bleu')
    plt.xlabel('epoch'); plt.ylabel('BLEU'); plt.legend(); plt.title(f'Validation BLEU ({exp_name})')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'bleu_curve_{exp_name}.png'); plt.close()

    # 学习率与梯度范数（可选）
    plt.figure()
    plt.plot(df['epoch'], df['lr'], label='lr')
    plt.xlabel('epoch'); plt.ylabel('learning rate'); plt.legend(); plt.title(f'Learning Rate ({exp_name})')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'lr_curve_{exp_name}.png'); plt.close()

    plt.figure()
    plt.plot(df['epoch'], df['train_grad_norm'], label='train_grad_norm')
    plt.xlabel('epoch'); plt.ylabel('grad norm'); plt.legend(); plt.title(f'Gradient Norm ({exp_name})')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'grad_norm_curve_{exp_name}.png'); plt.close()

# ================== 2) 消融实验对比 ==================
# 会自动收集 results/ 目录下所有 metrics_*.csv

ablation_files = sorted(RESULTS.glob('metrics_*.csv'))

def label_from_name(path: Path) -> str:
    """
    从文件名提取简短 label：
      metrics_baseline.csv  -> baseline
      metrics_no_pe.csv     -> no_pe
    """
    name = path.stem  # e.g. metrics_no_pe
    return name.replace('metrics_', '')

if ablation_files:
    # Val Loss 对比
    plt.figure()
    for path in ablation_files:
        df_a = pd.read_csv(path)
        lbl = label_from_name(path)
        plt.plot(df_a['epoch'], df_a['val_loss'], label=lbl)
    plt.xlabel('epoch'); plt.ylabel('val_loss')
    plt.legend(); plt.title('Val Loss (Ablation)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'ablation_val_loss.png'); plt.close()

    # Val PPL 对比
    plt.figure()
    for path in ablation_files:
        df_a = pd.read_csv(path)
        lbl = label_from_name(path)
        if 'val_ppl' not in df_a.columns:
            df_a['val_ppl'] = df_a['val_loss'].apply(math.exp)
        plt.plot(df_a['epoch'], df_a['val_ppl'], label=lbl)
    plt.xlabel('epoch'); plt.ylabel('val_ppl')
    plt.legend(); plt.title('Val Perplexity (Ablation)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'ablation_val_ppl.png'); plt.close()

    # Val BLEU 对比
    plt.figure()
    for path in ablation_files:
        df_a = pd.read_csv(path)
        lbl = label_from_name(path)
        plt.plot(df_a['epoch'], df_a['val_bleu'], label=lbl)
    plt.xlabel('epoch'); plt.ylabel('BLEU')
    plt.legend(); plt.title('Validation BLEU (Ablation)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'ablation_bleu.png'); plt.close()

print(f"所有图像已保存到: {PLOTS_DIR}")
