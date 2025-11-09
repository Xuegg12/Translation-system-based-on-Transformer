# train_visualization.py
import time, csv, os, math
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_dataloader
from model import TranslationModel
from tokenizer import ChineseTokenizer, EnglishTokenizer
from evaluate import evaluate as eval_bleu  # 复用你已有的 BLEU 评估


def grad_global_norm(model) -> float:
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += float(p.grad.data.norm(2).item() ** 2)
    return total_sq ** 0.5


@torch.no_grad()
def evaluate_loss(dataloader, model, loss_function, device):
    """验证集交叉熵（与训练同一计算方式，no grad）。"""
    model.eval()
    total, n = 0.0, 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        # 解码器输入与标签对齐
        decoder_input = targets[:, :-1]
        labels = targets[:, 1:].contiguous()
        # mask
        src_pad_mask = (inputs == model.src_embedding.padding_idx)
        tgt_pad_mask = (decoder_input == model.tgt_embedding.padding_idx)
        # 自回归mask（下三角）
        seq_len = decoder_input.size(1)
        tgt_mask = model.transformer.generate_square_subsequent_mask(seq_len).to(device)
        # 前向并计算loss
        logits = model(inputs, decoder_input, src_pad_mask, tgt_mask, tgt_pad_mask)  # [B,T,V]
        loss = loss_function(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        total += loss.item()
        n += 1
    return total / max(1, n)


def train_one_epoch(dataloader, model, optimizer, loss_function, device, writer, global_step):
    model.train()
    epoch_total_loss = 0.0
    epoch_total_gn = 0.0
    steps = 0

    for inputs, targets in tqdm(dataloader, desc='训练'):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        decoder_input = targets[:, :-1]
        labels = targets[:, 1:].contiguous()

        src_pad_mask = (inputs == model.src_embedding.padding_idx)
        tgt_pad_mask = (decoder_input == model.tgt_embedding.padding_idx)

        seq_len = decoder_input.size(1)
        tgt_mask = model.transformer.generate_square_subsequent_mask(seq_len).to(device)

        logits = model(inputs, decoder_input, src_pad_mask, tgt_mask, tgt_pad_mask)  # [B,T,V]
        loss = loss_function(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        loss.backward()

        # （可选）梯度裁剪，满足稳定性实践要求
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        gn = grad_global_norm(model)
        optimizer.step()

        epoch_total_loss += float(loss.item())
        epoch_total_gn += gn
        steps += 1

        # TensorBoard（step级）
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/loss_step', float(loss.item()), global_step)
        writer.add_scalar('train/grad_norm_step', gn, global_step)
        writer.add_scalar('train/lr_step', lr, global_step)
        global_step += 1

    avg_loss = epoch_total_loss / max(1, steps)
    avg_gn = epoch_total_gn / max(1, steps)
    return avg_loss, avg_gn, global_step


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DIR / 'en_vocab.txt')

    model = TranslationModel(
        zh_vocab_size=zh_tokenizer.vocab_size,
        en_vocab_size=en_tokenizer.vocab_size,
        zh_padding_index=zh_tokenizer.pad_token_id,
        en_padding_index=en_tokenizer.pad_token_id
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_function = torch.nn.CrossEntropyLoss(
        ignore_index=en_tokenizer.pad_token_id,
        label_smoothing=getattr(config, 'LABEL_SMOOTHING', 0.0)
    )

    train_loader = get_dataloader(train=True)
    val_loader = get_dataloader(train=False)

    exp_name = getattr(config, 'EXPERIMENT_NAME', 'exp')
    writer = SummaryWriter(log_dir=config.LOGS_DIR / exp_name)

    metrics_dir = getattr(config, 'RESULTS_DIR', (config.ROOT_DIR / 'results'))
    os.makedirs(metrics_dir, exist_ok=True)
    csv_path = metrics_dir / f'metrics_{exp_name}.csv'
    # header: 多加 train_ppl / val_ppl
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch',
                'train_loss', 'train_ppl', 'train_grad_norm',
                'val_loss', 'val_ppl', 'val_bleu',
                'lr'
            ])

    best_loss = float('inf')
    global_step = 0

    for epoch in range(1, config.EPOCHS + 1):
        start = time.time()

        # 训练一个 epoch
        train_loss, train_gn, global_step = train_one_epoch(
            train_loader, model, optimizer, loss_function, device, writer, global_step
        )
        train_ppl = math.exp(train_loss)

        # 验证（loss + BLEU）
        val_loss = evaluate_loss(val_loader, model, loss_function, device)
        val_ppl = math.exp(val_loss)
        bleu = eval_bleu(val_loader, model, zh_tokenizer, en_tokenizer, device)

        # 记录到 TensorBoard（epoch级）
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/ppl', train_ppl, epoch)
        writer.add_scalar('train/grad_norm', train_gn, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/ppl', val_ppl, epoch)
        writer.add_scalar('val/bleu', bleu, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        # 记录到 CSV
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch,
                f'{train_loss:.6f}', f'{train_ppl:.6f}', f'{train_gn:.6f}',
                f'{val_loss:.6f}', f'{val_ppl:.6f}', f'{bleu:.6f}',
                optimizer.param_groups[0]['lr']
            ])

        # 控制台信息
        dt = time.time() - start
        print(
            f'Epoch {epoch:02d} | '
            f'train_loss {train_loss:.4f} | train_ppl {train_ppl:.2f} | '
            f'val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f} | '
            f'BLEU {bleu:.4f} | time {dt:.1f}s'
        )

        # 保存最优
        if val_loss < best_loss:
            best_loss = val_loss
            ckpt_path = config.MODELS_DIR / f'model_{exp_name}.pt'
            torch.save(model.state_dict(), ckpt_path)
            print(f'模型保存成功: {ckpt_path}')
        else:
            print('模型无需保存')


if __name__ == '__main__':
    train()
