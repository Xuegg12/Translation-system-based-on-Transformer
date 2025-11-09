import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_dataloader
from model import TranslationModel
from tokenizer import ChineseTokenizer, EnglishTokenizer


def train_one_epoch(dataloader, model, optimizer, loss_function, device):
    model.train()
    epoch_total_loss = 0
    for inputs, targets in tqdm(dataloader, desc='训练'):
        inputs = inputs.to(device)
        # inputs.shape: [batch_size, seq_len]
        targets = targets.to(device)
        # targets.shape: [batch_size, seq_len]

        optimizer.zero_grad()

        # 解码器输入
        decoder_input = targets[:, :-1]
        # decoder_input.shape: [batch_size, tgt_len-1]
        # 源系列pad mask
        src_pad_mask = (inputs == model.src_embedding.padding_idx)
        # 目标系列pad mask
        tgt_pad_mask = (decoder_input == model.tgt_embedding.padding_idx)
        # tgt_mask 生成
        tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(device)

        decoder_outputs = model(inputs, decoder_input, src_pad_mask, tgt_mask, tgt_pad_mask)
        # decoder_outputs.shape: [batch_size, tgt_len-1, en_vocab_size]

        decoder_targets = targets[:, 1:]
        # decoder_targets.shape: [batch_size, tgt_len-1]

        # 计算损失
        loss = loss_function(decoder_outputs.reshape(-1, decoder_outputs.shape[-1])
                             , decoder_targets.reshape(-1))

        loss.backward()
        optimizer.step()
        epoch_total_loss += loss.item()
    return epoch_total_loss / len(dataloader)


def train():
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tokenizer
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DIR / 'en_vocab.txt')

    # 模型
    model = TranslationModel(zh_vocab_size=zh_tokenizer.vocab_size,
                             en_vocab_size=en_tokenizer.vocab_size,
                             zh_padding_index=zh_tokenizer.pad_token_id,
                             en_padding_index=en_tokenizer.pad_token_id).to(device)

    # 加载数据
    dataloader = get_dataloader()

    # 损失函数
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_id)

    # 优化器
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

    # tensorboard
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    best_loss = float('inf')
    for epoch in range(1, 1 + config.EPOCHS):
        print(f'========== Epoch: {epoch} ==========')
        avg_loss = train_one_epoch(dataloader, model, optimizer, loss_function, device)
        print(f'Loss: {avg_loss:.4f}')

        writer.add_scalar('Loss', avg_loss, epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'model.pt')
            print('模型保存成功')
        else:
            print('模型无需保存')


if __name__ == '__main__':
    train()
