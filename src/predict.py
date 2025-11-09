import torch
from tokenizer import ChineseTokenizer, EnglishTokenizer
import config
from model import TranslationModel


def predict_batch(input_tensor, model, zh_tokenizer, en_tokenizer, device):
    """
    批量预测
    :param input_tensor: 一批中文句子 [batch_size, seq_len]
    :param model:
    :param zh_tokenizer:
    :param en_tokenizer:
    :param device:
    :return: 一批英文句子,e.g.: [[4,6,7],[11,23,45,78,99],[88,99,26,55]]
    """
    model.eval()
    with torch.no_grad():
        # 编码
        src_pad_mask = (input_tensor == zh_tokenizer.pad_token_id)
        memory = model.encode(input_tensor, src_pad_mask)
        # memory.shape: [batch_size, src_len, d_model]

        # 解码
        batch_size = input_tensor.shape[0]
        decoder_input = torch.full((batch_size, 1), en_tokenizer.sos_token_id, device=device)
        # decoder_input.shape: [batch_size, 1]

        generated = [[] for _ in range(batch_size)]
        is_finished = [False for _ in range(batch_size)]
        for t in range(config.SEQ_LEN):
            tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(device)
            tgt_pad_mask = (decoder_input == en_tokenizer.pad_token_id)
            decoder_outputs = model.decode(decoder_input, memory, tgt_mask, src_pad_mask, tgt_pad_mask)
            # decoder_outputs.shape: [batch_size, tgt_len, en_vocab_size]

            last_decoder_output = decoder_outputs[:, -1, :]
            # last_decoder_output.shape: [batch_size, en_vocab_size]

            predict_indexes = torch.argmax(last_decoder_output, dim=-1)
            # predict_indexes.shape: [batch_size]

            # 处理每个时间步的预测结果
            for i in range(batch_size):
                if is_finished[i]:
                    continue
                else:
                    if predict_indexes[i].item() == en_tokenizer.eos_token_id:
                        is_finished[i] = True
                    else:
                        generated[i].append(predict_indexes[i].item())

            if all(is_finished):
                break
            decoder_input = torch.cat([decoder_input, predict_indexes.unsqueeze(1)], dim=1)
        return generated


def predict(user_input, model, zh_tokenizer, en_tokenizer, device):
    # 处理数据
    index_list = zh_tokenizer.encode(user_input, config.SEQ_LEN)
    input_tensor = torch.tensor([index_list]).to(device)
    # input_tensor.shape: (batch_size, seq_len)

    batch_result = predict_batch(input_tensor, model, zh_tokenizer, en_tokenizer, device)
    result = batch_result[0]
    return en_tokenizer.decode(result)


def run_predict():
    # 准备资源
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # tokenizer
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DIR / 'en_vocab.txt')

    # 模型
    model = TranslationModel(zh_vocab_size=zh_tokenizer.vocab_size,
                             en_vocab_size=en_tokenizer.vocab_size,
                             zh_padding_index=zh_tokenizer.pad_token_id,
                             en_padding_index=en_tokenizer.pad_token_id).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))

    # 运行预测
    print('中英翻译：（输入q或者quit退出）')
    while True:
        user_input = input('中文：')
        if user_input in ['q', 'quit']:
            break
        if user_input.strip() == '':
            continue
        result = predict(user_input, model, zh_tokenizer, en_tokenizer, device)
        print('英文：', result)


if __name__ == '__main__':
    run_predict()
