import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from tokenizer import ChineseTokenizer, EnglishTokenizer
import config
from model import TranslationModel
from dataset import get_dataloader
from predict import predict_batch


def evaluate(dataloader, model, zh_tokenizer, en_tokenizer, device):
    references = []  # [[[4,5,6,7]],[[5,6,7,8,9]],[[7,8,9]]]
    predictions = []  # [[4,5,6,7],[5,6,7,8,9],[7,8,9]]

    special_tokens = [en_tokenizer.sos_token_id, en_tokenizer.eos_token_id, en_tokenizer.pad_token_id]

    for inputs, targets in tqdm(dataloader, desc='评估'):
        inputs = inputs.to(device)
        # inputs.shape: [batch_size, seq_len]

        targets = targets.tolist()
        # 参考译文: [[*,*,*,*,*,*],[*,*,*,*,*,*],[*,*,*,*,*,*]]

        batch_result = predict_batch(inputs, model, zh_tokenizer, en_tokenizer, device)
        # 预测译文：batch_result.shape: [[4,6,7],[11,23,45,78,99],[88,99,26,55]...]

        # 处理预测结果
        predictions.extend(batch_result)

        # 获取参考译文
        references.extend([[[index for index in target if index not in special_tokens]] for target in targets])

    return corpus_bleu(references, predictions)


def run_evaluate():
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

    # 加载数据集
    dataloader = get_dataloader(train=False)

    bleu = evaluate(dataloader, model, zh_tokenizer, en_tokenizer, device)

    print(f'Bleu: {bleu}')


if __name__ == '__main__':
    run_evaluate()
