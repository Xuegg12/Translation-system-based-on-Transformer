import pandas as pd
from sklearn.model_selection import train_test_split

import config
from tokenizer import EnglishTokenizer, ChineseTokenizer


def process():
    print("开始处理数据")
    # 读取数据
    df = pd.read_csv(config.RAW_DATA_DIR / 'cmn.txt', sep='\t', header=None, usecols=[0, 1], names=['en', 'zh'],
                     encoding='utf-8')

    # 过滤数据
    df = df.dropna()
    df = df[df['en'].str.strip().ne('') & df['zh'].str.strip().ne('')]

    # 划分数据集
    train_df, test_df = train_test_split(df, test_size=0.2)

    # 构建词表
    ChineseTokenizer.build_vocab(train_df['zh'].tolist(), config.PROCESSED_DIR / 'zh_vocab.txt')
    EnglishTokenizer.build_vocab(train_df['en'].tolist(), config.PROCESSED_DIR / 'en_vocab.txt')

    # 构建Tokenizer对象
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DIR / 'en_vocab.txt')

    # 计算序列长度
    # zh_len = train_df['zh'].apply(lambda x: len(zh_tokenizer.tokenize(x))).max()
    # en_len = train_df['en'].apply(lambda x: len(en_tokenizer.tokenize(x))).max()
    # print(f'中文长度为：{zh_len}, 英文长度为：{en_len}')

    # 构建训练集
    train_df['zh'] = train_df['zh'].apply(lambda x: zh_tokenizer.encode(x, config.SEQ_LEN, add_sos_eos=False))
    train_df['en'] = train_df['en'].apply(lambda x: en_tokenizer.encode(x, config.SEQ_LEN, add_sos_eos=True))
    # 保存训练集
    train_df.to_json(config.PROCESSED_DIR / 'indexed_train.jsonl', orient='records', lines=True)

    # 构建测试集
    test_df['zh'] = test_df['zh'].apply(lambda x: zh_tokenizer.encode(x, config.SEQ_LEN, add_sos_eos=False))
    test_df['en'] = test_df['en'].apply(lambda x: en_tokenizer.encode(x, config.SEQ_LEN, add_sos_eos=True))
    # 保存测试集
    test_df.to_json(config.PROCESSED_DIR / 'indexed_test.jsonl', orient='records', lines=True)

    print("数据处理完成")


if __name__ == '__main__':
    process()
