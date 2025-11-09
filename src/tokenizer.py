from abc import abstractmethod

import nltk
from nltk import word_tokenize, TreebankWordDetokenizer
from tqdm import tqdm

# nltk.download('punkt_tab')


class BaseTokenizer:
    unk_token = '<unk>'
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)

        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}

        self.unk_token_id = self.word2index[self.unk_token]
        self.pad_token_id = self.word2index[self.pad_token]
        self.sos_token_id = self.word2index[self.sos_token]
        self.eos_token_id = self.word2index[self.eos_token]

    @staticmethod
    @abstractmethod
    def tokenize(text):
        pass

    def encode(self, text, seq_len, add_sos_eos=False):
        word_list = self.tokenize(text)

        if add_sos_eos:
            if len(word_list) == seq_len - 2:
                word_list = [self.sos_token] + word_list + [self.eos_token]
            elif len(word_list) < seq_len - 2:
                word_list = [self.sos_token] + word_list + [self.eos_token] + [self.pad_token] * (
                        seq_len - len(word_list) - 2)
            else:
                word_list = [self.sos_token] + word_list[:seq_len - 2] + [self.eos_token]
        else:
            # 补齐或截断到指定的seq_len
            if len(word_list) > seq_len:
                word_list = word_list[0:seq_len]
            elif len(word_list) < seq_len:
                word_list = word_list + [self.pad_token] * (seq_len - len(word_list))

        return [self.word2index.get(word, self.unk_token_id) for word in word_list]

    @classmethod
    def from_vocab(cls, vocab_file):
        # 1.加载词表文件
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_list = [line[:-1] for line in f.readlines()]
        print("词表加载完成")

        # 2.创建tokenizer对象
        return cls(vocab_list)

    @classmethod
    def build_vocab(cls, sentences, vocab_file):
        # 构建词表(用训练集)
        vocab_set = set()
        for sentence in tqdm(sentences, desc='构建词表'):
            for word in cls.tokenize(sentence):
                if word.strip() != '':  # 去掉不可见的token（space、\t等）
                    vocab_set.add(word)
        vocab_list = [cls.pad_token, cls.unk_token, cls.sos_token, cls.eos_token] + list(vocab_set)
        print(f'词表大小:{len(vocab_list)}')

        # 保存词表
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for word in vocab_list:
                f.write(word + '\n')
        print('词表保存完成')

    @abstractmethod
    def decode(self, word_ids):
        pass


class ChineseTokenizer(BaseTokenizer):
    @staticmethod
    def tokenize(text):
        return list(text)

    def decode(self, word_ids):
        word_list = [self.index2word[word_id] for word_id in word_ids]
        return ''.join(word_list)


class EnglishTokenizer(BaseTokenizer):
    def decode(self, word_ids):
        word_list = [self.index2word[word_id] for word_id in word_ids]
        return TreebankWordDetokenizer().detokenize(word_list)

    @staticmethod
    def tokenize(text):
        return word_tokenize(text)


if __name__ == '__main__':
    print(EnglishTokenizer.tokenize('hello world'))
