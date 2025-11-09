import math

import torch
from torch import nn
import config
from src import transformer


# class PositionEncoding(nn.Module):
#     def __init__(self, d_model, max_len=500):
#         super().__init__()
#         self.d_model = d_model
#         self.max_len = max_len
#
#         pos = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)  # pos.shape: (max_len, 1)
#         _2i = torch.arange(0, self.d_model, step=2, dtype=torch.float)  # _2i.shape: (d_model/2,)
#         div_term = torch.pow(10000, _2i / self.d_model)
#
#         sins = torch.sin(pos / div_term)  # sins.shape: (max_len, d_model/2)
#         coss = torch.cos(pos / div_term)  # coss.shape: (max_len, d_model/2)
#
#         pe = torch.zeros(self.max_len, self.d_model, dtype=torch.float)  # pe.shape: (max_len, d_model)
#
#         pe[:, 0::2] = sins
#         pe[:, 1::2] = coss
#
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         seq_len = x.size(1)
#         return x + self.pe[:seq_len]


class PositionEncoding(nn.Module):
    def __init__(self, dim_model, max_len=100):
        super().__init__()
        self.pe = self.build_pe(max_len, dim_model)

    def forward(self, x):
        # x.shape: [batch_size, sql_len, d_model]
        seq_len = x.shape[1]
        pe_part = self.pe[0:seq_len]
        # pe_part.shape: [seq_len, d_model]
        x = x + pe_part
        return x

    def build_pe(self, max_len, dim_model):
        print('Building PositionEncoding')
        pe = torch.zeros(max_len, dim_model, dtype=torch.float)

        for pos in range(max_len):
            for _2i in range(0, dim_model, 2):
                pe[pos, _2i] = math.sin(pos / math.pow(10000.0, _2i / dim_model))
                pe[pos, _2i + 1] = math.cos(pos / math.pow(10000.0, _2i / dim_model))

        return pe


class TranslationModel(nn.Module):
    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_index, en_padding_index):
        super().__init__()
        self.src_embedding = nn.Embedding(
            num_embeddings=zh_vocab_size,
            embedding_dim=config.DIM_MODEL,
            padding_idx=zh_padding_index
        )

        self.tgt_embedding = nn.Embedding(
            num_embeddings=en_vocab_size,
            embedding_dim=config.DIM_MODEL,
            padding_idx=en_padding_index
        )

        self.position_encoding = PositionEncoding(dim_model=config.DIM_MODEL)
        # 是否使用位置编码（由 config 控制）
        self.use_positional_encoding = getattr(config, 'USE_POSITION_ENCODING', True)

        # 这里你目前用的是 nn.Transformer，如果之后想换成手写的 transformer.Transformer 也可以
        self.transformer = nn.Transformer(
            d_model=config.DIM_MODEL,
            nhead=config.NUM_HEADS,
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.NUM_DECODER_LAYERS,
            batch_first=True
        )

        self.linear = nn.Linear(
            in_features=config.DIM_MODEL,
            out_features=en_vocab_size
        )

    def encode(self, src, src_pad_mask):
        # src.shape: [batch_size, sel_len]
        src_embed = self.src_embedding(src)
        # src_embed.shape: [batch_size, seq_len, d_model]
        if self.use_positional_encoding:
            src_embed = self.position_encoding(src_embed)
        # src_embed.shape: [batch_size, seq_len, d_model]

        memory = self.transformer.encoder(
            src=src_embed,
            src_key_padding_mask=src_pad_mask
        )
        return memory

    def decode(self, tgt, memory, tgt_mask, memory_pad_mask, tgt_pad_mask):
        tgt_embed = self.tgt_embedding(tgt)
        if self.use_positional_encoding:
            tgt_embed = self.position_encoding(tgt_embed)

        output = self.transformer.decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask
        )
        output = self.linear(output)
        return output

    def forward(self, src, tgt, src_pad_mask, tgt_mask, tgt_pad_mask):
        memory = self.encode(src, src_pad_mask)
        output = self.decode(tgt, memory, tgt_mask, src_pad_mask, tgt_pad_mask)
        return output


if __name__ == '__main__':
    model = TranslationModel(zh_vocab_size=100, en_vocab_size=100, zh_padding_index=0, en_padding_index=0)
    print(model)
