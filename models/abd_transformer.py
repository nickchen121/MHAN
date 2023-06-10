import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torch.autograd import Variable

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class LayerNorm(nn.Module):

    def __init__(self, feature, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature))
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class FeatEmbedding(nn.Module):

    def __init__(self, d_feat, d_model, dropout):
        super(FeatEmbedding, self).__init__()
        self.video_embeddings = nn.Sequential(
            LayerNorm(d_feat),
            nn.Dropout(dropout),
            nn.Linear(d_feat, d_model))

    def forward(self, x):
        return self.video_embeddings(x)


class TextEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model):
        super(TextEmbedding, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, dim, dropout, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.drop_out = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):

        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.drop_out(emb)
        return emb


def self_attention(query, key, value, dropout=None, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # mask的操作在QK之后，softmax之前
    if mask is not None:
        mask.cuda()
        scores = scores.masked_fill(mask == 0, -1e9)

    self_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        self_attn = dropout(self_attn)

    return torch.matmul(self_attn, value), self_attn


def self_attention_select(query, key, value, dropout=None, mask=None, d_k=None, select_num=20, is_matmul=None,
                          automatic_select_tag=None):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # mask的操作在QK之后，softmax之前
    if mask is not None:
        mask.cuda()
        scores = scores.masked_fill(mask == 0, -1e9)
    self_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        self_attn = dropout(self_attn)

    if automatic_select_tag:
        select_num = torch.nonzero(self_attn).shape[0] // (self_attn.shape[0] * self_attn.shape[1])

    if is_matmul:
        self_value = torch.matmul(self_attn, value)
        # print(tag, self_attn.shape, torch.nonzero(self_attn).shape[0] // (self_attn.shape[0] * self_attn.shape[1]))
        # print(f'self_attn.shape[0] , self_attn.shape[0: {self_attn.shape[0] , self_attn.shape[1]}')
        top_v, top_i = self_value.topk(select_num, dim=1, largest=True, sorted=True)

        return top_v, self_attn

    # print(tag, self_attn.shape, torch.nonzero(self_attn).shape[0] // (self_attn.shape[0] * self_attn.shape[1]))
    # print(f'self_attn.shape[0], self_attn.shape[0]: {self_attn.shape[0], self_attn.shape[1]}')
    # select_num = torch.nonzero(self_attn).shape[0] // (self_attn.shape[0] * self_attn.shape[1])
    top_v, top_i = self_attn.topk(select_num, dim=2, largest=True, sorted=True)

    # 通过索引选出分数最高的 select_num 个图片
    vv_list = []
    vv_tensor_list = []
    for i, v in enumerate(value):
        # print(v.shape)  # torch.Size([50, 640])
        for j in top_i[i]:
            vv = torch.index_select(v, 0, j)
            vv_list.append(vv)

        vv_stack = torch.stack(vv_list)
        vv_list.clear()
        vv_tensor_list.append(vv_stack)

    vv_tensor_stack = torch.stack(vv_tensor_list)

    return vv_tensor_stack, self_attn


class MultiHeadAttention(nn.Module):

    def __init__(self, head, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)
        self.d_k = d_model // head
        self.head = head
        self.d_model = d_model
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None

    def forward(self, query, key, value, mask=None, is_select=False, select_num=None, is_matmul=None,
                automatic_select_tag=None):

        if is_select:
            x, self.attn = self_attention_select(query, key, value, dropout=self.dropout, mask=mask, d_k=self.d_k,
                                                 select_num=select_num, is_matmul=is_matmul,
                                                 automatic_select_tag=automatic_select_tag)
            return x
        else:
            if mask is not None:
                # 多头注意力机制的线性变换层是4维，是把query[batch, frame_num, d_model]变成[batch, -1, head, d_k]
                # 再1，2维交换变成[batch, head, -1, d_k], 所以mask要在第一维添加一维，与后面的self attention计算维度一样
                mask = mask.unsqueeze(1)
            n_batch = query.size(0)
            query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 32, 64]
            key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]
            value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]
            x, self.attn = self_attention(query, key, value, dropout=self.dropout, mask=mask)

            # 变为三维， 或者说是concat head
            x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)

            return self.linear_out(x)


class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        # x = x.squeeze(2)
        return self.dropout(self.layer_norm(x + sublayer(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sublayer_connection[1](x, self.feed_forward)


class EncoderLayerNoAttention(nn.Module):
    def __init__(self, size, attn, feed_forward, dropout=0.1):
        super(EncoderLayerNoAttention, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        return self.sublayer_connection[1](x, self.feed_forward)


class DecoderLayer(nn.Module):

    def __init__(self, size, attn, feed_forward, sublayer_num, dropout=0.1, select_num=0):
        super(DecoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), sublayer_num)
        self.sublayer_connection_1 = SublayerConnection(size, dropout)
        self.layer_norm = LayerNorm(size)
        self.select_num = select_num
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory=None, r2l_trg_mask=None, layer_num=None,
                memory_cat=None):

        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, trg_mask))

        select_num = self.select_num

        if select_num == 0:
            tag = True
        else:
            tag = False

        if len(memory) == 4:
            memory_1, memory_2, memory_3, memory_4 = memory

            if layer_num == 1:
                # 大概范围：MSR-VTT: 35, MSVD:
                memory_1 = self.attn(x, memory_1, memory_1, src_mask[0], is_select=True, select_num=select_num,
                                     automatic_select_tag=tag)
                # 大概范围：MSR-VTT: 15, MSVD:
                memory_2 = self.attn(x, memory_2, memory_2, src_mask[1], is_select=True, select_num=select_num,
                                     automatic_select_tag=tag)
                # 大概范围：MSR-VTT: 30, MSVD:
                memory_4 = self.attn(memory_3, memory_4, memory_4, src_mask[3], is_select=True, select_num=select_num,
                                     is_matmul=True, automatic_select_tag=tag)
                memory_4 = torch.stack([memory_4] * int(memory_1.shape[1]), dim=1)
                # 大概范围：MSR-VTT: 15, MSVD:
                memory_3 = self.attn(x, memory_3, memory_3, src_mask[2], is_select=True, select_num=select_num,
                                     automatic_select_tag=tag)

                memory_cat = torch.cat([memory_1, memory_2, memory_3, memory_4], dim=2)
                # memory_cat = memory_1 + memory_2 + memory_3 + memory_4

        elif len(memory) == 3:
            memory_1, memory_2, memory_3 = memory

            if layer_num == 1:
                memory_1 = self.attn(x, memory_1, memory_1, src_mask[0], is_select=True, select_num=select_num,
                                     automatic_select_tag=tag)
                memory_2 = self.attn(x, memory_2, memory_2, src_mask[1], is_select=True, select_num=select_num,
                                     automatic_select_tag=tag)
                memory_3 = self.attn(x, memory_3, memory_3, src_mask[2], is_select=True, select_num=select_num,
                                     automatic_select_tag=tag)

                memory_cat = torch.cat([memory_1, memory_2, memory_3], dim=2)
                # memory_cat = memory_1+memory_2+memory_3

        elif len(memory) == 2:
            memory_1, memory_2 = memory

            if layer_num == 1:
                # memory_1 = self.attn(x, memory_1, memory_1, src_mask[0], is_select=True, select_num=select_num,
                #                      au8omatic_select_tag=tag)
                # memory_2 = self.attn(x, memory_2, memory_2, src_mask[1], is_select=True, select_num=select_num,
                #                      automatic_select_tag=tag)
                #
                memory_cat = torch.cat([memory_1, memory_2], dim=2)
                # memory_cat = memory_1 + memory_2


        else:
            raise "length of memory is error"

        # x = x.unsqueeze(2)
        x = self.sublayer_connection[1](x, lambda x: self.attn(x, memory_cat, memory_cat, None))

        if r2l_memory is not None:
            x = self.sublayer_connection[-2](x, lambda x: self.attn(x, r2l_memory, r2l_memory, r2l_trg_mask))

        return memory_cat, self.sublayer_connection[-1](x, self.feed_forward)


class Encoder(nn.Module):

    def __init__(self, n, encoder_layer):
        super(Encoder, self).__init__()
        self.encoder_layer = clones(encoder_layer, n)

    def forward(self, x, src_mask):
        for layer in self.encoder_layer:
            x = layer(x, src_mask)
        return x


class R2L_Decoder(nn.Module):

    def __init__(self, n, decoder_layer):
        super(R2L_Decoder, self).__init__()
        self.decoder_layer = clones(decoder_layer, n)

    def forward(self, x, memory, src_mask, r2l_trg_mask):
        layer_num = 1
        memory_cat = None
        for layer in self.decoder_layer:
            memory_cat, x = layer(x, memory, src_mask, r2l_trg_mask, layer_num=layer_num, memory_cat=memory_cat)
            layer_num += 1
        return x


class L2R_Decoder(nn.Module):

    def __init__(self, n, decoder_layer):
        super(L2R_Decoder, self).__init__()
        self.decoder_layer = clones(decoder_layer, n)

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask):
        layer_num = 1
        memory_cat = None
        for layer in self.decoder_layer:
            memory_cat, x = layer(x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask, layer_num=layer_num,
                                  memory_cat=memory_cat)
            layer_num += 1
        return x


def pad_mask(src, r2l_trg, trg, pad_idx):
    if isinstance(src, tuple):
        if len(src) == 4:
            src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
            src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
            src_object_mask = (src[2][:, :, 0] != pad_idx).unsqueeze(1)
            src_rel_mask = (src[3][:, :, 0] != pad_idx).unsqueeze(1)
            enc_src_mask = (src_image_mask, src_motion_mask, src_object_mask, src_rel_mask)
            # dec_src_mask = src_image_mask & src_motion_mask
            dec_src_mask = torch.stack([src_image_mask, src_motion_mask, src_rel_mask, src_rel_mask], dim=2)
            src_mask = (enc_src_mask, dec_src_mask)
        if len(src) == 3:
            src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
            src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
            src_object_mask = (src[2][:, :, 0] != pad_idx).unsqueeze(1)
            enc_src_mask = (src_image_mask, src_motion_mask, src_object_mask)
            dec_src_mask = src_image_mask & src_motion_mask
            src_mask = (enc_src_mask, dec_src_mask)
        if len(src) == 2:
            src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
            src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
            enc_src_mask = (src_image_mask, src_motion_mask)
            dec_src_mask = src_image_mask & src_motion_mask
            src_mask = (enc_src_mask, dec_src_mask)
    else:
        src_mask = (src[:, :, 0] != pad_idx).unsqueeze(1)
    if trg is not None:
        if isinstance(src_mask, tuple):
            trg_mask = (trg != pad_idx).unsqueeze(1) & subsequent_mask(trg.size(1)).type_as(src_image_mask.data)
            r2l_pad_mask = (r2l_trg != pad_idx).unsqueeze(1).type_as(src_image_mask.data)
            r2l_trg_mask = r2l_pad_mask & subsequent_mask(r2l_trg.size(1)).type_as(src_image_mask.data)
            return src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask
        else:
            trg_mask = (trg != pad_idx).unsqueeze(1) & subsequent_mask(trg.size(1)).type_as(src_mask.data)
            r2l_pad_mask = (r2l_trg != pad_idx).unsqueeze(1).type_as(src_mask.data)
            r2l_trg_mask = r2l_pad_mask & subsequent_mask(r2l_trg.size(1)).type_as(src_mask.data)
            return src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask  # src_mask[batch, 1, lens]  trg_mask[batch, 1, lens]

    else:
        return src_mask


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return (torch.from_numpy(mask) == 0).cuda()


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


class ABDTransformer(nn.Module):

    def __init__(self, vocab, d_feat, d_model, d_ff, n_heads, n_layers, dropout, feature_mode,
                 device='cuda', n_heads_big=128, select_num=0):
        super(ABDTransformer, self).__init__()

        self.vocab = vocab
        self.device = device
        self.feature_mode = feature_mode

        c = copy.deepcopy

        attn = MultiHeadAttention(n_heads, d_model, dropout)

        attn_big = MultiHeadAttention(n_heads_big, d_model, dropout)

        feed_forward = PositionWiseFeedForward(d_model, d_ff)

        if feature_mode == 'two':
            self.image_src_embed = FeatEmbedding(d_feat[0], d_model, dropout)
            self.motion_src_embed = FeatEmbedding(d_feat[1], d_model, dropout)
        elif feature_mode == 'three':
            self.image_src_embed = FeatEmbedding(d_feat[0], d_model, dropout)
            self.motion_src_embed = FeatEmbedding(d_feat[1], d_model, dropout)
            self.object_src_embed = FeatEmbedding(d_feat[2], d_model, dropout)
        elif feature_mode == 'four':
            self.image_src_embed = FeatEmbedding(d_feat[0], d_model, dropout)
            self.motion_src_embed = FeatEmbedding(d_feat[1], d_model, dropout)
            self.object_src_embed = FeatEmbedding(d_feat[2], d_model, dropout)
            self.rel_src_embed = FeatEmbedding(d_feat[3], d_model, dropout)

        self.trg_embed = TextEmbedding(vocab.n_vocabs, d_model)

        self.pos_embed = PositionalEncoding(d_model, dropout)

        self.encoder = Encoder(n_layers, EncoderLayer(d_model, c(attn), c(feed_forward), dropout))

        self.encoder_big = Encoder(n_layers, EncoderLayer(d_model, c(attn_big), c(feed_forward), dropout))

        self.encoder_no_attention = Encoder(n_layers,
                                            EncoderLayerNoAttention(d_model, c(attn), c(feed_forward), dropout))

        self.r2l_decoder = R2L_Decoder(n_layers, DecoderLayer(d_model, c(attn), c(feed_forward),
                                                              sublayer_num=3, dropout=dropout, select_num=select_num))
        self.l2r_decoder = L2R_Decoder(n_layers, DecoderLayer(d_model, c(attn), c(feed_forward),
                                                              sublayer_num=4, dropout=dropout, select_num=select_num))

        self.generator = Generator(d_model, vocab.n_vocabs)

    def encode(self, src, src_mask, feature_mode_two=False):

        if feature_mode_two:
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder(x1, src_mask[0])
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder(x2, src_mask[1])
            return x1, x2

        elif self.feature_mode == 'two':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder(x1, src_mask[0])

            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder(x2, src_mask[1])

            return x1, x2

        elif self.feature_mode == 'three':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder(x1, src_mask[0])

            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder(x2, src_mask[1])

            x3 = self.object_src_embed(src[2])
            x3 = self.encoder(x3, src_mask[2])

            return x1, x2, x3

        elif self.feature_mode == 'four':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder(x1, src_mask[0])

            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder(x2, src_mask[1])

            x3 = self.object_src_embed(src[2])
            x3 = self.encoder(x3, src_mask[2])

            x4 = self.rel_src_embed(src[3])
            x4 = self.encoder(x4, src_mask[3])

            return x1, x2, x3, x4

    def r2l_decode(self, r2l_trg, memory, src_mask, r2l_trg_mask):
        x = self.trg_embed(r2l_trg)
        x = self.pos_embed(x)
        return self.r2l_decoder(x, memory, src_mask, r2l_trg_mask)

    def l2r_decode(self, trg, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask):
        x = self.trg_embed(trg)
        x = self.pos_embed(x)
        return self.l2r_decoder(x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask)

    def forward(self, src, r2l_trg, trg, mask):
        src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask = mask
        enc_src_mask, dec_src_mask = src_mask

        # r2l_encoding_outputs = self.encode(src, enc_src_mask, feature_mode_two=True)
        encoding_outputs = self.encode(src, enc_src_mask)

        r2l_outputs = self.r2l_decode(r2l_trg, encoding_outputs, enc_src_mask, r2l_trg_mask)

        l2r_outputs = self.l2r_decode(trg, encoding_outputs, enc_src_mask, trg_mask, r2l_outputs, r2l_pad_mask)

        r2l_pred = self.generator(r2l_outputs)
        l2r_pred = self.generator(l2r_outputs)

        return r2l_pred, l2r_pred

    def r2l_beam_search_decode(self, batch_size, src, src_mask, model_encodings, beam_size, max_len):
        end_symbol = self.vocab.word2idx['<S>']
        start_symbol = self.vocab.word2idx['<S>']

        r2l_outputs = None

        hypotheses = [copy.deepcopy(torch.full((1, 1), start_symbol, dtype=torch.long,
                                               device=self.device)) for _ in range(batch_size)]
        completed_hypotheses = [copy.deepcopy([]) for _ in range(batch_size)]
        hyp_scores = [copy.deepcopy(torch.full((1,), 0, dtype=torch.float, device=self.device))
                      for _ in range(batch_size)]

        for iter in range(max_len + 1):
            if all([len(completed_hypotheses[i]) == beam_size for i in range(batch_size)]):
                break

            cur_beam_sizes, last_tokens, src_mask_l = [], [], []
            model_encodings_l_1, model_encodings_l_2, model_encodings_l_3, model_encodings_l_4 = [], [], [], []
            src_mask_l1, src_mask_l2, src_mask_l3, src_mask_l4 = [], [], [], []

            for i in range(batch_size):
                if hypotheses[i] is None:
                    cur_beam_sizes += [0]
                    continue
                cur_beam_size, decoded_len = hypotheses[i].shape
                cur_beam_sizes += [cur_beam_size]
                last_tokens += [hypotheses[i]]
                model_encodings_l_1 += [model_encodings[0][i:i + 1]] * cur_beam_size
                model_encodings_l_2 += [model_encodings[1][i:i + 1]] * cur_beam_size

                src_mask_l1 += [src_mask[0][i:i + 1]] * cur_beam_size
                src_mask_l2 += [src_mask[1][i:i + 1]] * cur_beam_size

                if len(model_encodings) == 4:
                    model_encodings_l_3 += [model_encodings[2][i:i + 1]] * cur_beam_size
                    src_mask_l3 += [src_mask[2][i:i + 1]] * cur_beam_size
                    model_encodings_l_4 += [model_encodings[3][i:i + 1]] * cur_beam_size
                    src_mask_l4 += [src_mask[3][i:i + 1]] * cur_beam_size
                elif len(model_encodings) == 3:
                    model_encodings_l_3 += [model_encodings[2][i:i + 1]] * cur_beam_size
                    src_mask_l3 += [src_mask[2][i:i + 1]] * cur_beam_size

            model_encodings_cur_1 = torch.cat(model_encodings_l_1, dim=0)
            model_encodings_cur_2 = torch.cat(model_encodings_l_2, dim=0)

            src_mask_cur1 = torch.cat(src_mask_l1, dim=0)
            src_mask_cur2 = torch.cat(src_mask_l2, dim=0)

            if model_encodings_l_4:
                src_mask_cur3 = torch.cat(src_mask_l3, dim=0)
                src_mask_cur4 = torch.cat(src_mask_l4, dim=0)
                src_mask_cur = (src_mask_cur1, src_mask_cur2, src_mask_cur3, src_mask_cur4)

                model_encodings_cur_3 = torch.cat(model_encodings_l_3, dim=0)
                model_encodings_cur_4 = torch.cat(model_encodings_l_4, dim=0)
                model_encodings_cur = (
                    model_encodings_cur_1, model_encodings_cur_2, model_encodings_cur_3, model_encodings_cur_4)
            elif model_encodings_l_3:
                src_mask_cur3 = torch.cat(src_mask_l3, dim=0)
                src_mask_cur = (src_mask_cur1, src_mask_cur2, src_mask_cur3)

                model_encodings_cur_3 = torch.cat(model_encodings_l_3, dim=0)
                model_encodings_cur = (
                    model_encodings_cur_1, model_encodings_cur_2, model_encodings_cur_3)
            else:
                src_mask_cur = (src_mask_cur1, src_mask_cur2)
                model_encodings_cur = (model_encodings_cur_1, model_encodings_cur_2)

            y_tm1 = torch.cat(last_tokens, dim=0)

            out = self.r2l_decode(Variable(y_tm1).to(self.device), model_encodings_cur, src_mask_cur,
                                  Variable(subsequent_mask(y_tm1.size(-1)).type_as(src[0].data)).to(self.device))

            r2l_outputs = out

            log_prob = self.generator(out[:, -1, :]).unsqueeze(1)
            _, decoded_len, vocab_sz = log_prob.shape
            log_prob = torch.split(log_prob, cur_beam_sizes, dim=0)

            new_hypotheses, new_hyp_scores = [], []
            for i in range(batch_size):
                if hypotheses[i] is None or len(completed_hypotheses[i]) >= beam_size:
                    new_hypotheses += [None]
                    new_hyp_scores += [None]
                    continue

                cur_beam_sz_i, dec_sent_len, vocab_sz = log_prob[i].shape
                cumulative_hyp_scores_i = (hyp_scores[i].unsqueeze(-1).unsqueeze(-1)
                                           .expand((cur_beam_sz_i, 1, vocab_sz)) + log_prob[i]).view(-1)

                live_hyp_num_i = beam_size - len(completed_hypotheses[i])
                top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(cumulative_hyp_scores_i, k=live_hyp_num_i)
                prev_hyp_ids, hyp_word_ids = top_cand_hyp_pos // self.vocab.n_vocabs, \
                                             top_cand_hyp_pos % self.vocab.n_vocabs

                new_hypotheses_i, new_hyp_scores_i = [], []
                for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids,
                                                                        top_cand_hyp_scores):
                    prev_hyp_id, hyp_word_id, cand_new_hyp_score = \
                        prev_hyp_id.item(), hyp_word_id.item(), cand_new_hyp_score.item()

                    new_hyp_sent = torch.cat(
                        (hypotheses[i][prev_hyp_id], torch.tensor([hyp_word_id], device=self.device)))
                    if hyp_word_id == end_symbol:
                        completed_hypotheses[i].append(Hypothesis(
                            value=[self.vocab.idx2word[a.item()] for a in new_hyp_sent[1:-1]],
                            score=cand_new_hyp_score))
                    else:
                        new_hypotheses_i.append(new_hyp_sent.unsqueeze(-1))
                        new_hyp_scores_i.append(cand_new_hyp_score)

                if len(new_hypotheses_i) > 0:
                    hypotheses_i = torch.cat(new_hypotheses_i, dim=-1).transpose(0, -1).to(self.device)
                    hyp_scores_i = torch.tensor(new_hyp_scores_i, dtype=torch.float, device=self.device)
                else:
                    hypotheses_i, hyp_scores_i = None, None
                new_hypotheses += [hypotheses_i]
                new_hyp_scores += [hyp_scores_i]
            hypotheses, hyp_scores = new_hypotheses, new_hyp_scores

        for i in range(batch_size):
            hyps_to_add = beam_size - len(completed_hypotheses[i])
            if hyps_to_add > 0:
                scores, ix = torch.topk(hyp_scores[i], k=hyps_to_add)
                for score, id in zip(scores, ix):
                    completed_hypotheses[i].append(Hypothesis(
                        value=[self.vocab.idx2word[a.item()] for a in hypotheses[i][id][1:]],
                        score=score))
            completed_hypotheses[i].sort(key=lambda hyp: hyp.score, reverse=True)
        return r2l_outputs, completed_hypotheses

    def beam_search_decode(self, src, beam_size, max_len):
        start_symbol = self.vocab.word2idx['<S>']
        end_symbol = self.vocab.word2idx['<S>']

        src_mask = pad_mask(src, r2l_trg=None, trg=None, pad_idx=self.vocab.word2idx['<PAD>'])

        batch_size = src[0].shape[0]
        enc_src_mask = src_mask[0]

        # r2l_model_encodings = self.encode(src, enc_src_mask, feature_mode_two=True)
        model_encodings = self.encode(src, enc_src_mask)

        r2l_memory, r2l_completed_hypotheses = self.r2l_beam_search_decode(batch_size, src, enc_src_mask,
                                                                           model_encodings=model_encodings,
                                                                           beam_size=beam_size, max_len=max_len)

        hypotheses = [copy.deepcopy(torch.full((1, 1), start_symbol, dtype=torch.long,
                                               device=self.device)) for _ in range(batch_size)]
        completed_hypotheses = [copy.deepcopy([]) for _ in range(batch_size)]
        hyp_scores = [copy.deepcopy(torch.full((1,), 0, dtype=torch.float, device=self.device))
                      for _ in range(batch_size)]

        for iter in range(max_len + 1):
            if all([len(completed_hypotheses[i]) == beam_size for i in range(batch_size)]):
                break

            cur_beam_sizes, last_tokens, src_mask_l, r2l_memory_l = [], [], [], []
            model_encodings_l_1, model_encodings_l_2, model_encodings_l_3, model_encodings_l_4 = [], [], [], []
            src_mask_l1, src_mask_l2, src_mask_l3, src_mask_l4 = [], [], [], []

            for i in range(batch_size):

                if hypotheses[i] is None:
                    cur_beam_sizes += [0]
                    continue

                cur_beam_size, decoded_len = hypotheses[i].shape
                cur_beam_sizes += [cur_beam_size]
                last_tokens += [hypotheses[i]]
                model_encodings_l_1 += [model_encodings[0][i:i + 1]] * cur_beam_size
                model_encodings_l_2 += [model_encodings[1][i:i + 1]] * cur_beam_size

                src_mask_l1 += [enc_src_mask[0][i:i + 1]] * cur_beam_size
                src_mask_l2 += [enc_src_mask[1][i:i + 1]] * cur_beam_size

                if len(model_encodings) == 4:
                    model_encodings_l_3 += [model_encodings[2][i:i + 1]] * cur_beam_size
                    src_mask_l3 += [enc_src_mask[2][i:i + 1]] * cur_beam_size
                    model_encodings_l_4 += [model_encodings[3][i:i + 1]] * cur_beam_size
                    src_mask_l4 += [enc_src_mask[3][i:i + 1]] * cur_beam_size
                elif len(model_encodings) == 3:
                    model_encodings_l_3 += [model_encodings[2][i:i + 1]] * cur_beam_size
                    src_mask_l3 += [enc_src_mask[2][i:i + 1]] * cur_beam_size

                r2l_memory_l += [r2l_memory[i: i + 1]] * cur_beam_size

            model_encodings_cur_1 = torch.cat(model_encodings_l_1, dim=0)
            model_encodings_cur_2 = torch.cat(model_encodings_l_2, dim=0)

            src_mask_cur1 = torch.cat(src_mask_l1, dim=0)
            src_mask_cur2 = torch.cat(src_mask_l2, dim=0)

            if model_encodings_l_4:

                src_mask_cur3 = torch.cat(src_mask_l3, dim=0)
                src_mask_cur4 = torch.cat(src_mask_l4, dim=0)
                src_mask_cur = (src_mask_cur1, src_mask_cur2, src_mask_cur3, src_mask_cur4)

                model_encodings_cur_3 = torch.cat(model_encodings_l_3, dim=0)
                model_encodings_cur_4 = torch.cat(model_encodings_l_4, dim=0)
                model_encodings_cur = (
                    model_encodings_cur_1, model_encodings_cur_2, model_encodings_cur_3, model_encodings_cur_4)
            elif model_encodings_l_3:
                src_mask_cur3 = torch.cat(src_mask_l3, dim=0)
                src_mask_cur = (src_mask_cur1, src_mask_cur2, src_mask_cur3)

                model_encodings_cur_3 = torch.cat(model_encodings_l_3, dim=0)
                model_encodings_cur = (
                    model_encodings_cur_1, model_encodings_cur_2, model_encodings_cur_3)

            else:
                src_mask_cur = (src_mask_cur1, src_mask_cur2)
                model_encodings_cur = (model_encodings_cur_1, model_encodings_cur_2)

            y_tm1 = torch.cat(last_tokens, dim=0)
            r2l_memory_cur = torch.cat(r2l_memory_l, dim=0)

            out = self.l2r_decode(Variable(y_tm1).to(self.device), model_encodings_cur, src_mask_cur,
                                  Variable(subsequent_mask(y_tm1.size(-1)).type_as(src[0].data)).to(self.device),
                                  r2l_memory_cur, r2l_trg_mask=None)

            log_prob = self.generator(out[:, -1, :]).unsqueeze(1)
            _, decoded_len, vocab_sz = log_prob.shape
            log_prob = torch.split(log_prob, cur_beam_sizes, dim=0)

            new_hypotheses, new_hyp_scores = [], []
            for i in range(batch_size):
                if hypotheses[i] is None or len(completed_hypotheses[i]) >= beam_size:
                    new_hypotheses += [None]
                    new_hyp_scores += [None]
                    continue

                cur_beam_sz_i, dec_sent_len, vocab_sz = log_prob[i].shape
                cumulative_hyp_scores_i = (hyp_scores[i].unsqueeze(-1).unsqueeze(-1)
                                           .expand((cur_beam_sz_i, 1, vocab_sz)) + log_prob[i]).view(-1)

                live_hyp_num_i = beam_size - len(completed_hypotheses[i])
                top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(cumulative_hyp_scores_i, k=live_hyp_num_i)
                prev_hyp_ids, hyp_word_ids = top_cand_hyp_pos // self.vocab.n_vocabs, \
                                             top_cand_hyp_pos % self.vocab.n_vocabs

                new_hypotheses_i, new_hyp_scores_i = [], []
                for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids,
                                                                        top_cand_hyp_scores):
                    prev_hyp_id, hyp_word_id, cand_new_hyp_score = \
                        prev_hyp_id.item(), hyp_word_id.item(), cand_new_hyp_score.item()

                    new_hyp_sent = torch.cat(
                        (hypotheses[i][prev_hyp_id], torch.tensor([hyp_word_id], device=self.device)))
                    if hyp_word_id == end_symbol:
                        completed_hypotheses[i].append(Hypothesis(
                            value=[self.vocab.idx2word[a.item()] for a in new_hyp_sent[1:-1]],
                            score=cand_new_hyp_score))
                    else:
                        new_hypotheses_i.append(new_hyp_sent.unsqueeze(-1))
                        new_hyp_scores_i.append(cand_new_hyp_score)

                if len(new_hypotheses_i) > 0:
                    hypotheses_i = torch.cat(new_hypotheses_i, dim=-1).transpose(0, -1).to(self.device)
                    hyp_scores_i = torch.tensor(new_hyp_scores_i, dtype=torch.float, device=self.device)
                else:
                    hypotheses_i, hyp_scores_i = None, None
                new_hypotheses += [hypotheses_i]
                new_hyp_scores += [hyp_scores_i]
            hypotheses, hyp_scores = new_hypotheses, new_hyp_scores

        for i in range(batch_size):
            hyps_to_add = beam_size - len(completed_hypotheses[i])
            if hyps_to_add > 0:
                scores, ix = torch.topk(hyp_scores[i], k=hyps_to_add)
                for score, id in zip(scores, ix):
                    completed_hypotheses[i].append(Hypothesis(
                        value=[self.vocab.idx2word[a.item()] for a in hypotheses[i][id][1:]],
                        score=score))
            completed_hypotheses[i].sort(key=lambda hyp: hyp.score, reverse=True)
        return r2l_completed_hypotheses, completed_hypotheses
