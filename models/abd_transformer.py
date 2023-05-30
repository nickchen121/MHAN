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

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 多头注意力机制的线性变换层是4维，是把query[batch, frame_num, d_model]变成[batch, -1, head, d_k]
            # 再1，2维交换变成[batch, head, -1, d_k], 所以mask要在第一维添加一维，与后面的self attention计算维度一样
            mask = mask.unsqueeze(1)
        n_batch = query.size(0)
        # if self.head == 1:
        #     x, self.attn = self_attention(query, key, value, dropout=self.dropout, mask=mask)
        # else:
        #     query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 32, 64]
        #     key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]
        #     value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]
        #
        #     x, self.attn = self_attention(query, key, value, dropout=self.dropout, mask=mask)
        #     # 变为三维， 或者说是concat head
        #     x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)

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

    def __init__(self, size, attn, feed_forward, sublayer_num, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), sublayer_num)

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory=None, r2l_trg_mask=None):
        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, trg_mask))
        x = self.sublayer_connection[1](x, lambda x: self.attn(x, memory, memory, src_mask))

        if r2l_memory is not None:
            x = self.sublayer_connection[-2](x, lambda x: self.attn(x, r2l_memory, r2l_memory, r2l_trg_mask))

        return self.sublayer_connection[-1](x, self.feed_forward)


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
        for layer in self.decoder_layer:
            x = layer(x, memory, src_mask, r2l_trg_mask)
        return x


class L2R_Decoder(nn.Module):

    def __init__(self, n, decoder_layer):
        super(L2R_Decoder, self).__init__()
        self.decoder_layer = clones(decoder_layer, n)

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask):
        for layer in self.decoder_layer:
            x = layer(x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask)
        return x


def pad_mask(src, r2l_trg, trg, pad_idx):
    if isinstance(src, tuple):
        if len(src) == 4:
            src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
            src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
            src_object_mask = (src[2][:, :, 0] != pad_idx).unsqueeze(1)
            src_rel_mask = (src[3][:, :, 0] != pad_idx).unsqueeze(1)
            enc_src_mask = (src_image_mask, src_motion_mask, src_object_mask, src_rel_mask)
            dec_src_mask = src_image_mask & src_motion_mask
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
                 device='cuda', n_heads_big=128):
        super(ABDTransformer, self).__init__()
        self.vocab = vocab
        self.device = device
        self.feature_mode = feature_mode

        c = copy.deepcopy

        # attn_no_heads = MultiHeadAttention(1, d_model, dropout)

        attn = MultiHeadAttention(n_heads, d_model, dropout)

        attn_big = MultiHeadAttention(n_heads_big, d_model, dropout)

        # attn_big2 = MultiHeadAttention(10, d_model, dropout)

        feed_forward = PositionWiseFeedForward(d_model, d_ff)

        if feature_mode == 'one':
            self.src_embed = FeatEmbedding(d_feat, d_model, dropout)
        elif feature_mode == 'two':
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

        # self.encoder_no_heads = Encoder(n_layers, EncoderLayer(d_model, c(attn_no_heads), c(feed_forward), dropout))

        self.encoder = Encoder(n_layers, EncoderLayer(d_model, c(attn), c(feed_forward), dropout))

        self.encoder_big = Encoder(n_layers, EncoderLayer(d_model, c(attn_big), c(feed_forward), dropout))

        # self.encoder_big2 = Encoder(n_layers, EncoderLayer(d_model, c(attn_big2), c(feed_forward), dropout))

        self.encoder_no_attention = Encoder(n_layers,
                                            EncoderLayerNoAttention(d_model, c(attn), c(feed_forward), dropout))

        self.r2l_decoder = R2L_Decoder(n_layers, DecoderLayer(d_model, c(attn), c(feed_forward),
                                                              sublayer_num=3, dropout=dropout))
        self.l2r_decoder = L2R_Decoder(n_layers, DecoderLayer(d_model, c(attn), c(feed_forward),
                                                              sublayer_num=4, dropout=dropout))

        self.generator = Generator(d_model, vocab.n_vocabs)

    def encode(self, src, src_mask, feature_mode_two=False):
        if self.feature_mode == 'two':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder_big(x1, src_mask[0])
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder_big(x2, src_mask[1])
            return x1 + x2
        if feature_mode_two:
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder_big(x1, src_mask[0])
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder_big(x2, src_mask[1])
            return x1 + x2
        if self.feature_mode == 'one':
            x = self.src_embed(src)
            x = self.pos_embed(x)
            return self.encoder(x, src_mask)
        elif self.feature_mode == 'two':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder_big(x1, src_mask[0])
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder_big(x2, src_mask[1])
            return x1 + x2
        elif self.feature_mode == 'three':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder(x1, src_mask[0])
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder(x2, src_mask[1])
            x3 = self.object_src_embed(src[2])
            x3 = self.pos_embed(x3)
            x3 = self.encoder(x3, src_mask[2])
            return x1 + x2 + x3
        elif self.feature_mode == 'four':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder(x1, src_mask[0])

            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder(x2, src_mask[1])

            x3 = self.object_src_embed(src[2])
            # x3 = self.pos_embed(x3)
            x3 = self.encoder(x3, src_mask[2])
            # x3 = self.encoder_no_attention(x3, src_mask[2])

            x4 = self.rel_src_embed(src[3])
            # x4 = self.pos_embed(x4)
            # x4 = self.encoder_no_
            # heads(x4, src_mask[3])
            x4 = self.encoder_no_attention(x4, src_mask[3])
            # x4 = self.encoder(x4, src_mask[3])
            return x1 + x2 + x3 + x4

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
        if self.feature_mode == 'one':
            encoding_outputs = self.encode(src, src_mask)
            r2l_outputs = self.r2l_decode(r2l_trg, encoding_outputs, src_mask, r2l_trg_mask)
            l2r_outputs = self.l2r_decode(trg, encoding_outputs, src_mask, trg_mask, r2l_outputs, r2l_pad_mask)

        elif self.feature_mode == 'two' or 'three' or 'four':
            enc_src_mask, dec_src_mask = src_mask
            r2l_encoding_outputs = self.encode(src, enc_src_mask, feature_mode_two=True)
            encoding_outputs = self.encode(src, enc_src_mask)

            r2l_outputs = self.r2l_decode(r2l_trg, r2l_encoding_outputs, dec_src_mask, r2l_trg_mask)
            l2r_outputs = self.l2r_decode(trg, encoding_outputs, dec_src_mask, trg_mask, r2l_outputs, r2l_pad_mask)

            # r2l_outputs = self.r2l_decode(r2l_trg, encoding_outputs, dec_src_mask, r2l_trg_mask)
            # l2r_outputs = self.l2r_decode(trg, encoding_outputs, dec_src_mask, trg_mask, None, None)
        else:
            raise "没有输出"

        r2l_pred = self.generator(r2l_outputs)
        l2r_pred = self.generator(l2r_outputs)

        return r2l_pred, l2r_pred

    def greedy_decode(self, batch_size, src_mask, memory, max_len):

        eos_idx = self.vocab.word2idx['<S>']
        r2l_hidden = None
        with torch.no_grad():
            output = torch.ones(batch_size, 1).fill_(eos_idx).long().cuda()
            for i in range(max_len + 2 - 1):
                trg_mask = subsequent_mask(output.size(1))
                dec_out = self.r2l_decode(output, memory, src_mask, trg_mask)  # batch, len, d_model
                r2l_hidden = dec_out
                pred = self.generator(dec_out)  # batch, len, n_vocabs
                next_word = pred[:, -1].max(dim=-1)[1].unsqueeze(1)  # pred[:, -1]([batch, n_vocabs])
                output = torch.cat([output, next_word], dim=-1)
        return r2l_hidden, output

    # 代码输入的是logits，而且考虑很周全（我感觉漏了考虑k和p都给了的情况，这应该是不合适的）
    # 巧妙地使用了torch.cumsum
    # 避免了一个词都选不出来的尴尬情况
    def top_k_top_p_filtering(self,logits, top_k, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (batch size, vocabulary size)
                if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
                Make sure we keep at least min_tokens_to_keep per batch example in the output
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits
    def r2l_beam_search_decode(self, batch_size, src, src_mask, model_encodings, beam_size, max_len):
        end_symbol = self.vocab.word2idx['<S>']
        start_symbol = self.vocab.word2idx['<S>']
        r2l_outputs = None

        # 1.1 Setup Src
        "src has shape (batch_size, sent_len)"
        "src_mask has shape (batch_size, 1, sent_len)"
        # src_mask = (src[:, :, 0] != self.vocab.word2idx['<PAD>']).unsqueeze(-2)  # TODO Untested
        "model_encodings has shape (batch_size, sentence_len, d_model)"
        # model_encodings = self.encode(src, src_mask)

        # 1.2 Setup Tgt Hypothesis Tracking
        "hypothesis is List(4 bt)[(cur beam_sz, dec_sent_len)], init: List(4 bt)[(1 init_beam_sz, dec_sent_len)]"
        "hypotheses[i] is shape (cur beam_sz, dec_sent_len)"
        hypotheses = [copy.deepcopy(torch.full((1, 1), start_symbol, dtype=torch.long,
                                               device=self.device)) for _ in range(batch_size)]
        "List after init: List 4 bt of List of len max_len_completed, init: List of len 4 bt of []"
        completed_hypotheses = [copy.deepcopy([]) for _ in range(batch_size)]
        "List len batch_sz of shape (cur beam_sz), init: List(4 bt)[(1 init_beam_sz)]"
        "hyp_scores[i] is shape (cur beam_sz)"
        hyp_scores = [copy.deepcopy(torch.full((1,), 0, dtype=torch.float, device=self.device))
                      for _ in range(batch_size)]  # probs are log_probs must be init at 0.
        # print(hypotheses)  # [tensor([[1]], device='cuda:0')]
        # print(completed_hypotheses)# [[]]
        # print(hypotheses)# [tensor([[1]], device='cuda:0')]

        # 2. Iterate: Generate one char at a time until maxlen
        for iter in range(max_len + 1):
            # print('iter',iter)
            if all([len(completed_hypotheses[i]) == beam_size for i in range(batch_size)]):
                break

            # 2.1 Setup the batch. Since we use beam search, each batch has a variable number (called cur_beam_size)
            # between 0 and beam_size of hypotheses live at any moment. We decode all hypotheses for all batches at
            # the same time, so we must copy the src_encodings, src_mask, etc the appropriate number fo times for
            # the number of hypotheses for each example. We keep track of the number of live hypotheses for each example.
            # We run all hypotheses for all examples together through the decoder and log-softmax,
            # and then use `torch.split` to get the appropriate number of hypotheses for each example in the end.
            cur_beam_sizes, last_tokens, model_encodings_l, src_mask_l = [], [], [], []
            for i in range(batch_size):  #1
                if hypotheses[i] is None:
                    cur_beam_sizes += [0]
                    continue
                # print(hypotheses[i].shape)
                cur_beam_size, decoded_len = hypotheses[i].shape
                # print('iter',iter,'cur_beam_size',cur_beam_size) #5
                cur_beam_sizes += [cur_beam_size]
                last_tokens += [hypotheses[i]]
                model_encodings_l += [model_encodings[i:i + 1]] * cur_beam_size
                src_mask_l += [src_mask[i:i + 1]] * cur_beam_size
                # print(batch_size) #1
                # print('cur_beam_size',cur_beam_size) #5
                # print('decoded_len)',decoded_len)   #5
                # print('last_tokens',last_tokens)
                # print('model_encodings_l',model_encodings_l)
                # print('src_mask_l',src_mask_l)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            model_encodings_cur = torch.cat(model_encodings_l, dim=0)
            src_mask_cur = torch.cat(src_mask_l, dim=0)
            y_tm1 = torch.cat(last_tokens, dim=0)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            if self.feature_mode == 'one':
                out = self.r2l_decode(Variable(y_tm1).to(self.device), model_encodings_cur, src_mask_cur,
                                      Variable(subsequent_mask(y_tm1.size(-1)).type_as(src.data)).to(self.device))
            elif self.feature_mode == 'two' or 'three' or 'four':
                out = self.r2l_decode(Variable(y_tm1).to(self.device), model_encodings_cur, src_mask_cur,
                                      Variable(subsequent_mask(y_tm1.size(-1)).type_as(src[0].data)).to(self.device))
            r2l_outputs = out

            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 50002 vocab_sz)"

            log_prob = self.generator(out[:, -1, :]).unsqueeze(1)
            # print(out.shape,out[:, -1, :].shape,log_prob.shape)
            # torch.Size([5, 2, 640])
            # torch.Size([5, 640])
            # torch.Size([5, 1, 9307])
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 50002 vocab_sz)"
            _, decoded_len, vocab_sz = log_prob.shape
            # log_prob = log_prob.reshape(batch_size, cur_beam_size, decoded_len, vocab_sz)
            "shape List(4 bt)[(cur_beam_sz_i, dec_sent_len, 50002 vocab_sz)]"
            "log_prob[i] is (cur_beam_sz_i, dec_sent_len, 50002 vocab_sz)"

            log_prob = torch.split(log_prob, cur_beam_sizes, dim=0)
            # for i in log_prob:
            #     print(iter,i.shape,cur_beam_sizes)


            # 2.2 Now we process each example in the batch. Note that the example may have already finished processing before
            # other examples (no more hypotheses to try), in which case we continue
            new_hypotheses, new_hyp_scores = [], []
            for i in range(batch_size):
                if hypotheses[i] is None or len(completed_hypotheses[i]) >= beam_size:
                    new_hypotheses += [None]
                    new_hyp_scores += [None]
                    continue

                # 2.2.1 We compute the cumulative scores for each live hypotheses for the example
                # hyp_scores is the old scores for the previous stage, and `log_prob` are the new probs for
                # this stage. Since they are log probs, we sum them instaed of multiplying them.
                # The .view(-1) forces all the hypotheses into one dimension. The shape of this dimension is
                # cur_beam_sz * vocab_sz (ex: 5 * 50002). So after getting the topk from it, we can recover the
                # generating sentence and the next word using: ix // vocab_sz, ix % vocab_sz.
                cur_beam_sz_i, dec_sent_len, vocab_sz = log_prob[i].shape
                "shape (vocab_sz,)"
                cumulative_hyp_scores_i = (hyp_scores[i].unsqueeze(-1).unsqueeze(-1)
                                           .expand((cur_beam_sz_i, 1, vocab_sz)) + log_prob[i]).view(-1)
                # print(cur_beam_sz_i, dec_sent_len, vocab_sz,cumulative_hyp_scores_i.shape) 5 1 9307 torch.Size([46535])
                # 2.2.2 We get the topk values in cumulative_hyp_scores_i and compute the current (generating) sentence
                # and the next word using: ix // vocab_sz, ix % vocab_sz.
                "shape (cur_beam_sz,)"
                live_hyp_num_i = beam_size - len(completed_hypotheses[i])
                nusa = torch.squeeze(log_prob[i], 1)
                _scores = self.top_k_top_p_filtering(nusa, 0, 0.85, min_tokens_to_keep=2)  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together to sample from all beam_idxs
                _scores = _scores.contiguous().view(
                    batch_size, cur_beam_sz_i * vocab_sz
                )  # (batch_size, num_beams * vocab_size)

                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                probs = F.softmax(_scores, dim=-1)
                # 采样
                next_tokens = torch.multinomial(probs, num_samples=live_hyp_num_i)  # (batch_size, num_beams * 2)
                # Compute next scores
                next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
                # sort the sampled vector to make sure that the first num_beams samples are the best
                next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)
                # print('next',next_scores.shape,next_tokens.shape)
                top_cand_hyp_scores=torch.squeeze(next_scores,0)
                top_cand_hyp_pos=torch.squeeze(next_tokens,0)


                "shape (cur_beam_sz,). Vals are between 0 and 50002 vocab_sz"
                # top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(cumulative_hyp_scores_i, k=live_hyp_num_i)
                # print('top',top_cand_hyp_scores.shape,top_cand_hyp_pos.shape)
                # print(top_cand_hyp_pos)
                "shape (cur_beam_sz,). prev_hyp_ids vals are 0 <= val < cur_beam_sz. hyp_word_ids vals are 0 <= val < vocab_len"
                prev_hyp_ids, hyp_word_ids = top_cand_hyp_pos // self.vocab.n_vocabs, \
                                             top_cand_hyp_pos % self.vocab.n_vocabs
                # print(top_cand_hyp_scores,top_cand_hyp_pos.size(),prev_hyp_ids,hyp_word_ids)
                # tensor([-1.1972, -2.7728, -3.1850, -3.7347, -3.8908],
                #                                           device='cuda:0')
                # torch.Size([5])
                # tensor([0, 0, 0, 0, 0], device='cuda:0')
                # tensor([735, 714, 1812, 94, 332], device='cuda:0')

                # 2.2.3 For each of the topk words, we append the new word to the current (generating) sentence
                # We add this to new_hypotheses_i and add its corresponding total score to new_hyp_scores_i
                new_hypotheses_i, new_hyp_scores_i = [], []  # Removed live_hyp_ids_i, which is used in the LSTM decoder to track live hypothesis ids
                for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids,
                                                                        top_cand_hyp_scores):
                    prev_hyp_id, hyp_word_id, cand_new_hyp_score = \
                        prev_hyp_id.item(), hyp_word_id.item(), cand_new_hyp_score.item()
                    # print(prev_hyp_id,hyp_word_id,cand_new_hyp_score)

                    new_hyp_sent = torch.cat(
                        (hypotheses[i][prev_hyp_id], torch.tensor([hyp_word_id], device=self.device)))
                    # print(new_hyp_sent)
                    # print(new_hyp_sent.size())
                    # print(new_hyp_sent.unsqueeze(-1).size())
                    if hyp_word_id == end_symbol:
                        completed_hypotheses[i].append(Hypothesis(
                            value=[self.vocab.idx2word[a.item()] for a in new_hyp_sent[1:-1]],
                            score=cand_new_hyp_score))
                    else:
                        new_hypotheses_i.append(new_hyp_sent.unsqueeze(-1))
                        new_hyp_scores_i.append(cand_new_hyp_score)

                # 2.2.4 We may find that the hypotheses_i for some example in the batch
                # is empty - we have fully processed that example. We use None as a sentinel in this case.
                # Above, the loops gracefully handle None examples.
                if len(new_hypotheses_i) > 0:
                    hypotheses_i = torch.cat(new_hypotheses_i, dim=-1).transpose(0, -1).to(self.device)
                    hyp_scores_i = torch.tensor(new_hyp_scores_i, dtype=torch.float, device=self.device)
                else:
                    hypotheses_i, hyp_scores_i = None, None
                new_hypotheses += [hypotheses_i]
                new_hyp_scores += [hyp_scores_i]
            # print(new_hypotheses, new_hyp_scores)
            hypotheses, hyp_scores = new_hypotheses, new_hyp_scores

        # 2.3 Finally, we do some postprocessing to get our final generated candidate sentences.
        # Sometimes, we may get to max_len of a sentence and still not generate the </s> end token.
        # In this case, the partial sentence we have generated will not be added to the completed_hypotheses
        # automatically, and we have to manually add it in. We add in as many as necessary so that there are
        # `beam_size` completed hypotheses for each example.
        # Finally, we sort each completed hypothesis by score.
        # print(completed_hypotheses)
        for i in range(batch_size):
            hyps_to_add = beam_size - len(completed_hypotheses[i])
            if hyps_to_add > 0:
                scores, ix = torch.topk(hyp_scores[i], k=hyps_to_add)
                for score, id in zip(scores, ix):
                    completed_hypotheses[i].append(Hypothesis(
                        value=[self.vocab.idx2word[a.item()] for a in hypotheses[i][id][1:]],
                        score=score))
            completed_hypotheses[i].sort(key=lambda hyp: hyp.score, reverse=True)
        # print(completed_hypotheses)
        return r2l_outputs, completed_hypotheses
    def r2l_beam_search_decode1(self, batch_size, src, src_mask, model_encodings, beam_size, max_len):
        end_symbol = self.vocab.word2idx['<S>']
        start_symbol = self.vocab.word2idx['<S>']

        r2l_outputs = None

        # 1.1 Setup Src
        "src has shape (batch_size, sent_len)"
        "src_mask has shape (batch_size, 1, sent_len)"
        # src_mask = (src[:, :, 0] != self.vocab.word2idx['<PAD>']).unsqueeze(-2)  # TODO Untested
        "model_encodings has shape (batch_size, sentence_len, d_model)"
        # model_encodings = self.encode(src, src_mask)

        # 1.2 Setup Tgt Hypothesis Tracking
        "hypothesis is List(4 bt)[(cur beam_sz, dec_sent_len)], init: List(4 bt)[(1 init_beam_sz, dec_sent_len)]"
        "hypotheses[i] is shape (cur beam_sz, dec_sent_len)"
        hypotheses = [copy.deepcopy(torch.full((1, 1), start_symbol, dtype=torch.long,
                                               device=self.device)) for _ in range(batch_size)]
        "List after init: List 4 bt of List of len max_len_completed, init: List of len 4 bt of []"
        completed_hypotheses = [copy.deepcopy([]) for _ in range(batch_size)]
        "List len batch_sz of shape (cur beam_sz), init: List(4 bt)[(1 init_beam_sz)]"
        "hyp_scores[i] is shape (cur beam_sz)"
        hyp_scores = [copy.deepcopy(torch.full((1,), 0, dtype=torch.float, device=self.device))
                      for _ in range(batch_size)]  # probs are log_probs must be init at 0.
        # print(hypotheses)  # [tensor([[1]], device='cuda:0')]
        # print(completed_hypotheses)# [[]]
        # print(hypotheses)# [tensor([[1]], device='cuda:0')]

        # 2. Iterate: Generate one char at a time until maxlen
        for iter in range(max_len + 1):
            # print('iter',iter)
            if all([len(completed_hypotheses[i]) == beam_size for i in range(batch_size)]):
                break

            # 2.1 Setup the batch. Since we use beam search, each batch has a variable number (called cur_beam_size)
            # between 0 and beam_size of hypotheses live at any moment. We decode all hypotheses for all batches at
            # the same time, so we must copy the src_encodings, src_mask, etc the appropriate number fo times for
            # the number of hypotheses for each example. We keep track of the number of live hypotheses for each example.
            # We run all hypotheses for all examples together through the decoder and log-softmax,
            # and then use `torch.split` to get the appropriate number of hypotheses for each example in the end.
            cur_beam_sizes, last_tokens, model_encodings_l, src_mask_l = [], [], [], []
            for i in range(batch_size):  #1
                if hypotheses[i] is None:
                    cur_beam_sizes += [0]
                    continue
                # print(hypotheses[i].shape)
                cur_beam_size, decoded_len = hypotheses[i].shape
                # print('iter',iter,'cur_beam_size',cur_beam_size) #5
                cur_beam_sizes += [cur_beam_size]
                last_tokens += [hypotheses[i]]
                model_encodings_l += [model_encodings[i:i + 1]] * cur_beam_size
                src_mask_l += [src_mask[i:i + 1]] * cur_beam_size
                # print(batch_size) #1
                # print('cur_beam_size',cur_beam_size) #5
                # print('decoded_len)',decoded_len)   #5
                # print('last_tokens',last_tokens)
                # print('model_encodings_l',model_encodings_l)
                # print('src_mask_l',src_mask_l)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            model_encodings_cur = torch.cat(model_encodings_l, dim=0)
            src_mask_cur = torch.cat(src_mask_l, dim=0)
            y_tm1 = torch.cat(last_tokens, dim=0)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            if self.feature_mode == 'one':
                out = self.r2l_decode(Variable(y_tm1).to(self.device), model_encodings_cur, src_mask_cur,
                                      Variable(subsequent_mask(y_tm1.size(-1)).type_as(src.data)).to(self.device))
            elif self.feature_mode == 'two' or 'three' or 'four':
                out = self.r2l_decode(Variable(y_tm1).to(self.device), model_encodings_cur, src_mask_cur,
                                      Variable(subsequent_mask(y_tm1.size(-1)).type_as(src[0].data)).to(self.device))
            r2l_outputs = out

            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 50002 vocab_sz)"

            log_prob = self.generator(out[:, -1, :]).unsqueeze(1)
            # print(out.shape,out[:, -1, :].shape,log_prob.shape)
            # torch.Size([5, 2, 640])
            # torch.Size([5, 640])
            # torch.Size([5, 1, 9307])
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 50002 vocab_sz)"
            _, decoded_len, vocab_sz = log_prob.shape
            # log_prob = log_prob.reshape(batch_size, cur_beam_size, decoded_len, vocab_sz)
            "shape List(4 bt)[(cur_beam_sz_i, dec_sent_len, 50002 vocab_sz)]"
            "log_prob[i] is (cur_beam_sz_i, dec_sent_len, 50002 vocab_sz)"

            log_prob = torch.split(log_prob, cur_beam_sizes, dim=0)
            # for i in log_prob:
            #     print(iter,i.shape,cur_beam_sizes)


            # 2.2 Now we process each example in the batch. Note that the example may have already finished processing before
            # other examples (no more hypotheses to try), in which case we continue
            new_hypotheses, new_hyp_scores = [], []
            for i in range(batch_size):
                if hypotheses[i] is None or len(completed_hypotheses[i]) >= beam_size:
                    new_hypotheses += [None]
                    new_hyp_scores += [None]
                    continue

                # 2.2.1 We compute the cumulative scores for each live hypotheses for the example
                # hyp_scores is the old scores for the previous stage, and `log_prob` are the new probs for
                # this stage. Since they are log probs, we sum them instaed of multiplying them.
                # The .view(-1) forces all the hypotheses into one dimension. The shape of this dimension is
                # cur_beam_sz * vocab_sz (ex: 5 * 50002). So after getting the topk from it, we can recover the
                # generating sentence and the next word using: ix // vocab_sz, ix % vocab_sz.
                cur_beam_sz_i, dec_sent_len, vocab_sz = log_prob[i].shape
                "shape (vocab_sz,)"
                cumulative_hyp_scores_i = (hyp_scores[i].unsqueeze(-1).unsqueeze(-1)
                                           .expand((cur_beam_sz_i, 1, vocab_sz)) + log_prob[i]).view(-1)
                # print(cur_beam_sz_i, dec_sent_len, vocab_sz,cumulative_hyp_scores_i.shape) 5 1 9307 torch.Size([46535])
                # 2.2.2 We get the topk values in cumulative_hyp_scores_i and compute the current (generating) sentence
                # and the next word using: ix // vocab_sz, ix % vocab_sz.
                "shape (cur_beam_sz,)"
                live_hyp_num_i = beam_size - len(completed_hypotheses[i])
                "shape (cur_beam_sz,). Vals are between 0 and 50002 vocab_sz"
                top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(cumulative_hyp_scores_i, k=live_hyp_num_i)
                "shape (cur_beam_sz,). prev_hyp_ids vals are 0 <= val < cur_beam_sz. hyp_word_ids vals are 0 <= val < vocab_len"
                prev_hyp_ids, hyp_word_ids = top_cand_hyp_pos // self.vocab.n_vocabs, \
                                             top_cand_hyp_pos % self.vocab.n_vocabs
                # print(top_cand_hyp_scores,top_cand_hyp_pos.size(),prev_hyp_ids,hyp_word_ids)
                # tensor([-1.1972, -2.7728, -3.1850, -3.7347, -3.8908],
                #                                           device='cuda:0')
                # torch.Size([5])
                # tensor([0, 0, 0, 0, 0], device='cuda:0')
                # tensor([735, 714, 1812, 94, 332], device='cuda:0')

                # 2.2.3 For each of the topk words, we append the new word to the current (generating) sentence
                # We add this to new_hypotheses_i and add its corresponding total score to new_hyp_scores_i
                new_hypotheses_i, new_hyp_scores_i = [], []  # Removed live_hyp_ids_i, which is used in the LSTM decoder to track live hypothesis ids
                for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids,
                                                                        top_cand_hyp_scores):
                    prev_hyp_id, hyp_word_id, cand_new_hyp_score = \
                        prev_hyp_id.item(), hyp_word_id.item(), cand_new_hyp_score.item()
                    # print(prev_hyp_id,hyp_word_id,cand_new_hyp_score)

                    new_hyp_sent = torch.cat(
                        (hypotheses[i][prev_hyp_id], torch.tensor([hyp_word_id], device=self.device)))
                    # print(new_hyp_sent)
                    # print(new_hyp_sent.size())
                    # print(new_hyp_sent.unsqueeze(-1).size())
                    if hyp_word_id == end_symbol:
                        completed_hypotheses[i].append(Hypothesis(
                            value=[self.vocab.idx2word[a.item()] for a in new_hyp_sent[1:-1]],
                            score=cand_new_hyp_score))
                    else:
                        new_hypotheses_i.append(new_hyp_sent.unsqueeze(-1))
                        new_hyp_scores_i.append(cand_new_hyp_score)

                # 2.2.4 We may find that the hypotheses_i for some example in the batch
                # is empty - we have fully processed that example. We use None as a sentinel in this case.
                # Above, the loops gracefully handle None examples.
                if len(new_hypotheses_i) > 0:
                    hypotheses_i = torch.cat(new_hypotheses_i, dim=-1).transpose(0, -1).to(self.device)
                    hyp_scores_i = torch.tensor(new_hyp_scores_i, dtype=torch.float, device=self.device)
                else:
                    hypotheses_i, hyp_scores_i = None, None
                new_hypotheses += [hypotheses_i]
                new_hyp_scores += [hyp_scores_i]
            # print(new_hypotheses, new_hyp_scores)
            hypotheses, hyp_scores = new_hypotheses, new_hyp_scores

        # 2.3 Finally, we do some postprocessing to get our final generated candidate sentences.
        # Sometimes, we may get to max_len of a sentence and still not generate the </s> end token.
        # In this case, the partial sentence we have generated will not be added to the completed_hypotheses
        # automatically, and we have to manually add it in. We add in as many as necessary so that there are
        # `beam_size` completed hypotheses for each example.
        # Finally, we sort each completed hypothesis by score.
        # print(completed_hypotheses)
        for i in range(batch_size):
            hyps_to_add = beam_size - len(completed_hypotheses[i])
            if hyps_to_add > 0:
                scores, ix = torch.topk(hyp_scores[i], k=hyps_to_add)
                for score, id in zip(scores, ix):
                    completed_hypotheses[i].append(Hypothesis(
                        value=[self.vocab.idx2word[a.item()] for a in hypotheses[i][id][1:]],
                        score=score))
            completed_hypotheses[i].sort(key=lambda hyp: hyp.score, reverse=True)
        # print(r2l_outputs)
        return r2l_outputs, completed_hypotheses

    def beam_search_decode(self, src, beam_size, max_len):
        """
                An Implementation of Beam Search for the Transformer Model.
                Beam search is performed in a batched manner. Each example in a batch generates `beam_size` hypotheses.
                We return a list (len: batch_size) of list (len: beam_size) of Hypothesis, which contain our output decoded sentences
                and their scores.
                :param src: shape (sent_len, batch_size). Each val is 0 < val < len(vocab_dec). The input tokens to the decoder.
                :param max_len: the maximum length to decode
                :param beam_size: the beam size to use
                :return completed_hypotheses: A List of length batch_size, each containing a List of beam_size Hypothesis objects.
                    Hypothesis is a named Tuple, its first entry is "value" and is a List of strings which contains the translated word
                    (one string is one word token). The second entry is "score" and it is the log-prob score for this translated sentence.
                Note: Below I note "4 bt", "5 beam_size" as the shapes of objects. 4, 5 are default values. Actual values may differ.
                """
        # 1. Setup
        start_symbol = self.vocab.word2idx['<S>']
        # print(start_symbol)  #1
        end_symbol = self.vocab.word2idx['<S>']
        # print(end_symbol)  #1

        # 1.1 Setup Src
        "src has shape (batch_size, sent_len)"
        "src_mask has shape (batch_size, 1, sent_len)"
        # src_mask = (src[:, :, 0] != self.vocab.word2idx['<PAD>']).unsqueeze(-2)  # TODO Untested
        src_mask = pad_mask(src, r2l_trg=None, trg=None, pad_idx=self.vocab.word2idx['<PAD>'])
        "model_encodings has shape (batch_size, sentence_len, d_model)"
        if self.feature_mode == 'one':
            batch_size = src.shape[0]
            model_encodings = self.encode(src, src_mask)
            r2l_memory, r2l_completed_hypotheses = self.r2l_beam_search_decode(batch_size, src, src_mask,
                                                                               model_encodings=model_encodings,
                                                                               beam_size=beam_size, max_len=max_len)
        elif self.feature_mode == 'two' or 'three' or 'four':
            # print('src', src.size())
            # print('src0',src[0].size())   # 1 60 1836
            # print('src1', src[1].size())  # 1 60 1324
            # print('src2', src[2].size())  # 1 60 1328
            # print('src3', src[3].size())  # 1 60 600
            # print('src_mask[0]',src_mask[0][0].size())  # 1 1 60
            # print('src_mask[1]', src_mask[0][1].size())  # 1 1 60
            # print('src_mask[2]', src_mask[0][2].size()) # 1 1 60
            # print('src_mask[3]', src_mask[0][3].size()) # 1 1 60
            # print('src_mask[1]',src_mask[1].size()) # 1 1 60

            batch_size = src[0].shape[0]
            enc_src_mask = src_mask[0]
            dec_src_mask = src_mask[1]
            r2l_model_encodings = self.encode(src, enc_src_mask, feature_mode_two=True)
            # print('r2l_model_encodings:',r2l_model_encodings.size()) # 1 60 640
            # model_encodings = r2l_model_encodings
            model_encodings = self.encode(src, enc_src_mask)
            # print('model_encodings:', model_encodings.size()) # 1 60 640
            r2l_memory, r2l_completed_hypotheses = self.r2l_beam_search_decode(batch_size, src, dec_src_mask,
                                                                               model_encodings=r2l_model_encodings,
                                                                               beam_size=beam_size, max_len=max_len)

        # 1.2 Setup r2l target output
        # r2l_memory, r2l_completed_hypotheses = self.r2l_beam_search_decode(batch_size, src, src_mask,
        #                                                                    model_encodings=model_encodings,
        #                                                                    beam_size=1, max_len=max_len)
        # r2l_memory, r2l_completed_hypotheses = self.greedy_decode(batch_size, src_mask, model_encodings, max_len)
        # beam_r2l_memory = [copy.deepcopy(r2l_memory) for _ in range(beam_size)]
        # 1.3 Setup Tgt Hypothesis Tracking
        "hypothesis is List(4 bt)[(cur beam_sz, dec_sent_len)], init: List(4 bt)[(1 init_beam_sz, dec_sent_len)]"
        "hypotheses[i] is shape (cur beam_sz, dec_sent_len)"
        hypotheses = [copy.deepcopy(torch.full((1, 1), start_symbol, dtype=torch.long,
                                               device=self.device)) for _ in range(batch_size)]
        "List after init: List 4 bt of List of len max_len_completed, init: List of len 4 bt of []"
        completed_hypotheses = [copy.deepcopy([]) for _ in range(batch_size)]
        "List len batch_sz of shape (cur beam_sz), init: List(4 bt)[(1 init_beam_sz)]"
        "hyp_scores[i] is shape (cur beam_sz)"
        hyp_scores = [copy.deepcopy(torch.full((1,), 0, dtype=torch.float, device=self.device))
                      for _ in range(batch_size)]  # probs are log_probs must be init at 0.

        # 2. Iterate: Generate one char at a time until maxlen
        for iter in range(max_len + 1):
            if all([len(completed_hypotheses[i]) == beam_size for i in range(batch_size)]):
                break

            # 2.1 Setup the batch. Since we use beam search, each batch has a variable number (called cur_beam_size)
            # between 0 and beam_size of hypotheses live at any moment. We decode all hypotheses for all batches at
            # the same time, so we must copy the src_encodings, src_mask, etc the appropriate number fo times for
            # the number of hypotheses for each example. We keep track of the number of live hypotheses for each example.
            # We run all hypotheses for all examples together through the decoder and log-softmax,
            # and then use `torch.split` to get the appropriate number of hypotheses for each example in the end.
            cur_beam_sizes, last_tokens, model_encodings_l, src_mask_l, r2l_memory_l = [], [], [], [], []
            for i in range(batch_size):
                if hypotheses[i] is None:
                    cur_beam_sizes += [0]
                    continue
                cur_beam_size, decoded_len = hypotheses[i].shape
                cur_beam_sizes += [cur_beam_size]
                last_tokens += [hypotheses[i]]
                model_encodings_l += [model_encodings[i:i + 1]] * cur_beam_size
                if self.feature_mode == 'one':
                    src_mask_l += [src_mask[i:i + 1]] * cur_beam_size
                elif self.feature_mode == 'two' or 'three' or 'four':
                    src_mask_l += [dec_src_mask[i:i + 1]] * cur_beam_size
                r2l_memory_l += [r2l_memory[i: i + 1]] * cur_beam_size
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            model_encodings_cur = torch.cat(model_encodings_l, dim=0)
            src_mask_cur = torch.cat(src_mask_l, dim=0)
            y_tm1 = torch.cat(last_tokens, dim=0)
            r2l_memory_cur = torch.cat(r2l_memory_l, dim=0)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            if self.feature_mode == 'one':
                out = self.l2r_decode(Variable(y_tm1).to(self.device), model_encodings_cur, src_mask_cur,
                                      Variable(subsequent_mask(y_tm1.size(-1)).type_as(src.data)).to(self.device),
                                      r2l_memory_cur, r2l_trg_mask=None)
            elif self.feature_mode == 'two' or 'three' or 'four':
                out = self.l2r_decode(Variable(y_tm1).to(self.device), model_encodings_cur, src_mask_cur,
                                      Variable(subsequent_mask(y_tm1.size(-1)).type_as(src[0].data)).to(self.device),
                                      r2l_memory_cur, r2l_trg_mask=None)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 50002 vocab_sz)"
            log_prob = self.generator(out[:, -1, :]).unsqueeze(1)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 50002 vocab_sz)"
            _, decoded_len, vocab_sz = log_prob.shape
            # log_prob = log_prob.reshape(batch_size, cur_beam_size, decoded_len, vocab_sz)
            "shape List(4 bt)[(cur_beam_sz_i, dec_sent_len, 50002 vocab_sz)]"
            "log_prob[i] is (cur_beam_sz_i, dec_sent_len, 50002 vocab_sz)"
            log_prob = torch.split(log_prob, cur_beam_sizes, dim=0)

            # 2.2 Now we process each example in the batch. Note that the example may have already finished processing before
            # other examples (no more hypotheses to try), in which case we continue
            new_hypotheses, new_hyp_scores = [], []
            for i in range(batch_size):
                if hypotheses[i] is None or len(completed_hypotheses[i]) >= beam_size:
                    new_hypotheses += [None]
                    new_hyp_scores += [None]
                    continue

                # 2.2.1 We compute the cumulative scores for each live hypotheses for the example
                # hyp_scores is the old scores for the previous stage, and `log_prob` are the new probs for
                # this stage. Since they are log probs, we sum them instaed of multiplying them.
                # The .view(-1) forces all the hypotheses into one dimension. The shape of this dimension is
                # cur_beam_sz * vocab_sz (ex: 5 * 50002). So after getting the topk from it, we can recover the
                # generating sentence and the next word using: ix // vocab_sz, ix % vocab_sz.
                cur_beam_sz_i, dec_sent_len, vocab_sz = log_prob[i].shape
                "shape (vocab_sz,)"
                cumulative_hyp_scores_i = (hyp_scores[i].unsqueeze(-1).unsqueeze(-1)
                                           .expand((cur_beam_sz_i, 1, vocab_sz)) + log_prob[i]).view(-1)

                # 2.2.2 We get the topk values in cumulative_hyp_scores_i and compute the current (generating) sentence
                # and the next word using: ix // vocab_sz, ix % vocab_sz.
                "shape (cur_beam_sz,)"
                live_hyp_num_i = beam_size - len(completed_hypotheses[i])
                "shape (cur_beam_sz,). Vals are between 0 and 50002 vocab_sz"
                top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(cumulative_hyp_scores_i, k=live_hyp_num_i)
                "shape (cur_beam_sz,). prev_hyp_ids vals are 0 <= val < cur_beam_sz. hyp_word_ids vals are 0 <= val < vocab_len"
                prev_hyp_ids, hyp_word_ids = top_cand_hyp_pos // self.vocab.n_vocabs, \
                                             top_cand_hyp_pos % self.vocab.n_vocabs

                # 2.2.3 For each of the topk words, we append the new word to the current (generating) sentence
                # We add this to new_hypotheses_i and add its corresponding total score to new_hyp_scores_i
                new_hypotheses_i, new_hyp_scores_i = [], []  # Removed live_hyp_ids_i, which is used in the LSTM decoder to track live hypothesis ids
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

                # 2.2.4 We may find that the hypotheses_i for some example in the batch
                # is empty - we have fully processed that example. We use None as a sentinel in this case.
                # Above, the loops gracefully handle None examples.
                if len(new_hypotheses_i) > 0:
                    hypotheses_i = torch.cat(new_hypotheses_i, dim=-1).transpose(0, -1).to(self.device)
                    hyp_scores_i = torch.tensor(new_hyp_scores_i, dtype=torch.float, device=self.device)
                else:
                    hypotheses_i, hyp_scores_i = None, None
                new_hypotheses += [hypotheses_i]
                new_hyp_scores += [hyp_scores_i]
            # print(new_hypotheses, new_hyp_scores)
            hypotheses, hyp_scores = new_hypotheses, new_hyp_scores

        # 2.3 Finally, we do some postprocessing to get our final generated candidate sentences.
        # Sometimes, we may get to max_len of a sentence and still not generate the </s> end token.
        # In this case, the partial sentence we have generated will not be added to the completed_hypotheses
        # automatically, and we have to manually add it in. We add in as many as necessary so that there are
        # `beam_size` completed hypotheses for each example.
        # Finally, we sort each completed hypothesis by score.
        for i in range(batch_size):
            hyps_to_add = beam_size - len(completed_hypotheses[i])
            if hyps_to_add > 0:
                scores, ix = torch.topk(hyp_scores[i], k=hyps_to_add)
                for score, id in zip(scores, ix):
                    completed_hypotheses[i].append(Hypothesis(
                        value=[self.vocab.idx2word[a.item()] for a in hypotheses[i][id][1:]],
                        score=score))
            completed_hypotheses[i].sort(key=lambda hyp: hyp.score, reverse=True)
        # print('completed_hypotheses', completed_hypotheses)
        return r2l_completed_hypotheses, completed_hypotheses




















    def beam_search_decode1(self, src, beam_size, max_len):
        """
                An Implementation of Beam Search for the Transformer Model.
                Beam search is performed in a batched manner. Each example in a batch generates `beam_size` hypotheses.
                We return a list (len: batch_size) of list (len: beam_size) of Hypothesis, which contain our output decoded sentences
                and their scores.
                :param src: shape (sent_len, batch_size). Each val is 0 < val < len(vocab_dec). The input tokens to the decoder.
                :param max_len: the maximum length to decode
                :param beam_size: the beam size to use
                :return completed_hypotheses: A List of length batch_size, each containing a List of beam_size Hypothesis objects.
                    Hypothesis is a named Tuple, its first entry is "value" and is a List of strings which contains the translated word
                    (one string is one word token). The second entry is "score" and it is the log-prob score for this translated sentence.
                Note: Below I note "4 bt", "5 beam_size" as the shapes of objects. 4, 5 are default values. Actual values may differ.
                """
        # 1. Setup
        start_symbol = self.vocab.word2idx['<S>']
        # print(start_symbol)  #1
        end_symbol = self.vocab.word2idx['<S>']
        # print(end_symbol)  #1

        # 1.1 Setup Src
        "src has shape (batch_size, sent_len)"
        "src_mask has shape (batch_size, 1, sent_len)"
        # src_mask = (src[:, :, 0] != self.vocab.word2idx['<PAD>']).unsqueeze(-2)  # TODO Untested
        src_mask = pad_mask(src, r2l_trg=None, trg=None, pad_idx=self.vocab.word2idx['<PAD>'])
        "model_encodings has shape (batch_size, sentence_len, d_model)"
        if self.feature_mode == 'one':
            batch_size = src.shape[0]
            model_encodings = self.encode(src, src_mask)
            r2l_memory, r2l_completed_hypotheses = self.r2l_beam_search_decode1(batch_size, src, src_mask,
                                                                               model_encodings=model_encodings,
                                                                               beam_size=beam_size, max_len=max_len)
        elif self.feature_mode == 'two' or 'three' or 'four':
            # print('src', src.size())
            # print('src0',src[0].size())   # 1 60 1836
            # print('src1', src[1].size())  # 1 60 1324
            # print('src2', src[2].size())  # 1 60 1328
            # print('src3', src[3].size())  # 1 60 600
            # print('src_mask[0]',src_mask[0][0].size())  # 1 1 60
            # print('src_mask[1]', src_mask[0][1].size())  # 1 1 60
            # print('src_mask[2]', src_mask[0][2].size()) # 1 1 60
            # print('src_mask[3]', src_mask[0][3].size()) # 1 1 60
            # print('src_mask[1]',src_mask[1].size()) # 1 1 60

            batch_size = src[0].shape[0]
            enc_src_mask = src_mask[0]
            dec_src_mask = src_mask[1]
            r2l_model_encodings = self.encode(src, enc_src_mask, feature_mode_two=True)
            # print('r2l_model_encodings:',r2l_model_encodings.size()) # 1 60 640
            # model_encodings = r2l_model_encodings
            model_encodings = self.encode(src, enc_src_mask)
            # print('model_encodings:', model_encodings.size()) # 1 60 640
            r2l_memory, r2l_completed_hypotheses = self.r2l_beam_search_decode(batch_size, src, dec_src_mask,
                                                                               model_encodings=r2l_model_encodings,
                                                                               beam_size=beam_size, max_len=max_len)

        # 1.2 Setup r2l target output
        # r2l_memory, r2l_completed_hypotheses = self.r2l_beam_search_decode(batch_size, src, src_mask,
        #                                                                    model_encodings=model_encodings,
        #                                                                    beam_size=1, max_len=max_len)
        # r2l_memory, r2l_completed_hypotheses = self.greedy_decode(batch_size, src_mask, model_encodings, max_len)
        # beam_r2l_memory = [copy.deepcopy(r2l_memory) for _ in range(beam_size)]
        # 1.3 Setup Tgt Hypothesis Tracking
        "hypothesis is List(4 bt)[(cur beam_sz, dec_sent_len)], init: List(4 bt)[(1 init_beam_sz, dec_sent_len)]"
        "hypotheses[i] is shape (cur beam_sz, dec_sent_len)"
        hypotheses = [copy.deepcopy(torch.full((1, 1), start_symbol, dtype=torch.long,
                                               device=self.device)) for _ in range(batch_size)]
        "List after init: List 4 bt of List of len max_len_completed, init: List of len 4 bt of []"
        completed_hypotheses = [copy.deepcopy([]) for _ in range(batch_size)]
        "List len batch_sz of shape (cur beam_sz), init: List(4 bt)[(1 init_beam_sz)]"
        "hyp_scores[i] is shape (cur beam_sz)"
        hyp_scores = [copy.deepcopy(torch.full((1,), 0, dtype=torch.float, device=self.device))
                      for _ in range(batch_size)]  # probs are log_probs must be init at 0.

        # 2. Iterate: Generate one char at a time until maxlen
        for iter in range(max_len + 1):
            if all([len(completed_hypotheses[i]) == beam_size for i in range(batch_size)]):
                break

            # 2.1 Setup the batch. Since we use beam search, each batch has a variable number (called cur_beam_size)
            # between 0 and beam_size of hypotheses live at any moment. We decode all hypotheses for all batches at
            # the same time, so we must copy the src_encodings, src_mask, etc the appropriate number fo times for
            # the number of hypotheses for each example. We keep track of the number of live hypotheses for each example.
            # We run all hypotheses for all examples together through the decoder and log-softmax,
            # and then use `torch.split` to get the appropriate number of hypotheses for each example in the end.
            cur_beam_sizes, last_tokens, model_encodings_l, src_mask_l, r2l_memory_l = [], [], [], [], []
            for i in range(batch_size):
                if hypotheses[i] is None:
                    cur_beam_sizes += [0]
                    continue
                cur_beam_size, decoded_len = hypotheses[i].shape
                cur_beam_sizes += [cur_beam_size]
                last_tokens += [hypotheses[i]]
                model_encodings_l += [model_encodings[i:i + 1]] * cur_beam_size
                if self.feature_mode == 'one':
                    src_mask_l += [src_mask[i:i + 1]] * cur_beam_size
                elif self.feature_mode == 'two' or 'three' or 'four':
                    src_mask_l += [dec_src_mask[i:i + 1]] * cur_beam_size
                r2l_memory_l += [r2l_memory[i: i + 1]] * cur_beam_size
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            model_encodings_cur = torch.cat(model_encodings_l, dim=0)
            src_mask_cur = torch.cat(src_mask_l, dim=0)
            y_tm1 = torch.cat(last_tokens, dim=0)
            r2l_memory_cur = torch.cat(r2l_memory_l, dim=0)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            if self.feature_mode == 'one':
                out = self.l2r_decode(Variable(y_tm1).to(self.device), model_encodings_cur, src_mask_cur,
                                      Variable(subsequent_mask(y_tm1.size(-1)).type_as(src.data)).to(self.device),
                                      r2l_memory_cur, r2l_trg_mask=None)
            elif self.feature_mode == 'two' or 'three' or 'four':
                out = self.l2r_decode(Variable(y_tm1).to(self.device), model_encodings_cur, src_mask_cur,
                                      Variable(subsequent_mask(y_tm1.size(-1)).type_as(src[0].data)).to(self.device),
                                      r2l_memory_cur, r2l_trg_mask=None)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 50002 vocab_sz)"
            log_prob = self.generator(out[:, -1, :]).unsqueeze(1)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 50002 vocab_sz)"
            _, decoded_len, vocab_sz = log_prob.shape
            # log_prob = log_prob.reshape(batch_size, cur_beam_size, decoded_len, vocab_sz)
            "shape List(4 bt)[(cur_beam_sz_i, dec_sent_len, 50002 vocab_sz)]"
            "log_prob[i] is (cur_beam_sz_i, dec_sent_len, 50002 vocab_sz)"
            log_prob = torch.split(log_prob, cur_beam_sizes, dim=0)

            # 2.2 Now we process each example in the batch. Note that the example may have already finished processing before
            # other examples (no more hypotheses to try), in which case we continue
            new_hypotheses, new_hyp_scores = [], []
            for i in range(batch_size):
                if hypotheses[i] is None or len(completed_hypotheses[i]) >= beam_size:
                    new_hypotheses += [None]
                    new_hyp_scores += [None]
                    continue

                # 2.2.1 We compute the cumulative scores for each live hypotheses for the example
                # hyp_scores is the old scores for the previous stage, and `log_prob` are the new probs for
                # this stage. Since they are log probs, we sum them instaed of multiplying them.
                # The .view(-1) forces all the hypotheses into one dimension. The shape of this dimension is
                # cur_beam_sz * vocab_sz (ex: 5 * 50002). So after getting the topk from it, we can recover the
                # generating sentence and the next word using: ix // vocab_sz, ix % vocab_sz.
                cur_beam_sz_i, dec_sent_len, vocab_sz = log_prob[i].shape
                "shape (vocab_sz,)"
                cumulative_hyp_scores_i = (hyp_scores[i].unsqueeze(-1).unsqueeze(-1)
                                           .expand((cur_beam_sz_i, 1, vocab_sz)) + log_prob[i]).view(-1)

                # 2.2.2 We get the topk values in cumulative_hyp_scores_i and compute the current (generating) sentence
                # and the next word using: ix // vocab_sz, ix % vocab_sz.
                "shape (cur_beam_sz,)"
                live_hyp_num_i = beam_size - len(completed_hypotheses[i])
                "shape (cur_beam_sz,). Vals are between 0 and 50002 vocab_sz"
                top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(cumulative_hyp_scores_i, k=live_hyp_num_i)
                "shape (cur_beam_sz,). prev_hyp_ids vals are 0 <= val < cur_beam_sz. hyp_word_ids vals are 0 <= val < vocab_len"
                prev_hyp_ids, hyp_word_ids = top_cand_hyp_pos // self.vocab.n_vocabs, \
                                             top_cand_hyp_pos % self.vocab.n_vocabs

                # 2.2.3 For each of the topk words, we append the new word to the current (generating) sentence
                # We add this to new_hypotheses_i and add its corresponding total score to new_hyp_scores_i
                new_hypotheses_i, new_hyp_scores_i = [], []  # Removed live_hyp_ids_i, which is used in the LSTM decoder to track live hypothesis ids
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

                # 2.2.4 We may find that the hypotheses_i for some example in the batch
                # is empty - we have fully processed that example. We use None as a sentinel in this case.
                # Above, the loops gracefully handle None examples.
                if len(new_hypotheses_i) > 0:
                    hypotheses_i = torch.cat(new_hypotheses_i, dim=-1).transpose(0, -1).to(self.device)
                    hyp_scores_i = torch.tensor(new_hyp_scores_i, dtype=torch.float, device=self.device)
                else:
                    hypotheses_i, hyp_scores_i = None, None
                new_hypotheses += [hypotheses_i]
                new_hyp_scores += [hyp_scores_i]
            # print(new_hypotheses, new_hyp_scores)
            hypotheses, hyp_scores = new_hypotheses, new_hyp_scores

        # 2.3 Finally, we do some postprocessing to get our final generated candidate sentences.
        # Sometimes, we may get to max_len of a sentence and still not generate the </s> end token.
        # In this case, the partial sentence we have generated will not be added to the completed_hypotheses
        # automatically, and we have to manually add it in. We add in as many as necessary so that there are
        # `beam_size` completed hypotheses for each example.
        # Finally, we sort each completed hypothesis by score.
        for i in range(batch_size):
            hyps_to_add = beam_size - len(completed_hypotheses[i])
            if hyps_to_add > 0:
                scores, ix = torch.topk(hyp_scores[i], k=hyps_to_add)
                for score, id in zip(scores, ix):
                    completed_hypotheses[i].append(Hypothesis(
                        value=[self.vocab.idx2word[a.item()] for a in hypotheses[i][id][1:]],
                        score=score))
            completed_hypotheses[i].sort(key=lambda hyp: hyp.score, reverse=True)
        # print('completed_hypotheses', completed_hypotheses)
        return r2l_completed_hypotheses, completed_hypotheses
















# inish the model load in CUDA. Try to enter Test Set.
# build onlyonce_iter::   0%|          | 0/59800 [00:00<?, ?it/s]
# 100%|██████████| 59800/59800 [01:37<00:00, 611.06it/s]
#   0%|          | 0/2990 [00:00<?, ?it/s][[Hypothesis(value=['park', 'a', 'in', 'crying', 'is', 'man', 'a', 'is', 'there'], score=-0.10984330624341965), Hypothesis(value=['park', 'a', 'of', 'front', 'in', 'stadium', 'a', 'near', 'guitar', 'plays'], score=-0.3920365571975708), Hypothesis(value=['park', 'a', 'in', 'basketball', 'playing', 'man', 'a'], score=-0.446031391620636), Hypothesis(value=['park', 'a', 'of', 'front', 'in', 'stadium', 'a', 'near', 'guitar', 'a'], score=-0.8148292899131775), Hypothesis(value=['park', 'a', 'in', 'basketball', 'playing', 'man'], score=-1.8244637250900269)]]
# completed_hypotheses [[Hypothesis(value=['there', 'is', 'a', 'person', 'is', 'walking', 'outside', 'and', 'filming', 'space'], score=-0.4037010967731476), Hypothesis(value=['there', 'is', 'a', 'person', 'is', 'walking', 'outside', 'and', 'filming', 'a'], score=-0.5641604065895081), Hypothesis(value=['there', 'is', 'a', 'person', 'is', 'playing', 'the', 'golf', 'creation'], score=-0.8466820120811462), Hypothesis(value=['there', 'is', 'a', 'person', 'is', 'playing', 'the', 'golf'], score=-2.1718363761901855), Hypothesis(value=['there', 'is', 'a', 'person', 'is', 'walking', 'outside', 'and', 'filming', 'space', 'cups'], score=tensor(-9.8324, device='cuda:0'))]]
#   0%|          | 1/2990 [00:00<21:53,  2.27it/s][[Hypothesis(value=['beatboxing', 'and', 'songs', 'singing', 'guy', 'a'], score=-0.15891645848751068), Hypothesis(value=['a', 'for', 'dancing', 'person', 'a'], score=-0.1841302514076233), Hypothesis(value=['bubbles', 'with', 'video', 'a', 'from', 'footage', 'then', 'and', 'flips'], score=-0.48116645216941833), Hypothesis(value=['bubbles', 'with', 'video', 'a', 'from', 'footage', 'then', 'and', 'dances'], score=-0.8349403142929077), Hypothesis(value=['bubbles', 'with', 'video', 'a', 'from', 'sync', 'a'], score=-2.3781938552856445)]]
#   0%|          | 2/2990 [00:00<18:16,  2.73it/s]completed_hypotheses [[Hypothesis(value=['man', 'dancing', 'while', 'playing', 'music', 'a', 'song', 'in', 'the', 'room'], score=-0.11141674965620041), Hypothesis(value=['guy', 'watching', 'a', 'tv', 'show'], score=-0.25194987654685974), Hypothesis(value=['guy', 'watching', 'a', 'bunch', 'of', 'a', 'webcam'], score=-0.34744420647621155), Hypothesis(value=['man', 'dancing', 'while', 'playing', 'music', 'a', 'video'], score=-0.4307742416858673), Hypothesis(value=['guy', 'watching', 'a', 'bunch', 'of', 'tricks'], score=-0.616431474685669)]]
# [[Hypothesis(value=['wrestling', 'are', 'men', 'two'], score=-0.105088010430336), Hypothesis(value=['played', 'being', 'is', 'match', 'wrestling', 'a'], score=-0.14491412043571472), Hypothesis(value=['played', 'being', 'is', 'match', 'wrestling', 'a', 'of', 'footage'], score=-0.19306915998458862), Hypothesis(value=['matches', 'wrestling', 'about', 'video', 'a'], score=-0.5089492797851562), Hypothesis(value=['wrestling', 'are', 'people'], score=-0.5977432727813721)]]
# completed_hypotheses [[Hypothesis(value=['man', 'asks', 'something', 'about', 'wrestling', 'matches'], score=-0.20213964581489563), Hypothesis(value=['man', 'asks', 'something', 'to', 'an', 'audience', 's', 'kombat'], score=-0.35753583908081055), Hypothesis(value=['man', 'asks', 'something', 'to', 'an', 'audience'], score=-0.5122723579406738), Hypothesis(value=['man', 'asks', 'something', 'about', 'wrestling'], score=-1.3053356409072876), Hypothesis(value=['the', 'man', 'paces'], score=-3.110365867614746)]]
#   0%|          | 3/2990 [00:00<15:13,  3.27it/s][[Hypothesis(value=['head', 'their', 'painting', 'is', 'woman', 'a'], score=-0.1366117000579834), Hypothesis(value=['head', 'their', 'painting', 'is', 'character', 's', 'child', 'young', 'a'], score=-0.16792869567871094), Hypothesis(value=['head', 'their', 'painting', 'is', 'character', 's', 'child', 'a'], score=-0.20381493866443634), Hypothesis(value=['paint', 'in', 'acting', 'kid', 'a'], score=-0.21076416969299316), Hypothesis(value=['head', 'their', 'painting', 'woman', 'a'], score=-0.7210447788238525)]]
#   0%|          | 4/2990 [00:01<13:44,  3.62it/s]completed_hypotheses [[Hypothesis(value=['a', 'girl', 'wearing', 'ears', 'hair', 'and', 'a', 'cloth', 'that'], score=-0.8330574035644531), Hypothesis(value=['a', 'girl', 'wearing', 'ears', 'hair', 'and', 'a', 'cloth', 'that', 'it'], score=-0.8352349996566772), Hypothesis(value=['a', 'girl', 'is', 'doing', 'something'], score=-0.916714072227478), Hypothesis(value=['a', 'girl', 'wearing', 'ears', 'hair', 'and', 'a', 'painting'], score=-1.115577220916748), Hypothesis(value=['a', 'girl', 'wearing', 'a', 'helmet', 'holds'], score=-4.85388708114624)]]
# [[Hypothesis(value=['together', 'guitar', 'a', 'playing', 'band', 'a'], score=-0.2944316565990448), Hypothesis(value=['together', 'guitar', 'a', 'playing', 'are', 'people'], score=-0.4885163903236389), Hypothesis(value=['together', 'guitar', 'the', 'playing', 'are', 'people'], score=-0.49148261547088623), Hypothesis(value=['together', 'guitar', 'a', 'playing', 'are', 'people', 'of', 'group', 'a', 'and'], score=-1.3485819101333618), Hypothesis(value=['exploding', 'beach', 'a', 'on', 'people'], score=-1.5606945753097534)]]
#   0%|          | 5/2990 [00:01<12:46,  3.90it/s]completed_hypotheses [[Hypothesis(value=['people', 'around', 'baseket', 'theme', 'instruments'], score=-0.5176293253898621), Hypothesis(value=['a', 'teenage', 'group', 'of', 'people', 'singing', 'and', 'dancing'], score=-0.687189519405365), Hypothesis(value=['a', 'teenage', 'group', 'of', 'people', 'singing', 'and', 'dancing', 'as', 'they'], score=-1.5120649337768555), Hypothesis(value=['a', 'teenage', 'group', 'of', 'people', 'singing', 'and', 'dancing', 'as', 'others', 'playing'], score=tensor(-3.3522, device='cuda:0')), Hypothesis(value=['a', 'teenage', 'group', 'of', 'people', 'singing', 'and', 'dancing', 'as', 'others', 'have'], score=tensor(-6.3242, device='cuda:0'))]]
# [[Hypothesis(value=['furniture', 'their', 'in', 'features', 'various', 'doing', 'is', 'man', 'a'], score=-0.1260627955198288), Hypothesis(value=['furniture', 'their', 'in', 'features', 'various', 'about', 'speaking', 'is', 'woman', 'a'], score=-0.1263742297887802), Hypothesis(value=['furniture', 'their', 'on', 'it', 'discussing', 'is', 'cat', 'a'], score=-0.137444406747818), Hypothesis(value=['furniture', 'their', 'in', 'features', 'various', 'about', 'talking', 'is', 'woman', 'a'], score=-0.14648763835430145), Hypothesis(value=['furniture', 'their', 'in', 'features', 'various', 'about', 'talking', 'is', 'woman'], score=-1.8389613628387451)]]
#   0%|          | 6/2990 [00:01<11:56,  4.16it/s]completed_hypotheses [[Hypothesis(value=['there', 'are', 'here', 'video', 'of', 'friends', 'walking', 'on', 'the', 'hallway'], score=-0.2514185309410095), Hypothesis(value=['there', 'are', 'here', 'video', 'of', 'friends', 'walking', 'on', 'the', 'jeans'], score=-0.2575567960739136), Hypothesis(value=['a', 'woman', 'walking', 'around', 'a', 'bed'], score=-1.05106782913208), Hypothesis(value=['a', 'woman', 'walks', 'on', 'a', 'gives', 'full', 'of', 'lots', 'of', 'shoes'], score=tensor(-1.4687, device='cuda:0')), Hypothesis(value=['there', 'are', 'here', 'video', 'of', 'friends', 'walking', 'on', 'the', 'jeans', 'with'], score=tensor(-4.9144, device='cuda:0'))]]
# [[Hypothesis(value=['terrorism', 'of', 'country', 'a', 'shows', 'video', 'a'], score=-0.11995705962181091), Hypothesis(value=['news', 'the', 'discusses', 'newscaster', 'a'], score=-0.1307632327079773), Hypothesis(value=['terrorism', 'of', 'country', 'a', 'shows', 'someone'], score=-0.14084479212760925), Hypothesis(value=['terrorism', 'of', 'country', 'a', 'shows', 'video'], score=-2.338298797607422), Hypothesis(value=['news', 'the', 'discusses', 'reporter', 'news'], score=-2.4015603065490723)]]
# completed_hypotheses [[Hypothesis(value=['the', 'news', 'shows', 'two', 'people', 'in', 'which', 'are', 'bbc'], score=-0.6144518852233887), Hypothesis(value=['the', 'news', 'shows', 'two', 'these', 'text', 'reporting'], score=-1.5745999813079834), Hypothesis(value=['a', 'man', 'asks', 'news'], score=-2.4476864337921143), Hypothesis(value=['the', 'news', 'shows', 'two', 'these', 'text', 'reporting', 'that', 'some', 'innovation', 'place'], score=tensor(-7.2850, device='cuda:0')), Hypothesis(value=['the', 'news', 'shows', 'two', 'people', 'are', 'laughing', 'and', 'seeing', 'infowars', 'personalities'], score=tensor(-9.6573, device='cuda:0'))]]
#   0%|          | 7/2990 [00:01<11:01,  4.51it/s][[Hypothesis(value=['game', 'football', 'a', 'on', 'talked', 'being', 'man', 'a'], score=-0.1794201135635376), Hypothesis(value=['players', 'football', 'different', 'of', 'picture', 'a'], score=-0.641808807849884), Hypothesis(value=['game', 'football', 'a', 'on', 'commentating', 'man'], score=-1.5006954669952393), Hypothesis(value=['players', 'football', 'different', 'of', 'picture', 'a', 'with', 'himself', 'to', 'talking', 'man'], score=tensor(-1.5201, device='cuda:0')), Hypothesis(value=['game', 'football', 'a', 'on', 'talked', 'being', 'man'], score=-1.5605648756027222)]]
#   0%|          | 8/2990 [00:01<10:27,  4.75it/s]completed_hypotheses [[Hypothesis(value=['people', 'are', 'playing', 'a', 'soccer', 'game'], score=-0.09148353338241577), Hypothesis(value=['people', 'are', 'playing', 'a', 'soccer', 'competition'], score=-0.09952165186405182), Hypothesis(value=['people', 'are', 'playing', 'sports'], score=-0.1130099669098854), Hypothesis(value=['people', 'are', 'playing', 'a', 'video', 'game'], score=-0.11460267007350922), Hypothesis(value=['man', 'describes', 'his', 'worst', 'team'], score=-0.5327538847923279)]]
# [[Hypothesis(value=['hills', 'the', 'through', 'drives', 'car', 'a', 'driving', 'is', 'person', 'a'], score=-0.10141713917255402), Hypothesis(value=['hills', 'the', 'through', 'drives', 'car', 'the', 'in', 'man', 'a'], score=-0.16216497123241425), Hypothesis(value=['hills', 'the', 'on', 'drives', 'he', 'as', 'commentary', 'and', 'service'], score=-0.4817068874835968), Hypothesis(value=['hills', 'the', 'on', 'drives', 'he', 'as', 'commentary', 'and', 'road', 'the', 'down'], score=tensor(-1.2729, device='cuda:0')), Hypothesis(value=['hills', 'the', 'on', 'drives', 'he', 'as'], score=-5.400332450866699)]]
#   0%|          | 9/2990 [00:02<10:38,  4.67it/s]completed_hypotheses [[Hypothesis(value=['many', 'men', 'are', 'racing', 'the', 'car', 'with', 'jeanclaude', 'in', 'the'], score=-0.19049349427223206), Hypothesis(value=['guy', 'is', 'riding', 'the', 'car', 'in', 'the', 'car', 'center'], score=-0.2277527004480362), Hypothesis(value=['guy', 'is', 'riding', 'the', 'car', 'in', 'the', 'woods'], score=-0.3862661123275757), Hypothesis(value=['guy', 'is', 'riding', 'the', 'car', 'in', 'the', 'woods', 'and', 'playing'], score=-0.4720209240913391), Hypothesis(value=['many', 'men', 'are', 'racing', 'a', 'car'], score=-1.260828971862793)]]
# [[Hypothesis(value=['challenge', 'mirror', 'the', 'discussing', 'is', 'woman', 'a'], score=-0.1173776388168335), Hypothesis(value=['challenge', 'mirror', 'a', 'with', 'up', 'going', 'is', 'woman', 'a'], score=-0.14700284600257874), Hypothesis(value=['challenge', 'mirror', 'the', 'discussing', 'girl', 'a'], score=-0.2099861204624176), Hypothesis(value=['challenge', 'mirror', 'a', 'with', 'makeup', 'different', 'up', 'putting', 'is', 'woman', 'a'], score=tensor(-0.4814, device='cuda:0')), Hypothesis(value=['challenge', 'mirror', 'a', 'with', 'makeup', 'different', 'up', 'putting', 'is', 'woman', 'young'], score=tensor(-3.1909, device='cuda:0'))]]
# completed_hypotheses [[Hypothesis(value=['beauty', 'makeup', 'video', 'of', 'a', 'woman', 'in', 'video', 'of', 'a'], score=-0.31434598565101624), Hypothesis(value=['there', 'is', 'a', 'woman', 'is', 'explaining', 'about', 'halloween', 'scarf'], score=-0.4247782826423645), Hypothesis(value=['beauty', 'makeup', 'video', 'of', 'a', 'woman', 'in', 'video', 'while'], score=-1.720238208770752), Hypothesis(value=['beauty', 'makeup', 'video', 'of', 'a', 'woman'], score=-3.120375156402588), Hypothesis(value=['beauty', 'makeup', 'video', 'of', 'a', 'woman', 'in', 'video', 'of', 'a', 'kitchen'], score=tensor(-8.0527, device='cuda:0'))]]
#   0%|          | 10/2990 [00:02<10:24,  4.77it/s][[Hypothesis(value=['shouting', 'are', 'characters', 'cartoon', 'two'], score=-0.10130414366722107), Hypothesis(value=['supermarket', 'a', 'around', 'walking', 'are', 'animals', 'cartoon', 'two'], score=-0.12255831807851791), Hypothesis(value=['supermarket', 'a', 'around', 'walking', 'is', 'character', 'cartoon', 'a'], score=-0.13509775698184967), Hypothesis(value=['shouting', 'are', 'characters', 'cartoon'], score=-0.5471217036247253), Hypothesis(value=['supermarket', 'a', 'around', 'walking', 'are', 'animals', 'cartoon'], score=-0.7177591919898987)]]
#   0%|          | 11/2990 [00:02<10:07,  4.90it/s]completed_hypotheses [[Hypothesis(value=['disney', 'characters', 'are', 'serving', 'food'], score=-0.5758137702941895), Hypothesis(value=['a', 'cartoon', 'cutouts', 'just', 'a', 'cartoon', 'clip'], score=-0.6554723978042603), Hypothesis(value=['a', 'cartoon', 'cutouts', 'just', 'a', 'hot', 'dog', 'trip', 'by', 'getting'], score=-1.2517539262771606), Hypothesis(value=['a', 'cartoon', 'cutouts', 'just', 'a', 'box'], score=-1.3906673192977905), Hypothesis(value=['a', 'cartoon', 'cutouts', 'just', 'a', 'hot', 'dog', 'trip'], score=-1.4168572425842285)]]
# [[Hypothesis(value=['it', 'on', 'working', 'while', 'pot', 'a', 'in', 'food', 'cooking', 'is'], score=-0.5575807094573975), Hypothesis(value=['it', 'on', 'working', 'while', 'pot', 'a', 'in', 'food', 'cooking', 'woman'], score=-0.6600722074508667), Hypothesis(value=['it', 'on', 'working', 'while', 'pot', 'a', 'in', 'food', 'cooks', 'woman', 'a'], score=tensor(-0.7326, device='cuda:0')), Hypothesis(value=['it', 'on', 'items', 'potatoes', 'adding', 'and', 'pot', 'a', 'in'], score=-2.4017226696014404), Hypothesis(value=['it', 'on', 'items', 'potatoes', 'adding', 'and', 'pot', 'a', 'riding'], score=-2.634864568710327)]]
# completed_hypotheses [[Hypothesis(value=['there', 'is', 'a', 'woman', 'in', 'a', 'tank', 'is', 'making', 'chicken'], score=-0.3826378285884857), Hypothesis(value=['there', 'is', 'a', 'woman', 'in', 'a', 'tank', 'is', 'cooking', 'something'], score=-0.57960045337677), Hypothesis(value=['there', 'is', 'a', 'man', 'is', 'cooking', 'egg', 'and', 'browns', 'in', 'a'], score=tensor(-1.4076, device='cuda:0')), Hypothesis(value=['there', 'is', 'a', 'man', 'is', 'cooking', 'egg', 'and', 'browns', 'in', 'the'], score=tensor(-1.5205, device='cuda:0')), Hypothesis(value=['two', 'women', 'cooking', 'a', 'video', 'render', 'of', 'something', 'on', 'tv', 'outside'], score=tensor(-6.9939, device='cuda:0'))]]
#   0%|          | 12/2990 [00:02<10:18,  4.81it/s][[Hypothesis(value=['makeup', 'showing', 'person', 'a'], score=-0.10415434092283249), Hypothesis(value=['liquid', 'from', 'someone', 'mask', 'a', 'putting', 'woman', 'a'], score=-0.11486265063285828), Hypothesis(value=['liquid', 'from', 'someone', 'mask', 'a', 'putting', 'man', 'a'], score=-0.12116972357034683), Hypothesis(value=['it', 'puts', 'woman', 'a'], score=-0.19698958098888397), Hypothesis(value=['liquid', 'from', 'someone', 'mask', 'a', 'putting', 'man'], score=-2.3683412075042725)]]
#   0%|          | 13/2990 [00:02<10:05,  4.92it/s]completed_hypotheses [[Hypothesis(value=['man', 'is', 'putting', 'sort', 'of', 'flour', 'in', 'a', 'glass', 'log'], score=-0.2893237769603729), Hypothesis(value=['man', 'showing', 'how', 'to', 'make', 'beauty', 'face'], score=-0.3205130100250244), Hypothesis(value=['man', 'showing', 'how', 'to', 'make', 'beauty', 'pipe'], score=-0.43298041820526123), Hypothesis(value=['man', 'showing', 'how', 'to', 'make', 'a', 'cup'], score=-0.5580348968505859), Hypothesis(value=['person', 'cleaning', 'bottles', 'icing'], score=-0.9279813766479492)]]
# [[Hypothesis(value=['blog', 'mine', 'a', 'about', 'talking', 'is', 'woman', 'a'], score=-0.33166736364364624), Hypothesis(value=['screen', 'a', 'on', 'options', 'some', 'man', 'a', 'while', 'tv', 'on'], score=-0.38354629278182983), Hypothesis(value=['woman', 'the', 'to', 'talking', 'man', 'a'], score=-0.48866286873817444), Hypothesis(value=['screen', 'a', 'on', 'options', 'some', 'man', 'a', 'to', 'speaking', 'woman', 'a'], score=tensor(-1.1376, device='cuda:0')), Hypothesis(value=['screen', 'a', 'on', 'options', 'some', 'man', 'a'], score=-1.5594065189361572)]]
# completed_hypotheses [[Hypothesis(value=['the', 'man', 'talking', 'to', 'a', 'woman', 'about', 'his', 'name', 'by'], score=-0.39500826597213745), Hypothesis(value=['the', 'man', 'talking', 'to', 'a', 'woman', 'about', 'something', 'on', 'the'], score=-0.657130241394043), Hypothesis(value=['the', 'man', 'talking', 'to', 'a', 'woman', 'about', 'his', 'name'], score=-0.6750906705856323), Hypothesis(value=['the', 'man', 'talking', 'to', 'a', 'woman', 'about', 'something'], score=-1.1701793670654297), Hypothesis(value=['a', 'woman', 'and', 'woman', 'are', 'talking', 'with', 'each', 'other', 'in', 'a'], score=tensor(-1.1719, device='cuda:0'))]]
#   0%|          | 14/2990 [00:03<10:17,  4.82it/s][[Hypothesis(value=['products', 'some', 'about', 'talking', 'lady', 'a'], score=-0.20954248309135437), Hypothesis(value=['products', 'some', 'about', 'talking', 'is', 'man', 'a'], score=-0.22086693346500397), Hypothesis(value=['phone', 'cell', 'a', 'to', 'things', 'cut', 'to', 'trying', 'are', 'people'], score=-0.3457977771759033), Hypothesis(value=['phone', 'cell', 'a', 'to', 'things', 'cut', 'to', 'trying', 'are', 'woman', 'a'], score=tensor(-0.8191, device='cuda:0')), Hypothesis(value=['phone', 'cell', 'a', 'to', 'things', 'cut', 'can', 'that'], score=-2.694697856903076)]]
# completed_hypotheses [[Hypothesis(value=['still', 'photos', 'of', 'a', 'woman', 'looking', 'at', 'a', 'table'], score=-0.4841245412826538), Hypothesis(value=['a', 'girls', 'restaurants', 'and', 'it', 'the', 'chip', 'grading', 'a'], score=-1.2208800315856934), Hypothesis(value=['a', 'girls', 'restaurants', 'and', 'it', 'the', 'chip'], score=-1.7927132844924927), Hypothesis(value=['a', 'girls', 'restaurants', 'and', 'it', 'the', 'need', 'an', 'blowing', 'a', 'book'], score=tensor(-2.7194, device='cuda:0')), Hypothesis(value=['a', 'girls', 'restaurants', 'and', 'it', 'the', 'need', 'an', 'blowing', 'a', 'in'], score=tensor(-6.6114, device='cuda:0'))]]
#   1%|          | 15/2990 [00:03<10:10,  4.87it/s][[Hypothesis(value=['giving', 'is', 'man', 'a'], score=-0.15432985126972198), Hypothesis(value=['speech', 'a', 'giving', 'is', 'clinton', 'hillary', 'of', 'crowd', 'a'], score=-0.19272364675998688), Hypothesis(value=['around', 'jumping', 'crowd', 'a'], score=-0.6018294095993042), Hypothesis(value=['speech', 'a', 'giving', 'is', 'clinton', 'hillary', 'of', 'front', 'in'], score=-0.9106165170669556), Hypothesis(value=['speech', 'a', 'giving', 'is'], score=-6.544086456298828)]]
# completed_hypotheses [[Hypothesis(value=['hillary', 'clinton', 'is', 'bush', 'on', 'stage', 'watching', 'a', 'crowd'], score=-0.28226152062416077), Hypothesis(value=['hillary', 'clinton', 'is', 'bush', 'on', 'stage', 'watching', 'a', 'desk'], score=-0.3441372215747833), Hypothesis(value=['hillary', 'clinton', 'is', 'bush', 'on', 'stage', 'watching', 'a', 'tv'], score=-0.5092308521270752), Hypothesis(value=['hillary', 'clinton', 'gives', 'speech'], score=-1.5285980701446533), Hypothesis(value=['men', 'are', 'coats', 'on', 'stage', 'standing'], score=-2.45729398727417)]]
#   1%|          | 16/2990 [00:03<09:49,  5.04it/s][[Hypothesis(value=['mall', 'the', 'at', 'together', 'talk', 'people', 'young', 'two'], score=-0.14877662062644958), Hypothesis(value=['mall', 'the', 'at', 'are', 'women', 'some'], score=-0.2011871039867401), Hypothesis(value=['mall', 'the', 'at', 'are', 'women', 'and', 'girl', 'a', 'and', 'man', 'a'], score=tensor(-0.5016, device='cuda:0')), Hypothesis(value=['mall', 'the', 'at', 'together', 'talk', 'people', 'young'], score=-0.8840498924255371), Hypothesis(value=['mall', 'the', 'at', 'are', 'women', 'and', 'girl', 'a', 'and', 'men', 'two'], score=tensor(-1.7157, device='cuda:0'))]]
# completed_hypotheses [[Hypothesis(value=['some', 'people', 'are', 'are', 'running', 'in', 'the', 'kitchen'], score=-0.29517069458961487), Hypothesis(value=['some', 'people', 'are', 'are', 'running', 'next', 'to', 'each', 'other'], score=-0.4937620460987091), Hypothesis(value=['some', 'people', 'are', 'are', 'running', 'next', 'to', 'a', 'woman', 'in', 'a'], score=tensor(-0.5411, device='cuda:0')), Hypothesis(value=['some', 'people', 'are', 'talking'], score=-1.1457351446151733), Hypothesis(value=['a', 'movie', 'depicting', 'daughter'], score=-2.542421579360962)]]
#   1%|          | 17/2990 [00:03<10:08,  4.88it/s][[Hypothesis(value=['aisle', 'large', 'a', 'in', 'up', 'walking', 'are', 'models', 'some'], score=-0.12994055449962616), Hypothesis(value=['aisle', 'large', 'a', 'in', 'up', 'walking', 'are', 'models', 'few', 'a'], score=-0.22433674335479736), Hypothesis(value=['catwalk', 'a', 'down', 'walks', 'lady', 'a', 'as', 'stage', 'on', 'walking'], score=-0.6836610436439514), Hypothesis(value=['catwalk', 'a', 'down', 'walks', 'lady', 'a', 'as', 'stage', 'the', 'down', 'walking'], score=tensor(-0.9067, device='cuda:0')), Hypothesis(value=['york', 'red', 'a', 'in', 'stage', 'a', 'down', 'walks', 'woman'], score=-0.9079339504241943)]]
# completed_hypotheses [[Hypothesis(value=['as', 'a', 'women', 'are', 'walking', 'in', 'the', 'black', 'language', 'and'], score=-0.756597101688385), Hypothesis(value=['as', 'a', 'women', 'are', 'walking', 'down', 'the', 'runway'], score=-1.0752569437026978), Hypothesis(value=['as', 'a', 'women', 'are', 'walking', 'in', 'the', 'black', 'language', 'and', 'white'], score=tensor(-4.7607, device='cuda:0')), Hypothesis(value=['as', 'a', 'women', 'are', 'walking', 'in', 'the', 'black', 'shirt', 'walks', 'front'], score=tensor(-7.6563, device='cuda:0')), Hypothesis(value=['as', 'a', 'women', 'are', 'walking', 'in', 'the', 'black', 'language', 'and', 'covering'], score=tensor(-9.8219, device='cuda:0'))]]
#   1%|          | 18/2990 [00:03<10:11,  4.86it/s][[Hypothesis(value=['background', 'the', 'in', 'driving', 'car', 'a', 'is', 'there'], score=-0.09192206710577011), Hypothesis(value=['background', 'the', 'in', 'driving', 'is', 'person', 'a'], score=-0.14731092751026154), Hypothesis(value=['background', 'the', 'in', 'driving', 'car', 'a', 'is', 'it'], score=-0.17000655829906464), Hypothesis(value=['sailing', 'is', 'snow', 'the', 'of', 'view', 'the'], score=-0.2117520272731781), Hypothesis(value=['sailing', 'is', 'snow', 'the', 'of', 'footage'], score=-0.43443763256073)]]
#   1%|          | 19/2990 [00:04<10:13,  4.84it/s]completed_hypotheses [[Hypothesis(value=['desert', 'locations', '10', 'through', 'an', 'region', 'a', 'highway'], score=-0.3827904462814331), Hypothesis(value=['desert', 'locations', '10', 'through', 'an', 'region', 'and', 'the', 'brown', 'terrain'], score=-0.41329944133758545), Hypothesis(value=['a', 'vehicle', 'falls', 'down', 'the', 'road'], score=-0.6103153824806213), Hypothesis(value=['desert', 'locations', '10', 'through', 'an', 'region', 'and', 'the', 'to', 'see', 'the'], score=tensor(-2.1433, device='cuda:0')), Hypothesis(value=['desert', 'locations', '10', 'through', 'an', 'region', 'and', 'the', 'brown'], score=-2.277238130569458)]]
# [[Hypothesis(value=['bars', 'of', 'front', 'in', 'chinchilla', 'a', 'showing', 'is', 'woman', 'a'], score=-0.12342997640371323), Hypothesis(value=['dinner', 'speaking', 'is', 'woman', 'a'], score=-0.14728128910064697), Hypothesis(value=['bars', 'of', 'front', 'in', 'something', 'about', 'talking', 'lady', 'a'], score=-0.16721269488334656), Hypothesis(value=['bars', 'of', 'front', 'in', 'chinchilla', 'a', 'about', 'talking', 'woman'], score=-0.7845392227172852), Hypothesis(value=['bars', 'of', 'front', 'in', 'chinchilla', 'a', 'showing', 'is', 'woman'], score=-0.8923512697219849)]]
# completed_hypotheses [[Hypothesis(value=['woman', 'is', 'giving', 'interview', 'for', 'an', 'exercise', 'hall'], score=-0.19730965793132782), Hypothesis(value=['woman', 'is', 'giving', 'interview', 'for', 'her', 'voices'], score=-0.24995210766792297), Hypothesis(value=['woman', 'is', 'giving', 'interview', 'for', 'different', 'things'], score=-0.300008088350296), Hypothesis(value=['there', 'is', 'a', 'middle', 'of', 'a', 'woman', 'is', 'talking', 'to', 'a'], score=tensor(-1.4416, device='cuda:0')), Hypothesis(value=['there', 'is', 'a', 'middle', 'of', 'a', 'woman', 'is', 'talking', 'to', 'each'], score=tensor(-3.2393, device='cuda:0'))]]
#   1%|          | 20/2990 [00:04<10:23,  4.77it/s][[Hypothesis(value=['cartoons', 'at', 'talking', 'is', 'woman', 'a'], score=-0.2524891197681427), Hypothesis(value=['him', 'to', 'next', 'talking', 'men', 'cartoon', 'two', 'with', 'show', 'tv', 'a'], score=tensor(-0.6983, device='cuda:0')), Hypothesis(value=['him', 'to', 'next', 'talking', 'men', 'cartoon', 'two', 'with', 'show', 'television', 'cartoon'], score=tensor(-1.1855, device='cuda:0')), Hypothesis(value=['cartoons', 'of', 'cartoon'], score=-2.825117588043213), Hypothesis(value=['cartoons', 'about'], score=-7.110711097717285)]]
# completed_hypotheses [[Hypothesis(value=['the', 'kings', 'clip', 'depicting', 'meryl', 'cartoon', 'characters'], score=-0.34028127789497375), Hypothesis(value=['the', 'kings', 'clip', 'depicting', 'songs'], score=-0.4019407331943512), Hypothesis(value=['it', 'is', 'an', 'animated', 'video'], score=-0.7021492719650269), Hypothesis(value=['the', 'man', 'is', 'sitting', 'at', 'his', 'lunch', 'and', 'talking'], score=-0.8274210095405579), Hypothesis(value=['the', 'man', 'is', 'sitting', 'at', 'his', 'lunch'], score=-0.9707951545715332)]]
#   1%|          | 21/2990 [00:04<10:26,  4.74it/s][[Hypothesis(value=['computer', 'a', 'on', 'learning', 'is', 'man', 'a'], score=-0.137648344039917), Hypothesis(value=['education', 'at', 'studying', 'is', 'classroom', 'the'], score=-1.2679890394210815), Hypothesis(value=['education', 'at', 'studying', 'is', 'classroom', 'his', 'in', 'man'], score=-2.4246737957000732), Hypothesis(value=['education', 'at', 'studying', 'is', 'classroom', 'his'], score=-2.5059826374053955), Hypothesis(value=['education', 'at'], score=-6.9079132080078125)]]
# completed_hypotheses [[Hypothesis(value=['people', 'are', 'sitting', 'in', 'a', 'computer', 'that', 'teasers', 'have', 'pictures'], score=-0.25400665402412415), Hypothesis(value=['people', 'are', 'sitting', 'in', 'a', 'computer', 'show'], score=-0.4906710088253021), Hypothesis(value=['a', 'teacher', 'is', 'sitting', 'at', 'office'], score=-0.9952025413513184), Hypothesis(value=['people', 'are', 'sitting', 'in', 'a', 'computer', 'that', 'teasers'], score=-1.824989676475525), Hypothesis(value=['people', 'are', 'sitting', 'in', 'a', 'computer', 'that', 'teasers', 'are'], score=-1.8925540447235107)]]
#   1%|          | 22/2990 [00:04<10:19,  4.79it/s][[Hypothesis(value=['potatoes', 'boil', 'to', 'how', 'showing', 'is', 'lady', 'a'], score=-0.169455885887146), Hypothesis(value=['potatoes', 'boil', 'to', 'how', 'showing', 'is', 'girl', 'a'], score=-0.17403599619865417), Hypothesis(value=['potatoes', 'garlic', 'like', 'put', 'not', 'be', 'to', 'spices', 'adding', 'is'], score=-1.0910091400146484), Hypothesis(value=['potatoes', 'garlic', 'like', 'put', 'not', 'be', 'to', 'how', 'of', 'show'], score=-1.1071360111236572), Hypothesis(value=['potatoes', 'garlic', 'like', 'put', 'not', 'be', 'to', 'how', 'of', 'show', 'cooking'], score=tensor(-1.2336, device='cuda:0'))]]
#   1%|          | 23/2990 [00:04<10:22,  4.77it/s]completed_hypotheses [[Hypothesis(value=['how', 'to', 'cook', 'pasta', 'in', 'a', 'pot', 'with', 'tanks'], score=-0.16632963716983795), Hypothesis(value=['how', 'to', 'cook', 'pasta', 'in', 'a', 'pot', 'with', 'cornstarch'], score=-0.2282889038324356), Hypothesis(value=['how', 'to', 'cook', 'pasta', 'recipe', 'using', 'a', 'recipe'], score=-0.306397020816803), Hypothesis(value=['how', 'to', 'cook', 'pasta', 'in', 'a', 'pot'], score=-0.5775007009506226), Hypothesis(value=['how', 'to', 'cook', 'pasta', 'recipe'], score=-1.2557083368301392)]]
# [[Hypothesis(value=['states', 'interviewed', 'being', 'about', 'speak', 'girl', 'a'], score=-0.1975124031305313), Hypothesis(value=['states', 'interviewed', 'being', 'kids', 'from', 'clip', 'a'], score=-0.2153930366039276), Hypothesis(value=['states', 'interviewed', 'being', 'about', 'speak', 'kids'], score=-0.685401976108551), Hypothesis(value=['states', 'interviewed', 'being', 'kids'], score=-1.1049909591674805), Hypothesis(value=['states', 'interviewed', 'being', 'about', 'speak', 'girl', 'young'], score=-1.57078218460083)]]
#   1%|          | 24/2990 [00:05<10:04,  4.91it/s]completed_hypotheses [[Hypothesis(value=['girl', 'hosting', 'the', 'hood', 'and', 'let', 'celebrities', 'are', 'being', 'interviewed'], score=-0.2526220977306366), Hypothesis(value=['girl', 'hosting', 'a', 'vine', 'language'], score=-0.28689196705818176), Hypothesis(value=['girl', 'hosting', 'the', 'voice', 'clips'], score=-0.32078948616981506), Hypothesis(value=['girl', 'hosting', 'the', 'hood', 'videos'], score=-0.369079053401947), Hypothesis(value=['girl', 'hosting', 'the', 'hood', 'and', 'let', 'celebrities', 'are', 'being', 'share', 'germany'], score=tensor(-9.8281, device='cuda:0'))]]
# [[Hypothesis(value=['hall', 'the', 'during', 'fly', 'person', 'a'], score=-0.11217339336872101), Hypothesis(value=['planes', 'make', 'to', 'trying', 'woman', 'a'], score=-0.12124702334403992), Hypothesis(value=['something', 'doing', 'person', 'a'], score=-0.13619567453861237), Hypothesis(value=['hall', 'the', 'in', 'is', 'she', 'when', 'combing', 'is', 'woman', 'young', 'a'], score=tensor(-1.0370, device='cuda:0')), Hypothesis(value=['hall', 'the', 'in', 'is', 'she', 'when', 'combing', 'is', 'woman'], score=-1.645118236541748)]]
# completed_hypotheses [[Hypothesis(value=['the', 'person', 'has', 'been', 'judged', 'in', 'the', 'room'], score=-0.14519613981246948), Hypothesis(value=['the', 'person', 'has', 'been', 'judged', 'for', 'a', 'box'], score=-0.1949514001607895), Hypothesis(value=['the', 'person', 'has', 'been', 'judged', 'in', 'the', 'home'], score=-0.24546144902706146), Hypothesis(value=['the', 'person', 'has', 'been', 'cd'], score=-0.8860096335411072), Hypothesis(value=['the', 'person', 'has', 'been', 'judged'], score=-1.224090576171875)]]
#   1%|          | 25/2990 [00:05<10:13,  4.83it/s][[Hypothesis(value=['sports', 'gymnast', 'on', 'going', 'is', 'girl', 'young', 'a'], score=-0.11705617606639862), Hypothesis(value=['school', 'on', 'moves', 'some', 'showing', 'is', 'woman', 'a'], score=-0.12363868951797485), Hypothesis(value=['school', 'on', 'moves', 'some', 'showing', 'is', 'gymnast', 'a'], score=-0.1259077787399292), Hypothesis(value=['sports', 'gymnast', 'on', 'going', 'is', 'girl', 'young'], score=-2.8507308959960938), Hypothesis(value=['sports', 'gymnast', 'on', 'going', 'is', 'woman'], score=-2.990597724914551)]]
# completed_hypotheses [[Hypothesis(value=['a', 'girl', 'is', 'performing', 'on', 'stage'], score=-0.27885138988494873), Hypothesis(value=['a', 'girl', 'is', 'applying', 'gymnastics', 'activities'], score=-0.3418574333190918), Hypothesis(value=['little', 'girl', 's', 'watching', 'gymnastics'], score=-0.4536621868610382), Hypothesis(value=['girls', 'doing', 'gymnastics'], score=-0.5385003685951233), Hypothesis(value=['a', 'girl', 'is', 'applying', 'gymnastics'], score=-0.607836127281189)]]
#


