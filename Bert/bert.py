import torch
import torch.nn as nn
import torch.optim as optim
import re
import math
import numpy as np
from random import *

def make_batch():
    batch = []
    positive = negative = 0
    while positive != batch_size/2 or negative != batch_size/2:
        tokens_a_index, tokens_b_index= randrange(len(sentences)), randrange(len(sentences)) # sample random index in sentences
        tokens_a, tokens_b= token_list[tokens_a_index], token_list[tokens_b_index]
        input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM
        n_pred =  min(max_pred, max(1, int(round(len(input_ids) * 0.15)))) # 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word_dict['[CLS]'] and token != word_dict['[SEP]']]
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = word_dict['[MASK]'] # make mask
            elif random() < 0.5:  # 10%
                index = randint(0, vocab_size - 1) # random index in vocabulary
                input_ids[pos] = word_dict[num_dict[index]] # replace

        # Zero Paddings
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # NotNext
            negative += 1
    return batch

# def make_batch():
#     batch = []
#     positive = negative = 0 # 用于计数正例（IsNext）和负例（NotNext）的数量
#     while positive!= batch_size/2 or negative != batch_size/2:
#         tokens_a_index,tokens_b_index = randrange(len(sentences)),randrange(len(sentences)) # 随机选择两个句子的索引
#         tokens_a,tokens_b = token_list[tokens_a_index],token_list[tokens_b_index]
#         input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']]
#         segment_ids = [0] * (1+ len(tokens_a) + 1)+[1] * (len(tokens_b) + 1)
#         n_pred = min(max_pred,max(1,int(round(len(input_ids)*0.15)))) # 遮盖的数量
#         cand_maked_pos = [i for i,token in enumerate(input_ids) if token != word_dict['[CLS]'] and token!= word_dict['[SEP]']] #找到可以进行遮盖的候选位置
#         shuffle(cand_maked_pos) # 打乱他们
#         masked_tokens,masked_pos = [],[]
#         # 选定的候选位置上进行遮盖操作。
#         for pos in cand_maked_pos[:n_pred]:
#             masked_pos.append(pos)
#             masked_tokens.append(input_ids[pos])
#             if random() < 0.8:
#                 input_ids[pos] = word_dict['[MASK]']
#             elif random() < 0.5:
#                 index = randint(0,vocab_size-1)
#                 input_ids[pos] = word_dict[num_dict[index]]
#
#         # 将输入标记和句子片段标识的长度补齐到maxlen
#         n_pad = maxlen - len(input_ids)
#         input_ids.append([0]*n_pad)
#         segment_ids.extend([0]*n_pad)
#
#         # 将遮盖的数量补齐
#         if max_pred > n_pred:
#             n_pad = max_pred - n_pred
#             masked_tokens.extend([0]*n_pad)
#             masked_pos.extend([0]*n_pad)
#
#         # 根据选定的句子索引以及正例和负例的计数，将生成的批次数据添加到batch中
#         if tokens_a_index+1==tokens_b_index and positive < batch_size / 2:
#             batch.append([input_ids,segment_ids,masked_tokens,masked_pos,True])
#             positive +=1
#         elif tokens_a_index+1 !=tokens_b_index and negative < batch_size/2:
#             batch.append([input_ids,segment_ids,masked_tokens,masked_pos,False])
#             negative +=1
#     return batch


def get_attn_pad_mask(seq_q,seq_k):
    batch_size,len_q = seq_q.size()
    batch_size,len_k = seq_k.size()
    # eq(zero) is PAD token
    # pad_attn_mask:[batch_size,1,len_k]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # pad_attn_mask:[batch_size x len_q x len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self,Q,K,V,attn_mask):
        #Q: [batch_size,n_head,seq_q,d_model]
        #k: [batch_size,n_head,seq_k,d_model]
        #V: [batch_size,n_head,seq_v,d_model]

        # Q: [batch_size,n_head,seq_q,seq_k]
        scores = torch.matmul(Q,K.transpose(-1,-2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        # atten: [batch_size,n_head,seq_q,seq_k]
        attn = nn.Softmax(dim=-1)(scores)
        # context:[batch_size,n_head,seq_q,n_model]
        context = torch.matmul(attn, V)
        return context,attn

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding,self).__init__()
        self.tok_embed = nn.Embedding(vocab_size,d_model) # 这里的weight应该是[vocab_size,d_model]
        self.pos_embed = nn.Embedding(maxlen,d_model)
        self.seg_embed = nn.Embedding(n_segments,d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self,x,seg):
        # x:[batch_size,seq_len]
        seq_len = x.size(1)
        pos = torch.arange(seq_len,dtype=torch.long)
        #[seq_len] -> [batch_size,seq_len]
        pos = pos.unsqueeze(0).expand_as(x)
        # embedding: [batch_size,seq_len,d_model]
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)
        

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention,self).__init__()
        self.W_Q = nn.Linear(d_model,d_k * n_heads)
        self.W_K = nn.Linear(d_model,d_k * n_heads)
        self.W_V = nn.Linear(d_model,d_v * n_heads)

    def forward(self,Q,K,V,attn_mask):
        residual,batch_size = Q,Q.size(0)
        # proj:[batch_size,seq_len,d_model]-> [batch_size,seq_len,d_k*n_heads]
        # view: ->[batch_size, seq_len , n_heads,d_k]
        # trans: -> [batch_size,n_head,len_q,d_k]
        q_s = self.W_Q(Q).view(batch_size,-1,n_heads,d_k).transpose(1,2)
        # [batch_size, n_head, len_k, d_k]
        k_s = self.W_Q(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        # [batch_size, n_head, len_v, d_k]
        v_s = self.W_Q(V).view(batch_size, -1, n_heads, d_k).transpose(1,2)

        # attn_mask:[batch_size,seq_q,seq_k]
        # ---> [batch_size,n_head,seq_q,seq_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1,n_heads,1,1)
        # context: [batch_size,n_heads,len_q,d_v]
        # attn: [batch_size,n_heads,len_q,len_k]
        context,attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        # context: [batch_size,len_q,n_head*d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        # attn: [batch_size,n_heads, len_q x d_model]
        output = nn.Linear(n_heads*d_v,d_model)(context)
        return nn.LayerNorm(d_model)(output + residual), attn  # output: [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    def forward(self,x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer,self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self,enc_inputs,enc_self_attn_mask):
        # [batch_size,seq_len,d_model]
        enc_outputs,attn = self.enc_self_attn(enc_inputs,enc_inputs,enc_inputs,enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs,attn


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BERT(nn.Module):
    def __init__(self):
        super(BERT,self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Linear(d_model,d_model)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(d_model,d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model,2)

        # decoder is shared with embedding  layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab,n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim,n_vocab,bias = False)  # 这里的weight其实是[n_vocab ,n_dim]
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self,input_ids,segement_ids,masked_pos):
        # input_ids:[batch_size,seq_len]
        # segement_ids:[batch_size,seq_len]
        # output: [batch_size,seq_len,n_hidden]
        output = self.embedding(input_ids,segement_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids,input_ids)
        for layer in self.layers:
            # output:[batch_size,seq_len,n_hidden]
            # enc_self_attn: [batch_size,n_head,seq_q,seq_k]
            output,enc_self_attn = layer(output,enc_self_attn_mask)
            # 只取出来output中[CLS]对应的embeding，进行前向传播
            # h_pooled：[batch_size,d_model]
            h_pooled = self.activ1(self.fc(output[:, 0]))
            # [batch_size, 2]
            logits_clsf = self.classifier(h_pooled)

            # masked_pos: [batch_size, max_pred] -> # [batch_size, max_pred, d_model]
            masked_pos = masked_pos[:,:,None].expand(-1, -1, output.size(-1))
            # output:[batch_size,seq_len,n_hidden]
            # h_masked: [batch_size, max_pred, d_model],取出来那些掩码的位置的表征向量
            h_masked = torch.gather(output,1,masked_pos)
            h_masked = self.norm(self.activ2(self.linear(h_masked)))
            # [batch_size, max_pred, n_vocab]
            logits_lm = self.decoder(h_masked) + self.decoder_bias
            #  logits_lm: [batch_size,max_pred,n_vocab] logits_clsf:[batch_size,2]
            return logits_lm, logits_clsf


if __name__=="__main__":
    # BERT Parameters
    maxlen = 30 # 一个句子的长度
    batch_size = 6
    max_pred = 5  # 预测的最多token数
    n_layers = 6
    n_heads = 12
    d_model = 768
    d_ff = 768 *4
    d_k = d_v = 64
    n_segments = 2

    text = (
        'Hello, how are you? I am Romeo.\n'
        'Hello, Romeo My name is Juliet. Nice to meet you.\n'
        'Nice meet you too. How are you today?\n'
        'Great. My baseball team won the competition.\n'
        'Oh Congratulations, Juliet\n'
        'Thanks you Romeo'
    )
    sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')  # filter '.', ',', '?', '!'
    word_list = list(set(" ".join(sentences).split()))
    word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    for i,w in enumerate(word_list):
        word_dict[w] = i+4
    num_dict = {i:w for i,w in enumerate(word_dict)}
    vocab_size = len(word_dict)

    token_list = list() # 存放每个句子的词汇id序列
    for sentence in sentences:
        arr = [word_dict[s] for s in sentence.split()]
        token_list.append(arr)

    model = BERT()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch = make_batch()
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(*batch))

    for epoch in range(100):
        optimizer.zero_grad()
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
        # logits_lm: [batch_size,max_pred,n_vocab] logits_clsf:[batch_size,2]
        # masked_tokens: [batch_size,max_pred]
        # logits_lm.transpose(1, 2): [batch_size,n_vocab,max_pred]
        loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)  # for masked LM
        loss_lm = (loss_lm.float()).mean()
        loss_clsf = criterion(logits_clsf, isNext)  # for sentence classification
        loss = loss_lm + loss_clsf
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Predict mask tokens ans isNext
    # [1,seq_len]
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(batch[0]))
    print(text)
    print([num_dict[w.item()] for w in input_ids[0] if num_dict[w.item()] != '[PAD]'])

    # logits_lm: batch_size(1) * max_pred(5) * vobsize(29)
    # logits_clsf: batch_size(1) * 2
    logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
    # batch_size(1) * max_pred  ---> [max_pred]
    # batch_size(1) * max_pred(5) * vobsize(29)--max(2)[1]---> batch_size(1) * max_pred(5) ->[0]--> max_pred(5)
    logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
    print('masked tokens list : ', [num_dict[pos.item()] for pos in masked_tokens[0] if pos.item() != 0])
    print('predict masked tokens list : ', [num_dict[pos] for pos in logits_lm if pos != 0])

    # batch_size(1)
    logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
    print('isNext : ', True if isNext else False)
    print('predict isNext : ', True if logits_clsf else False)



