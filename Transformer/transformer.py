import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib as plt


def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)

def get_sinusoid_encoding_table(n_position,d_model):
    # 于计算位置position和隐藏单元hid_idx之间的角度。
    def cal_angle(position,hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    # 获取给定位置position的Sinusoid编码向量
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table) #[src_len+1,d_model]


def get_attn_pad_mask(seq_q,seq_k):
    # batch_size * len_q
    batch_size,len_q = seq_q.size()
    batch_size,len_k = seq_k.size()
    # batch_size * 1 * len_k
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # pad_attn_mask:[batch_size,len_q,len_k]
    return pad_attn_mask.expand(batch_size,len_q,len_k)

def get_attn_subsequent_mask(seq):
    # seq: batch_size,tgt_len
    # batch_size,tgt_len,tgt_len
    attn_shape = [seq.size(0),seq.size(1),seq.size(1)]
    # 上三角全是1，下三角全是0
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    return subsequent_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention,self).__init__()

    def forward(self,Q,K,V,attn_mask):
        # Q: [batch_size,n_heads,len_q,d_k]
        # score: [batch_size,n_head,len_q,len_k]
        # attn_mask: [batch_size,n_head,len_q,len_k]
        scores = torch.matmul(Q,K.transpose(-1,-2) / np.sqrt(d_k))
        scores.masked_fill_(attn_mask, -1e9)
        # attn: [batch_size,n_head,len_q,len_k]
        attn = nn.Softmax(dim=-1)(scores)
        # V:[batch_size,n_head,len_k,d_v]
        # context:[batch_size,n_head,len_q,d_v]
        context = torch.matmul(attn,V)
        return context,attn



class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention,self).__init__()
        self.W_Q = nn.Linear(d_model,d_k * n_heads)
        self.W_K = nn.Linear(d_model,d_k * n_heads)
        self.W_V = nn.Linear(d_model,d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v,d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self,Q,K,V,attn_mask):
        # q: [batch_size x len_q x d_model]
        # k: [batch_size x len_k x d_model]
        # v: [batch_size x len_k x d_model]
        # attn_mask: [batch_size,len_q,len_k]
        residual, batch_size = Q,Q.size()
        # q_s: [batch_size,n_heads,len_q,d_k]
        # q_s = self.W_Q(Q).view(batch_size,-1,n_heads,d_k)
        # q_s = q_s.transpose(1,2)
        # k_s = self.W_K(K).view(batch_size,-1,n_heads,d_k).transpose(1,2)
        # v_s = self.W_V(V).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        q_s = self.W_Q(Q)
        q_s = q_s.view(batch_size, -1, n_heads, d_k)
        q_s = q_s.transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)


        # attn_mask: [batch_size,n_head,len_q,len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1,n_heads,1,1)
        # context: [batch_size,n_head,len_q,d_v]
        # attn: [batch_size,n_head,len_q,len_k]
        context,attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        # context: [batch_size,len_q,n_head*len_k]
        context = context.transpose(1,2).contiguous().view(batch_size,-1,n_heads * d_v)
        # output: [batch_size,len_q,d_model]
        output = self.liear(context)
        # [batch_size,len_q,d_model]
        # [batch_size,n_head,len_q,len_k]
        return self.layer_norm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self,inputs):
        #inputs,residual: [batch_size,len_q,d_model]
        redisual = inputs
        # 输入特征图的形状应该是(batch_size, in_channels, input_length)。
        # output: [batch_size,d_ff,len_q]
        output = nn.ReLU()(self.conv1(inputs.transpose(1,2)))
        # output: [batch_size,len_q,d_model]
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + redisual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer,self).__init__()
        self.enc_self_attns = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self,enc_inputs,enc_self_attn_mask):
        # enc_outputs: [batch_size x len_q x d_model]
        enc_outputs, attn = self.enc_self_attns(enc_inputs, enc_inputs, enc_inputs,enc_self_attn_mask)
        # enc_outputs: [batch_size x len_q x d_model]
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer,self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self,dec_input,enc_outputs,dec_self_attn_mask,dec_enc_attn_mask):
        # dec_input: batch_size * tgt_len * d_model
        # enc_output: batch_size * src_len * d_model
        # dec_self_attn_mask: batch_size * tgt_len * tgt_len
        # dec_enc_attn_mask: batch_size * tgt_len * src_len
        dec_output,dec_self_attn = self.dec_self_attn(dec_input,dec_input,dec_input,dec_self_attn_mask)
        dec_output,dec_enc_attn = self.dec_enc_attn(dec_output,enc_outputs,enc_outputs,dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output,dec_self_attn,dec_enc_attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size,d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len+1, d_model),freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    # enc_input: batch_size * src_len
    def forward(self,enc_inputs):
        # enc_output: batch_size * src_len * d_model
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(torch.LongTensor([[1,2,3,4,0]]))
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs,enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs,enc_self_attns

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size,d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len + 1, d_model), freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self,dec_inputs,enc_input,enc_output):
        # enc_input: batch_size * src_len
        # enc_output: batch_size * src_len * d_model
        # dec_input: batch_size * tgt_len
        # dec_output:batch_size * tgt_len * d_model
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5, 1, 2, 3, 4]]))
        # batch_size,tgt_len,tgt_len
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        # batch_size,tgt_len,tgt_len
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs,enc_input)

        dec_self_attns, dec_enc_attns = [],[]
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_output, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs,dec_self_attns,dec_enc_attns




class Transformer(nn.Module):
    def __init__(self):
        super(Transformer,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
    def forward(self,enc_input,dec_input):
        # enc_input: batch_size * src_len
        # enc_output: batch_size * src_len * d_model
        # enc_self_attns: batch_size,n_head,src_len,src_len
        # dec_input: batch_size * tgt_len
        enc_outputs,enc_self_attns = self.encoder(enc_input)
        dec_outputs,dec_self_attns,dec_enc_attns = self.decoder(dec_input,enc_input,enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

def showgraph(attn):
    # [1,q_k,q_v]
    attn = attn[-1].squeeze(0)[0]
    # [q_k,q_v]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads)) # [n_heads, n_heads]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels(['']+sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels(['']+sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()


if __name__=="__main__":
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5
    tgt_len = 5

    d_model = 512 # Embedding size
    d_ff = 2048 # FeedForward dimension
    d_k=d_v = 64 # dimension of K,V
    n_layers = 6 # number of Encoder of Decoder layer
    n_heads = 8 # number of heads in Multi-Head Attention

    model = Transformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    enc_inputs,dec_inputs,target_batch = make_batch(sentences)

    for epoch in range(20):
        optimizer.zero_grad()
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    predict, _, _, _ = model(enc_inputs, dec_inputs)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    print('first head of last state enc_self_attns')
    showgraph(enc_self_attns)

    print('first head of last state dec_self_attns')
    showgraph(dec_self_attns)

    print('first head of last state dec_enc_attns')
    showgraph(dec_enc_attns)
