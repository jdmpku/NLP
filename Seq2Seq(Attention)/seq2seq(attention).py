import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def make_batch():
    input_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[0].split()]]]
    output_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[1].split()]]]
    target_batch = [[word_dict[n] for n in sentences[2].split()]]
    # make tensor
    return torch.FloatTensor(input_batch), torch.FloatTensor(output_batch), torch.LongTensor(target_batch)

class Attention(nn.Module):
    def __init__(self):
        super(Attention,self).__init__()
        self.enc_cell = nn.RNN(input_size=n_class,hidden_size=n_hidden,dropout=0.5)
        self.dec_cell = nn.RNN(input_size=n_class,hidden_size=n_hidden,dropout=0.5)

        self.attn = nn.Linear(n_hidden,n_hidden)
        self.out =nn.Linear(n_hidden*2,n_class)

    def forward(self,enc_input,hidden,dec_input):
        enc_inputs = enc_input.transpose(0,1)
        dec_inputs = dec_input.transpose(0,1)

        # enc_output: [seq_length,batch_size(1),num_direction*n_hidden]
        # enc_hidden:[num_layer*num_direction,batch_size,n_hidden]
        enc_outputs,enc_hidden = self.enc_cell(enc_inputs,hidden)

        trained_attn = []
        hidden = enc_hidden
        n_step = len(dec_inputs)
        model = torch.empty([n_step,1,n_class])

        for i in range(n_step):
            # dec_output : [n_step(1), batch_size(=1), num_directions(=1) * n_hidden]
            # hidden : [num_layers(=1) * num_directions(=1), batch_size(=1), n_hidden]
            dec_output,hidden = self.dec_cell(dec_inputs[i].unsqueeze(0),hidden)
            attn_weights = self.get_att_weight(dec_output, enc_outputs)  # attn_weights : [1, 1, n_step]
            # dec_lenght * encode_length
            trained_attn.append(attn_weights.squeeze().data.numpy())

            # [2,3,4] X [2,4,5] = [2,3,5]
            # [1, 1, n_step] * [batch_size(1),n_step,num_direction*n_hidden] = [batch_size(1),1,num_direction*n_hidden]
            context = attn_weights.bmm(enc_outputs.transpose(0,1))
            dec_output = dec_output.squeeze(0)  # dec_output : [batch_size(=1), num_directions(=1) * n_hidden]
            context = context.squeeze(1)  # [batch_size(1), num_directions(=1) * n_hidden]
            # model[i]:[batch_size(1),n_class]
            model[i] = self.out(torch.cat((dec_output, context), 1))
        # model:[batch_size(1),seq_len,n_class]
        return model.transpose(0, 1).squeeze(0), trained_attn

    def get_att_weight(self,dec_output,enc_outputs):
        n_step = len(enc_outputs)
        attn_scores = torch.zeros(n_step)

        for i in range(n_step):
            # eec_output[i]: [batch_size(1),n_hidden]
            # dec_output: [batch_size(1),n_hidden]
            attn_scores[i] = self.get_att_score(dec_output,enc_outputs[i])
        return F.softmax(attn_scores).view(1,1,-1)

    def get_att_score(self,dec_output,enc_output):
        score = self.attn(enc_output)  # score : [batch_size, n_hidden]
        return torch.dot(dec_output.view(-1), score.view(-1))  # inner product make scalar value





if __name__=='__main__':
    n_steps = 5
    n_hidden = 128

    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w:i for i,w in enumerate(word_list)}
    number_dict = {i:w for i,w in enumerate(word_list)}
    n_class = len(word_dict)

    hidden = torch.zeros(1,1,n_hidden)

    model = Attention()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    input_batch, output_batch, target_batch = make_batch()

    for epoch in range(2000):
        optimizer.zero_grad()
        output, _ = model(input_batch, hidden, output_batch)

        loss = criterion(output, target_batch.squeeze(0))
        if (epoch + 1) % 400 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    #Test
    test_batch = [np.eye(n_class)[[word_dict[n] for n in 'SPPPP']]]
    test_batch = torch.FloatTensor(test_batch)
    predict, trained_attn = model(input_batch, hidden, test_batch)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    # Show Attention
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(trained_attn, cmap='viridis')
    ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14})
    ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()


