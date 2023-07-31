import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def make_batch():
    input_batch = []
    target_batch = []
    for seq in seq_data:
        input = [word_dict[n] for n in seq[:-1]]
        target = word_dict[seq[-1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)
    return input_batch,target_batch
class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM,self).__init__()
        self.lstm = nn.LSTM(input_size = n_class,hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self,X):
        # X:[batch_size,n_step,n_class]
        input = X.transpose(0,1) #input:[n_step,batch_size,n_class]
        hidden_state = torch.zeros(1, len(X), n_hidden)  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        cell_state = torch.zeros(1, len(X), n_hidden)  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]

        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model




if __name__=="__main__":
    n_steps = 3 # num of cell
    n_hidden = 128 # number of hidden unit in one cell

    char_arr =[c for c in 'abcdefghijklmnopqrstuvwxyz']
    word_dict = {n:i for i,n in enumerate(char_arr)}
    number_dict = {i:n for i,n in enumerate(char_arr)}
    n_class = len(word_dict) # number of vocab

    seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']

    model = TextLSTM()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    input_batch,target_batch = make_batch()

    input_batch, target_batch = make_batch()
    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)
    # Training
    for epoch in range(1000):
        optimizer.zero_grad()

        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    inputs = [sen[:3] for sen in seq_data]

    #batch_size * 1
    predict = model(input_batch).data.max(1, keepdim=True)[1]
    #因为predict还是一个张量，所以要点item
    print(inputs, '->', [number_dict[n.item()] for n in predict.squeeze()])