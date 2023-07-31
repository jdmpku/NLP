import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN,self).__init__()
        self.num_filters_total = num_filters * len(filter_size)
        self.W = nn.Embedding(vocab_size,embedding_dim=2)
        self.Weight = nn.Linear(self.num_filters_total,num_class,bias=False)
        self.Bias = nn.Parameter(torch.ones([num_class]))
        self.filter_list = nn.ModuleList([nn.Conv2d(1, num_filters, (size, embedding_size)) for size in filter_size])

    def forward(self,X):
        embedded_chars = self.W(X) # [batch_size, sequence_length, embedding_size]
        embedded_chars = embedded_chars.unsqueeze(1)  #[batch, channel(=1), sequence_length, embedding_size]
        pooled_outputs = []
        for i,conv in enumerate(self.filter_list):
            # conv : [input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]
            h = F.relu(conv(embedded_chars)) #h: [batch_size,channel(3),sequence_length-filter_size[i]+1,1]
            mp = nn.MaxPool2d((sequence_length-filter_size[i]+1,1)) #h: [batch_size,channel(3),1,1]
            # pooled : [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3)]
            pooled = mp(h).permute(0, 3, 2, 1)
            pooled_outputs.append(pooled)
        h_pool = torch.cat(pooled_outputs,len(filter_size))  # [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3) * 3]
        h_pool_flat = torch.reshape(h_pool, [-1,self.num_filters_total])  # [batch_size(=6), output_height * output_width * (output_channel * 3)]
        model = self.Weight(h_pool_flat) + self.Bias  # [batch_size, num_classes]
        return model

if __name__=="__main__":
    embedding_size = 2
    sequence_length = 3
    num_class = 2
    filter_size = [2,2,2]
    num_filters = 3

    sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
    labels = [1,1,1,0,0,0]

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w:i for i,w in enumerate(word_list)}
    vocab_size = len(word_dict)
    model = TextCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    inputs = torch.LongTensor([np.asarray([word_dict[n] for n in sen.split()]) for sen in sentences])
    targets = torch.LongTensor([out for out in labels])  #target : [batch_size]

    for epoch in range(5000):
        optimizer.zero_grad()
        output = model(inputs) # output : [batch_size, num_classes]
        # target 将真实标签表示为整数索引形式（从0开始）
        loss = criterion(output, targets)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

        # Test
    test_text = 'sorry hate you'
    tests = [np.asarray([word_dict[n] for n in test_text.split()])]
    test_batch = torch.LongTensor(tests)

    # Predict
    # batch_size * num_class
    print(model(test_batch).data)
    #max:对行取最大值，[1]返回最大值的索引，[0]返回最大值
    predict = model(test_batch).data.max(1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_text, "is Bad Mean...")
    else:
        print(test_text, "is Good Mean!!")





