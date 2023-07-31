import torch
import torch.nn as nn
import torch.optim as optim

def make_batch():
    input_batch = []
    target_batch = []
    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(input)
        target_batch.append(target)
    return input_batch,target_batch

# Model
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM,self).__init__()
        # 定义了一个嵌入层，用于将输入的离散化整数表示的单词或者字符转换为密集的联系向量表示
        self.C = nn.Embedding(n_class,m)
        # 对嵌入层的输出进行线性变换
        self.H = nn.Linear(n_steps*m,n_hidden,bias=False)
        # 定义了一个可学习的参数，用于在计算tanh时使用
        self.d = nn.Parameter(torch.ones(n_hidden))
        # 用于将tanh的输出映射回词汇表大小的向量
        self.U = nn.Linear(n_hidden,n_class,bias=False)
        # 用于将嵌入层的输入直接映射回词汇表大小的向量
        self.W = nn.Linear(n_steps*m,n_class,bias=False)
        # 定义了一个可学习的偏置参数
        self.b = nn.Parameter(torch.ones(n_class))
    def forward(self,X):
        X = self.C(X)   # X:[batch_size,n_step,m]
        X = X.view(-1,n_steps*m) #X:[batch_size,n_steps*m]
        tanh = torch.tanh(self.d + self.H(X)) #X:[batch_size,hidden_size]
        output = self.b + self.W(X) + self.U(tanh) #X:[batch_size,n_class]
        return output

if __name__=='__main__':
    n_steps = 3 #每个样本中有多少个单词
    n_hidden = 2 #隐藏层的维度
    m = 2 #embedding的维度

    sentences = ["i like smart dog", "i love smart coffee", "i hate smart milk"]

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w:i for i,w in enumerate(word_list)}
    number_dict = {i:w for i,w in enumerate(word_list)}
    n_class = len(word_dict) # 词表的大小

    model = NNLM()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    input_batch,target_batch = make_batch()
    input_batch = torch.LongTensor(input_batch)  #batch_size * n_steps
    target_batch = torch.LongTensor(target_batch) #batch_size * 1

    for epoch in range(5000):
        optimizer.zero_grad()
        output = model(input_batch)

        # output : [batch_size, n_class], target_batch : [batch_size]
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # Predict
    predict = model(input_batch).data.max(1, keepdim=True)[1]
    # Test
    print([sen.split()[:3] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])












