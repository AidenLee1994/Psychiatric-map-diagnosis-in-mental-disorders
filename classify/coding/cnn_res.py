from model.cnn import CNN
from util.cnn_dataset import get_data
import torch.nn as nn
from torch import optim
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from util.d_index import multi_compute_measure
import numpy as np
learning_rate=0.02


train_data_set,test_data_set=get_data(1)
train_loader=DataLoader(train_data_set,batch_size=10,shuffle=True)
test_loader=DataLoader(test_data_set,batch_size=10,shuffle=True)


model = CNN()
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

epochs = 700
for epoch in range(epochs):
    for data in train_loader:
        img, label = data

        # 与全连接网络不同，卷积网络不需要将所像素矩阵转换成一维矩阵
        # img = img.view(img.size(0), -1)

        if torch.cuda.is_available():
            img = img.to(torch.float32).cuda()
            label = label.to(torch.float32).cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        out = model(img)

        loss = criterion(out, label.long())

        print_loss = loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch += 1
    if epoch % 50 == 0:
        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

model.eval()
eval_loss = 0
eval_acc = 0
eval_dindex=0
for data in test_loader:
    img, label = data

    # 与全连接网络不同，卷积网络不需要将所像素矩阵转换成一维矩阵
    # img = img.view(img.size(0), -1)

    if torch.cuda.is_available():
        img = img.to(torch.float32).cuda()
        label = label.to(torch.float32).cuda()
    else:
        img = Variable(img)
        label = Variable(label)

    out = model(img)
    loss = criterion(out, label.long())
    eval_loss += loss.data.item() * label.size(0)
    _, pred = torch.max(out, 1)

    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()

    ans = multi_compute_measure(3,pred.to('cpu').numpy().astype(np.int16), label.to('cpu').numpy().astype(np.int16))
    eval_dindex+=ans[1]
print('Test Loss: {:.6f}, Acc: {:.6f}, dindex: {:.6f}'.format(
    eval_loss / (len(train_data_set)),
    eval_acc / (len(test_data_set)),
    eval_dindex / (len(test_loader))
))