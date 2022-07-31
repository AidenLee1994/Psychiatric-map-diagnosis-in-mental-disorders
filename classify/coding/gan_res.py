import numpy as np

from model.gan import discriminator,generator
from util.gan_dataset import get_data
import torch.nn as nn
from torch import optim
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from util.d_index import multi_compute_measure


learning_rate=0.02
batch_size=10
z_dimension = 100

train_data_set,test_data_set=get_data(1)
train_loader=DataLoader(train_data_set,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_data_set,batch_size=batch_size,shuffle=True)



D=discriminator()
G=generator()
if torch.cuda.is_available():
    D=D.cuda()
    G=G.cuda()

criterion = nn.CrossEntropyLoss()

D_optimizer = optim.Adam(D.parameters(),lr=learning_rate)
G_optimizer = optim.Adam(G.parameters(),lr=learning_rate)

epochs = 500
for epoch in range(epochs):
    for data in train_loader:
        img, label = data

        # 与全连接网络不同，卷积网络不需要将所像素矩阵转换成一维矩阵
        # img = img.view(img.size(0), -1)

        if torch.cuda.is_available():
            img = img.to(torch.float32).cuda()
            label = label.to(torch.float32).cuda()
            fake_label = torch.from_numpy(np.array([3]*batch_size)).cuda()
            D_z = Variable(torch.randn(batch_size, z_dimension)).cuda()
            G_z = Variable(torch.randn(batch_size, z_dimension)).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
            fake_label = torch.from_numpy(np.array([3] * batch_size))
            D_z = Variable(torch.randn(batch_size, z_dimension))
            G_z = Variable(torch.randn(batch_size, z_dimension))
        #计算真实图片
        real_out=D(img)
        real_loss=criterion(real_out,label.long())

        #计算假图片
        fake_img=G(D_z).detach()
        fake_out=D(fake_img)
        fake_loss=criterion(fake_out,fake_label.long())

        #计算判别器的损失，更新
        D_loss=real_loss+fake_loss

        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        #更新生成器
        fake_img = G(G_z)
        fake_output = D(fake_img)
        G_loss = criterion(fake_output, fake_label.long())
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

    if epoch % 50 == 0:
        print('epoch: {}, D loss: {:.4}  G loss: {:.4}'.format(epoch, D_loss.data.item(), G_loss.data.item()))

D.eval()
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

    out = D(img)
    loss = criterion(out, label.long())
    eval_loss += loss.data.item() * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
    ans = multi_compute_measure(3, pred.to('cpu').numpy().astype(np.int16), label.to('cpu').numpy().astype(np.int16))
    eval_dindex+=ans[1]
print('Test Loss: {:.6f}, Acc: {:.6f}, dindex: {:.6f}'.format(
    eval_loss / (len(train_data_set)),
    eval_acc / (len(test_data_set)),
    eval_dindex / (len(test_loader))
))