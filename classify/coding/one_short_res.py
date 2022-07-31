from model.one_short import SiameseNet
from util.one_short_dataset import get_data
from torch.nn.functional import binary_cross_entropy_with_logits
from torch import optim
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from util.d_index import multi_compute_measure
import numpy as np

learning_rate=0.02


train_data_set,test_data_set=get_data()
train_loader=DataLoader(train_data_set,batch_size=10,shuffle=True)
test_loader=DataLoader(test_data_set,batch_size=10,shuffle=True)


model = SiameseNet()
if torch.cuda.is_available():
    model = model.cuda()


optimizer = optim.Adam(model.parameters(),lr=learning_rate)

epochs = 2
for epoch in range(epochs):
    for data in train_loader:
        img1,img2, label = data
        if torch.cuda.is_available():
            img1 = img1.to(torch.float32).cuda()
            img2 = img2.to(torch.float32).cuda()
            label = label.view(-1,1).to(torch.float32).cuda()
        else:
            img1 = Variable(img1)
            img2 = Variable(img2)
            label = Variable(label.view(-1,1))

        out = model(img1,img2)

        loss = binary_cross_entropy_with_logits(out, label)



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(epoch)
    if epoch % 50 == 0:
        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

model.eval()
eval_loss = 0
eval_acc = 0
eval_dindex=0
for data in test_loader:
    img1,img2, label = data

    # 与全连接网络不同，卷积网络不需要将所像素矩阵转换成一维矩阵
    # img = img.view(img.size(0), -1)

    if torch.cuda.is_available():
        img1 = img1.to(torch.float32).cuda()
        img2 = img2.to(torch.float32).cuda()
        label = label.view(-1, 1).to(torch.float32).cuda()
    else:
        img1 = Variable(img1)
        img2 = Variable(img2)
        label = Variable(label.view(-1,1))

    out = model(img1,img2)
    loss = binary_cross_entropy_with_logits(out, label)
    eval_loss += loss.data.item() * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label.squeeze(1)).sum()
    eval_acc += num_correct.item()
    ans=multi_compute_measure(2,label.squeeze(1).to('cpu').numpy().astype(np.int16),pred.to('cpu').numpy().astype(np.int16))

    eval_dindex+=ans[1]
print('Test Loss: {:.6f}, Acc: {:.6f}, dindex: {:.6f}'.format(
    eval_loss / (len(train_data_set)),
    eval_acc / (len(test_data_set)),
    eval_dindex / (len(test_loader))
))