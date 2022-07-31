from sklearn.svm import SVC
from util.pmap_dataset import get_data
from util.d_index import multi_compute_measure
from tqdm import tqdm
learning_rate=0.02


model = SVC()
epochs = 1000
percent=1
for epoch in tqdm(range(epochs)):
    train_data_set, _ = get_data(percent)
    img=train_data_set[0]
    label=train_data_set[1]
    model.fit(img,label)


eval_dindex=0
eval_acc=0
for epoch in tqdm(range(epochs)):
    _, test_data_set = get_data(percent)
    img = test_data_set[0]
    label = test_data_set[1]
    out=model.predict(img)
    ans = multi_compute_measure(3, out, label)
    eval_acc+=ans[0]
    eval_dindex+=ans[1]

print(' Acc: {:.6f}, dindex: {:.6f}'.format(
    eval_acc / (epochs),
    eval_dindex / (epochs)
))


