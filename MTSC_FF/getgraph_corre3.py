import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import argparse

def extract_data(data):
    res_data = []
    res_labels = []
    for t_data, t_label in data:
        t_data = np.array([d.tolist() for d in t_data])
        t_label = t_label.decode("utf-8")
        res_data.append(t_data)
        res_labels.append(t_label)
    return np.array(res_data).swapaxes(1, 2), np.array(res_labels)

from Teoriginal_corre3 import getTEgraph1
def graph_feature(data):
    x = data
    # print(x.shape)  # 275
    # x = x.reshape(n, -1)
    # print('x', x.shape)  # 5,55

    x = getTEgraph1(x)
    #print('123x', x.shape)
    # G = {'feature': data, 'graph': x}
    # G = {}
    # G['feature'] = torch.tensor(data).cuda()
    # G['graph'] = torch.tensor(x).cuda()
    return x
parser = argparse.ArgumentParser(description='MF-Net for MTSC')
parser.add_argument('--data_path', type=str, default='D:\桌面\多元时间序列数据集\Multivariate_arff')#./data
parser.add_argument('--cache_path', type=str, default='./cache')
args = parser.parse_args()
# random_seed(args.seed)

####
archive_name='AtrialFibrillation'####
#BasicMotions
#ArticularyWordRecognition
#def load_UEA(archive_name, args):

# load from cache
cache_path = f'{args.cache_path}/{archive_name}.dat'##需要建一个
# if os.path.exists(cache_path) is True:
#     print('load form cache....')
#     train_x, train_y, test_x, test_y, num_class = torch.load(cache_path)
if 1==2:
    print(1)

# load from arff
else:
    train_data = \
        loadarff(open(f'{args.data_path}/{archive_name}/{archive_name}_TRAIN.arff', 'r', encoding='UTF-8'))[0]
    test_data = \
        loadarff(open(f'{args.data_path}/{archive_name}/{archive_name}_TEST.arff', 'r', encoding='UTF-8'))[0]

    train_x, train_y = extract_data(train_data)##y为标签
    test_x, test_y = extract_data(test_data)
    train_x[np.isnan(train_x)] = 0
    test_x[np.isnan(test_x)] = 0

   # 归一化
    # scaler = StandardScaler()
    # scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
    # train_x = scaler.transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)
    # test_x = scaler.transform(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape)

    # # 放到0-Numclass
    # labels = np.unique(train_y)##标签
    # num_class = len(labels)
    # # print(num_class)
    # transform = {k: i for i, k in enumerate(labels)}
    # train_y = np.vectorize(transform.get)(train_y)
    # test_y = np.vectorize(transform.get)(test_y)

    #torch.save((train_x, train_y, test_x, test_y, num_class), cache_path)

#图结构获取
f_g_train, f_g_test = [], []
print(train_x.shape)  #
#input()
#printtt(type(x_train))
i=1
for data in train_x:
    print('图获取train{0}+'.format(i))
    # printtt(1)
    G = graph_feature(data)
    f_g_train.append(G)
    i=i+1
j=1
for data in test_x:
    print('图获取test+'.format(j))
    G = graph_feature(data)
    f_g_test.append(G)
    j=j+1

f_g_test = np.array(f_g_test)
f_g_train = np.array(f_g_train)
print('生成的图结构',f_g_train.shape)


dir_name='./corre3graph/{0}'.format(archive_name)
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)

with open('./corre3graph/{0}/{1}_train.txt'.format(archive_name,archive_name), 'w') as outfile:
    for slice_2d in f_g_train:
        np.savetxt(outfile, slice_2d, fmt = '%f', delimiter = ',')
print('写入train')
with open('./corre3graph/{0}/{1}_test.txt'.format(archive_name,archive_name), 'w') as outfile:
    for slice_2d in f_g_test:
        np.savetxt(outfile, slice_2d, fmt = '%f', delimiter = ',')
print('写入test')

c = np.loadtxt('./corre3graph/{0}/{1}_train.txt'.format(archive_name,archive_name), delimiter = ',').\
    reshape((f_g_train.shape[0], f_g_train.shape[1], f_g_train.shape[2]))
print('读出train',c.shape)#(40, 6, 6)