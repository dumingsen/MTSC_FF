import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt


def extract_data(data):
    res_data = []
    res_labels = []
    for t_data, t_label in data:
        t_data = np.array([d.tolist() for d in t_data])
        t_label = t_label.decode("utf-8")
        res_data.append(t_data)
        res_labels.append(t_label)
    return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    # swapaxes的用法就是交换轴的位置，前后两个的位置没有关系。


# from getfrequency import read_template


def load_UEA(archive_name, args):
    # train_data = loadarff(open(f'D:/FTP/chengrj/time_series/data/Multivariate_arff/{dataset}/{dataset}_TRAIN.arff','r',encoding='UTF-8'))[0]
    # test_data = loadarff(open(f'D:/FTP/chengrj/time_series/data/Multivariate_arff/{dataset}/{dataset}_TEST.arff','r',encoding='UTF-8'))[0]

    # load from cache
    cache_path = f'{args.cache_path}/{archive_name}.dat'  ##需要建一个
    if os.path.exists(cache_path) is True:
        print('load form cache....')
        train_x, train_y, test_x, test_y, num_class = torch.load(cache_path)


    # load from arff
    else:
        train_data = \
            loadarff(open(f'{args.data_path}/{archive_name}/{archive_name}_TRAIN.arff', 'r', encoding='UTF-8'))[0]
        test_data = \
            loadarff(open(f'{args.data_path}/{archive_name}/{archive_name}_TEST.arff', 'r', encoding='UTF-8'))[0]

        train_x, train_y = extract_data(train_data)  ##y为标签
        test_x, test_y = extract_data(test_data)
        train_x[np.isnan(train_x)] = 0
        test_x[np.isnan(test_x)] = 0

        scaler = StandardScaler()
        scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
        train_x = scaler.transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)
        test_x = scaler.transform(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape)

        # 放到0-Numclass
        labels = np.unique(train_y)  ##标签
        num_class = len(labels)
        # print(num_class)
        transform = {k: i for i, k in enumerate(labels)}
        train_y = np.vectorize(transform.get)(train_y)
        test_y = np.vectorize(transform.get)(test_y)

        torch.save((train_x, train_y, test_x, test_y, num_class), cache_path)

    # #图结构获取
    # f_g_train, f_g_test = [], []
    # print(train_x.shape)  #
    # #input()
    # #printtt(type(x_train))
    # for data in train_x:
    #     # printtt(1)
    #     G = graph_feature(data)
    #     f_g_train.append(G)
    #
    # for data in test_x:
    #     G = graph_feature(data)
    #     f_g_test.append(G)
    printt('train_x', train_x.shape)  # (40, 100, 6)

    ##邻接矩阵数据
    show=1
    if show==1:

        #图结构1 spearm
        f_g_train = np.loadtxt('/root/MSCN/corre1graph/{0}/{1}_train.txt'.format(archive_name, archive_name),
                               delimiter=','). \
            reshape((train_x.shape[0], train_x.shape[2], train_x.shape[2]))

        f_g_test = np.loadtxt('/root/MSCN/corre1graph/{0}/{1}_test.txt'.format(archive_name, archive_name),
                              delimiter=','). \
            reshape((test_x.shape[0], test_x.shape[2], test_x.shape[2]))
        f_g_test = np.array(f_g_test)
        f_g_train = np.array(f_g_train)
        printt('生成的图结构', f_g_train.shape)

        #图结构2 kendall
        f_g_train2 = np.loadtxt('/root/MSCN/corre2graph/{0}/{1}_train.txt'.format(archive_name, archive_name),
                               delimiter=','). \
            reshape((train_x.shape[0], train_x.shape[2], train_x.shape[2]))

        f_g_test2 = np.loadtxt('/root/MSCN/corre2graph/{0}/{1}_test.txt'.format(archive_name, archive_name),
                              delimiter=','). \
            reshape((test_x.shape[0], test_x.shape[2], test_x.shape[2]))
        f_g_test2 = np.array(f_g_test2)
        f_g_train2 = np.array(f_g_train2)
        printt('生成的图结构', f_g_train2.shape)

        ##频域图像
        dir_name1 = '/root/freq_image/{0}/Train'.format(archive_name)#/root/freq_image
        dir_name2 = '/root/freq_image/{0}/Test'.format(archive_name)
        freq_train = read_template(dir_name1,f_g_train.shape[1])
        freq_test = read_template(dir_name2,f_g_train.shape[1])
        printt('训练频域图像',freq_train.shape)
        printt('测试频域图像', freq_test.shape)
        #input()
        #各个数据集的预处理操作
        # input()
        # TrainDataset = DealDataset(train_x, train_y)
        # TestDataset = DealDataset(test_x, test_y)
        TrainDataset = subDataset(train_x, f_g_train,f_g_train2, freq_train, train_y)
        TestDataset = subDataset(test_x, f_g_test,f_g_test2,freq_test, test_y)

    # return TrainDataset,TestDataset,len(labels)
    # DataLoader是Pytorch中用来处理模型输入数据的一个工具类。组合了数据集（dataset） + 采样器(sampler)，
    # 并在数据集上提供单线程或多线程(num_workers )的可迭代对象
    # dataset (Dataset) – 决定数据从哪读取或者从何读取；
    # batchszie：批大小，决定一个epoch有多少个Iteration；

    #!!!
    train_loader = DataLoader(dataset=TrainDataset,
                              batch_size=args.batch_size,
                              shuffle=False)#
    test_loader = DataLoader(dataset=TestDataset,
                             batch_size=args.batch_size,
                             shuffle=False)
    #printt('1')
    return train_loader, test_loader, num_class

def load_UEA1(archive_name, args):
    # train_data = loadarff(open(f'D:/FTP/chengrj/time_series/data/Multivariate_arff/{dataset}/{dataset}_TRAIN.arff','r',encoding='UTF-8'))[0]
    # test_data = loadarff(open(f'D:/FTP/chengrj/time_series/data/Multivariate_arff/{dataset}/{dataset}_TEST.arff','r',encoding='UTF-8'))[0]

    # load from cache
    cache_path = f'{args.cache_path}/{archive_name}.dat'  ##需要建一个
    if os.path.exists(cache_path) is True:
        print('load form cache....')
        train_x, train_y, test_x, test_y, num_class = torch.load(cache_path)


    # load from arff
    else:
        train_data = \
            loadarff(open(f'{args.data_path}/{archive_name}/{archive_name}_TRAIN.arff', 'r', encoding='UTF-8'))[0]
        test_data = \
            loadarff(open(f'{args.data_path}/{archive_name}/{archive_name}_TEST.arff', 'r', encoding='UTF-8'))[0]

        train_x, train_y = extract_data(train_data)  ##y为标签
        test_x, test_y = extract_data(test_data)
        train_x[np.isnan(train_x)] = 0
        test_x[np.isnan(test_x)] = 0

        scaler = StandardScaler()
        scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
        train_x = scaler.transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)
        test_x = scaler.transform(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape)

        # 放到0-Numclass
        labels = np.unique(train_y)  ##标签
        num_class = len(labels)
        # print(num_class)
        transform = {k: i for i, k in enumerate(labels)}
        train_y = np.vectorize(transform.get)(train_y)
        test_y = np.vectorize(transform.get)(test_y)

        torch.save((train_x, train_y, test_x, test_y, num_class), cache_path)

    # #图结构获取
    # f_g_train, f_g_test = [], []
    # print(train_x.shape)  #
    # #input()
    # #printtt(type(x_train))
    # for data in train_x:
    #     # printtt(1)
    #     G = graph_feature(data)
    #     f_g_train.append(G)
    #
    # for data in test_x:
    #     G = graph_feature(data)
    #     f_g_test.append(G)
    printt('train_x', train_x.shape)  # (40, 100, 6)

    ##邻接矩阵数据
    show=1
    if show==1:
        #input()
        #各个数据集的预处理操作
        # input()
        # TrainDataset = DealDataset(train_x, train_y)
        # TestDataset = DealDataset(test_x, test_y)
        TrainDataset = DealDataset(train_x, train_y)
        TestDataset = DealDataset(test_x,  test_y)

    # return TrainDataset,TestDataset,len(labels)
    # DataLoader是Pytorch中用来处理模型输入数据的一个工具类。组合了数据集（dataset） + 采样器(sampler)，
    # 并在数据集上提供单线程或多线程(num_workers )的可迭代对象
    # dataset (Dataset) – 决定数据从哪读取或者从何读取；
    # batchszie：批大小，决定一个epoch有多少个Iteration；
    train_loader = DataLoader(dataset=TrainDataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=TestDataset,
                             batch_size=args.batch_size,
                             shuffle=True)

    return train_loader, test_loader, num_class


from model.layer import printt
from Teoriginal1 import getTEgraph1

import cv2
import torch
import cv2 as cv
array_of_img = []
def read_template(dir_name1,channel):
    # # 读取工程文件夹下存放图片的文件夹的图片名
    # imgList = os.listdir(r"./" + directory_name)
    # imgList.sort(key=lambda x: int(x.split('.')[0]))  # 按照数字顺序排列图片名
    # print(imgList)  # 打印出来看看效果，不想看也可以删掉这句

    path = dir_name1
    path_list = os.listdir(path)
    imgList = path_list
    print(path_list)
    imgList_sort=sorted(imgList)#,key=lambda x:int(x[0:4]
    # imgList.sort()
    if imgList_sort==imgList:
        printt('相等')
    else:
        printt('不相等')
    print('====================================================')
    print(imgList_sort)
    # 这个for循环是用来读入图片放入array_of_img这个列表中，因为前面的操作都只是图片名而已，图片是其实未读入的
    for count in range(0, len(imgList_sort)):

        filename = imgList[count]
        imgg = cv2.imread(dir_name1 + "/" + filename)  # 根据图片名读入图片
        #printt('imgg',imgg)
        #input()
        img = np.zeros(imgg.shape, dtype=np.float32)
        #归一化
        cv.normalize(imgg, img, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        #printt('img',img)
        array_of_img.append(img)  # 这一步是将图片放入array_of_img中，如果不需要的可以和这个列表的定义一起删掉

    print('array_of_img', np.array(array_of_img).shape)
    img = np.array(array_of_img)
    img = img.reshape(img.shape[0] // channel, -1, img.shape[1], img.shape[2], img.shape[3])
    print('img', img.shape)
    img1 = torch.permute(torch.as_tensor(img), (0, 1, 4, 3, 2))
    print('img1', img1.shape)

    # cv2.imshow("test", array_of_img[3])  # 显示第四张图片，严重是否正确，可以更改中括号里的数字多验证几张图片
    #cv2.imshow("test", img[1][1])
    array_of_img.clear()
    #printt('1')
    return np.array(img1)
def graph_feature(data):
    x = data
    # print(x.shape)  # 275
    # x = x.reshape(n, -1)
    # print('x', x.shape)  # 5,55

    x = getTEgraph1(x)
    # print('123x', x.shape)
    # G = {'feature': data, 'graph': x}
    # G = {}
    # G['feature'] = torch.tensor(data).cuda()
    # G['graph'] = torch.tensor(x).cuda()
    return x


class subDataset1(Dataset):
    def __init__(self, Feature_1, Feature_2, Label):
        self.Feature_1 = torch.from_numpy(Feature_1)
        self.Feature_2 = torch.from_numpy(Feature_2)
        self.Label = Label

    def __len__(self):
        return len(self.Label)

    def __getitem__(self, index):
        Feature_1 = torch.Tensor(self.Feature_1[index])
        Feature_2 = torch.Tensor(self.Feature_2[index])
        Label = torch.Tensor(self.Label[index])
        return Feature_1, Feature_2, Label


import cv2 as cv
import pathlib

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class subDataset3(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, x, z, i, y):
        self.x_data = torch.from_numpy(x)  # 原始
        self.graph = torch.from_numpy(z)  # 矩阵

        self.i = torch.from_numpy(i)  # 频域图片
        self.y_data = torch.from_numpy(y)  # 标签
        self.len = x.shape[0]
        self.x_data = self.x_data.transpose(2, 1)

    def __getitem__(self, index):
        # img = cv.imread(self.i[index])
        # # img = cv.resize(img, (224, 224))
        # img = img / 255.
        # img = (img - self.mean) / self.std
        # # img = np.transpose(img, [2, 0, 1])

        return self.x_data[index], self.graph[index], self.i[index], self.y_data[index]

    def __len__(self):
        return self.len


class subDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, x, z,z2, i, y):
        self.x_data = torch.from_numpy(x)  # 原始
        self.graph = torch.from_numpy(z)  # 矩阵1
        self.graph2 = torch.from_numpy(z2)  # 矩阵2
        self.i = torch.from_numpy(i)  # 频域图片
        self.y_data = torch.from_numpy(y)  # 标签
        self.len = x.shape[0]
        self.x_data = self.x_data.transpose(2, 1)

    def __getitem__(self, index):
        # img = cv.imread(self.i[index])
        # # img = cv.resize(img, (224, 224))
        # img = img / 255.
        # img = (img - self.mean) / self.std
        # # img = np.transpose(img, [2, 0, 1])

        return self.x_data[index], self.graph[index],self.graph2[index], self.i[index], self.y_data[index]

    def __len__(self):
        return self.len

class subDataset2(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, x, z, y):
        self.x_data = torch.from_numpy(x)
        self.graph = torch.from_numpy(z)
        self.y_data = torch.from_numpy(y)
        self.len = x.shape[0]
        self.x_data = self.x_data.transpose(2, 1)

    def __getitem__(self, index):
        return self.x_data[index], self.graph[index], self.y_data[index]

    def __len__(self):
        return self.len


class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, x, y):
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
        self.len = x.shape[0]
        self.x_data = self.x_data.transpose(2, 1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def num_class(self):
        return len(set(self.y_data))
def print(*args, **kwargs):
    flag = False
    # flag = False
    if flag:
        print(*args, **kwargs)
    else:
        pass

if __name__ == '__main__':
    load_UEA('Ering')
