import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import argparse
from freq_ import get_freq
from torchvision import datasets, models, transforms
import torchvision.datasets


def extract_data(data):
    res_data = []
    res_labels = []
    for t_data, t_label in data:
        t_data = np.array([d.tolist() for d in t_data])
        t_label = t_label.decode("utf-8")
        res_data.append(t_data)
        res_labels.append(t_label)
    return np.array(res_data).swapaxes(1, 2), np.array(res_labels)


from Teoriginal1 import getTEgraph1


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


parser = argparse.ArgumentParser(description='MF-Net for MTSC')
parser.add_argument('--data_path', type=str, default='/root/MF-Net-1202/data/Multivariate_arff')  # ./data \root\MF-Net-1202\data\Multivariate_arff
parser.add_argument('--cache_path', type=str, default='./cache')
args = parser.parse_args()
# random_seed(args.seed)

####ArticularyWordRecognition
#AtrialFibrillation
#StandWalkJump
archive_name = 'ArticularyWordRecognition'  ####
# BasicMotions
# ArticularyWordRecognition
#ArticularyWordRecognition
#BasicMotions
#HandMovementDirection
#NATOPS
#SelfRegulationSCP2

# def load_UEA(archive_name, args):

# load from cache
cache_path = f'{args.cache_path}/{archive_name}.dat'  ##需要建一个
# if os.path.exists(cache_path) is True:
#     print('load form cache....')
#     train_x, train_y, test_x, test_y, num_class = torch.load(cache_path)
if 1 == 2:
    print(1)

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

# torch.save((train_x, train_y, test_x, test_y, num_class), cache_path)

# 图结构获取

dir_name1 = './freq_image/{0}/Train'.format(archive_name)
if not os.path.isdir(dir_name1):
    os.makedirs(dir_name1)
dir_name2 = './freq_image/{0}/Test'.format(archive_name)
if not os.path.isdir(dir_name2):
    os.makedirs(dir_name2)

f_g_train, f_g_test = [], []
print(train_x.shape)  # 数据维度
# input()
# printtt(type(x_train))
i = 1
train = True

# core 训练集
# for data in train_x:
#     print('freq_train_{0}'.format(i))
#     # printtt(1)

#     #获得数据集对应频域图像
#     get_freq(data,i,archive_name,train=True)


#     # G = graph_feature(data)#
#     # f_g_train.append(G)
#     i = i + 1

# core 测试集
j = 1
# for data in test_x:
#     print('freq_test_{0}'.format(j))

#     get_freq(data, j, archive_name, train=False)

#     # G = graph_feature(data)
#     # f_g_test.append(G)
#     j = j + 1

# f_g_test = np.array(f_g_test)
# f_g_train = np.array(f_g_train)
# print('生成的图结构', f_g_train.shape)


import os

path = dir_name1
path_list = os.listdir(path)
print(path_list)

import cv2
import torch
import cv2 as cv
import pathlib

array_of_img = []  # 用来存放图片的列表，不需要的可以删除
def read_template(dir_name1):
    # # 读取工程文件夹下存放图片的文件夹的图片名
    # imgList = os.listdir(r"./" + directory_name)
    # imgList.sort(key=lambda x: int(x.split('.')[0]))  # 按照数字顺序排列图片名
    # print(imgList)  # 打印出来看看效果，不想看也可以删掉这句

    path = dir_name1
    path_list = os.listdir(path)
    imgList = path_list
    #print(path_list)
    imgList_sort=sorted(imgList)#,key=lambda x:int(x[0:4])
    # imgList.sort()
    if imgList_sort==imgList:
        print('相等')
    else:
        print('不相等')
    print('====================================================')
    print(imgList_sort)
    # 这个for循环是用来读入图片放入array_of_img这个列表中，因为前面的操作都只是图片名而已，图片是其实未读入的
    for count in range(0, len(imgList_sort)):

        filename = imgList[count]
        img = cv2.imread(dir_name1 + "/" + filename)  # 根据图片名读入图片
        array_of_img.append(img)  # 这一步是将图片放入array_of_img中，如果不需要的可以和这个列表的定义一起删掉
    print('array_of_img', np.array(array_of_img).shape)
    img = np.array(array_of_img)
    img = img.reshape(img.shape[0] // 2, -1, img.shape[1], img.shape[2], img.shape[3])
    print('img', img.shape)
    img1 = torch.permute(torch.as_tensor(img), (0, 1, 4, 3, 2))
    print('img1', img1.shape)

    # cv2.imshow("test", array_of_img[3])  # 显示第四张图片，严重是否正确，可以更改中括号里的数字多验证几张图片
    #cv2.imshow("test", img[1][1])

    return np.array(img1)


read_template(dir_name1)
cv2.waitKey(0)

##数据预处理
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class subDataset(Dataset):
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
        img = cv.imread(self.i[index])
        # img = cv.resize(img, (224, 224))
        img = img / 255.
        img = (img - self.mean) / self.std
        # img = np.transpose(img, [2, 0, 1])

        return self.x_data[index], self.graph[index], self.img[index], self.y_data[index]

    def __len__(self):
        return self.len


#train_set = subDataset(train_dir)


##数据集预处理操作
class Hotdog(Dataset):
    def __init__(self, path):
        data_root = pathlib.Path(path)
        all_image_paths = list(data_root.glob('*/*'))
        self.all_image_paths = [str(path) for path in all_image_paths]  ##

        label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
        label_to_index = dict((label, index) for index, label in enumerate(label_names))
        self.all_image_labels = [label_to_index[path.parent.name] for path in all_image_paths]
        self.mean = np.array(mean).reshape((1, 1, 3))
        self.std = np.array(std).reshape((1, 1, 3))

    def __getitem__(self, index):
        img = cv.imread(self.all_image_paths[index])
        img = cv.resize(img, (224, 224))
        img = img / 255.
        img = (img - self.mean) / self.std
        img = np.transpose(img, [2, 0, 1])
        label = self.all_image_labels[index]
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label)
        return img, label

    def __len__(self):
        return len(self.all_image_paths)

# train_set = Hotdog(train_dir)
# test_set = Hotdog(test_dir)

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
# train_augs = transforms.Compose([
#     # transforms.RandomResizedCrop(size=224),
#     # transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean, std)
# ])
# from torchvision.datasets import ImageFolder
# dataset_train = ImageFolder(dir_name1,transform=train_augs)
# dataset_test=ImageFolder(dir_name1,transform=train_augs)
#
# print('train',dataset_train.shape)
