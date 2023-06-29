# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:53

@author: mick.yi

入口类

"""
import argparse
import os
import re

import cv2
import numpy as np
import torch
#from skimage import io
from torch import nn
from torchvision import models

from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from interpretability.guided_back_propagation import GuidedBackPropagation

from model.net import gtnet


def get_net(net_name, weight_path=None):
    """
    根据网络名称获取模型
    :param net_name: 网络名称
    :param weight_path: 与训练权重路径
    :return:
    """
    pretrain = weight_path is None  # 没有指定权重路径，则加载默认的预训练权重
    if net_name in ['vgg', 'vgg16']:
        net = models.vgg16(pretrained=pretrain)
    elif net_name == 'vgg19':
        net = models.vgg19(pretrained=pretrain)
    elif net_name in ['resnet', 'resnet50']:
        net = models.resnet50(pretrained=pretrain)
    elif net_name == 'resnet101':
        net = models.resnet101(pretrained=pretrain)
    elif net_name in ['densenet', 'densenet121']:
        net = models.densenet121(pretrained=pretrain)
    elif net_name in ['inception']:
        net = models.inception_v3(pretrained=pretrain)
    elif net_name in ['mobilenet_v2']:
        net = models.mobilenet_v2(pretrained=pretrain)
    elif net_name in ['shufflenet_v2']:
        net = models.shufflenet_v2_x1_0(pretrained=pretrain)
    else:
        print(net_name)
        # raise ValueError('invalid network name:{}'.format(net_name))

    # 加载指定路径的权重参数
    if weight_path is not None and net_name.startswith('densenet'):
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(weight_path)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        net.load_state_dict(state_dict)

    # 自定义的模型
    elif weight_path is not None:
        net = torch.load(weight_path, map_location=torch.device('cpu'))  #
        # net.load_state_dict(torch.load(weight_path))
        # for (name, m) in net.named_modules():
        #     print(name)
        #     # module.EncoderList.2.layers.0.1.mlp_block.fn.fn.net.2
        #     # if name == 'attention.0.EncoderList.0.layers.0.0.mlp_block.fn.fn.net.2':  #
        #     #     target_layers = [m]
        #     #     print(1)
    return net


def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def prepare_input(image):
    image = image.copy()

    # 归一化
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    print('image3', image.shape)  # (224, 224, 4)
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = image[np.newaxis, ...]  # 增加batch维

    return torch.tensor(image, requires_grad=True)


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    mask=mask.transpose(1, 0) 
    #mask=np.flip(mask,axis=1)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    #heatmap = heatmap[..., ::-1]  # gbr to rgb
    #heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    print('mask', mask.shape)  # 64, 128
    print('生成cam图 heatmap', heatmap.shape)  # (64, 128, 3)
    # 合并heatmap到原始图像
    print('image2', image.shape)  # (64, 128, 3)
    # image=image.reshape(2,64,128,3)

    cam = heatmap  + np.float32(image)  # image[0].cpu().detach().numpy() #.transpose(1, 0, 2) #[::-1]

    return norm_image(cam), (heatmap * 255).astype(np.uint8)


def gen_camm(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    mask=mask.transpose(1, 0) 
    #mask=np.flip(mask,axis=1)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    print('生成cam图 heatmap1', heatmap.shape)

    #heatmap = heatmap[..., ::-1]  # gbr to rgb
    #heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    print('mask', mask.shape)  # 64, 128
    print('生成cam图 heatmap2', heatmap.shape)  # (64, 128, 3)
    # 合并heatmap到原始图像
    print('image4', image.shape)  # (64, 128, 3)
    # image=image.reshape(2,64,128,3)
    cam = heatmap + np.float32(image)  # image[0].cpu().detach().numpy()#.transpose(1, 0, 2) 
    # heatmap = np.float32(heatmap).transpose(1,0,2)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)


def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    print('grad', grad.shape)
    grad = grad.data.numpy()
    gb = np.transpose(grad)  # , (1, 2, 0)
    return gb


def save_image(image_dicts, input_image_name, network, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        width=image.shape[1]
        height=image.shape[0]
        image=cv2.resize(image,(width*2,height*2),interpolation=cv2.INTER_CUBIC) 
        # io.imsave
        cv2.imwrite(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        #cv2.imwrite(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


from torch.autograd import Variable
from dataloader.read_UEA import load_UEA


def main(args):
    # 输入
    img = cv2.imread(args.image_path)  # io
    print('image1', img.shape)  # (64, 128, 3)
    #img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
    img = np.float32(cv2.resize(img, (256, 128))) / 255  # 224 224 img.shape[0] #原始图像
    #inputs = prepare_input(img)  # 用于模型输入

    # 输出图像
    image_dict = {}
    # 网络
    net = get_net(args.network, args.weight_path)

    file_name = args.model_path.split('/')[-1].split(' ')[0]
    print('batch', args.batch_size)
    train_loader, test_loader, num_class = load_UEA(file_name, args)  #

    for j, (x, z, z2, ii, y) in enumerate(test_loader):
        # i=ii[0][0]
        # i=i.unsqueeze(dim=0)
        # i = i.unsqueeze(dim=0)#[1, 1, 3, 128, 64])

        i = ii
        print('fre img', i.shape)  # [1, 2, 3, 128, 64]

        x = Variable(x).float().to(args.device)
        z = Variable(z).float().to(args.device)
        z2 = Variable(z2).float().to(args.device)
        i = Variable(i).float().to(args.device)
        i = torch.tensor(i, requires_grad=True)
        y = Variable(y).to(args.device)
        print('图像i  ', i.shape)  # 1, 2, 3, 128, 64])

        # Grad-CAM
        print('========================================================')
        print('args.layer_name', args.layer_name)  # cbam_img.sa.sigmoid
        layer_name = get_last_conv_name(net) if args.layer_name is None else args.layer_name
        grad_cam = GradCAM(net, layer_name)
        mask = grad_cam(x, z, z2, i, args.class_id)  # cam mask ###inputs
        print('mask', mask.shape)  # (64, 128)
        image_dict['cam'], image_dict['heatmap'] = gen_cam(img, mask)  ##img
        grad_cam.remove_handlers()

        # Grad-CAM++
        print('===========================================================')
        grad_cam_plus_plus = GradCamPlusPlus(net, layer_name)
        mask_plus_plus = grad_cam_plus_plus(x, z, z2, i, args.class_id)  # cam mask inputs
        print('mask_plus_plus', mask_plus_plus.shape)  # mask_plus_plus (128, 64)
        image_dict['cam++'], image_dict['heatmap++'] = gen_camm(img, mask_plus_plus)
        grad_cam_plus_plus.remove_handlers()

        # # GuidedBackPropagation
        # gbp = GuidedBackPropagation(net)
        # i.grad.zero_()  # 梯度置零 inputs
        # grad = gbp(x, z, z2, i)  # inputs
        # gb = gen_gb(grad[0])  ##需要一张图片
        # image_dict['gb'] = norm_image(gb)
        #
        # # 生成Guided Grad-CAM
        # mask = mask[..., np.newaxis]
        # print('mask', mask.shape)
        # cam_gb = gb * mask  # .reshape(128,64,1)#
        # image_dict['cam_gb'] = norm_image(cam_gb)

        save_image(image_dict, os.path.basename(args.image_path), args.network, args.output_dir)
        input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='/root/MF-Net-1202/data/Multivariate_arff')
    parser.add_argument('--cache_path', type=str, default='/root/MF-Net-1202/cache')
    parser.add_argument('--model_path', type=str,
                        default=r'/root/saved_model/AtrialFibrillation/AtrialFibrillation batch=2 length=3072 time=2023-03-24-22_01_03.pkl')
    #/root/saved_model/AtrialFibrillation/AtrialFibrillation batch=2 length=3072 time=2023-03-24-20_45_44.pkl
    parser.add_argument('--device', type=str, default='cpu' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--network', type=str, default='MSCN',
                        help='ImageNet classification network')
    parser.add_argument('--image_path', type=str, default='/root/MSCN/examples/00002002.png',#pic1.jpg #00001001.png
                        help='input image path')
    parser.add_argument('--weight-path', type=str,
                        default='/root/saved_model/AtrialFibrillation/AtrialFibrillation batch=2 length=3072 time=2023-03-24-22_01_03.pkl',
                        help='weight path of the model')
    parser.add_argument('--layer-name', type=str, default='cbam_img.sa.sigmoid',
                        # cbam_img.sa.conv1 cbam_img.sa.sigmoid
                        help='last convolutional layer name')
    parser.add_argument('--class-id', type=int, default=None,
                        help='class id')
    parser.add_argument('--output-dir', type=str, default='/root/MSCN/results',
                        help='output directory to save results')
    arguments = parser.parse_args()

    main(arguments)
