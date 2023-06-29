import os

import numpy as np
import matplotlib.pyplot as plt
import pywt


def get_freq(x1, i, archive_name, train):
    print(train)
    sampling_rate =len(x1) #1024  # 采样频率
    print('x1',x1.shape)
    print('sampling_rate',sampling_rate)
    x1=x1.transpose(1,0)
    print('x1',x1.shape)
    a=1
    for x in x1:
        t = np.arange(0, 1.0, 1.0 / sampling_rate)  # 0-1.0之间的数，步长为1.0/sampling_rate
        f1 = 100  # 频率
        f2 = 200
        f3 = 300
        data = x
        # 分段函数
        # data = np.piecewise(t,[t<1,t<0.8,t<0.3],
        #                     [lambda t : np.sin(2 * np.pi * f1 * t),
        #                      lambda t : np.sin(2 * np.pi * f2 * t),
        #                      lambda t : np.sin(2 * np.pi * f3 * t)])
        wavename = "cgau8"  # 小波函数
        totalscal = 256  # totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
        fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
        cparam = 2 * fc * totalscal  # 常数c
        scales = cparam / np.arange(totalscal, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
        [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)  # 连续小波变换模块

        # plt.figure(figsize=(8, 4))
        # plt.subplot(211)  # 第一整行
        # plt.plot(t, data)
        # plt.xlabel(u"time(s)")
        # plt.title(u"300Hz 200Hz 100Hz Time spectrum")
        # plt.subplot(212)  # 第二整行

        #
        plt.figure(figsize=(16, 8))
        plt.contourf(t, frequencies, abs(cwtmatr))
        plt.axis('off')
        # plt.ylabel(u"freq(Hz)")
        # plt.xlabel(u"time(s)")
        plt.subplots_adjust(hspace=0.4)  # 调整边距和子图的间距 hspace为子图之间的空间保留的高度，平均轴高度的一部分


        if train == True:
            #dir_name2 = './freq_image/{0}/Train/Sample{1}'.format(archive_name,i)
            # dir_name2 = './freq_image/{0}/Train'.format(archive_name)
            # if not os.path.isdir(dir_name2):
            #     os.makedirs(dir_name2)

            i = str(i).zfill(4)
            a=str(a).zfill(3)
            #plt.savefig('./freq_image/{0}/Train/Sample{1}_Dim_{2}.png'.format(archive_name,i, a),dpi=8)
            plt.savefig('./freq_image/{0}/Train/{1}{2}.png'.format(archive_name, i, a), dpi=8)
        else:
            #dir_name2 = './freq_image/{0}/Test/Sample{1}'.format(archive_name,i)
            # dir_name2 = './freq_image/{0}/Test'.format(archive_name)
            # if not os.path.isdir(dir_name2):
            #     os.makedirs(dir_name2)
            i = str(i).zfill(4)
            a = str(a).zfill(3)
            plt.savefig('./freq_image/{0}/Test/{1}{2}.png'.format(archive_name, i,a), dpi=8)
        a = int(a) + 1

        # plt.show()
        plt.close()
