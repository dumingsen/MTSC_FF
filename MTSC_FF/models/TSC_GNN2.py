import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from layers import GraphAttentionLayer, SpGraphAttentionLayer
# from torch_geometric.nn import GCNConv
from models.layer import DeGINConv, DenseGCNConv, DenseSAGEConv, DenseGraphConv, rkGraphConv
# from torch_geometric.nn import GATConv
from process import preprocess
# from Teoriginal import getTEgraph
from pearsonr import getTEgraph
from models.trans import Transformer
from models.gru import GRU
from model.layer import printtt


def print(*args, **kwargs):
    flag = False
    # flag = False
    if flag:
        print(*args, **kwargs)
    else:
        pass


class Model(nn.Module):
    judg = 1

    def __init__(self, args, num_class, judge):  # data
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        # ==================================================================================
        self.preprocess = preprocess(55, 55, 64)
        # self.getTe = getTEgraph()

        # =======================================================================================
        # A = np.loadtxt(args.A)
        # A = np.array(A, dtype=np.float32)
        # # A = np.ones((args.n_e,args.n_e),np.int8)
        # # A[A>0.05] = 1
        # A = A / np.sum(A, 0)
        self.A_new = np.zeros((args.batch_size, args.n_e, args.n_e), dtype=np.float32)
        self.A_new1 = np.zeros((4, args.n_e, args.n_e), dtype=np.float32)  #
        # for i in range(args.batch_size):  ###每个batch的图
        #     A_new[i, :, :] = A

        # self.A = torch.from_numpy(A_new).cuda()
        # ==================================================================================
        # self.adjs = [self.A]
        # self.num_adjs = args.num_adj
        # if self.num_adjs > 1:
        #     A = np.loadtxt(args.B)
        #     A = np.array(A, dtype=np.float32)
        #     # A = np.ones((args.n_e,args.n_e),np.int8)
        #     # A[A>0.05] = 1
        #     A = A / np.sum(A, 1)
        #     A_new = np.zeros((args.batch_size, args.n_e, args.n_e), dtype=np.float32)
        #     for i in range(args.batch_size):
        #         A_new[i, :, :] = A
        #
        #     self.B = torch.from_numpy(A_new).cuda()
        #     A = np.ones((args.n_e, args.n_e), np.int8)
        #     A = A / np.sum(A, 1)
        #     A_new = np.zeros((args.batch_size, args.n_e, args.n_e), dtype=np.float32)
        #     for i in range(args.batch_size):
        #         A_new[i, :, :] = A
        #     self.C = torch.from_numpy(A_new).cuda()
        #     self.adjs = [self.A, self.B, self.C]
        # ================================================================================================================
        # self.A = torch.from_numpy(A_new)
        self.n_e = args.n_e
        self.decoder = args.decoder
        # self.attention_mode = args.attention_mode
        # if self.decoder != 'GAT':
        ##The hyper-parameters are applied to all datasets in all horizons

        self.conv1 = nn.Conv2d(1, args.channel_size, kernel_size=(1, args.k_size[0]), padding=(0, args.k_size[0] // 2),
                               stride=1)
        self.conv2 = nn.Conv2d(1, args.channel_size, kernel_size=(1, args.k_size[1]), padding=(0, args.k_size[1] // 2),
                               stride=1)
        self.conv3 = nn.Conv2d(1, args.channel_size, kernel_size=(1, args.k_size[2]), padding=(0, args.k_size[2] // 2),
                               stride=1)
        self.bn = nn.BatchNorm1d(5)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(5, 2, ceil_mode=True,padding=1)
        ###fcn
        self.conv1_1 = nn.Conv2d(1, args.channel_size, kernel_size=(1, args.k_size[0]), padding=(0, args.k_size[0] // 2),
                               stride=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(1, args.channel_size, kernel_size=(1, args.k_size[1]), padding=(0, args.k_size[1] // 2),
                               stride=1)
        self.relu1_2 = nn.ReLU(inplace=True)

        #self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # self.maxpool1 = nn.MaxPool2d(kernel_size = (1,args.k_size[0]),stride=1)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=(1, args.k_size[1]), stride=1)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=(1, args.k_size[2]), stride=1)
        # self.dropout = nn.Dropout(p=0.1)

        # d = (len(args.k_size) * (args.window) - sum(args.k_size) + len(args.k_size)) * args.channel_size

        # skip_mode = args.skip_mode
        self.BATCH_SIZE = args.batch_size
        self.dropout = 0.1
        # if self.decoder == 'GCN':
        #     self.gcn1 = DenseGCNConv(d, args.hid1)#10
        #     self.gcn2 = DenseGCNConv(args.hid1, args.hid2)#40 10
        #     self.gcn3 = DenseGCNConv(args.hid2, 1)

        self.judge = 1

        if self.decoder == 'GNN':
            # self.gnn0 = DenseGraphConv(d, h0)
            d = 990
            self.gnn1 = DenseGraphConv(d, 400)  # 40
            # self.gnn1 = DenseGraphConv(d, args.hid1)#40
            self.gnn2 = DenseGraphConv(400, 200)  # 40 10
            self.gnn3 = DenseGraphConv(200, 200)

            # n = 990
            # self.gnn11 = DenseGraphConv(d, 400)  # 40
            # # self.gnn1 = DenseGraphConv(d, args.hid1)#40
            # self.gnn22 = DenseGraphConv(400, 200)  # 40 10
            # self.gnn33 = DenseGraphConv(200, 200)
            # # self.gnn11 = DenseGraphConv(n, args.hid1)  # 40
            # # # self.gnn1 = DenseGraphConv(d, args.hid1)#40
            # # self.gnn22 = DenseGraphConv(args.hid1, args.hid2)  # 40 10
            # # self.gnn33 = DenseGraphConv(args.hid2, args.hid2)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(200),  ##62,normalized_shape：归一化的维度，int（最后一维）list（list里面的维度）#hidden_dim[-1]
            nn.Linear(200, num_class)  # hidden_dim[-1]
        )
        self.Transformer = Transformer(990, 1, 0.1)
        self.gru = GRU(5, 200)
        self.gru1 = GRU(5, 990)

    def skip_connect_out(self, x2, x1):
        return self.ff(torch.cat((x2, x1), 1)) if self.skip_mode == "concat" else x2 + x1

    def forward(self, x, x_graph):
        print('特征', x.shape)
        print('图', x_graph.shape)
        print('图',x_graph[0])

        # ==================================================================================
        # x = self.preprocess(x)
        x2 = x.reshape(x.size(0), 5, -1)  # 输入x torch.Size([8, 5, 55])
        print('输入x2', x2.shape)

        # print('self.A_new', self.A_new.shape)
        if x.size(0) == 8:
            for i, A in enumerate(x_graph):
                # .cpu()
                A = A / torch.sum(A, 0)
                #  ###每个batch的图
                self.A_new[i, :, :] = A
            A_new = self.A_new
            A_new = torch.from_numpy(A_new)  # .cuda()

        if x.size(0) == 4:
            for i, A in enumerate(x_graph):
                A = A / torch.sum(A, 0)
                #  ###每个batch的图
                self.A_new1[i, :, :] = A
            A_new = self.A_new1
            A_new = torch.from_numpy(A_new)  # .cuda()
        # ==================================================================================
        # c = x.permute(0, 2, 1)
        c = x2.unsqueeze(1)
        print('c', c.shape)  # [1, 1, 5, 55]) [4, 1, 5, 55])
        # if self.decoder != 'GAT':
        a1 = self.bn(self.relu(self.conv1(c).permute(0, 2, 1, 3).reshape(x.size(0), self.n_e, -1)))  # 多尺度
        print('a1', a1.shape)  # [8, 5, 318] [8, 5, 660]
        a2 = self.bn(self.relu(self.conv2(c).permute(0, 2, 1, 3).reshape(x.size(0), self.n_e, -1)))
        a3 = self.bn(self.relu(self.conv3(c).permute(0, 2, 1, 3).reshape(x.size(0), self.n_e, -1)))
        # a1 = self.dropout(a1)
        # a2 = self.dropout(a2)
        # a3 = self.dropout(a3)
        # x_conv = F.relu(torch.cat([a1,a2],2))
        x_conv = F.relu(torch.cat([a1, a2, a3], 2))
        print('多尺度x_conv', x_conv.shape)  # [1, 5, 1836] [8, 5, 918] 8, 5, 1980
        # x_conv=F.relu(torch.cat([a1,a2,a3,a4,a5],2))
        # print(x_conv.shape)

        x_conv=self.pool(x_conv)
        #x_conv = self.gru1(x_conv)
        #x_conv = self.Transformer(x_conv)
       # printtt(x_conv.shape)#[8, 5, 400])

        ##GCN1
        # x1=F.relu(torch.bmm(self.A,x_conv).bmm(self.w1))
        # x2=F.relu(torch.bmm(self.A,x1).bmm(self.w2))
        # x3=F.relu(torch.bmm(self.A,x2).bmm(self.w3))

        if self.decoder == 'GCN':
            # x1 = F.relu(self.gcn1(x_conv,self.A))
            x1 = F.relu(self.gcn1(x_conv, A_new))  # self.A
            x2 = F.relu(self.gcn2(x1, A_new))
            x3 = self.gcn3(x2, A_new)
            x3 = x3.squeeze()
        if x.size(0) == 8:
            if self.decoder == 'GNN':
                # x0 = F.relu(self.gnn0(x_conv,self.A))

                print('judge', self.judge)
                x11 = F.relu(self.gnn1(x_conv, A_new))  # self.A
                print('x11', x11.shape)  # [1, 5, 40])
                x22 = F.relu(self.gnn2(x11, A_new))
                print('x22', x22.shape)  # 1, 5, 10]) 8, 5, 100
                x33 = self.gnn3(x22, A_new)
                print('x33', x33.shape)  # [1, 5, 1])
                x33 = x33.squeeze(dim=1)
        if x.size(0) == 4:
            if self.decoder == 'GNN':
                # x0 = F.relu(self.gnn0(x_conv,self.A))

                print('judge', self.judge)
                x11 = F.relu(self.gnn11(x_conv, A_new))  # self.A
                print('x11', x11.shape)  # [1, 5, 40])
                x22 = F.relu(self.gnn22(x11, A_new))
                print('x22', x22.shape)  # 1, 5, 10])
                x33 = self.gnn33(x22, A_new)
                print('x33', x33.shape)  # [1, 5, 1])   8, 5, 100])
                x33 = x33.squeeze(dim=1)

        #x33=self.Transformer(x33) 效果不好

        x33 = self.gru(x33)
        # x33=x33.transpose(1,2)
        print('x33', x33.shape)#[8, 5, 10]) [8, 5, 200]
        #return
        fl1 = x33.mean(dim=[1])
        print('mean fl', fl1.shape)
        fl = self.mlp_head(fl1)

        print('fl', fl.shape)
        return fl
