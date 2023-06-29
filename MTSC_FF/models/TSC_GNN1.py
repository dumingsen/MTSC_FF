import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from layers import GraphAttentionLayer, SpGraphAttentionLayer
# from torch_geometric.nn import GCNConv
from models.layer import DeGINConv, DenseGCNConv, DenseSAGEConv, DenseGraphConv, rkGraphConv
# from torch_geometric.nn import GATConv
from process import preprocess
from Teoriginal import getTEgraph
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

class Model(nn.Module):

    def __init__(self, args,num_class):  # data
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        # ==================================================================================
        self.preprocess = preprocess(55, 55, 64)
        self.getTe = getTEgraph()

        # =======================================================================================
        # A = np.loadtxt(args.A)
        # A = np.array(A, dtype=np.float32)
        # # A = np.ones((args.n_e,args.n_e),np.int8)
        # # A[A>0.05] = 1
        # A = A / np.sum(A, 0)
        self.A_new = np.zeros((args.batch_size, args.n_e, args.n_e), dtype=np.float32)  #
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
        self.conv1 = nn.Conv2d(1, args.channel_size, kernel_size=(1, args.k_size[0]), stride=1)
        self.conv2 = nn.Conv2d(1, args.channel_size, kernel_size=(1, args.k_size[1]), stride=1)
        self.conv3 = nn.Conv2d(1, args.channel_size, kernel_size=(1, args.k_size[2]), stride=1)

        # self.maxpool1 = nn.MaxPool2d(kernel_size = (1,args.k_size[0]),stride=1)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=(1, args.k_size[1]), stride=1)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=(1, args.k_size[2]), stride=1)
        # self.dropout = nn.Dropout(p=0.1)

        #d = (len(args.k_size) * (args.window) - sum(args.k_size) + len(args.k_size)) * args.channel_size
        d=1836

        # skip_mode = args.skip_mode
        self.BATCH_SIZE = args.batch_size
        self.dropout = 0.1
        if self.decoder == 'GCN':
            self.gcn1 = DenseGCNConv(d, args.hid1)#10
            self.gcn2 = DenseGCNConv(args.hid1, args.hid2)#40 10
            self.gcn3 = DenseGCNConv(args.hid2, 1)

        if self.decoder == 'GNN':
            # self.gnn0 = DenseGraphConv(d, h0)
            self.gnn1 = DenseGraphConv(d, args.hid1)#40
            self.gnn2 = DenseGraphConv(args.hid1, args.hid2)#40 10
            self.gnn3 = DenseGraphConv(args.hid2, args.hid2)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(args.hid2),  ##62,normalized_shape：归一化的维度，int（最后一维）list（list里面的维度）#hidden_dim[-1]
            nn.Linear(args.hid2, num_class)  # hidden_dim[-1]
        )
    def skip_connect_out(self, x2, x1):
        return self.ff(torch.cat((x2, x1), 1)) if self.skip_mode == "concat" else x2 + x1

    def forward(self, x):

        # ==================================================================================
        # x = self.preprocess(x)
        x2 = x.reshape(self.BATCH_SIZE, 5, -1)  # 输入x torch.Size([8, 5, 55])
        print('输入x2', x2.shape)

        print('self.A_new', self.A_new.shape)
        for i, x1 in enumerate(x2):
            A = self.getTe(x1)
            A = A / np.sum(A, 0)
            #  ###每个batch的图
            self.A_new[i, :, :] = A
        A_new = self.A_new
        A_new = torch.from_numpy(A_new)#.cuda()

        # ==================================================================================
        # c = x.permute(0, 2, 1)
        c = x2.unsqueeze(1)
        print('c',c.shape)#[1, 1, 5, 55])
        # if self.decoder != 'GAT':
        a1 = self.conv1(c).permute(0, 2, 1, 3).reshape(self.BATCH_SIZE, self.n_e, -1)#多尺度
        print('a1',a1.shape)
        a2 = self.conv2(c).permute(0, 2, 1, 3).reshape(self.BATCH_SIZE, self.n_e, -1)
        a3 = self.conv3(c).permute(0, 2, 1, 3).reshape(self.BATCH_SIZE, self.n_e, -1)
        # a1 = self.dropout(a1)
        # a2 = self.dropout(a2)
        # a3 = self.dropout(a3)
        # x_conv = F.relu(torch.cat([a1,a2],2))
        x_conv = F.relu(torch.cat([a1, a2, a3], 2))
        print('多尺度x_conv',x_conv.shape)#[1, 5, 1836]
        # x_conv=F.relu(torch.cat([a1,a2,a3,a4,a5],2))
        # print(x_conv.shape)

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

        if self.decoder == 'GNN':
            # x0 = F.relu(self.gnn0(x_conv,self.A))
            x11 = F.relu(self.gnn1(x_conv, A_new))  # self.A
            print('x11',x11.shape)#[1, 5, 40])
            x22 = F.relu(self.gnn2(x11, A_new))
            print('x22', x22.shape)#1, 5, 10])
            x33 = self.gnn3(x22, A_new)
            print('x33', x33.shape)#[1, 5, 1])
            x33 = x33.squeeze(dim=1)
#
        fl1 = x33.mean(dim=[1])
        print('mean fl', fl1.shape)#1, 1]
        fl = self.mlp_head(fl1)

        print('fl', fl.shape)#[1, 4
        return fl
