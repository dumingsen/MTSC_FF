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
from models.layers import GraphAttentionLayer, SpGraphAttentionLayer
from models.mixhop import MixHopConv

# from torch_geometric.nn import GATConv




from models.layer_mixhop import mixprop
from model.attention import attentionNet


class Modelmuti(nn.Module):
    judg = 1

    def __init__(self, args, num_class, t=1,  # 长度
                 down_dim=1024,  # length = 1536 * 2，降维维度
                 hidden_dim=(96, 62),  ##192
                 layers1=(2, 2, 6, 2),
                 heads=(3, 6, 12, 24),
                 channels=1,
                 num_classes=1,
                 head_dim=32,
                 window_size=1,
                 downscaling_factors=(4, 2, 2, 2),  # 代表多长的时间作为一个特征
                 relative_pos_embedding=True,
                 wa=1,
                 prob=1,
                 mask=1, ):  # data
        super(Modelmuti, self).__init__()
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
        self.channels = channels
        self.conv1 = nn.Conv2d(1, args.channel_size, kernel_size=(1, args.k_size[0]), padding=(0, args.k_size[0] // 2),
                               stride=1)
        self.conv2 = nn.Conv2d(1, args.channel_size, kernel_size=(1, args.k_size[1]), padding=(0, args.k_size[1] // 2),
                               stride=1)
        self.conv3 = nn.Conv2d(1, args.channel_size, kernel_size=(1, args.k_size[2]), padding=(0, args.k_size[2] // 2),
                               stride=1)
        self.bn = nn.BatchNorm1d(self.channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(self.channels, 2, ceil_mode=True, padding=(0, args.k_size[0] // 2))
        ###fcn

        # self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # self.maxpool1 = nn.MaxPool2d(kernel_size = (1,args.k_size[0]),stride=1)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=(1, args.k_size[1]), stride=1)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=(1, args.k_size[2]), stride=1)
        # self.dropout = nn.Dropout(p=0.1)

        # d = (len(args.k_size) * (args.window) - sum(args.k_size) + len(args.k_size)) * args.channel_size
        d = 400
        conv_channels = 400
        residual_channels = 200
        gcn_depth = 2
        dropout = 0.3
        propalpha = 0.05

        # skip_mode = args.skip_mode
        self.BATCH_SIZE = args.batch_size
        self.dropout = 0.1

        in_dim = 400
        hid_dim = 200
        p = [0, 1, 2]
        inputdropout = 0.3
        activation = torch.tanh
        batchnorm = True
        if self.decoder == 'mixhop':
            # 1
            self.mixhop = nn.ModuleList()
            self.mixhop.append(
                MixHopConv(
                    in_dim,
                    hid_dim,
                    p,
                    inputdropout,
                    activation,
                    batchnorm,
                )
            )
        # self.swintransformer = attentionNet(
        #     t=t,  # 长度 t
        #     down_dim=down_dim,  # length = 1536 * 2，降维维度
        #     hidden_dim=hidden_dim,
        #     layers=layers1,
        #
        #     heads=heads,
        #     channels=channels,#channels
        #     num_classes=num_classes,
        #     head_dim=head_dim,
        #     window_size=window_size,
        #     downscaling_factors=downscaling_factors,  # 代表多长的时间作为一个特征
        #     wa=wa,
        #     prob=prob,
        #     mask=mask,
        #
        # )
        if self.decoder == 'graphattentionlayer':
            # 每次接受一个图 1
            self.GraphAttentionLayer = GraphAttentionLayer(d, args.hid1, 0.1, 0.1)

        if self.decoder == 'GCN':
            # 多
            self.gcn1 = DenseGCNConv(d, args.hid1)
            self.gcn2 = DenseGCNConv(args.hid1, args.hid2)
            self.gcn3 = DenseGCNConv(args.hid2, args.hid2)

        a = t * 12 * 3 
        b = t * 12
        if self.decoder == 'GNN':
            # 多
            # self.gnn0 = DenseGraphConv(d, h0)
            self.gnn1 = DenseGraphConv(b, b)

            self.gnn2 = DenseGraphConv(b, b//2)
            self.gnn3 = DenseGraphConv(b//2, b//2)

        if self.decoder == 'rGNN':
            #
            self.num_adjs = 1
            self.attention_mode = 'naive'
            self.gc1 = rkGraphConv(self.num_adjs, d, args.hid1, self.attention_mode, aggr='mean')
            self.gc2 = rkGraphConv(self.num_adjs, args.hid1, args.hid2, self.attention_mode, aggr='mean')
            self.gc3 = rkGraphConv(self.num_adjs, args.hid2, args.hid2, self.attention_mode, aggr='mean')

        # self.hw = args.highway_window
        # if (self.hw > 0):
        #     self.highway = nn.Linear(self.hw, 1)

        if self.decoder == 'SAGE':
            # 多
            self.sage1 = DenseSAGEConv(d, args.hid1)
            self.sage2 = DenseSAGEConv(args.hid1, args.hid2)
            self.sage3 = DenseSAGEConv(args.hid2, args.hid2)

        if self.decoder == 'GIN':
            ginnn = nn.Sequential(
                nn.Linear(d, args.hid1),
                # nn.ReLU(True),
                # nn.Linear(args.hid1, args.hid2),
                nn.ReLU(True),
                nn.Linear(args.hid1, args.hid2),
                nn.ReLU(True)
            )
            self.gin = DeGINConv(ginnn)
        # if self.decoder == 'GAT':
        # self.gatconv1 = GATConv(d, args.hid1)
        # self.gatconv2 = GATConv(args.hid1, args.hid2)
        # self.gatconv3 = GATConv(args.hid2, args.hid2)

        if self.decoder == 'mixhop1':
            # 1
            self.gconv1 = nn.ModuleList()
            self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
            self.gconv1.append(mixprop(residual_channels, residual_channels, gcn_depth, dropout, propalpha))

            self.gconv2 = nn.ModuleList()
            self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
            self.gconv2.append(mixprop(residual_channels, residual_channels, gcn_depth, dropout, propalpha))

        self.mlp_head = nn.Sequential(
            #nn.LayerNorm(t),  ##t
            nn.Linear(t, num_class)  # hidden_dim[-1] t
        )
        self.Transformer = Transformer(990, 1, 0.1)

        self.gru = GRU(1,channels *t)#(self.channels, t)

        self.gru1 = GRU(5, 990)

        self.end_linear = nn.Linear(a, t)
        self.mid_linear = nn.Linear(a, b)

    def skip_connect_out(self, x2, x1):
        return self.ff(torch.cat((x2, x1), 1)) if self.skip_mode == "concat" else x2 + x1

    def forward(self, x, x_graph):
        print('特征', x.shape)
        print('图', x_graph.shape)
        print('图', x_graph[0])
        print(type(x))
        print(type(x_graph))

       
# ==================================================================================
        # x = self.preprocess(x)
        x2 = x.reshape(x.size(0), self.channels, -1)  # 输入x torch.Size([8, 5, 55])
        print('输入x2', x2.shape)

        # print('self.A_new', self.A_new.shape)
        for i, A in enumerate(x_graph):
            # .cpu()
            A = A #/ torch.sum(A, 0)
            A = A.reshape(1, A.size(0), A.size(1))
            #  ###每个batch的图
            # self.A_new[i, :, :] = A
            if i == 0:
                A_new = A
            else:
                A_new = torch.cat((A_new, A), dim=0)
        # A_new = self.A_new
        A_new=A_new
        # A_new = map(A / torch.sum(A, 0), x_graph)
        # A_new = torch.from_numpy(A_new)  # .cuda()
        print(A_new.shape)
 # ==================================================================================


        # c = x.permute(0, 2, 1)
        c = x2.unsqueeze(1)
        print('c', c.shape)  # [1, 1, 5, 55]) [4, 1, 5, 55])
        # if self.decoder != 'GAT':
        a1 = self.bn(self.relu(self.conv1(c).permute(0, 2, 1, 3).reshape(x.size(0), self.channels, -1)))  # 多尺度
        print('a1', a1.shape)  # [8, 5, 318] [8, 5, 660]
        a2 = self.bn(self.relu(self.conv2(c).permute(0, 2, 1, 3).reshape(x.size(0), self.channels, -1)))
        a3 = self.bn(self.relu(self.conv3(c).permute(0, 2, 1, 3).reshape(x.size(0), self.channels, -1)))
        # a1 = self.dropout(a1)
        # a2 = self.dropout(a2)
        # a3 = self.dropout(a3)
        # x_conv = F.relu(torch.cat([a1,a2],2))
        x_conv = F.relu(torch.cat([a1, a2, a3], 2))
        print('多尺度x_conv', x_conv.shape)  # [1, 5, 1836] [8, 5, 918] 8, 5, 1980
        # x_conv=F.relu(torch.cat([a1,a2,a3,a4,a5],2))
        # print(x_conv.shape)

        # x_conv = self.pool(x_conv)

        # x_conv = self.gru1(x_conv)
        # x_conv = self.Transformer(x_conv)
        # print(x_conv.shape)#[8, 5, 400])
        x_conv = self.mid_linear(x_conv)

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

            x1 = F.relu(self.gnn1(x_conv, A_new))  # self.A
            print('x1', x1.shape)  # [1, 5, 40])
            x2 = F.relu(self.gnn2(x1, A_new))
            print('x2', x2.shape)  # 1, 5, 10]) 8, 5, 100
            x3 = self.gnn3(x2, A_new)
            print('x3', x3.shape)  # [1, 5, 1])
            x3 = x3.squeeze(dim=1)
            #x1 torch.Size([8, 6, 828])
            # x2 torch.Size([8, 6, 414])
            # x3 torch.Size([8, 6, 414])

        if self.decoder == 'mixhop1':
            x3 = self.gconv1[0](x_conv, A_new) + self.gconv2[0](x_conv, A_new.transpose(1, 0))

        if self.decoder == 'mixhop':
            x3 = self.mixhop[0](A_new, x_conv)

        if self.decoder == 'graphattentionlayer':
            x1 = F.relu(self.GraphAttentionLayer(x_conv, A_new))
            x2 = F.relu(self.GraphAttentionLayer(x1, A_new))
            x3 = self.GraphAttentionLayer(x2, A_new)
            x3 = x3.squeeze()

        adjs = A_new
        if self.decoder == 'rGNN':
            x1 = F.relu(self.gc1(x_conv, adjs))
            # x1 = F.dropout(x1, self.dropout)
            x2 = F.relu(self.gc2(x1, adjs))
            x3 = F.relu(self.gc3(x2, adjs))
            # x3 = F.dropout(x2, self.dropout)
            x3 = x3.squeeze()

        if self.decoder == 'SAGE':
            x1 = F.relu(self.sage1(x_conv, A_new))
            x2 = F.relu(self.sage2(x1, A_new))
            x3 = F.relu(self.sage3(x2, A_new))
            x3 = x3.squeeze()

        if self.decoder == 'GIN':
            x3 = F.relu(self.gin(x_conv, A_new))
            x3 = x3.squeeze()

        if self.decoder == 'GAT':
            x1 = F.relu(self.gatconv1(x_conv, self.edge_index))
            x2 = F.relu(self.gatconv2(x1, self.edge_index))
            x3 = F.relu(self.gatconv3(x2, self.edge_index))
            x3 = x3.squeeze()

        # x3=self.Transformer(x3) 效果不好
        x3=self.end_linear(x_conv.to(torch.float32))
        #x3 = self.gru(x3)
#================================================================================================
        # x=x.reshape(x.size(0), 1,x.size(1))
        # print(x.shape)
        _,x3=self.swintransformer(x3)#8, 5, 100
        #x3 = self.gru(x)

        # x33=x33.transpose(1,2)
        print('x3', x3.shape)  # [8, 5, 10]) [8, 5, 200]
        
        #fl1=x3.reshape(x3.shape[0],-1)
        fl1 = x3.mean(dim=[1])
        print('mean fl', fl1.shape)
        fl = self.mlp_head(fl1.float())

        print('fl', fl.shape)
        return fl


def print(*args, **kwargs):
    flag = False
    # flag = False
    if flag:
        print(*args, **kwargs)
    else:
        pass