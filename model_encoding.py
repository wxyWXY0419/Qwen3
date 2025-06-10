import torch
"""
This module implements Graph Convolutional Networks (GCN) and Fast Graph Transformer Networks (FastGTNs) for graph-based learning tasks.
Classes:
    - GCN: A simple Graph Convolutional Network with two layers.
    - FastGTNs: A multi-layer Fast Graph Transformer Network.
    - FastGTN: A single layer of Fast Graph Transformer Network.
    - FastGTLayer: A layer in the Fast Graph Transformer Network.
    - FastGTConv: A convolution operation in the Fast Graph Transformer Network.
GCN:
    Methods:
        - __init__(self, in_channels, out_channels): Initializes the GCN with input and output channels.
        - forward(self, x, edge_index): Forward pass of the GCN.
FastGTNs:
    Methods:
        - __init__(self, num_edge_type, w_in, num_class, num_nodes, args=None): Initializes the FastGTNs with various parameters.
        - forward(self, A, X, target_x, target, num_nodes=None, eval=False, args=None, n_id=None, node_labels=None, epoch=None): Forward pass of the FastGTNs.
FastGTN:
    Methods:
        - __init__(self, num_edge_type, w_in, num_class, num_nodes, args=None, pre_trained=None): Initializes a single FastGTN layer.
        - forward(self, A, X, num_nodes, eval=False, node_labels=None, epoch=None): Forward pass of the FastGTN layer.
FastGTLayer:
    Methods:
        - __init__(self, in_channels, out_channels, num_nodes, first=True, args=None, pre_trained=None): Initializes a FastGTLayer.
        - forward(self, H_, A, A_, num_nodes, epoch=None, layer=None): Forward pass of the FastGTLayer.
FastGTConv:
    Methods:
        - __init__(self, in_channels, out_channels, num_nodes, args=None, pre_trained=None): Initializes a FastGTConv.
        - reset_parameters(self): Resets the parameters of the convolution.
        - forward(self, A, num_nodes, epoch=None, layer=None): Forward pass of the FastGTConv.
Parameters:
    - in_channels: Number of input features per node.
    - out_channels: Number of output classes.
    - x: Node feature matrix with shape [num_nodes, in_channels].
    - edge_index: Graph connectivity in COO format with shape [2, num_edges].
    - num_edge_type: Number of edge types in the graph.
    - w_in: Input feature dimension.
    - num_class: Number of output classes.
    - num_nodes: Number of nodes in the graph.
    - args: Additional arguments for the model.
    - A: Adjacency matrix.
    - X: Node feature matrix.
    - target_x: Target node indices.
    - target: Target labels.
    - eval: Evaluation mode flag.
    - n_id: Node IDs.
    - node_labels: Node labels.
    - epoch: Current epoch number.
    - pre_trained: Pre-trained model weights.
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from gcn import GCNConv
import torch_sparse
from torch_geometric.utils import softmax
from utils import _norm, generate_non_local_graph
from model_GTN import FastGTNs

from torch_geometric.nn import GCNConv as GCNNomal


device = f'cuda' if torch.cuda.is_available() else 'cpu'


class EncodingNet(nn.Module):
    def __init__(self, num_edge_type, w_in, num_class, num_nodes, embedding_w,args=None):
        super(EncodingNet, self).__init__()
        self.args = args
        self.num_nodes = num_nodes
        self.num_FastGTN_layers = args.num_FastGTN_layers
        self.num_channels = args.num_channels

        self.gtn_w_out=args.node_dim

        fastGTNs = []
        for i in range(args.num_FastGTN_layers):
            if i == 0:
                fastGTNs.append(FastGTN(num_edge_type, w_in, num_class, num_nodes, args))
            else:
                fastGTNs.append(FastGTN(args.num_channels, self.gtn_w_out, num_class, num_nodes, args))

        self.fastGTNs = nn.ModuleList(fastGTNs)
        self.linear = nn.Linear(self.gtn_w_out, num_class)
        gcns=[]
        for i in range(args.num_FastGTN_layers):
            gcns.append(GCNLayer(self.gtn_w_out,self.gtn_w_out))
        self.gcns=nn.ModuleList(gcns)
        self.token_emdedding_linear = nn.Linear(self.gtn_w_out, embedding_w)
        self.loss = nn.CrossEntropyLoss()
        if args.dataset == "PPI":
            self.m = nn.Sigmoid()
            self.loss = nn.BCELoss()
        else:    
            self.loss = nn.CrossEntropyLoss()
        

    # GTN+GCN 堆叠多个
    #  +线性层
    def forward(self, A, X, target_x, target, num_nodes=None, eval=False, args=None, n_id=None, node_labels=None, epoch=None):
        if num_nodes == None:
            num_nodes = self.num_nodes
        # print('Len-A1:',len(A))
        # print('A1:',A[0])
        H_, Ws ,A_= self.fastGTNs[0](A, X, num_nodes=num_nodes, epoch=epoch)
        H_=self.gcns[0](A_,H_,num_nodes)

        # print('HHHHHH')
        # print('H:',len(H_),len(H_[0]))
        # print('H_:',H_)
        # print('A2:',A_)
        # print(self.num_FastGTN_layers)
        for i in range(1, self.num_FastGTN_layers):
            H_, Ws ,A_= self.fastGTNs[i](A_, H_, num_nodes=num_nodes)
            H_=self.gcns[i](A_,H_,num_nodes)
            # print('A',i+3,':',A_)
        
        token_embed=self.token_emdedding_linear(H_)
        # print('token:',token_embed)
        # print('A:',A_)
        y = self.linear(H_[target_x])
        if eval:
            return y
        else:
            if self.args.dataset == 'PPI':
                loss = self.loss(self.m(y), target)
            else:
                loss = self.loss(y, target.squeeze())
        # 训练时将额外返回loss,权重,token表征
        return loss, y, Ws,token_embed



class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNNomal(in_channels, 16)
        self.conv2 = GCNNomal(16, out_channels)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)
    
class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer,self).__init__()
        self.gcn=GCN(in_channels, out_channels)

    def forward(self,A,x,num_nodes):
        for i, (edge_index,edge_value) in enumerate(A):
            if i == 0:
                total_edge_index = edge_index
                total_edge_value = edge_value
            else:
                total_edge_index = torch.cat((total_edge_index, edge_index), dim=1)
                total_edge_value = torch.cat((total_edge_value, edge_value))
            
        index, value = torch_sparse.coalesce(total_edge_index.detach(), total_edge_value, m=num_nodes, n=num_nodes, op='add')
        output=self.gcn(x,index,value)
        return output
            

# Example usage:
# in_channels: Number of input features per node
# out_channels: Number of output classes
# x: Node feature matrix with shape [num_nodes, in_channels]
# edge_index: Graph connectivity in COO format with shape [2, num_edges]


# model = GCN(in_channels=dataset.num_features, out_channels=dataset.num_classes)
# out = model(data.x, data.edge_index)
class FastGTNs(nn.Module):
    def __init__(self, num_edge_type, w_in, num_class, num_nodes, args=None):
        super(FastGTNs, self).__init__()
        self.args = args
        self.num_nodes = num_nodes
        self.num_FastGTN_layers = args.num_FastGTN_layers
        self.num_channels = args.num_channels
        fastGTNs = []
        for i in range(args.num_FastGTN_layers):
            if i == 0:
                fastGTNs.append(FastGTN(num_edge_type, w_in, num_class, num_nodes, args))
            else:
                fastGTNs.append(FastGTN(args.num_channels, args.node_dim, num_class, num_nodes, args))
        self.fastGTNs = nn.ModuleList(fastGTNs)
        self.linear = nn.Linear(args.node_dim, num_class)
        self.loss = nn.CrossEntropyLoss()
        if args.dataset == "PPI":
            self.m = nn.Sigmoid()
            self.loss = nn.BCELoss()
        else:    
            self.loss = nn.CrossEntropyLoss()
        
    def forward(self, A, X, target_x, target, num_nodes=None, eval=False, args=None, n_id=None, node_labels=None, epoch=None):
        if num_nodes == None:
            num_nodes = self.num_nodes
        # print('Len-A1:',len(A))
        # print('A1:',A[0])
        H_, Ws ,A_= self.fastGTNs[0](A, X, num_nodes=num_nodes, epoch=epoch)
        # print('H_:',H_)
        print('H:',len(H_),len(H_[0]))
        print('A2:',A_)
        print(self.num_FastGTN_layers)
        for i in range(1, self.num_FastGTN_layers):
            H_, Ws ,A_= self.fastGTNs[i](A_, H_, num_nodes=num_nodes)
            print('A',i+3,':',A_)
        y = self.linear(H_[target_x])
        if eval:
            return y
        else:
            if self.args.dataset == 'PPI':
                loss = self.loss(self.m(y), target)
            else:
                loss = self.loss(y, target.squeeze())
        return loss, y, Ws



class FastGTN(nn.Module):
    def __init__(self, num_edge_type, w_in, num_class, num_nodes, args=None, pre_trained=None):
        super(FastGTN, self).__init__()
        if args.non_local:
            num_edge_type += 1
        self.num_edge_type = num_edge_type
        self.num_channels = args.num_channels
        self.num_nodes = num_nodes
        self.w_in = w_in
        args.w_in = w_in
        self.w_out = args.node_dim
        self.num_class = num_class
        self.num_layers = args.num_layers
        # print('in:',self.w_in)
        # print('out:',self.w_out)
        
        if pre_trained is None:
            layers = []
            for i in range(self.num_layers):
                if i == 0:
                    layers.append(FastGTLayer(num_edge_type, self.num_channels, num_nodes, first=True, args=args))
                else:
                    layers.append(FastGTLayer(num_edge_type, self.num_channels, num_nodes, first=False, args=args))
            self.layers = nn.ModuleList(layers)
        else:
            layers = []
            for i in range(self.num_layers):
                if i == 0:
                    layers.append(FastGTLayer(num_edge_type, self.num_channels, num_nodes, first=True, args=args, pre_trained=pre_trained[i]))
                else:
                    layers.append(FastGTLayer(num_edge_type, self.num_channels, num_nodes, first=False, args=args, pre_trained=pre_trained[i]))
            self.layers = nn.ModuleList(layers)
        
        self.Ws = []
        for i in range(self.num_channels):
            self.Ws.append(GCNConv(in_channels=self.w_in, out_channels=self.w_out).weight)
        self.Ws = nn.ParameterList(self.Ws)

        self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)

        feat_trans_layers = []
        for i in range(self.num_layers+1):
            feat_trans_layers.append(nn.Sequential(nn.Linear(self.w_out, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 64)))
        self.feat_trans_layers = nn.ModuleList(feat_trans_layers)

        self.args = args

        self.out_norm = nn.LayerNorm(self.w_out)
        self.relu = torch.nn.ReLU()



        self.A_= []
        # 定义矩阵的大小
        n = num_nodes  # 你可以根据需要修改 n 的值
        for i in range(self.num_channels):
            self.A_.append(torch.eye(n).to(device))
        # print("A_",self.num_channels)
        # print("A_",self.A_)



    def forward(self, A, X, num_nodes, eval=False, node_labels=None, epoch=None):        
        Ws = []
        X_ = [X@W for W in self.Ws]
        H = [X@W for W in self.Ws]
        # print('H:',H)
        
        A_=self.A_
        # print('self.num_layers:',self.num_layers)
        for i in range(self.num_layers):
            if self.args.non_local:
                g = generate_non_local_graph(self.args, self.feat_trans_layers[i], torch.stack(H).mean(dim=0), A, self.num_edge_type, num_nodes)
                deg_inv_sqrt, deg_row, deg_col = _norm(g[0].detach(), num_nodes, g[1])
                g[1] = softmax(g[1],deg_row)
                if len(A) < self.num_edge_type:
                    A.append(g)
                else:
                    A[-1] = g
            
            H, W ,A_= self.layers[i](H, A, A_,num_nodes, epoch=epoch, layer=i+1)
            Ws.append(W)
        
        for i in range(self.num_channels):
            if i==0:
                H_ = F.relu(self.args.beta * (X_[i]) + (1-self.args.beta) * H[i])
            else:
                if self.args.channel_agg == 'concat':
                    H_ = torch.cat((H_,F.relu(self.args.beta * (X_[i]) + (1-self.args.beta) * H[i])), dim=1)
                elif self.args.channel_agg == 'mean':
                    H_ = H_ + F.relu(self.args.beta * (X_[i]) + (1-self.args.beta) * H[i])
        if self.args.channel_agg == 'concat':
            H_ = F.relu(self.linear1(H_))
        elif self.args.channel_agg == 'mean':
            H_ = H_ /self.args.num_channels
        
        
        for i in range(len(A_)):
            # 获取非零元素的索引
            edge_index = torch.nonzero(A_[i], as_tuple=False).t()

            # 获取非零元素的值
            edge_weight = A_[i][edge_index[0], edge_index[1]]
            A_[i]=(edge_index,edge_weight)
        # print('A_:#############3',A_)
        return H_, Ws,A_

class FastGTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_nodes, first=True, args=None, pre_trained=None):
        super(FastGTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        self.num_nodes = num_nodes
        if pre_trained is not None:
            self.conv1 = FastGTConv(in_channels, out_channels, num_nodes, args=args, pre_trained=pre_trained.conv1)
        else:
            self.conv1 = FastGTConv(in_channels, out_channels, num_nodes, args=args)
        self.args = args
        self.feat_transfrom = nn.Sequential(nn.Linear(args.w_in, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 64))
    def forward(self, H_, A, A_ ,num_nodes, epoch=None, layer=None):
        result_A, W1 = self.conv1(A, num_nodes, epoch=epoch, layer=layer)
        W = [W1]
        Hs = []
        As_=[]
        # print("result_A:",len(result_A))
        # print("A_:",len(A_))
        for i in range(len(result_A)):
            a_edge, a_value = result_A[i]
            # print('FastGTNLayer-',i,':-A',result_A[i])

            mat_a = torch.sparse_coo_tensor(a_edge, a_value, (num_nodes, num_nodes)).to(a_edge.device)

            Adj=torch.sparse.mm(mat_a, A_[i])
            As_.append(Adj)
            # # 获取非零元素的索引
            # edge_index = torch.nonzero(Adj, as_tuple=False).t()

            # # 获取非零元素的值
            # edge_weight = Adj[edge_index[0], edge_index[1]]

            # print('FastGTNLayer-',i,':-As',edge_index,edge_weight)
            # print("I:",i,"  Adj:",Adj)

            H = torch.sparse.mm(mat_a, H_[i])
            # print("H:",H)
            Hs.append(H)
        # print('over')
        return Hs, W, As_

class FastGTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_nodes, args=None, pre_trained=None):
        super(FastGTConv, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels))
        
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.num_nodes = num_nodes

        self.reset_parameters()

        if pre_trained is not None:
            with torch.no_grad():
                self.weight.data = pre_trained.weight.data
        
    def reset_parameters(self):
        n = self.in_channels
        nn.init.normal_(self.weight, std=0.1)
        if self.args.non_local and self.args.non_local_weight != 0:
            with torch.no_grad():
                self.weight[:,-1] = self.args.non_local_weight
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)               

    def forward(self, A, num_nodes, epoch=None, layer=None):
        
        weight = self.weight
        filter = F.softmax(weight, dim=1)   
        num_channels = filter.shape[0]
        results = []
        for i in range(num_channels):
            for j, (edge_index,edge_value) in enumerate(A):
                if j == 0:
                    total_edge_index = edge_index
                    total_edge_value = edge_value*filter[i][j]
                else:
                    total_edge_index = torch.cat((total_edge_index, edge_index), dim=1)
                    total_edge_value = torch.cat((total_edge_value, edge_value*filter[i][j]))
            
            index, value = torch_sparse.coalesce(total_edge_index.detach(), total_edge_value, m=num_nodes, n=num_nodes, op='add')
            results.append((index, value))
        
        return results, filter



