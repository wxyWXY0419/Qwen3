# import sys
# sys.path.append('/home/lankefeng/桌面/RCA-1/GTN/Graph_Transformer_Networks')

import torch
import numpy as np
import torch.nn as nn
# from model_gtn import GTN
from model_encoding import FastGTNs,EncodingNet
from utils_data import convert_data_from_RG,tokens_transverter
from model_llm import model_llm

import pickle
import argparse
from torch_geometric.utils import add_self_loops
from sklearn.metrics import f1_score as sk_f1_score
from utils import init_seed, _norm,f1_score,print_cuda_info
import copy

# from sklearn.metrics import f1_score
# from torch_geometric.utils import add_self_loops

# 你的其他代码

def train(A,node_features,labels,A_mux,args):
    init_seed(seed=777)
    # args.num_FastGTN_layers
    # args.node_dim
    # args.dataset
    # args.non_local
    # args.num_layers
    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    runs = args.runs
# !!!!!!!!!!!!!!!!!!!!!!1
    # with open('../data/%s/labels.pkl' % args.dataset,'rb') as f:
    #     labels = pickle.load(f)

     
    # num_nodes = edges[0].shape[0]
    # args.num_nodes = num_nodes
    # build adjacency matrices for each edge type
    # print_cuda_info()
    
    
    num_edge_type = len(A)
    node_features = torch.from_numpy(node_features).type(torch.cuda.FloatTensor)
    # 从labels种提取训练、测试、验证集的节点编号和标签
    if True:
        train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.cuda.LongTensor)
        print('train_node:', train_node)
        train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.cuda.LongTensor)
        print('train_target:',train_target)
        valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.cuda.LongTensor)
        valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.cuda.LongTensor)
        test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.cuda.LongTensor)
        test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.cuda.LongTensor)
        num_classes = np.max([torch.max(train_target).item(), torch.max(valid_target).item(), torch.max(test_target).item()])+1
        # 计算所有节点的类别数
    final_f1, final_micro_f1 = [], []

    for l in range(runs):
        # initialize a model       
        if True:
            # 若当前图的边类型数超过原始定义，则裁剪多余的边类型。
            while len(A) > num_edge_type:
                del A[-1]
            # 创建 EncodingNet 模型（自定义 GNN 模型），传入边类型数、输入特征维度、类别数、节点数等参数
            model = EncodingNet(num_edge_type=len(A),
                            w_in = node_features.shape[1],
                            num_class=num_classes,
                            num_nodes = node_features.shape[0],
                            embedding_w=768,
                            args = args)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        model.cuda()
        # print_cuda_info()
        loss = nn.CrossEntropyLoss()
        Ws = []
        
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        best_train_f1, best_micro_train_f1 = 0, 0
        best_val_f1, best_micro_val_f1 = 0, 0
        best_test_f1, best_micro_test_f1 = 0, 0
        
        for i in range(epochs):
            # print('Epoch ',i)
            model.zero_grad()
            model.train()
            # print("train")
            # print_cuda_info()
            

            loss,y_train,W,token = model(A, node_features, train_node, train_target, epoch=i)

            train_f1 = torch.mean(f1_score(torch.argmax(y_train.detach(),dim=1), train_target, num_classes=num_classes)).cpu().numpy()
            sk_train_f1 = sk_f1_score(train_target.detach().cpu(), np.argmax(y_train.detach().cpu(), axis=1), average='micro')
            # print(W)
            # print('Train - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(loss.detach().cpu().numpy(), train_f1, sk_train_f1))
            loss.backward()
            optimizer.step()
            model.eval()
            # Valid
            with torch.no_grad():

                val_loss, y_valid,_ ,token = model.forward(A, node_features, valid_node, valid_target, epoch=i)

                token_list=tokens_transverter(token,A_mux,22)
                print('token_list:',len(token))
                print(len(token_list[0]))
                # print(token_list[0])
                val_f1 = torch.mean(f1_score(torch.argmax(y_valid,dim=1), valid_target, num_classes=num_classes)).cpu().numpy()
                sk_val_f1 = sk_f1_score(valid_target.detach().cpu(), np.argmax(y_valid.detach().cpu(), axis=1), average='micro')
                # print('Valid - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1, sk_val_f1))

                test_loss, y_test,W ,token = model.forward(A, node_features, test_node, test_target, epoch=i)
                test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=num_classes)).cpu().numpy()
                sk_test_f1 = sk_f1_score(test_target.detach().cpu(), np.argmax(y_test.detach().cpu(), axis=1), average='micro')
                # print('Test - Loss: {}, Macro_F1: {}, Micro_F1:{} \n'.format(test_loss.detach().cpu().numpy(), test_f1, sk_test_f1))
            if sk_val_f1 > best_micro_val_f1:
                best_val_loss = val_loss.detach().cpu().numpy()
                best_test_loss = test_loss.detach().cpu().numpy()
                best_train_loss = loss.detach().cpu().numpy()
                best_train_f1 = train_f1
                best_val_f1 = val_f1
                best_test_f1 = test_f1
                best_micro_train_f1 = sk_train_f1
                best_micro_val_f1 = sk_val_f1
                best_micro_test_f1 = sk_test_f1

            
            # print('Run {}'.format(l),"Epoch: ",i)
            # print('--------------------Best Result-------------------------')
            # print('Train - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}'.format(best_test_loss, best_train_f1, best_micro_train_f1))
            # print('Valid - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}'.format(best_val_loss, best_val_f1, best_micro_val_f1))
            # print('Test - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}'.format(best_test_loss, best_test_f1, best_micro_test_f1))

        print('Run {}'.format(l))
        print('--------------------Best Result-------------------------')
        print('Train - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}'.format(best_test_loss, best_train_f1, best_micro_train_f1))
        print('Valid - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}'.format(best_val_loss, best_val_f1, best_micro_val_f1))
        print('Test - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}'.format(best_test_loss, best_test_f1, best_micro_test_f1))
        final_f1.append(best_test_f1)
        final_micro_f1.append(best_micro_test_f1)

    print('--------------------Final Result-------------------------')
    print('Test - Macro_F1: {:.4f}+{:.4f}, Micro_F1:{:.4f}+{:.4f}'.format(np.mean(final_f1), np.std(final_f1), np.mean(final_micro_f1), np.std(final_micro_f1)))
