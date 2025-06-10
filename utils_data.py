import numpy as np
from  createRG import Real_Graph
import torch
from torch_geometric.utils import add_self_loops

from utils import init_seed, _norm,f1_score


# 转换实况知识图谱的数据格式
def convert_data_from_RG(RG):
    node_features = [[x for x in RG.get_vertex_attributes(v).values()] for v in RG.get_vertices()]
    print('node_fea:',node_features)
    #ps:type也有漏的情况，估计是add_edge
    # 找到最长的行长度
    max_length = max(len(row) for row in node_features)
    print('number',RG.node_num)
    print('leb',len(RG.get_vertices()))
    print("max_length",max_length)
    num_nodes=RG.node_num
    # 使用 numpy.pad 补齐每一行
    node_features = np.array([np.pad(row, (0, max_length - len(row)), 'constant') for row in node_features])

####### ############################################################
# 
# 
# #########################    Important   #########################
# 
# 
# ##################################################################
    # 去除第一列,但实际上去除了前二列
    node_features = node_features[:, 2:]
    def str_to_num(s):
        try:
            return float(s)
        except ValueError:
            return s  # 如果转换失败，返回原始字符串
    vectorized_str_to_num = np.vectorize(str_to_num)
    node_features = vectorized_str_to_num(node_features)

    print("node",node_features)#预期type，attr,补0
    edge_all=RG.get_edges()
    # print("edge_all",edge_all)
    edge_index=RG.get_all_edge_types_with_index()
    edge=[[] for i in edge_index]
    print(edge)


    # 构造edges,A_mux为关系矩阵
    # A_mux[i,j]表示i节点和j节点之间的边的类型
    # A_mux[i,j]=t表示i节点和j节点之间有一条边，且该边的类型为t
    A_mux=np.full((num_nodes, num_nodes), -1)
    for a,b,t in edge_all:
        a=RG.get_vertex_number(a)
        b=RG.get_vertex_number(b)
        t=edge_index[t]
        if a==-1 or b==-1:
            print(a,b,t)
        edge[t].append([a,b])
        A_mux[a,b]=t
    edges=[[] for i in edge_index] 
    for i in range(len(edge)):
        edges[i]=np.array(edge[i]).transpose()


    # print(edges)
    print('xxxxxxxx')
    A = []
    for i,edge in enumerate(edges):
        edge_tmp = torch.from_numpy(edge).type(torch.cuda.LongTensor)
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
        print(edge_tmp,value_tmp)
        # normalize each adjacency matrix
        # if args.model == 'FastGTN' and args.dataset != 'AIRPORT':
        if True:
            edge_tmp, value_tmp = add_self_loops(edge_tmp, edge_attr=value_tmp, fill_value=1e-20, num_nodes=num_nodes)#添加自环
            deg_inv_sqrt, deg_row, deg_col = _norm(edge_tmp.detach(), num_nodes, value_tmp.detach())
            value_tmp = deg_inv_sqrt[deg_row] * value_tmp#归一化
        A.append((edge_tmp,value_tmp))
    edge_tmp = torch.stack((torch.arange(0,num_nodes),torch.arange(0,num_nodes))).type(torch.cuda.LongTensor)#全点的自环
    value_tmp = torch.ones(num_nodes).type(torch.cuda.FloatTensor)
    A.append((edge_tmp,value_tmp))
    # print(A)
    labels = np.random.choice(num_nodes, size=(3, num_nodes // 3), replace=False)
    labels = np.array([[i, 1] for i in labels.flatten()]).reshape(3, -1, 2)
    print(labels[0])
    print('len(label[0]):',len(labels[0]))
    return A,node_features,labels,A_mux
    # A有9种类型的边，构造出9个图，每个图包含该类型的所有边
    # A[0]=(
    #       [[0,1,0,1,2],
    #        [1,2,0,1,2]] ,                 [1,1,1e-20,1e-20,1e-20])
    # 1代表存在该关系，1e-20代表自环

    # node_features是一个二维数组，每一行代表一个节点的特征向量

    # labels是一个三维数组，len(labels)=3,表示训练集、验证集和测试集三个集的标签
    # labels[i]是一个二维数组，表示第i个集合（比如labels[0]就是训练集的标签）的标签，里面有某些节点的标签[节点索引，标签（默认为1）]
    # 注意！！！本部分代码，labels并没有相关数据，所以这里默认都是1

    # A_mux是一个二维数组，表示每组节点的边的类型


    def create_tensor_from_X(X, n, order):
        """
        Create a new 2D tensor by selecting n rows from X in a specified order.

        Parameters:
        X (torch.Tensor): The input tensor from which rows are selected.
        n (int): The number of rows to select.
        order (list): The order in which to select rows from X.

        Returns:
        torch.Tensor: A new 2D tensor with n rows selected from X.
        """
        if len(order) < n:
            raise ValueError("The length of the order list must be at least n.")
        
        selected_rows = [X[order[i]] for i in range(n)]
        new_tensor = torch.stack(selected_rows)
        
        return new_tensor
def tokens_transverter(X,A,begin_index,max_length=4096):
    token_list=[]
    # print(A)
    # print(len(A))
    visited = set()
    queue = [begin_index]
    token_list.append(X[begin_index])
    visited.add(begin_index)

    while queue and len(token_list) < max_length:
        current_node = queue.pop(0)
        # Find neighbors in the current_node row
        row_neighbors = (A[current_node, :] != -1).nonzero()[0].tolist()

        # Find neighbors in the current_node column
        col_neighbors = (A[:, current_node] != -1).nonzero()[0].tolist()

        # Combine row and column neighbors
        neighbors = list(set(row_neighbors + col_neighbors))
        # neighbors = (A[current_node] != -1).nonzero()[0].tolist()
        # print(current_node,neighbors)
        
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                token_list.append(X[neighbor])
                
                if len(token_list) >= max_length:
                    break
    # print('sdf:',token_list)
    # print('token_list-len:',len(token_list))
    # print('token_list[0]-len:',len(token_list[0]))
    return token_list
    