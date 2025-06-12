
# Example usage:
from createBG import Basic_Graph
from createRG import DataSet,Real_Graph
from utils_data import convert_data_from_RG,tokens_transverter
import argparse
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='FastGTN',
                        help='Model')
    parser.add_argument('--dataset', type=str, default='DBLP',
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='hidden dimensions')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='number of GT/FastGT layers')
    parser.add_argument('--runs', type=int, default=5,
                        help='number of runs')
    parser.add_argument("--channel_agg", type=str, default='mean')
    parser.add_argument("--remove_self_loops", action='store_true', help="remove_self_loops")
    # Configurations for FastGTNs
    parser.add_argument("--non_local", action='store_true', help="use non local operations")
    parser.add_argument("--non_local_weight", type=float, default=0, help="weight initialization for non local operations")
    parser.add_argument("--beta", type=float, default=0, help="beta (Identity matrix)")
    parser.add_argument('--K', type=int, default=1,
                        help='number of non-local negibors')
    parser.add_argument("--pre_train", action='store_true', help="pre-training FastGT layers")
    parser.add_argument('--num_FastGTN_layers', type=int, default=3,
                        help='number of FastGTN layers')

    args = parser.parse_args()

    # Create a basic graph
    g = Basic_Graph()
    g.clear_graph()
     #导入基础知识图谱的相关数据
    g.load_edges_from_file("Basic_Graph.txt") 
    g.load_vertex_attributes_from_file("Basic_Graph_Attributes.txt")
    # 保存基础知识图谱,后续复用
    g.save_graph("Basic_Graph.json")


    #Create DataSet,为采集到的数据
    data = DataSet()
    data.load_all_data('data/4-flow_dealed.csv','data/4-cs_metric_low_bandwidth_3.csv','data/4-es_metric_low_bandwidth_3.csv','data/4-user_metric_low_bandwidth_3.csv')
    # x=data.get_data()
    # for key in x:
    #     print(key,x[key])


    #Create Real Graph
    rg = Real_Graph(g,data)  #由基础知识图谱和采集到的数据构建的真实知识图谱
    rg.init_RG()

    ################## 可以打印出来看看数据结构 #########################
    # print("Vertices:", rg.get_vertices())
    # print("Edges:", rg.get_edges())
    # print("Vertices:", len(rg.get_vertices()))
    # print("Edges:", len(rg.get_edges()))
    # print("Edge Types with Index:", rg.get_all_edge_types_with_index())
    # for vertex in rg.get_vertices():
    #     print(f"Attributes of vertex {vertex}:", rg.get_vertex_attributes(vertex))
    # rg.visualize_graph()


    # 将图的数据结构转换为神经网络模型所需的格式
    # PS:labels为训练集、测试集、验证集的节点编号和标签,在目前实现的代码中,没人都是1,还没有读取真实采集的数据
    A,node_features,labels,A_mux=convert_data_from_RG(rg)

    # print('len-A:',len(A))
    # print("A[0]:",A[0])  # A有9种类型的边，构造出9个图，每个图包含该类型的所有边
    # print('len(node_features):',len(node_features))
    # print("node_features:",node_features)
    # print("labels:",labels)
    # print("A_mux:",A_mux)

    # 训练模型 
    train(A,node_features,labels,A_mux,args)
    