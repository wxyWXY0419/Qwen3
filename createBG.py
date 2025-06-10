import json
import sys
import copy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
# Basic Graph 定义，以邻接表的形式存储图，并且边有类别

# Assign an index to each edge type
class Basic_Graph:
    def __init__(self):
        self.graph = {}
        self.edge_types = {} # 边的类别
        self.edge_type_index = {} # 边的类别索引
        self.next_edge_type_index = 0 # 下一个边的索引
        self.attribute = {} # 顶点的属性

    # 添加顶点
    def add_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = []
    # 获取顶点
    def get_vertices(self):
        return list(self.graph.keys())

    # 添加边
    def add_edge(self, vertex1, vertex2, edge_type):
        i=0
        if vertex1 not in self.graph:
            print(vertex1,' is null-1')
            i=1
            self.add_vertex(vertex1)
        if vertex2 not in self.graph:
            print(vertex2,' is null-2')
            i=2
            self.add_vertex(vertex2)
        if vertex1 in self.graph and vertex2 in self.graph:
            self.graph[vertex1].append((vertex2, edge_type))
            if edge_type not in self.edge_types: # 如果边的类别不存在
                self.edge_types[edge_type] = []
                self.edge_type_index[edge_type] = self.next_edge_type_index
                self.next_edge_type_index += 1
            self.edge_types[edge_type].append((vertex1, vertex2))
        return i

    # 获取边(node,node,edge_type)
    def get_edges(self):
        edges = []
        for vertex in self.graph:
            for neighbor, edge_type in self.graph[vertex]:
                edges.append((vertex, neighbor, edge_type))
        return edges
    
    # 获取所有边的类别索引
    def get_all_edge_types_with_index(self):
        return self.edge_type_index

    # 获取边的类别
    def get_edges_by_type(self, edge_type):
        return self.edge_types.get(edge_type, [])

    # 获取边的类别索引
    def get_edge_type_index(self, edge_type):
        return self.edge_type_index.get(edge_type, -1)
    

    # 添加顶点属性
    def add_vertex_attribute(self, vertex, attribute_name, attribute_value):
        if vertex not in self.attribute:
            self.attribute[vertex] = {}
        self.attribute[vertex][attribute_name] = attribute_value
    # 设置顶点属性
    def set_vertex_attribute(self, vertex, attribute_name, attribute_value):
        if vertex in self.attribute:
            self.attribute[vertex][attribute_name] = attribute_value
    def add_vertex_attributes_default(self, vertex, attribute_name):
        self.add_vertex_attribute(vertex, attribute_name, 0)
    # 获取顶点属性
    def get_vertex_attribute(self, vertex, attribute_name):
        return self.attribute.get(vertex, {}).get(attribute_name, None)
    # 获取顶点所有属性
    def get_vertex_attributes(self, vertex):
        # print(self.attribute.get(vertex, {}))
        return self.attribute.get(vertex, {})
    # 删除顶点属性
    def remove_vertex_attribute(self, vertex, attribute_name):
        if vertex in self.attribute and attribute_name in self.attribute[vertex]:
            del self.attribute[vertex][attribute_name]
            if not self.attribute[vertex]:  # 如果该顶点没有其他属性，删除该顶点的属性字典
                del self.attribute[vertex]

    # 删除顶点
    def remove_vertex(self, vertex):
        if vertex in self.graph:
            # Remove all edges to this vertex
            for v in self.graph:
                self.graph[v] = [edge for edge in self.graph[v] if edge[0] != vertex]
            # Remove the vertex
            del self.graph[vertex]
            # Remove edges from edge_types
            for edge_type in self.edge_types:
                self.edge_types[edge_type] = [edge for edge in self.edge_types[edge_type] if edge[0] != vertex and edge[1] != vertex]

    # 删除边
    def remove_edge(self, vertex1, vertex2, edge_type):
        if vertex1 in self.graph:
            self.graph[vertex1] = [edge for edge in self.graph[vertex1] if not (edge[0] == vertex2 and edge[1] == edge_type)]
            if edge_type in self.edge_types:
                self.edge_types[edge_type] = [edge for edge in self.edge_types[edge_type] if not (edge[0] == vertex1 and edge[1] == vertex2)]
    

    def find_father_node(self,subnode):
        # print('subnode:',subnode)
        # 从BG中找到节点
        father_node = []
        relations_set=[]
        attributes_set={}

        
        x = copy.deepcopy(self.graph[subnode]) 
        attributes_set.update(self.get_vertex_attributes(subnode))
        
        index = 0
        while index < len(x):
            neighbor, edge_type = x[index]
            if edge_type == 'father-class':
                x.extend(copy.deepcopy(self.graph[neighbor]))
                attributes_set.update(self.get_vertex_attributes(neighbor))
            index += 1
        for r in x:
            if r[1] == 'father-class':
                father_node.append(r[0])
            else:
                relations_set.append(r)
        # print('relation:',relations_set)
        father_node = list(set(father_node))
        return relations_set,attributes_set,father_node
    
    def judge_father_node(self,subnode,father_node):
        relations_set,attributes_set,father_nodes=self.find_father_node(subnode)
        if father_node in father_nodes:
            return True
        else:
            return False




    # 清空图
    def clear_graph(self):
        self.graph.clear()
        self.edge_types.clear()
        self.edge_type_index.clear()
        self.next_edge_type_index = 0
    def __str__(self):
        return str(self.graph)

    # 保存图
    def save_graph(self, filename):
        data = {
            'graph': self.graph,
            'edge_types': self.edge_types,
            'edge_type_index': self.edge_type_index,
            'next_edge_type_index': self.next_edge_type_index,
            'attribute': self.attribute
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    # 加载图
    def load_graph(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            self.graph = data['graph']
            self.edge_types = data['edge_types']
            self.edge_type_index = data['edge_type_index']
            self.next_edge_type_index = data['next_edge_type_index']
            self.attribute = data['attribute']
    
    # 从文件读取边并添加到图中
    def load_edges_from_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                vertex1, vertex2, edge_type = line.strip().split()
                self.add_vertex(vertex1)
                self.add_vertex(vertex2)
                self.add_edge(vertex1, vertex2, edge_type)
    
    # 从文件读取顶点属性并添加到图中
    def load_vertex_attributes_from_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                v=parts[0]
                for i in range(1,len(parts)):
                    self.add_vertex_attributes_default(v, parts[i])
    # 可视化图结构
    # def visualize_graph(self):
    #     G = nx.DiGraph()  # 创建一个有向图

    #     # 添加顶点和边
    #     for vertex in self.graph:
    #         G.add_node(vertex)
    #         for neighbor, edge_type in self.graph[vertex]:
    #             G.add_edge(vertex, neighbor, label=edge_type)

    #     # # 获取边的标签
    #     edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}

    #     pos = nx.circular_layout(G) # 选择布局
    #     nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)
    #     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    #     plt.title("Basic Graph Visualization")
    #     plt.show()

    def visualize_graph(self, output_file='graph.html'):
        G = nx.DiGraph()  # 创建一个有向图
        # 添加顶点和边
        for vertex in self.graph:
            G.add_node(vertex)
            for neighbor, edge_type in self.graph[vertex]:
                G.add_edge(vertex, neighbor, label=edge_type, edge_type=edge_type)

        # 创建 pyvis 网络
        net = Network(height='1400px',width='100%',bgcolor='#222222',filter_menu=True,select_menu=True, notebook=False, directed=True,cdn_resources='remote',font_color="white") # 定义网络类 
        # net = Network(notebook=False)

        # 将 networkx 图添加到 pyvis 网络
        net.from_nx(G)

        # 设置节点和边的样式
        for node in net.nodes:
            node['title'] = node['id']
            node['label'] = node['id']
            node['size'] = 20

        for edge in net.edges:
            edge['title'] = edge['label']
            edge['label'] = edge['label']
        # net.show_buttons()

# 显示按钮
        net.show_buttons(filter_=True)


        # 添加 JavaScript 和 CSS 代码以实现动态筛选和布局
        # filter_buttons = """
        # <style>
        #     .container {
        #         display: flex;
        #         height: 100vh;
        #     }
        #     .graph {
        #         flex: 3;
        #     }
        #     .controls {
        #         flex: 1;
        #         padding: 10px;
        #         overflow-y: auto;
        #     }
        #     .controls button {
        #         display: block;
        #         margin-bottom: 10px;
        #         width: 100%;
        #     }
        # </style>
        # <script type="text/javascript">
        # function filterEdges(edgeType) {
        #     var edges = document.getElementsByClassName('edge');
        #     for (var i = 0; i < edges.length; i++) {
        #         var edge = edges[i];
        #         if (edgeType === 'all' || edge.getAttribute('title') === edgeType) {
        #             edge.style.display = 'block';
        #         } else {
        #             edge.style.display = 'none';
        #         }
        #     }
        # }
        # </script>
        # <div class="container">
        #     <div class="graph" id="graph"></div>
        #     <div class="controls">
        #         <button onclick="filterEdges('all')">显示所有边</button>
        #         <button onclick="filterEdges('某类边')">显示某类边</button>
        #         <!-- 添加更多按钮以筛选不同类型的边 -->
        #     </div>
        # </div>
        # """

        # net.html = net.html.replace('<body>', '<body>' + filter_buttons)

        # 生成并保存 HTML 文件
        net.show(output_file)
    

# Example usage:
if __name__ == "__main__":
    g = Basic_Graph()
    g.clear_graph()
    g.load_edges_from_file("Basic_Graph.txt")
    g.load_vertex_attributes_from_file("Basic_Graph_Attributes.txt")
    print("Vertices:", g.get_vertices())
    print("Edges:", g.get_edges())
    print("Edge Types with Index:", g.get_all_edge_types_with_index())
    for vertex in g.get_vertices():
        print(f"Attributes of vertex {vertex}:", g.get_vertex_attributes(vertex))
    g.save_graph("Basic_Graph.json")

    g.visualize_graph()