from createBG import Basic_Graph
import copy
import pandas as pd

class Real_Graph(Basic_Graph):
    def __init__(self,BG,Data):
        super().__init__()
        self.BG=BG
        self.Data=Data
        self.node_num=0

    def init_RG(self):
        for id in self.Data.get_data(): #id=flow1,1,2,3,4
            self.add_data(id)
        pass

    # 添加顶点
    def add_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = []
            self.add_vertex_attribute(vertex,'number',self.node_num)
            self.node_num+=1
    def get_vertex_number(self, vertex):
        x=self.attribute.get(vertex, {})
        return x.get('number',-1)

    # 中英文名称对齐
    def add_data(self,id,name_t=None,type_t=None):
        # 明确data对应的节点类型（Basic_Graph）X
        name,type=self.Data.get_node_data(id)
        # print('name:',name,'type:',type)

        if name_t==None:
            name_t=name
        # 判断data对应的节点是否已经在RG中,如果在则返回
        if name_t in self.graph:
            # print('Error: 节点已存在',name)
            return
        if name_t!=name:
            self.add_virtual_data(type_t,name_t,False)
        
        #明确node
        data=self.Data.get_node_all_message(id,type)
        if data==None:
            print('Error: 未找到对应节点信息',name)
            return self.add_virtual_data(type,name,False)
        # print('data:',data)
        # 聚合X及其父类，得到关系和属性
        relations_set,attributes_set,father_nodes=self.BG.find_father_node(self.Data.convert_name(type))
        # print('relations_set:',relations_set)
        # print('attributes_set:',attributes_set)
        # 创建RG节点
        self.add_vertex(name)
        self.add_vertex_attribute(name,'type',type)
        # 添加RG节点属性
        for key in attributes_set:
            value=data.get(key,attributes_set[key])# 如果data中没有对应属性，则取默认值
            self.add_vertex_attribute(name,key,value)# 添加属性，及属性值
        # print('attr:',self.get_vertex_attributes(name))

        # 添加RG节点关系
        for relation in relations_set:
            neighbor,edge_type=relation
            if edge_type=='father-class':
                print('Error: 关系类型错误',edge_type)
            targetId,target_name=self.Data.get_target_from_relation(type,id,edge_type,neighbor)
            # print('targetId:',targetId)
            if targetId!=None:
                # name,type=self.Data.get_node_data(id)
                # target_name,target_type=self.Data.get_node_data(targetId)
                # # print('target_name:',target_name,'target_type:',target_type)
                # # print('neighbor:',neighbor)
                # if self.BG.judge_father_node(self.Data.convert_name(target_type),neighbor):     # 保证关系的正确性
                #     name_temp=self.add_data(targetId)#已存在则无作为
                #     i=self.add_edge(name,target_name,edge_type)

                #     if i != 0:
                #         print('Error-0: 创建虚拟节点失败',i)
                #         print(name_temp,target_name,target_type)
                # else:
                #     #创建虚拟节点
                #     print('Error: 找到目标id，但其类型与关系对象类型不匹配',neighbor,target_name)
                #     # self.add_edge(name,neighbor,edge_type)
                name_temp=self.add_data(targetId,target_name,neighbor)#已存在则无作为
                i=self.add_edge(name,target_name,edge_type)

                if i != 0:
                    print('Error-0: 创建虚拟节点失败',i)
                    # print(name_temp,target_name,target_type)
            else:
                neighbor_name=self.add_virtual_data(neighbor,name)
                i=self.add_edge(name,neighbor_name,edge_type)
                if i != 0:
                    print('Error: 创建虚拟节点失败',i)
        return name

    def add_virtual_data(self,neighbor,yuan_name,flag=True):
        if flag==False:
            name=yuan_name
        # 创建虚拟节点
        else:
            name=yuan_name+'_'+neighbor
        self.add_vertex(name)
        # self.add_vertex_attribute(name,'number',self.node_num)
        # self.node_num+=1
        self.add_vertex_attribute(name,'type',neighbor)
        neighbor=self.Data.convert_name(neighbor)
        # 添加虚拟节点属性

        relations_set,attributes_set,father_nodes=self.BG.find_father_node(neighbor)
        # 添加RG节点属性
        for key in attributes_set:
            value=attributes_set[key]# 如果data中没有对应属性，则取默认值
            self.add_vertex_attribute(name,key,value)# 添加属性，及属性值
        # attributes_set=self.BG.get_vertex_attributes(neighbor)

        # for key in attributes_set:
        #     value=attributes_set[key]
        #     self.add_vertex_attribute(neighbor,key,value)

        # 添加虚拟节点关系
        # relations = copy.deepcopy(self.BG.graph[neighbor])
        for relation in relations_set:
            neighbor, edge_type = relation
            neighbor_name=self.add_virtual_data(neighbor,name)
            i=self.add_edge(name, neighbor_name, edge_type)
            if i != 0:
                print('Error-2: 创建虚拟节点失败',i)
        return name


    def find_node_from_BG(self,name,type):
        type_in_BG=self.Data.convert_name(type)
        print(type_in_BG)
        # 从BG中找到节点
        edges_set=[]
        attributes_set={}
        
        x = copy.deepcopy(self.BG.graph[type_in_BG]) 
        attributes_set.update(self.BG.get_vertex_attributes(type_in_BG))
        
        index = 0
        while index < len(x):
            neighbor, edge_type = x[index]
            if edge_type == '父类':
                x.extend(copy.deepcopy(self.BG.graph[neighbor]))
                attributes_set.update(self.BG.get_vertex_attributes(neighbor))
            index += 1
        # print('relation:',x)
        return x,attributes_set
    
    def search_object(self,x,edge_type):
        # 查找对象
        y=x
        return y
    # 在这里添加子类的方法和属性

class DataSet:
    def __init__(self):
        self.data=[]
        self.flow_list={}
        self.node_list={}
        self.cs_list={}
        self.es_list={}
        self.user_list={}
        self.camera_list={}
        self.name_map={'flow':'Flow','cs':'Center-Server','es':'Edge-Server','user':'User','camera':'Camera'}


    def add_data(self,name,type):
        self.data.append({'name':name,'type':type})
    def get_data(self):
        return self.node_list
    # 加载所有数据
    def load_all_data(self,flow_path,cs_path,es_path,user_path):
        self.load_cs_data(cs_path)
        self.load_es_data(es_path)
        self.load_user_data(user_path)
        self.load_flow_data(flow_path)
        pass
    
    def load_flow_data(self,path):
        metric = pd.read_csv(path,sep=',', header=None, names=['flow_id', 'sim_time', 'source_address','destination_address','source_port','destination_port','protocol',
                                                       'source_id','destination_id','tx_packets','rx_packets','duration','throughput','mean_delay','mean_jitter','packet_loss_rate','runID','cameraId','anomaly_score','is_anomaly'])
        source_list={}
        dest_list={}
        for index, flow in metric.iterrows():#前提：metric只含有最新的一次数据
            # print('index:',index)
            # print('flow:',flow['flow_id'])
            # 在这里处理每一行
            # if flow['sim_time']>set_time:
            #         break
            flow_id='flow'+str(flow['flow_id'])
            sim_time=flow['sim_time']
            source_address=flow['source_address']
            destination_address=flow['destination_address']
            source_port=flow['source_port']
            destination_port=flow['destination_port']
            protocol=flow['protocol']
            source_id=flow['source_id']
            destination_id=flow['destination_id']
            tx_packets=flow['tx_packets']
            rx_packets=flow['rx_packets']
            duration=flow['duration']
            throughput=flow['throughput']
            mean_delay=flow['mean_delay']
            mean_jitter=flow['mean_jitter']
            packet_loss_rate=flow['packet_loss_rate']
            runID=flow['runID']
            cameraId=flow['cameraId']
            anomaly_score=flow['anomaly_score']
            is_anomaly=flow['is_anomaly']
            self.flow_list[flow_id] = {
                'sim_time': sim_time,
                'source_address': source_address,
                'destination_address': destination_address,
                'source_port': source_port,
                'destination_port': destination_port,
                'protocol': protocol,
                'source': source_id,
                'destination': destination_id,
                'tx_packets': tx_packets,
                'rx_packets': rx_packets,
                'duration': duration,
                'throughput': throughput,
                'mean_delay': mean_delay,
                'mean_jitter': mean_jitter,
                'packet_loss_rate': packet_loss_rate,
                'runID': runID,
                'cameraId': cameraId,
                'anomaly_score': anomaly_score,
                'is_anomaly': is_anomaly
            }
            self.load_node_data(flow_id,'flow')
            source_list.setdefault(source_id, []).append(flow_id)
            dest_list.setdefault(destination_id, []).append(flow_id)

        for flow_id, flow_data in self.flow_list.items():
            flow_data['source_list'] = source_list.get(flow_id, [])
            flow_data['dest_list'] = dest_list.get(flow_id, [])
                # Process the flow data as needed
        # print('flow_list:',self.flow_list)

    def get_target_from_relation(self,type,id,relation,target):
        nodeId=None
        attr=None
        if type=='flow':
            first=self.flow_list[id]
            if relation=='source':#中英文对齐
                nodeId=first['source']
            elif relation=='destination':
                nodeId=first['destination']
            elif relation=='increase':
                attr=target
                if target=='Send-queue':
                    nodeId=first['source']
                elif target=='Receive-queue':
                    nodeId=first['destination']
            else:
                return nodeId,None
            target_name,target_type=self.get_node_data(nodeId)
            if attr!=None:
                name_temp=target_name+'_'+attr
            else:
                name_temp=target_name
            return nodeId,name_temp
            
        else:
            return nodeId,None

    def load_node_data(self,id,type):
        if type=='flow':
            self.node_list[id]={
                'name':id,
                'type':type,
            }
        else:
            self.node_list[id]={
                'name':type+str(id),
                'type':type
            }
    
    def convert_name(self,type):
        if type in self.name_map:
            return self.name_map[type]
        else:
            return type
    def judge_type_including(self,sub_type,father_type,bg):
        sub_type=self.convert_name(sub_type)
        self.BG.graph[sub_type]

    #根据id获取节点的的name，type   
    def get_node_data(self,id):
        type='camera'
        x= self.node_list.get(id, {'name':type+str(id),'type':type})
        return x['name'],x['type']

    def get_node_all_message(self,id,type):
        if type=='flow':
            return self.flow_list[id]
        elif type=='cs':
            return self.cs_list[id]
        elif type=='es':
            return self.es_list[id]
        elif type=='user':
            return self.user_list[id]
        # elif type=='camera':
        #     return self.camera_list[id]
        else:
            return None

    def load_cs_data(self,path):
        metric = pd.read_csv(path, header=None, names=['cs_id', 'sim_time', 'link_num','in_rate','out_rate'])
        for index, cs in metric.iterrows():
            cs_id=cs['cs_id']
            sim_time=cs['sim_time']
            link_num=cs['link_num']
            in_rate=cs['in_rate']
            out_rate=cs['out_rate']
            self.cs_list[cs_id] = {
                'sim_time': sim_time,
                'link_num': link_num,
                'in_rate': in_rate,
                'out_rate': out_rate
            }
            self.load_node_data(cs_id,'cs')
        pass
    def load_es_data(self,path):
        metric = pd.read_csv(path, header=None, names=['es_id', 'sim_time', 'link_num','in_rate','out_rate'])
        for index, es in metric.iterrows():
            es_id=es['es_id']
            sim_time=es['sim_time']
            link_num=es['link_num']
            in_rate=es['in_rate']
            out_rate=es['out_rate']
            self.es_list[es_id] = {
                'sim_time': sim_time,
                'link_num': link_num,
                'in_rate': in_rate,
                'out_rate': out_rate
            }
            self.load_node_data(es_id,'es')
        pass
    def load_user_data(self,path):
        metric = pd.read_csv(path, header=None, names=['user_id', 'sim_time', 'frame_rate','throughput','mean_delay','mean_jitter','packet_loss_rate'])
        for index, user in metric.iterrows():
            user_id=user['user_id']
            sim_time=user['sim_time']
            frame_rate=user['frame_rate']
            throughput=user['throughput']
            mean_delay=user['mean_delay']
            mean_jitter=user['mean_jitter']
            packet_loss_rate=user['packet_loss_rate']
            self.user_list[user_id] = {
                'sim_time': sim_time,
                'frame_rate': frame_rate,
                'throughput': throughput,
                'mean_delay': mean_delay,
                'mean_jitter': mean_jitter,
                'packet_loss_rate': packet_loss_rate
            }
            self.load_node_data(user_id,'user')
        pass
