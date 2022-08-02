import re
import sys
sys.path.append("./logsum")
from tarjan import get_cut_nodes_and_edges,get_connected_components
import numpy as np
import queue

class Edge():
    __slots__ = ["key","value","cnt","u","v"]
    def __init__(self,key,value=None):
        self.key = key
        self.value = value
        self.u = None
        self.v = None
        self.cnt = 0

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Edge{%d->%s->%d}"%(self.u.id,self.key,self.v.id)


class Node():
    __slots__ = ["id","name","fa","son","conn","vec","jump","flow","belong","redirect","max_depth","visited","removed","binds"]
    def __init__(self,name=None):
        self.id = 0
        self.name = name
        self.fa = set()
        self.son = dict()
        self.conn = set()
        self.vec = np.zeros([27])
        self.jump = set()
        self.flow = 0
        self.belong = None
        self.redirect = None
        self.max_depth = 0
        self.visited = False
        self.removed = False
        self.binds = set()

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Node#%d"%(self.id)


class PrefixGraph():
    def __init__(self,root_name="ROOT"):
        self.node_set = set()
        self.edge_set = set()
        self.node_id = 0
        self.template_list = []
        self.BEGIN_VALUE = "$BEGIN$"
        self.END_VALUE = "$END$"
        self.OMIT_SYMBOL = "<*>"
        self.root = self.create_node(root_name)

    def create_node(self,node_name=None):
        if node_name==None:
            node_name = "Node#%d" % len(self.node_set)
        node = Node(node_name)
        self.node_id += 1
        node.id = self.node_id
        node.belong = node
        node.redirect = node
        self.node_set.add(node)
        return node

    def create_edge(self,edge_key=None,edge_name=None):
        if edge_key==None:
            edge_key = "Edge#%d" % len(self.edge_set)
        if edge_name==None:
            edge_name = edge_key
        edge = Edge(edge_key,edge_name)
        self.edge_set.add(edge)
        return edge

    def add_edge(self,node_fa,node_son,key,value):
        edge = self.create_edge(key,value)
        edge.u = node_fa
        edge.v = node_son
        node_son.fa.add(edge)
        node_fa.son[key] = edge

    def pre_process(self,line,sep):
        line = line.replace('\n','')
        line = re.sub(r'\s+',' ',line)
        mask = re.compile(r'\w*\d+\w*')
        line_keys = re.sub(mask,'#',line)
        line_values = line
        result = []
        origin = []
        raw_val = line_values.split(sep)
        res_val = line_keys.split(sep)
        for i in range(len(res_val)):
            if len(res_val[i])==0:
                continue
            result.append(res_val[i])
            origin.append(raw_val[i])
        return result,origin

    def line_to_vec(self,result,origin):
        vec = [self.BEGIN_VALUE]
        names = [None]
        for i in range(len(result)):
            vec.append(result[i])
            names.append(origin[i])
        vec.append(self.END_VALUE)
        names.append(" ")
        return vec,names

    def fun_to_every_node(self,fun):
        for node in self.node_set:
            fun(node)

    def _insert(self,node,depth,vec,names):
        if depth>=len(vec):
            return
        if vec[depth] in node.conn:
            self._insert(node,depth+1,vec,names)
            return
        if vec[depth] not in node.son:
            node_son = self.create_node(names[depth])
            self.add_edge(node,node_son,vec[depth],names[depth])
        node.son[vec[depth]].cnt += 1
        self._insert(node.son[vec[depth]].v,depth+1,vec,names)

    def insert(self,line,sep=' '):
        result,origin = self.pre_process(line,sep=sep)
        vec,names = self.line_to_vec(result,origin)
        if len(vec)<=2:
            return
        self._insert(self.root,0,vec,names)

    def _match():
        return

    def match(self,line,sep=' '):
        return

    def get_char_id(self,ch):
        x = ord(ch)-96
        return x if 1<=x and x<=26 else 0

    def update_word_vec(self,vec,key,alpha):
        alpha = alpha/len(key)
        for e in key:
            vec[self.get_char_id(e)] += alpha

    def _traversal(self,node):
        node.vec = np.zeros([27])
        node.visited = True
        if len(node.son)!=0:
            node_cnt = 0
            for edge in node.son.values():
                node_cnt += edge.cnt
            node.flow = node_cnt
            for (key,edge) in node.son.items():
                alpha = 1.0*edge.cnt/node_cnt
                edge.v.max_depth = max(node.max_depth+1,edge.v.max_depth)
                if edge.v.visited==False:
                    son_vec = self._traversal(edge.v)
                else:
                    son_vec = edge.v.vec
                node.vec += 1.0*son_vec*alpha
                self.update_word_vec(node.vec,key,alpha)
        return node.vec

    def calc_similarity(self,node_1,node_2):
        sum_v1 = max(np.sum(node_1.vec),1)
        sum_v2 = max(np.sum(node_2.vec),1)
        return np.sqrt(np.sum((node_1.vec/sum_v1-node_2.vec/sum_v2)**2))

    def subgraph_alignment_two_pointer(self,node_1,node_2,gamma):
        branch_1 = node_1
        branch_2 = node_2
        min_dist = self.calc_similarity(node_1,node_2)
        if self._belong(node_1)==self._belong(node_2):
            min_dist = float("inf")
        valid_merge = False
        router = set()
        while True:
            router.add(node_1)
            router.add(node_2)
            son_set_1 = set([edge.v for edge in node_1.son.values()])
            son_set_2 = set([edge.v for edge in node_2.son.values()])
            merged_sons = set([s.belong for s in son_set_1])&set([s.belong for s in son_set_2])
            key_set_1 = set([k for (k,e) in node_1.son.items() if e.v not in merged_sons])
            key_set_2 = set([k for (k,e) in node_2.son.items() if e.v not in merged_sons])
            common_sons = len(key_set_1&key_set_2)
            total_sons = len(key_set_1|key_set_2)
            if total_sons==0 or len(key_set_1)==0 or len(key_set_2)==0:
                break
            if common_sons/total_sons>=0.2:
                if self._belong(node_1)!=self._belong(node_2) and min_dist<=gamma:
                    valid_merge = True
                    break
            if len(node_1.son)==0 or len(node_2.son)==0:
                break
            if self.END_VALUE in node_1.son or self.END_VALUE in node_2.son:
                break
            
            son_dist = [[float("inf"),None],[float("inf"),None]]
            move_flag = False
            for son in son_set_1:
                if self._belong(son) in merged_sons:
                    continue
                dist = self.calc_similarity(son,node_2)
                if dist<son_dist[0][0]:
                    son_dist[0] = [dist,son]
            for son in son_set_2:
                if self._belong(son) in merged_sons:
                    continue
                dist = self.calc_similarity(node_1,son)
                if dist<son_dist[1][0]:
                    son_dist[1] = [dist,son]
            if son_dist[0][0]<=son_dist[1][0] and son_dist[0][0]<min_dist:
                min_dist,node_1 = son_dist[0]
                move_flag = True
                if node_1 in router:
                    min_dist = float("inf")
                    move_flag = False
                    break
            elif son_dist[1][0]<=son_dist[0][0] and son_dist[1][0]<min_dist:
                min_dist,node_2 = son_dist[1]
                move_flag = True
                if node_2 in router:
                    min_dist = float("inf")
                    move_flag = False
                    break
            if self._belong(node_1)==self._belong(node_2):
                min_dist = float("inf")
            if move_flag==False:
                break
        router.add(node_1)
        router.add(node_2)
        if valid_merge==False:
            min_dist = float("inf")
        return min_dist,node_1,node_2,list(router)

    def construct(self,gamma=0.5):
        self.clear_vis_flag()
        self._traversal(self.root)
        self.clear_vis_flag()

    def build(self,gamma=0.1,early_stop=10,MAX_ITERS=64):
        iters = 0
        total_merges = 0
        while iters<MAX_ITERS:
            iters += 1
            self.clear_vis_flag()
            self._traversal(self.root)
            self.clear_vis_flag()
            self.update_jump_link()
            flag = self._merge(self.root,gamma=gamma)
            total_merges += flag
            self.update_connectivity()
            _nodes,_edges = self.clear_unreachable()
            complete = self.close_stream_test()
            if flag==0 or (complete==True and early_stop is not None and iters>=early_stop):
                break
        _nodes,_edges = self.clear_unreachable(drop=True)
        complete = self.close_stream_test()
        result = {
        "n_iters":iters,
        "n_merges":total_merges,
        "n_nodes":len(_nodes),
        "n_edges":len(_edges),
        "valid":complete,
        }
        return result

    def _belong(self,node):
        if node!=node.belong:
            node.belong = self._belong(node.belong)
        return node.belong

    def _redirect(self,node):
        if node!=node.redirect:
            node.redirect = self._belong(node.redirect)
        return node.redirect

    def union_belong(self,node_0,node_1):
        self._belong(node_1).belong = self._belong(node_0)

    def union_belong_for_reconnect(self,node_main,node_sub):
        self._redirect(node_sub).redirect = self._redirect(node_main)

    def get_all_binds(self,node,remove=True):
        node_binds = {node}
        def _search(u):
            for v in u.binds:
                if v not in node_binds:
                    node_binds.add(v)
                    _search(v)
            if remove:
                u.binds = set()
            return
        _search(node)
        node_binds.remove(node)
        return node_binds

    def node_binding(self,node_1,node_2):
        node_1.binds.add(node_2)
        node_2.binds.add(node_1)

    def _merge(self,node,gamma,reconnect=False):
        merge_count = 0
        node.visited = True
        sum_flow = node.flow
        drop_keys = set()
        for key,edge in node.son.items():
            if edge.cnt is None:
                drop_keys.add(key)
        for key in drop_keys:
            node.son.pop(key)
        node_binds = self.get_all_binds(node)
        drop_binds = set([node])
        add_binds = set()
        for target in node_binds:
            if target==node:
                continue
            sum_flow += target.flow
            target_fa = list(target.fa)
            for edge in target_fa:
                node.fa.add(edge)
                edge.v = node
            target.fa = set()
            target_son = list(target.son.items())
            for key,edge in target_son:
                if key not in node.son:
                    if edge.v.removed==True:
                        continue
                    node.son[key] = edge
                    edge.u = node
                    target.son.pop(key)
                else:
                    if edge.v.removed==True:
                        continue
                    related = node.son[key]
                    related.cnt += edge.cnt
                    if related.v!=edge.v:
                        if edge.v==node:
                            add_binds.add(related.v)
                        elif related.v==node:
                            add_binds.add(edge.v)
                        else:
                            self.node_binding(related.v,edge.v)
                    edge.cnt = None
                    if edge in edge.v.fa:
                        edge.v.fa.remove(edge)
            target.son = dict()
            drop_binds.add(target)
            target.removed = True
            add_binds |= target.binds
        if len(node_binds)>0 and sum_flow>0:
            node.vec *= node.flow
            for target in node_binds:
                node.vec += target.vec*target.flow
            node.vec /= sum_flow
            node.flow = sum_flow

        node.binds = node_binds-drop_binds
        for e in add_binds:
            self.node_binding(node,e)
        beta = 0.5
        candidates = []
        for key,edge in node.son.items():
            isolated = True
            for target in candidates:
                root_core = target
                root_sub = edge.v
                if root_core==root_sub:
                    isolated = False
                    break
                dist,merge_core,merge_sub,router = self.subgraph_alignment_two_pointer(root_core,root_sub,gamma)
                head_length = node.max_depth
                body_length = max(merge_core.max_depth,merge_sub.max_depth)-head_length
                tail_length = max(np.sum(merge_core.vec),np.sum(merge_sub.vec))-1  
                if body_length+tail_length>0 and body_length/(body_length+tail_length)>beta:
                    continue
                if dist<=gamma:
                    self.node_binding(merge_core,merge_sub)
                    merge_count += 1
                    isolated = False
                    break
            if isolated==True:
                candidates.append(edge.v)

        iter_key_edge = list(node.son.items())
        for key,edge in iter_key_edge:
            if key not in node.son:
                continue
            if edge.v.visited==False and edge.v.removed==False:
                merge_count += self._merge(edge.v,gamma)
        return merge_count

    def clear_unreachable(self,drop=False):
        reachable_nodes = set()
        reachable_edges = set()
        def _dfs(node):
            if node not in reachable_nodes:
                reachable_nodes.add(node)
                for e in node.son.values():
                    if e not in reachable_edges:
                        reachable_edges.add(e)
                    if e.v not in reachable_nodes:
                        _dfs(e.v)
        _dfs(self.root)
        if drop==True:
            for node in self.node_set:
                if node in reachable_nodes:
                    fa_list = list(node.fa)
                    for fa in fa_list:
                        if fa not in reachable_edges:
                            node.fa.remove(fa)
                    del fa_list
            self.node_set &= reachable_nodes
            self.edge_set &= reachable_edges
        return self.node_set&reachable_nodes,self.edge_set&reachable_edges

    def clear_vis_flag(self):
        for node in self.node_set:
            node.visited = False

    def update_jump_link(self):
        self.clear_unreachable()
        cut_nodes,cut_edges = get_cut_nodes_and_edges(self.root,self.edge_set)
        visited = set()
        def dfs(node,pre):
            visited.add(node)
            if node in cut_nodes:
                pre = node
            for edge in node.son.values():
                if edge.v in cut_nodes:
                    pre.jump.add(edge.v)
                if edge.v not in visited:
                    dfs(edge.v,pre)
            return
        for node in self.node_set:
            node.jump = set()
            node.belong = node
            node.redirect = node
        dfs(self.root,self.root)
        for edge in self.edge_set:
            if edge not in cut_edges:
                self.union_belong(edge.u,edge.v)
        return

    def template_extract(self,return_map=False):
        cut_nodes,cut_edges = get_cut_nodes_and_edges(self.root,self.edge_set)
        def trace_back(node):
            ans = []
            last_mark = None
            last_edge = None
            while len(node.fa)>0:
                fa_set = set([e.key for e in node.fa])
                pre = None
                for e in node.fa:
                    if pre is None or e.u.max_depth>pre.u.max_depth:
                        pre = e
                if pre in cut_edges or (len(fa_set)==1 and last_edge in cut_edges):
                    mark = pre.key
                else:
                    mark = self.OMIT_SYMBOL
                if pre.key==self.BEGIN_VALUE or pre.key==self.END_VALUE:
                    mark = self.OMIT_SYMBOL
                if len(node.conn)>0 and last_mark!=self.OMIT_SYMBOL:
                    ans.append(self.OMIT_SYMBOL)
                if mark!=self.OMIT_SYMBOL or last_mark!=self.OMIT_SYMBOL:
                    ans.append(mark)
                    last_mark = mark
                node = pre.u
                last_edge = pre
            return " ".join(ans[::-1])

        templates = set()
        counts = dict()
        template_map = dict()
        for edge in self.edge_set:
            if edge.key==self.END_VALUE:
                tt = trace_back(edge.v)
                templates.add(tt)
                counts[tt] = edge.cnt
                template_map[tt] = edge

        if return_map==True:
            return template_map
        return templates,counts

    def update_connectivity(self):
        node_set,edge_set = self.clear_unreachable()
        n_self_loop = 0
        for edge in edge_set:
            if edge.u==edge.v:
                node = edge.u
                node.conn.add(edge.key)
                if edge in node.fa:
                    node.fa.remove(edge)
                if edge.key in node.son:
                    node.son.pop(edge.key)
                n_self_loop += 1
        node_set,edge_set = self.clear_unreachable()
        con_groups = get_connected_components(self.root,edge_set)
        for group in con_groups:
            out_edges = set()
            in_edges = set()
            vnode = self.create_node("VCONNECT"+str(len(group)))

            inside_edges = set()
            for node in group:
                if len(node.conn)>0:
                    vnode.conn |= node.conn
                for e in node.son.values():
                    if e.v not in group:
                        out_edges.add(e)
                    else:
                        vnode.conn.add(e.key)
                        inside_edges.add(e)
                for e in node.fa:
                    if e.u not in group:
                        in_edges.add(e)
                    else:
                        vnode.conn.add(e.key)
                        inside_edges.add(e)
            
            for edge in inside_edges:
                edge.u.son.pop(edge.key)
                edge.v.fa.remove(edge)

            for edge in in_edges:
                edge.v = vnode
                vnode.fa.add(edge)

            for node in group:
                self.node_binding(vnode,node)

        self.clear_vis_flag()
        def _reconnect(node):
            node.visited = True
            if len(node.conn)>=1 and type(node.son)==set:
                for edge in node.son:
                    if edge.v.visited==False:
                        _reconnect(edge.v)
                edge_list = list(node.son)
                node.son = dict()
                for i in range(len(edge_list)):
                    if edge_list[i].key not in node.son:
                        node.son[edge_list[i].key] = edge_list[i]
                    else:
                        self.node_binding(node,edge_list[i].u)
            else:
                for edge in node.son.values():
                    if edge.v.visited==False:
                        _reconnect(edge.v)
        self.clear_vis_flag()
        check_flag = get_connected_components(self.root,edge_set)
        return

    def close_stream_test(self):
        node_set,edge_set = self.clear_unreachable()
        error_cnt = 0
        for node in node_set:
            in_stream = 0
            out_stream = 0
            for fa in node.fa:
                if fa.cnt==0:
                    in_stream = None
                    error_cnt += 1
                    break
                else:
                    in_stream += fa.cnt
            for son in node.son.values():
                if son.cnt==0:
                    out_stream = None
                    error_cnt += 1
                    break
                else:
                    out_stream += son.cnt
            if in_stream==out_stream:
                continue

            elif in_stream==0:
                if node.name!="ROOT":
                    error_cnt += 1
            elif out_stream==0:
                if node.name!=" ":
                    error_cnt += 1
            else:
                error_cnt += 1
        passed = 0
        if error_cnt==0:
            passed += 1
        check_flag = get_connected_components(self.root,edge_set)
        self_loop = False
        for e in edge_set:
            if e.u==e.v:
                self_loop = True
        if len(check_flag)==0 and self_loop==False:
            passed += 1
        check_flag = 0
        for node in node_set:
            for edge in node.fa:
                if edge.v!=node:
                    check_flag += 1
        if check_flag==0:
            passed += 1
        return passed==3
