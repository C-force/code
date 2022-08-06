import re
import numpy as np
from union_find import UnionFind
from tarjan import get_cut_nodes_and_edges
from queue import Queue

Format_Regex = {
    "Number":re.compile(r'^(?P<left>\D*[^\w-]+|)(?P<value>-?\d+(\.\d+)?)(?P<right>|\W\D*)$'),
    # "IPAddr":re.compile(r'^(?P<left>\D*)(?P<value>\d+\.\d+\.\d+\.\d+:\d+)(?P<right>\D*)$'),
}

Format_Fix_Value = {
    "Word":re.compile(r'^(?P<left>\W*)(?P<value>[a-zA-Z]+)(?P<right>\W*)$'),
    "Symbol":re.compile(r'^(?P<left>)(?P<value>\W+)(?P<right>)$'),
}

def is_word_edge(key):
    return re.match(Format_Fix_Value["Word"],key) is not None

def is_symbol_edge(key):
    return re.match(Format_Fix_Value["Symbol"],key) is not None

def convert_to_format(word):
    SPECIAL_CHAR = ["","\\#","\\&","\\*"]
    fmt_name = "String"
    res = None
    for rname,regex in Format_Regex.items():
        res = re.match(regex,word)
        if res is not None:
            res = res.groupdict()
            fmt_name = rname
            break
    fmt_bits = 0
    for ch in word:
        if ch.isdigit():
            fmt_bits |= 1
        elif ch.isalpha():
            fmt_bits |= 2
    fmt = re.sub(r'[a-zA-Z0-9]+',SPECIAL_CHAR[fmt_bits],word)
    if res is None:
        res = {"value":word}
    return fmt,fmt_name,res

def calc_format_compatability(fmt0,fmt1):
    fset0 = set(fmt0)
    fset1 = set(fmt1)
    return len((fset0|fset1)-(fset0&fset1))

def auto_split(log,sep=" ",sub=False):
    token_count = [log.count(token) for token in sep]
    token_sep = sep[np.argmax(token_count)]
    if sub==True:
        return log.replace(token_sep," ")
    else:
        return log.replace(token_sep," ").split(" ")

def prefix_graph_proprocessor(log):
    res = re.sub(r'\w*\d+\w*','#',log)
    return res

def weighted_random_choice(elements,n_select=1,weights=None,random_state=None):
    N = len(elements)
    if weights is None:
        weights = [1/N]*N
    if type(elements)==set:
        elements = {e:1/N for e in elements}
    elif type(elements)!=dict:
        elements = {elements[i]:weights[i] for i in range(len(elements))}
    targets = list(elements.keys())
    sum_weights = sum(list(elements.values()))
    weights = [elements[t]/sum_weights for t in targets]
    np.random.seed(random_state)
    random_value = np.random.random(n_select)
    results = []
    for k in range(n_select):
        sum_proba = 0
        for i in range(N):
            sum_proba += weights[i]
            if random_value[k]<=sum_proba:
                results.append(targets[i])
                break
    return results

class EdgeVar():
    def __init__(self):
        self.id = 0
        self.type = None
        self.subtype = None
        self.format_set = set()
        self.value_set = set()
        self.frequent = 0
        
    def __repr__(self):
        return str(self)
    
    def __str__(self):
        if len(self.value_set)==1:
            return "<Constant#%d>"%(self.id)
        elif len(self.value_set)<5:
            return "<Enumerated#%d>"%(self.id)
        else:
            return "<%s#%d>"%(self.type,self.id)
    
    def update(self,fmt,fmt_type,fmt_regex):
        if self.type is None or fmt_type=="String":
            self.type = fmt_type
        self.format_set.add(fmt)
        self.value_set.add(fmt_regex["value"])

class LogSummary():
    def __init__(self):
        self.name = ""
        self.uf = UnionFind(0)
        self.variable_map = {}
        self.unique_var_id = []
        self.unique_event_id = {}
        self.num_log_collected = 0
        self.template_freq = {}
        self.template_map = {}
        self.global_edge_var_id = 0
        self.edge_addin_attr = None
        self.get_edge_by_id = None
        return
    
    def register_edge(self,edge,fmt,fmt_type,fmt_regex):
        var = EdgeVar()
        var.id = self.edge_addin_attr[str(edge)]["id"]
        var.frequent = edge.cnt
        var.type = fmt_type
        self.variable_map[var.id] = var
        self.uf.append()
        return var
    
    def drop_var(self,var_id):
        self.variable_map.pop(var_id)
        return
    
    def adjacency_summary(self,graph):
        cut_nodes,cut_edges = get_cut_nodes_and_edges(graph.root,graph.edge_set)
        for node in graph.node_set:
            fa_core = []
            for e in node.fa:
                if e in cut_edges:
                    continue
                merged_flag = False
                for s in fa_core:
                    if calc_format_compatability(self.edge_addin_attr[str(e)]["fmt"],self.edge_addin_attr[str(s)]["fmt"])==0:
                        l_branch = min(self.edge_addin_attr[str(e)]["id"],self.edge_addin_attr[str(s)]["id"])
                        r_branch = max(self.edge_addin_attr[str(e)]["id"],self.edge_addin_attr[str(s)]["id"])
                        self.uf.union(l_branch,r_branch)
                        merged_flag = True
                        break
                if merged_flag==False:
                    fa_core.append(e)
            son_core = []
            for k,e in node.son.items():
                if e in cut_edges:
                    continue
                merged_flag = False
                for s in son_core:
                    if calc_format_compatability(self.edge_addin_attr[str(e)]["fmt"],self.edge_addin_attr[str(s)]["fmt"])==0:
                        l_branch = min(self.edge_addin_attr[str(e)]["id"],self.edge_addin_attr[str(s)]["id"])
                        r_branch = max(self.edge_addin_attr[str(e)]["id"],self.edge_addin_attr[str(s)]["id"])
                        self.uf.union(l_branch,r_branch)
                        merged_flag = True
                        break
                    elif graph.END_VALUE in e.v.son or graph.END_VALUE in s.v.son:
                        l_branch = min(self.edge_addin_attr[str(e)]["id"],self.edge_addin_attr[str(s)]["id"])
                        r_branch = max(self.edge_addin_attr[str(e)]["id"],self.edge_addin_attr[str(s)]["id"])
                        self.uf.union(l_branch,r_branch)
                        merged_flag = True
                        break
                if merged_flag==False:
                    son_core.append(e)
        return
    
    def sequence_summary(self,graph,params):
        traversed_count = {str(e):0 for e in graph.edge_set}
        traversed_node_count = {str(v):0 for v in graph.node_set}
        repeat_count = 0
        merge_route_count = 0
        
        def random_walk(weighted=False):
            node = graph.root
            fmt_seq = []
            while len(node.son)>0:
                if weighted==True:
                    wt_all = sum([edge.cnt for edge in node.son.values()])
                    prop = {key:edge.cnt/wt_all for key,edge in node.son.items()}
                else:
                    prop = {key:1 for key in node.son}
                if len(prop)>=2:
                    key = weighted_random_choice(prop,1)[0]
                else:
                    key = list(prop.keys())[0]
                next_edge = node.son[key]
                traversed_count[str(next_edge)] += 1
                edge_stack.append(next_edge)
                if self.edge_addin_attr[str(next_edge)]["is_word_or_symbol"]==True:
                    fmt_seq.append(key)
                else:
                    fmt_seq.append(self.edge_addin_attr[str(next_edge)]["fmt"])
                traversed_node_count[str(next_edge.v)] += 1
                if key!=graph.END_VALUE:
                    node = next_edge.v
                else:
                    break
            return fmt_seq
        
        def random_walk_bidirect(center_edge,weighted=False):
            traversed_count[str(center_edge)] += 1
            node = center_edge.v
            traversed_node_count[str(node)] += 1
            fmt_seq = []
            edge_stack_tail = []
            while len(node.son)>0:
                if weighted==True:
                    wt_all = sum([edge.cnt for edge in node.son.values()])
                    prop = {key:edge.cnt/wt_all for key,edge in node.son.items()}
                else:
                    prop = {key:1 for key in node.son}
                if len(prop)>=2:
                    key = weighted_random_choice(prop,1)[0]
                else:
                    key = list(prop.keys())[0]
                next_edge = node.son[key]
                traversed_count[str(next_edge)] += 1
                edge_stack_tail.append(next_edge)
                if self.edge_addin_attr[str(next_edge)]["is_word_or_symbol"]==True:
                    fmt_seq.append(key)
                else:
                    fmt_seq.append(self.edge_addin_attr[str(next_edge)]["fmt"])
                traversed_node_count[str(next_edge.v)] += 1
                if key!=graph.END_VALUE:
                    node = next_edge.v
                else:
                    break
            
            node = center_edge.u
            fmt_seq = fmt_seq[::-1]
            if self.edge_addin_attr[str(center_edge)]["is_word_or_symbol"]==True:
                fmt_seq.append(center_edge.key)
            else:
                fmt_seq.append(self.edge_addin_attr[str(center_edge)]["fmt"])

            edge_stack_front = []
            while len(node.fa)>0:
                fa_dict = {edge.key:edge for edge in node.fa}
                if weighted==True:
                    wt_all = sum([edge.cnt for edge in fa_dict.values()])
                    prop = {key:edge.cnt/wt_all for key,edge in fa_dict.items()}
                else:
                    prop = {key:1 for key in fa_dict}
                if len(prop)>=2:
                    key = weighted_random_choice(prop,1)[0]
                else:
                    key = list(prop.keys())[0]
                next_edge = fa_dict[key]
                traversed_count[str(next_edge)] += 1
                edge_stack_front.append(next_edge)
                if self.edge_addin_attr[str(next_edge)]["is_word_or_symbol"]==True:
                    fmt_seq.append(key)
                else:
                    fmt_seq.append(self.edge_addin_attr[str(next_edge)]["fmt"])
                traversed_node_count[str(next_edge.u)] += 1
                if key!=graph.BEGIN_VALUE:
                    node = next_edge.u
                else:
                    break
            
            fmt_seq = fmt_seq[::-1]
            edge_stack.extend(edge_stack_front[::-1])
            edge_stack.append(center_edge)
            edge_stack.extend(edge_stack_tail)
            return fmt_seq
        
        sentence_format = {}
        num_batch = params["num"] if "num" in params else 10
        wt_flag = params["wt"] if "wt" in params else False
        bidirect = params["bi"] if "bi" in params else True
        MAX_ROUTE = len(graph.edge_set)*num_batch
        edge_list = sorted(list(graph.edge_set),key=lambda x:str(x))
        for route_id in range(MAX_ROUTE):
            edge_stack = []
            if bidirect==True:
                fmt_seq = random_walk_bidirect(edge_list[route_id%len(edge_list)],weighted=wt_flag)
            else:
                fmt_seq = random_walk(weighted=wt_flag)
            fmts = " ".join(fmt_seq)
            if "ngram" in params and len(fmt_seq)>=params["ngram"]:
                ngram = params["ngram"]
                for k in range(len(fmt_seq)-ngram+1):
                    fmts = " ".join(fmt_seq[k:k+ngram])
                    if fmts not in sentence_format:
                        sentence_format[fmts] = [edge_stack[k+e] for e in range(ngram)]
                    else:
                        for e,s in zip(sentence_format[fmts],edge_stack[k:k+ngram]):
                            if calc_format_compatability(self.edge_addin_attr[str(e)]["fmt"],self.edge_addin_attr[str(s)]["fmt"])==0:
                                l_branch = min(self.edge_addin_attr[str(e)]["id"],self.edge_addin_attr[str(s)]["id"])
                                r_branch = max(self.edge_addin_attr[str(e)]["id"],self.edge_addin_attr[str(s)]["id"])
                                if l_branch!=r_branch:
                                    self.uf.union(l_branch,r_branch)
                                    merge_route_count += 1
                        repeat_count += 1
                        pass
                continue
            
            if fmts not in sentence_format:
                sentence_format[fmts] = [e for e in edge_stack]
            else:
                for e,s in zip(sentence_format[fmts],edge_stack):
                    if calc_format_compatability(self.edge_addin_attr[str(e)]["fmt"],self.edge_addin_attr[str(s)]["fmt"])==0:
                        l_branch = min(self.edge_addin_attr[str(e)]["id"],self.edge_addin_attr[str(s)]["id"])
                        r_branch = max(self.edge_addin_attr[str(e)]["id"],self.edge_addin_attr[str(s)]["id"])
                        if l_branch!=r_branch:
                            self.uf.union(l_branch,r_branch)
                            merge_route_count += 1
                repeat_count += 1
                pass
        print("Unvisited Edges:",sum([1 if item==0 else 0 for item in traversed_count.values()]))
        print("Unvisited Nodes:",sum([1 if item==0 else 0 for item in traversed_node_count.values()]))
        print("Merge Route Count:",merge_route_count)
        return

    def update_template_frequency(self,graph):
        self.template_freq = {}
        self.template_map = {}
        self.num_log_collected = 0
        end_edges = []
        for edge in graph.edge_set:
            if edge.key==graph.END_VALUE:
                end_edges.append(edge)
        end_edges = sorted(end_edges,key=lambda x:self.edge_addin_attr[str(x)]["id"])
        
        for i in range(len(end_edges)):
            edge_id = self.edge_addin_attr[str(end_edges[i])]["id"]
            conn_id = self.uf.find(edge_id)-1
            if edge_id==conn_id:
                if str(end_edges[i]) not in self.template_map:
                    self.template_map[str(end_edges[i])] = "C%d"%(len(self.template_map))
            target = str(self.get_edge_by_id[conn_id])
            if target not in self.template_freq:
                self.template_freq[target] = 0
            self.template_freq[target] += end_edges[i].cnt
            self.num_log_collected += end_edges[i].cnt
        return
    
    def merge_var(self,target,source):
        if source.type!=target.type and source.type=="String":
            target.type = source.type
        target.frequent += source.frequent
        target.format_set |= source.format_set
        target.value_set |= source.value_set
        return

    def update_var_relationship(self,graph):
        self.unique_var_id = []
        for edge in graph.edge_set:
            edge_id = self.edge_addin_attr[str(edge)]["id"]
            conn_id = self.uf.find(edge_id)-1
            if edge_id==conn_id:
                self.unique_var_id.append(edge_id)
            else:
                self.merge_var(self.variable_map[conn_id],self.variable_map[edge_id])
                self.drop_var(edge_id)
        return

    def summarize(self,graph,params={}):
        print("Number of Vars:",len(self.uf.uf)-1)

        # Adjacency Summary
        self.adjacency_summary(graph)
        self.uf.update()
        print("Number of Vars:",self.uf.sets_count,"after Adjacency Summary")

        # Sequence Summary
        self.sequence_summary(graph,params)
        self.uf.update()
        print("Number of Vars:",self.uf.sets_count,"after Sequence Summary")
        
        self.update_template_frequency(graph)
        self.update_var_relationship(graph)
        print("Summarize Complete!")
        return
    

def register_graph_attributes(graph,logsum,register=True):
    logsum.global_edge_var_id = 0
    logsum.edge_addin_attr = {}
    logsum.get_edge_by_id = {}
    cut_nodes,cut_edges = get_cut_nodes_and_edges(graph.root,graph.edge_set)
    for e in graph.edge_set:
        logsum.edge_addin_attr[str(e)] = {
            "id":logsum.global_edge_var_id,
            "fmt":None,
            "is_word_or_symbol":None,
            "is_cut_edge":False,
            "var":None,
        }
        logsum.get_edge_by_id[logsum.global_edge_var_id] = e
        token = e.value
        logsum.edge_addin_attr[str(e)]["is_word_or_symbol"] = is_word_edge(token) or is_symbol_edge(token)
        fmt,fmt_type,fmt_regex = convert_to_format(token)
        logsum.edge_addin_attr[str(e)]["fmt"] = fmt
        logsum.edge_addin_attr[str(e)]["is_cut_edge"] = e in cut_edges
        if register==True:
            logsum.edge_addin_attr[str(e)]["var"] = logsum.register_edge(e,fmt,fmt_type,fmt_regex)
        else:
            logsum.edge_addin_attr[str(e)]["var"] = logsum.varmap[logsum.global_edge_var_id]
        logsum.global_edge_var_id += 1
    return 

def search_header_by_frequency(graph,logsum,header_format):
    ground_truth = header_format.replace(" <Content>","").strip()
    ground_truth_names = ground_truth.split(" ")
    header_length = len(ground_truth_names)
    num_all_logs = list(graph.root.son.values())[0].cnt
    frequency_error_coef = 0.8
    eq = Queue()
    eq.put(graph.root)
    depth_map = {str(graph.root):0}
    depth_edge_list = []
    while not eq.empty():
        node = eq.get()
        cur_depth = depth_map[str(node)]
        for edge in node.son.values():
            if len(depth_edge_list)<=cur_depth:
                depth_edge_list.append(set())
            conn_id = logsum.uf.find(logsum.edge_addin_attr[str(edge)]["id"])-1
            var = logsum.variable_map[conn_id]
            depth_edge_list[cur_depth].add(var)
            if str(edge.v) not in depth_map:
                eq.put(edge.v)
                depth_map[str(edge.v)] = cur_depth+1
    max_header_length = 64
    header_score = []
    for k in range(1,min(len(depth_edge_list),max_header_length)):
        top_score = max([item.frequent/num_all_logs for item in depth_edge_list[k]])
        header_score.append(top_score)
    last_split_point = header_length
    for i in range(1,min(header_length+1,len(header_score))):
        if header_score[i] < header_score[i-1]*frequency_error_coef:
            last_split_point = i-1
    header_label = [j<last_split_point for j in range(len(header_score))]
    for i in range(len(header_label)):
        if header_score[i]>=0.99:
            header_label[i] = True
    return ground_truth_names,header_label

def match_single_log(graph,log):
    def _match(node,word_vec,pos=0):
        key = word_vec[pos]
        if key==graph.END_VALUE:
            if key in node.son:
                edge = node.son[key]
                edge_stack.append(edge)
                return str(edge)
            else:
                pass
        if key in node.son:
            edge = node.son[key]
            edge_stack.append(edge)
            flag = _match(edge.v,word_vec,pos+1)
            if flag is not None:
                return flag
            else:
                edge_stack.pop()
        if len(node.conn)>0:
            edge_stack.append(None)
            return _match(node,word_vec,pos+1)
        return None
    
    raw_log = auto_split(log,sub=True)
    word_vec = [graph.BEGIN_VALUE]+re.split(r'\s+',prefix_graph_proprocessor(raw_log))+[graph.END_VALUE]
    edge_stack = []
    res = _match(graph.root,word_vec,0)
    if res is None:
        edge_stack = []
        res = _match(graph.root,word_vec)
    if res is None:
        edge_stack = [None]
    match_result = {
        "success_flag":res,
        "raw_log":raw_log,
        "word_vec":re.split(r'\s+',raw_log),
        "edge_stack":edge_stack[1:-1],
        "destination":edge_stack[-1],
    }
    return match_result

def get_structured_match_result(logsum,graph,log,update_model=True):
    res = match_single_log(graph,log)
    if res['success_flag'] is None:
        return {"TemplateID":None,"Unmatched":True,"Content":log}
    if type(res['success_flag'])==str and res['success_flag'][0]=="@":
        return {"TemplateID":None,"Unmatched":True,"Content":log}
    edge_stack = res['edge_stack']
    word_vec = res['word_vec']
    end_point = res['destination']
    
    varname_list = []
    format_list = []
    varobj_list = []
    
    var_dict = {}
    is_template_word = []
    for pos in range(len(word_vec)):
        edge = edge_stack[pos]
        if edge is None:
            varobj_list.append(None)
            format_list.append("<*>")
            varname_list.append("<*>")
            is_template_word.append(False)
            continue
        if "#" in edge.key:
            is_template_word.append(False)
        else:
            is_template_word.append(True)
        word = word_vec[pos]
        var_id = logsum.uf.find(logsum.edge_addin_attr[str(edge)]["id"])-1
        var = logsum.variable_map[var_id]
        fmt,fmt_type,fmt_regex = convert_to_format(word)
        if update_model==True:
            var.update(fmt,fmt_type,fmt_regex)
        
        varobj_list.append(var)
        var_dict[var.id] = fmt_regex["value"]
        
        if var.type in Format_Regex:
            varname_list.append(fmt_regex["left"]+str(var)+fmt_regex["right"])
        elif var.type=="String" and len(var.value_set)<=4:
            varname_list.append(word)
        else:
            varname_list.append(str(var))

        if var.type in Format_Regex:
            format_list.append(fmt_regex["left"]+"<"+str(var.type)+">"+fmt_regex["right"])
        elif logsum.edge_addin_attr[str(edge)]["is_word_or_symbol"]:
            format_list.append(word)
        else:
            format_list.append(fmt.replace("\\#","#").replace("\\&","*").replace("\\*","*"))
            
    formats = " ".join(format_list)
    leaf_id = logsum.uf.find(logsum.edge_addin_attr[res['success_flag']]["id"])-1
    leaf_edge = logsum.get_edge_by_id[leaf_id]
    match_obj = {
        "TemplateID":logsum.template_map[str(leaf_edge)],
        "Frequency":round(logsum.template_freq[str(leaf_edge)]/logsum.num_log_collected,8),
        "Unmatched":False,
        "Content":log,
        "TemplateWord":is_template_word,
        "Formats":formats,
        "FieldList":varobj_list,
        "ValueDict":var_dict,
    }
    return match_obj
