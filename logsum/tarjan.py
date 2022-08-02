def get_cut_nodes_and_edges(root_node,edge_set):
    link,dfn,low = {},{},{}
    edge_dict = {}
    global_time = [0]
    for edge in edge_set:
        a,b = edge.u,edge.v
        if (a,b) not in edge_dict:
            edge_dict[(a,b)] = set()
            edge_dict[(b,a)] = set()
        edge_dict[(a,b)].add(edge)
        edge_dict[(b,a)].add(edge)
        if a not in link:
            link[a] = []
        link[a].append(b)
        if b not in link:
            link[b] = []
        link[b].append(a)
        dfn[a],dfn[b] = 0x7fffffff,0x7fffffff
        low[a],low[b] = 0x7fffffff,0x7fffffff
    pst = root_node

    cutting_points,cutting_edges = set(),set()

    def _dfs(cur,prev,root):
        global_time[0] += 1
        dfn[cur],low[cur] = global_time[0],global_time[0]
        children_cnt = 0
        flag = False
        for nxt in link[cur]:
            if nxt!=prev:
                if dfn[nxt]==0x7fffffff:
                    children_cnt += 1
                    _dfs(nxt,cur,root)
                    if cur!=root and low[nxt]>=dfn[cur]:
                        flag = True
                    low[cur] = min(low[cur],low[nxt])
                    if low[nxt]>dfn[cur]:
                        if len(edge_dict[(cur,nxt)])==1:
                            cutting_edges.add(list(edge_dict[(cur,nxt)])[0])
                else:
                    low[cur] = min(low[cur],dfn[nxt])

        if flag or (cur==root and children_cnt>=2):
            cutting_points.add(cur)

    _dfs(pst,None,pst)
    return cutting_points,cutting_edges

def get_connected_components(root_node,edge_set):
    link,dfn,low = {},{},{}
    components = []
    global_time = [0]
    for edge in edge_set:
        a,b = edge.u,edge.v
        if a not in link:
            link[a] = []
        link[a].append(b)
        if b not in link:
            link[b] = []
        dfn[a],dfn[b] = 0x7fffffff,0x7fffffff
        low[a],low[b] = 0x7fffffff,0x7fffffff
    pst = root_node
    visited = [[]]

    def _dfs(cur):
        global_time[0] += 1
        dfn[cur],low[cur] = global_time[0],global_time[0]
        children_cnt = 0
        visited[0].append(cur)
        for nxt in link[cur]:
            if dfn[nxt]==0x7fffffff:
                _dfs(nxt)
                low[cur] = min(low[cur],low[nxt])
            elif nxt in set(visited[0]):
                low[cur] = min(low[cur],dfn[nxt])
        if dfn[cur]==low[cur]:
            components.append(set())
            for i in range(len(visited[0]),0,-1):
                pos = visited[0][i-1]
                components[-1].add(pos)
                if pos==cur:
                    break
            visited[0] = visited[0][:len(visited[0])-len(components[-1])]

    _dfs(pst)

    connected_component = [group for group in components if len(group)>1]
    return connected_component
