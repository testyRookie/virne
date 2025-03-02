import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community

def split_graph_into_n_connected_subgraphs(G, n):
    # 确保图是连通的
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    
    # 使用 Louvain 算法检测社区
    communities = list(community.greedy_modularity_communities(G))
    
    # 调整社区数量为 n
    def adjust_communities(communities, n):
        while len(communities) > n:
            communities.sort(key=len)
            communities[0] = communities[0].union(communities[1])
            communities.pop(1)
        
        while len(communities) < n:
            communities.sort(key=len, reverse=True)
            largest_community = list(communities[0])
            split_point = len(largest_community) // 2
            communities[0] = set(largest_community[:split_point])
            communities.append(set(largest_community[split_point:]))
        
        return communities
    
    communities = adjust_communities(communities, n)
    
    # 提取子图
    subgraphs = [G.subgraph(community).copy() for community in communities]
    
    return subgraphs

# 示例使用
G = nx.karate_club_graph()
n = 3
subgraphs = split_graph_into_n_connected_subgraphs(G, n)

# 为每个子图分配颜色
colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow']  # 颜色列表
node_color_map = {}
edge_color_map = {}

for i, subgraph in enumerate(subgraphs):
    color = colors[i % len(colors)]  # 循环使用颜色列表
    for node in subgraph.nodes():
        node_color_map[node] = color
    for edge in subgraph.edges():
        edge_color_map[edge] = color

# 将颜色映射到节点和边
node_colors = [node_color_map[node] for node in G.nodes()]
# edge_colors = [edge_color_map[edge] for edge in G.edges()]

# 绘制图
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)  # 布局算法
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
# nx.draw_networkx_edges(G, pos, edge_color=edge_colors)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

plt.title(f"Graph Split into {n} Connected Subgraphs")
plt.axis('off')  # 关闭坐标轴
plt.savefig('graph_split.png')