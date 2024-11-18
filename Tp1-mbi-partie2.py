
import matplotlib.pyplot as plt
import networkx as nx

class Graph:
    def __init__(self):
        self.graph = {}
        self.heuristics = {}

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []
            self.heuristics[node] = 0

    def add_edge(self, start, end, cost):
        if start in self.graph:
            self.graph[start].append((end, cost))
        else:
            self.graph[start] = [(end, cost)]

    def set_heuristic(self, node, value):
        self.heuristics[node] = value

    def matrice_adjacance(self):
        nodes = sorted(self.graph.keys())
        n = len(nodes)
        matrix = [[0 for _ in range(n)] for _ in range(n)]

        for start, edges in self.graph.items():
            for end, cost in edges:
                i, j = nodes.index(start), nodes.index(end)
                matrix[i][j] = 1

        print("Adjacency Matrix:")
        for row in matrix:
            print(row)

    def f_path(self, path):
        g_cost = sum(cost for _, cost in path)
        h_cost = self.heuristics[path[-1][0]]
        return g_cost + h_cost

    def a_star_search(self, start, end):
        queue = [[(start, 0)]]
        visited = set()

        while queue:
            queue.sort(key=lambda x: self.f_path(x))
            path = queue.pop(0)
            node = path[-1][0]

            if node in visited:
                continue

            visited.add(node)

            if node == end:
                return path

            for neighbor, cost in self.graph.get(node, []):
                new_path = path + [(neighbor, cost)]
                queue.append(new_path)

        return []
    
    def search_bfs(self,start, end):
        visited=[]
        queue=[[start]]
        while queue:
            path = queue.pop(0) # queue
            node = path[-1]
            if node in visited:
                continue
            visited.append(node)
            if node == end:
                return path
            else:
                adjacent_node = self.graph.get(node,[])
                for neighbor, _ in adjacent_node:
                    new_path = path.copy()
                    new_path.append(neighbor)
                    queue.append(new_path)


    def dfs_algorithem(self,start,end):
        visited =[]
        stack=[[start]]
        while stack:
            path=stack.pop() # dfs
            node=path[-1]
            if node in visited:
                continue
            visited.append(node)
            if node == end:
                return path
            else:
                adjancy_node = self.graph.get(node,[])
                for neighbor, _ in adjancy_node:
                    new_path = list(path)
                    new_path.append(neighbor)
                    stack.append(new_path)

    def path_cost(self,path):
        total_cost=0
        for (node, cost) in path:
            total_cost += cost 
        return total_cost


    def ucs_algorithem(self,start,end):
        visited =[]
        queue=[[(start,0)]]
        while queue:
            queue.sort(key=self.path_cost)
            path=queue.pop(0) # dfs
            node=path[-1][0]
            if node in visited:
                continue
            visited.append(node)
            if node == end:
                return path
            else:
                adjancy_node = self.graph.get(node,[])
                for neighbor, cost in self.graph.get(node, []):
                    if neighbor not in visited:
                        new_path = list(path)  
                        new_path.append((neighbor, cost))
                        queue.append(new_path)

    def draw_solution(self, solution_path, title, color):
    # Create a new figure and axis explicitly because dynamicaly the title doesnt display idk why
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create a directed graph with networkx
        nx_graph = nx.DiGraph()

        # Add all edges from the graph to the NetworkX graph
        for start, neighbors in self.graph.items():
            for end, cost in neighbors:
                nx_graph.add_edge(start, end, weight=cost)

        # Get the solution path 
        solution_edges = [(solution_path[i], solution_path[i+1]) for i in range(len(solution_path) - 1)] if solution_path else []

        # Get positions for the nodes in the graph
        pos = nx.spring_layout(nx_graph)

        # Draw the nodes
        nx.draw(nx_graph, pos, ax=ax, with_labels=True, node_color="yellow", 
                node_size=2000, font_size=15, font_weight="bold", arrowsize=15)

        # default edge color is black
        nx.draw_networkx_edges(nx_graph, pos, ax=ax, edgelist=nx_graph.edges(), 
                                edge_color="black", arrows=True)

        # Highlight solution edges 
        nx.draw_networkx_edges(nx_graph, pos, ax=ax, edgelist=solution_edges, 
                                edge_color=color, width=2.5, arrows=True)

        cost = nx.get_edge_attributes(nx_graph, 'weight')
        nx.draw_networkx_edge_labels(nx_graph, pos, ax=ax, edge_labels=cost, 
                                    font_size=12, font_color="blue")

        ax.set_title(title, fontsize=16, fontweight="bold")
        
        plt.tight_layout()
        plt.show(block=True)

# Create the graph
g = Graph()

# Add nodes and edges
for node in ['S', 'A', 'B', 'C', 'D','G']:
    g.add_node(node)

g.add_edge('S', 'A', 1)
g.add_edge('S','G',12)
g.add_edge('A','B',3)
g.add_edge('A', 'C', 1)
g.add_edge('B', 'D', 3)
g.add_edge('C', 'D', 1)
g.add_edge('C', 'G', 2)
g.add_edge('D', 'G', 3)


g.draw_solution([],"Graph Visualization",'')
g.matrice_adjacance()
solution_bfs=g.search_bfs('S','G')
print('solution with bsf is:', solution_bfs)
g.draw_solution(solution_bfs,'Bfs Graph with Solution Path Highlighted','lime')

solution_dfs=g.dfs_algorithem('S','G')
print('solution with dfs is:', solution_dfs)
g.draw_solution(solution_bfs,'dfs Graph with Solution Path Highlighted','aqua')

solution_ucs=g.ucs_algorithem('S','G')
print('solution with ucs is:', solution_ucs)
print('The path cost of ucs is:',g.path_cost(solution_ucs))
g.draw_solution(solution_bfs,'UCS Graph with Solution Path Highlighted','deeppink')


solution_a_start=g.a_star_search('S','G')

print('solution with A* is:', solution_a_start)
print('the F-cost of the A* algorithem',g.f_path(solution_a_start))
g.draw_solution(solution_bfs,'A* Graph with Solution Path Highlighted','red')


