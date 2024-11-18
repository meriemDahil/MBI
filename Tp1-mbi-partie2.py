import matplotlib.pyplot as plt
import networkx as nx

class Graph:
    def __init__(self):
        self.graph = {}
        self.heuristics = {}

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []

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

    def draw_solution(self):
        # Create a directed graph with networkx
        nx_graph = nx.DiGraph()

        # Add all edges from the graph to the NetworkX graph
        for start, neighbors in self.graph.items():
            for end, cost in neighbors:
                nx_graph.add_edge(start, end, weight=cost)

        # Get the solution path 
      

        # Get positions for the nodes in the graph
        pos = nx.spring_layout(nx_graph)

        # Draw the nodes
        nx.draw(nx_graph, pos, with_labels=True, node_color="yellow", node_size=2000, font_size=15, font_weight="bold", arrowsize=17)

        # Draw all edges in default black
        nx.draw_networkx_edges(nx_graph, pos, edgelist=nx_graph.edges(), edge_color="black", arrows=True)


        # Draw edge with costs
        cost = nx.get_edge_attributes(nx_graph, 'weight')
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=cost, font_size=12, font_color="blue")

        # Display the graph
        plt.title("Graph with Solution Path Highlighted")
        plt.show()


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

g.draw_solution()
g.matrice_adjacance()

