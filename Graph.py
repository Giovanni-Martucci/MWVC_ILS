import networkx as nx
import matplotlib.pyplot as plt
import os
import shutil


class Graph:
    def __init__(self, path_file):
        self.graph = nx.Graph()
        self.numNodes = 0
        self.listNodeWeights = []
        self.listNodeWeights_dict = {}
        self.listDegrees = []
        self.listDegreesWeightRatio = []
        self.adjList = []
        self.unc_edges = []
        self.path_file = path_file
        self.all_edges_of_graph = []


        edges = []
        current_node = 0
        f = open(path_file, "r")
        for n, x in enumerate(f):
            if n == 0:
                self.numNodes = int(x)
            elif n == 1:
                self.listNodeWeights = x.split()
                self.listNodeWeights_dict = {nodeWeight: i for i, nodeWeight in enumerate(self.listNodeWeights)}
                # self.graph.add_nodes_from(self.listNodeWeights)
                self.graph.add_nodes_from([i for i in range(self.numNodes)])
            else:
                raw_line = x.split()
                indices = [i for i, x in enumerate(raw_line) if x == "1"]
                for node in indices:
                    # edges.append((self.listNodeWeights[current_node], self.listNodeWeights[node]))
                    edges.append((current_node, node))
                current_node += 1
        self.graph.add_edges_from(edges)

        for node in self.graph.nodes:
            self.listDegrees.append(self.graph.degree[node])
            self.listDegreesWeightRatio.append(round(self.graph.degree[node] / int(self.listNodeWeights[node]), 3))

        self.adjList = [(n, list_a) for n, list_a in self.graph.adjacency()]
        self.all_edges_of_graph = self.graph.edges #(60) oppure = edges (120)

    def outside_edges(self, solution):
        unc_edges = []
        for i, f_node in enumerate(self.listNodeWeights):
            for s_node in self.adjList[i][1]:
                # index_s_node = self.listNodeWeights_dict[s_node]
                if solution[i] != 1 and solution[s_node] != 1:
                    unc_edges.append((i, s_node))
        return unc_edges

    def remove_outside_edges(self, node_added, unc_edges):
        for adj_node in self.adjList[node_added][1]:
            pair_1 = (node_added, adj_node)
            pair_2 = (adj_node, node_added)
            for i in [pair_1, pair_2]:
                try:
                    unc_edges.remove(i)
                except:
                    pass
        return unc_edges

    def save_output(self, initial_weight, final_weight, final_sol, coordX, coordY):
        path = self.path_file[6:-4]
        plt.figure(1)
        color_map = []
        for node, _ in enumerate(self.listNodeWeights):
            if node in final_sol:
                color_map.append('green')
            else:
                color_map.append('blue')
        nx.draw(self.graph, node_color=color_map, with_labels=True)
        plt.text(-0.8, 0.8, "Initial weight {}".format(initial_weight), fontsize=9,
                 bbox=dict(facecolor='red', alpha=0.5))
        plt.text(-0.8, 0.5, "Final weight {}".format(final_weight), fontsize=9, bbox=dict(facecolor='green', alpha=0.5))

        plt.figure(2)
        plt.plot(coordX, coordY)
        # naming the x axis
        plt.xlabel('x - Iter')
        # naming the y axis
        plt.ylabel('y - Solution')

        # giving a title to my graph
        plt.title('Minimum Weights Vertex Cover!')


        # function to show the plot
        #plt.show()
        #x = input("[*] - Salvo in png?")


        if not os.path.exists("output/{}_result".format(path)):
            os.mkdir("output/{}_result".format(path))
        else:
            shutil.rmtree("output/{}_result".format(path))
            os.mkdir("output/{}_result".format(path))
        plt.figure(1).savefig("output/{}_result/output_{}_graph.png".format(path, path))
        plt.figure(2).savefig("output/{}_result/output_{}_whole_plot.png".format(path, path))
        plt.figure(1).clear()
        plt.figure(2).clear()







