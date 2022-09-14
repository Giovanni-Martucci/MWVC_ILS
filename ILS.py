from numpy import random
import matplotlib.pyplot as plt
import networkx as nx
from queue import PriorityQueue


class ILS:
    def __init__(self):
        self.solution = []

    @staticmethod
    def get_initial_solution(graph):
        total_weight = 0
        selected_nodes = []
        selected_edges = []

        solution = [0 for _ in range(graph.numNodes)]

        for edge in graph.all_edges_of_graph:
            s_edge = edge[0]  # source
            d_edge = edge[1]  # destination

            if solution[s_edge] != 0 or solution[d_edge] != 0:
                continue

            if graph.listNodeWeights[s_edge] < graph.listNodeWeights[d_edge]:
                solution[s_edge] = 1
                total_weight += int(graph.listNodeWeights[s_edge])
                selected_nodes.append(s_edge)
                selected_edges.append(edge)
            else:
                solution[d_edge] = 1
                total_weight += int(graph.listNodeWeights[d_edge])
                selected_nodes.append(d_edge)
                selected_edges.append(edge)

        return solution

    @staticmethod
    def building_queue(unc_edges, graph):
        node_queue = PriorityQueue()
        nodes = set()
        degrees = {}
        for first_node, second_node in unc_edges:
            for elem in [first_node, second_node]:
                if elem in degrees:
                    degrees[elem] += 1
                else:
                    degrees[elem] = 1
                nodes.add(elem)
        for node in nodes:
            order = float(degrees[node]) / float(graph.listNodeWeights[node])
            node_queue.put((-order, (order, node)))
        return node_queue

    @staticmethod
    def greedy_initial_solution(solution, graph):
        unc_edges = graph.outside_edges(solution)
        node_queue = ILS.building_queue(unc_edges, graph)
        while len(unc_edges) > 0:
            node_index = node_queue.get()[1][1]
            solution[node_index] = 1
            unc_edges = graph.remove_outside_edges(node_index, unc_edges)
            node_queue = ILS.building_queue(unc_edges, graph)
        return solution

    @staticmethod
    def get_weight_solution(solution, graph):
        weight = 0
        len_sol = len(solution)
        for i in range(len_sol):
            if solution[i] == 1:
                weight += int(graph.listNodeWeights[i])
        return weight

    @staticmethod
    def index_random_choice(max_number):
        value = random.uniform(0, max_number - 1)
        return value

    @staticmethod
    def rand_neighbor(node_chose, solution, value_neighbor, graph):
        current_neighbor = 0
        max_tries = 10
        list_node = graph.adjList[node_chose][1]
        len_list = len(list_node)
        curr_index = -10
        for tries in range(max_tries):
            rand_index = ILS.index_random_choice(len_list)
            for adj_node in list_node:
                if curr_index != rand_index:
                    current_neighbor = adj_node
                    curr_index += 1
            if solution[current_neighbor] == value_neighbor:
                return current_neighbor
        return node_chose

    @staticmethod
    def perturbation(num_elems_to_change, solution):
        for i in range(num_elems_to_change):
            rand_index = int(ILS.index_random_choice(len(solution)))
            if solution[rand_index] == 1:
                solution[rand_index] = 0
            else:
                solution[rand_index] = 1
        return solution

    def second_perturbation(self, solution, unc_edges):
        unselected_nodes = []
        if 0 in solution:
            rand_index_remove = ILS.index_random_choose(len(solution))
            rand_index_insert = ILS.index_random_choose(len(solution))
            solution[rand_index_remove] = 0
            solution[rand_index_insert] = 1
        else:
            rand_index = ILS.index_random_choose(len(solution))
            solution[rand_index] = 0

    @staticmethod
    def valid_solution(solution, graph):
        for first_node in range(graph.numNodes):
            for second_node in graph.adjList[first_node][1]:
                # index_snd_node = self.listNodeWeights_dict[second_node]
                if solution[first_node] != 1 and solution[second_node] != 1:
                    return False
        return True

    @staticmethod
    def valid_node_chosen(node_removed, solution, graph):
        for snd_node in graph.adjList[node_removed][1]:
            if solution[snd_node] != 1:
                return False
        return True

    @staticmethod
    def eps_scheduling(eps, current_weights, best_weights, max_eps, min_eps, current_iter, iter_without_improvements,
                    value_chosen, max_iter_without_improvs):
        value_to_divide = [3, 3, 3]
        new_eps = 0
        if current_weights < best_weights:
            parameter = ((best_weights - current_weights) * 2)
            new_eps = eps - parameter
            if new_eps > min_eps:
                return new_eps
            else:
                return min_eps
        else:
            new_eps = eps
            if int(current_iter) % 6000 == 0 and iter_without_improvements > max_iter_without_improvs:  # 6000
                less = new_eps / value_to_divide[value_chosen]
                new_eps = new_eps - less   # instead for + less  less = new_eps / 20
            if new_eps <= 20:
                new_eps = 100 * random.uniform(1, 4)
            '''if new_eps >= max_eps:
                new_eps = max_eps'''
            return new_eps

    @staticmethod
    def local_search(current_solution, current_weight, graph, num_obj_fun, max_obj_fun,
                     max_iter_ls, num_swaps, current_swap, curr_iter):
        curr_iters = curr_iter
        num_obj_fun += 1
        if num_obj_fun >= max_obj_fun:
            return current_solution, current_weight, num_obj_fun, current_swap, curr_iters, True
        prev_weight = current_weight
        neighbor = []
        neighbor_weight = current_weight
        best_local_search_solution = current_solution.copy()
        best_local_search_weights = current_weight
        ls_blocked = False
        node_removed = False
        for ls_iter in range(max_iter_ls):
            if not ls_blocked:
                node_removed = False
                # explore neighbors of selected node
                for i in range(len(current_solution)):
                    if current_solution[i] != 0:
                        neighbor = current_solution.copy()
                        neighbor[i] = 0
                        result_selection_node_to_remove = ILS.valid_node_chosen(i, neighbor, graph)
                        if result_selection_node_to_remove:
                            neighbor_weight = current_weight - int(graph.listNodeWeights[i])
                            num_obj_fun += 1
                            if neighbor_weight < best_local_search_weights:
                                node_removed = True
                                best_local_search_solution = neighbor.copy()
                                current_weight = neighbor_weight
                            if num_obj_fun >= max_obj_fun:
                                current_solution = best_local_search_solution.copy()
                                current_weight = best_local_search_weights
                                return current_solution, current_weight, num_obj_fun, current_swap, curr_iters, True
                for i in range(int(num_swaps)):
                    if not node_removed:
                        random_index = int(ILS.index_random_choice(len(current_solution)))
                        if current_solution[random_index] != 0:
                            rand_node_remove = random_index
                            rand_node_insert = ILS.rand_neighbor(rand_node_remove, current_solution, 0, graph)
                        else:
                            rand_node_insert = random_index
                            rand_node_remove = ILS.rand_neighbor(rand_node_insert, current_solution, 1, graph)
                        if graph.listNodeWeights[rand_node_insert] >= graph.listNodeWeights[rand_node_remove]:
                            continue
                        neighbor = current_solution.copy()
                        neighbor[rand_node_remove] = 0
                        neighbor[rand_node_insert] = 1
                        if ILS.valid_node_chosen(rand_node_remove, neighbor, graph):
                            neighbor_weight = current_weight - int(graph.listNodeWeights[rand_node_remove]) + int(
                                graph.listNodeWeights[rand_node_insert])
                            num_obj_fun += 1
                            if neighbor_weight < best_local_search_weights:
                                current_swap += 1
                                best_local_search_solution = neighbor.copy()
                                best_local_search_weights = neighbor_weight
                            if num_obj_fun >= max_obj_fun:
                                current_solution = best_local_search_solution.copy()
                                current_weight = best_local_search_weights
                                return current_solution, current_weight, num_obj_fun, current_swap, curr_iters, True
                current_solution = best_local_search_solution.copy()
                current_weight = best_local_search_weights
                if current_weight == prev_weight:
                    ls_blocked = True
                prev_weight = current_weight
            curr_iters += 1
        current_weight = best_local_search_weights
        current_solution = best_local_search_solution.copy()

        return current_solution, current_weight, num_obj_fun, current_swap, curr_iters, False

    @staticmethod
    def iterative_local_search(solution, graph, best_solution, best_weight, eps, min_eps, max_eps, pert_min,
                               pert_max, num_obj_fun, max_obj_f, max_iter_ls,
                               num_swaps, current_swaps, iter_without_imprvs, value_chosen):
        current_iter = 0
        while not ILS.valid_solution(solution, graph):
            unc_edges = graph.outside_edges(solution)
            candidateNodes = []
            for edge in unc_edges:
                s_node = edge[0]
                d_node = edge[1]
                if solution[s_node] != 1:
                    candidateNodes.append(s_node)
                if solution[d_node] != 1:
                    candidateNodes.append(d_node)

            rand_val = ILS.index_random_choice(2)

            if rand_val == 1:
                best_weight_index = candidateNodes[0]
                for node in candidateNodes:
                    if graph.listNodeWeights[node] < graph.listNodeWeights[best_weight_index]:
                        best_weight_value = node
                    # iter += 1
                solution[best_weight_index] = 1
            else:
                best_num_neighbors_value = candidateNodes[0]
                for node in candidateNodes:
                    if len(graph.adjList[node][1]) > len(graph.adjList[best_num_neighbors_value][1]):
                        best_num_neighbors_value = node
                    # iter += 1
                solution[best_num_neighbors_value] = 1

        current_weight = ILS.get_weight_solution(solution, graph)

        current_solution, current_weights, num_obj_fun, current_swaps, current_iter, max_eval_reached = \
            ILS.local_search(solution, current_weight, graph, num_obj_fun, max_obj_f,
                             max_iter_ls, num_swaps, current_swaps, current_iter)

        num_elems_to_change = ILS.change_number_n_ele(eps, min_eps, max_eps, pert_min, pert_max,
                                                      iter_without_imprvs, graph.numNodes, value_chosen, current_iter)

        if current_weights < best_weight:
            best_weight = current_weights
            best_solution = current_solution.copy()
            objfun_final = num_obj_fun
            iter_without_imprvs = 0
        else:
            iter_without_imprvs += 1

        return current_solution, current_weight, best_solution, best_weight, num_elems_to_change, num_obj_fun, current_swaps, iter_without_imprvs

    def acceptance_criteria_old(self, prev_sol, new_sol, graph):
        if ILS.get_weight_solution(prev_sol, graph) > ILS.get_weight_solution(new_sol, graph):
            return new_sol

        if ILS.index_random_choice(2) == 1:
            return new_sol
        return prev_sol

    @staticmethod
    def acceptance_criteria(best_solution, prev_solution, current_solution, eps, iter_without_impvs,
                            max_iter_without_improvs, graph, accept, iter_acc, current_iter, value_chosen):

        chosen_list_min = [10000, 30000, 60000]
        chosen_list_middle = [12000, 35000, 70000]
        chosen_list_max = [15000, 40000, 80000]

        best_weight = ILS.get_weight_solution(best_solution, graph)
        prev_weight = ILS.get_weight_solution(prev_solution, graph)
        current_weight = ILS.get_weight_solution(current_solution, graph)

        if accept > 10000:
            accept = 10000

        if current_weight < prev_weight:
            return iter_acc, accept, True
        # Se risulta peggiore di un epsilon del migliore risultato viene accettata
        if current_weight < (best_weight + eps) and iter_without_impvs > max_iter_without_improvs: #and current_iter < chosen_list_max[value_chosen]:
            return iter_acc, accept, True
        '''if (iter_without_impvs % 50 == 0 and current_iter < chosen_list_min[value_chosen]) or (
                iter_without_impvs % 200 == 0 and int(chosen_list_min[value_chosen]) < current_iter <
                int(chosen_list_middle[value_chosen])) or (
                iter_without_impvs % 2000 == 0 and int(chosen_list_middle[value_chosen]) < current_iter <
                int(chosen_list_max[value_chosen])):
            print("\n\nAccept\n\n")
            #if iter_acc % 30 == 0:
            #    accept = accept * 2
            #    iter_acc = 0
            #else:
            #    iter_acc += 1
            #return iter_acc, accept, True'''
        return iter_acc, accept, False

    @staticmethod
    def print_info(initial_weight, final_weight, final_sol, coordX, coordY, graph):
        plt.figure(1)
        color_map = []
        for node, _ in enumerate(graph.listNodeWeights):
            if node in final_sol:
                color_map.append('green')
            else:
                color_map.append('blue')
        nx.draw(graph, node_color=color_map, with_labels=True)
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

        plt.figure(3)
        plt.plot(coordX[:201], coordY[:201])
        # naming the x axis
        plt.xlabel('x - Iter 1000')
        # naming the y axis
        plt.ylabel('y - Solution')

        # giving a title to my graph
        plt.title('Minimum Weights Vertex Cover!')

        # function to show the plot
        plt.show()
        x = input("[*] - Salvo in png?")
        if x in ["si", "1"]:
            plt.savefig("output_{}.png".format(graph.path_file))

    @staticmethod
    def eps_init(current_weight, min_weight):
        max_eps = current_weight / 2
        min_eps = current_weight / 5
        if min_eps < (2 * min_weight):
            min_eps = (2 * min_weight)
        if max_eps < min_eps:
            max_eps = (2 * min_eps)
        return max_eps, max_eps, min_eps

    @staticmethod
    def num_change_init(num_nodes):
        pert_max = num_nodes / 6
        pert_min = num_nodes / 12
        if pert_min == 0:
            pert_min = 1
        if pert_max <= pert_min:
            pert_max = pert_min
        return int(pert_max), pert_max, pert_min

    @staticmethod
    def change_number_n_ele(eps, min_eps, max_eps, pert_min_changes, pert_max_changes, iter_without_improvements,
                            numNodes, value_chosen, current_iter):

        chosen_list_max = [15000, 40000, 80000]

        improvs = float(1.0 - (eps - min_eps) / (max_eps - min_eps))
        new_num_elems_to_change = pert_max_changes - int(improvs * (pert_max_changes - pert_min_changes))
        '''if iter_without_improvements % 15000 == 0 and current_iter < chosen_list_max[value_chosen]:
            new_num_elems_to_change = numNodes / 2'''

        return int(new_num_elems_to_change)

    def choose_eps(self, a, b):
        result = ((b - a) * 100) / a
        return int(result)
