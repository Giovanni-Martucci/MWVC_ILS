from Graph import Graph
from ILS import ILS
import pandas as pd
import time
import os


def main():
    df = pd.DataFrame(
        columns=['Id', 'Initial_Weight', 'Final_Weight', 'Final_Solution', 'Final_Eps', 'Run_time', 'Valid_Solution'])

    max_eval_list = [20000, 50000, 100000]
    index_max = int(input("Select the max value to iterate from these {}\nNumber between 1,2,3 (position list): "
                          .format(max_eval_list))) - 1
    print("[*] - Chosen: {}".format(int(max_eval_list[index_max])))

    value_to_take_coords = [150, 300, 1000]

    for file in sorted(os.listdir("input")):
        max_eval = int(max_eval_list[index_max])
        iter_without_improvements = 0
        num_obj_func = 0
        max_obj_func = 8000  # Num max for object_fun
        max_iter_ls = 800
        max_iter_without_imprvs = 500

        # Plotting data
        coordX = []
        coordY = []

        # Create Graph from file
        graph = Graph("input/{}".format(file))

        # initial_solution = ILS.get_initial_solution(graph)

        # Start time to exec
        start = time.time()

        # Greedy initial solution
        new_solution = ILS.greedy_initial_solution([0 for _ in range(graph.numNodes)], graph)
        new_weight = ILS.get_weight_solution(new_solution, graph)

        initial_solution = new_solution.copy()


        #coordX.append(1)
        #coordY.append(new_weight)

        # Used to calculate the different eps
        worst_solution = [1 for _ in range(graph.numNodes)]
        worst_weight = ILS.get_weight_solution(worst_solution, graph)

        best_solution = new_solution.copy()
        best_weight = new_weight

        prev_solution = new_solution.copy()

        # iterBsToRet = 1

        # Used to calculate the different eps
        min_weight = min(list(map(int, graph.listNodeWeights)))
        # eps = choose_eps(best_weight, worst_weight)
        eps_2, max_eps, min_eps = ILS.eps_init(int(best_weight), int(min_weight))

        num_elems_to_change, pert_max, pert_min = ILS.num_change_init(int(graph.numNodes))

        # Numero di sostituzioni massime in una iterazione di local search
        num_swaps = int(graph.numNodes) / 8
        if num_swaps < 10:
            num_swaps = 10

        current_swaps = 0
        accept = 50
        iter_acc = 1

        for current_iter in range(max_eval):
            # if eps < 25:
            perturbedSolution = ILS.perturbation(num_elems_to_change, new_solution)
            # else:
            #   perturbedSolution = ILS.second_perturbation(current_solution)

            new_solution, new_weight, best_solution, best_weight, num_elems_to_change, num_obj_func, current_swaps, \
            iter_without_improvements = ILS.iterative_local_search(perturbedSolution, graph, best_solution, best_weight,
                                                                   eps_2, min_eps, max_eps, pert_min,
                                                                   pert_max, num_obj_func, max_obj_func,
                                                                   max_iter_ls, num_swaps, current_swaps,
                                                                   iter_without_improvements, index_max)

            # eps = choose_eps(ILS.get_weight_solution(new_solution, graph), worst_weight)
            eps_2 = ILS.eps_scheduling(eps_2, new_weight, best_weight, max_eps, min_eps, current_iter, iter_without_improvements,
                                       index_max, max_iter_without_imprvs)
            iter_acc, accept, accepted = ILS.acceptance_criteria(best_solution, prev_solution, new_solution, eps_2,
                                                                 iter_without_improvements, max_iter_without_imprvs,
                                                                 graph, accept, iter_acc, current_iter, index_max)
            if accepted:
                if int(new_weight) < int(best_weight):
                    best_weight = new_weight
                    best_solution = new_solution.copy()
                    objfun_last = num_obj_func
                prev_solution = new_solution
                if current_iter % value_to_take_coords[index_max] == 0 or current_iter == (max_eval - 100):
                    coordY.append(new_weight)
                    coordX.append(current_iter + 1)
            else:
                new_solution = best_solution.copy()
                new_weight = best_weight

            '''coordY.append(ILS.get_weight_solution(new_solution, graph))
            coordX.append(current_iter + 1)'''

            print(
                "[*] - Best_weights: {} -- Curr_weights: {} -- eps: {} -- Iter_without_improv: {} -- Iter: {} -- "
                "Elem_change: {}".format(
                    ILS.get_weight_solution(best_solution, graph), ILS.get_weight_solution(new_solution, graph), eps_2,
                    iter_without_improvements, current_iter, num_elems_to_change))

        coordY.append(best_weight)
        coordX.append(current_iter + 1)
        end = time.time()
        final_time = end - start

        print_solution = [i for i in range(len(best_solution)) if best_solution[i] != 0]

        print("[*] - Soluzione migliore trovata: {}".format(print_solution))
        print("[1] - Soluzione migliore trovata: {}".format(best_solution))
        print("[2] - Soluzione migliore trovata: {}".format(
            [graph.listNodeWeights[i] for i in range(len(best_solution)) if best_solution[i] != 0]))
        print("[*] - Peso: {}".format(ILS.get_weight_solution(best_solution, graph)))

        init_weight = ILS.get_weight_solution(initial_solution, graph)
        final_weight = ILS.get_weight_solution(best_solution, graph)
        graph.save_output(init_weight, final_weight, print_solution, coordX, coordY)

        valid_solution = ILS.valid_solution(best_solution, graph)

        df.loc[len(df)] = [graph.path_file[6:], init_weight, final_weight, print_solution, eps_2, final_time,
                           valid_solution]
        print(
            "[*] - Id: {} -- init_weight: {} -- final_weight: {} -- print_solution: {} -- eps: {} -- final_time: {} "
            "-- valid: {}".format(
                graph.path_file[6:], init_weight, final_weight, print_solution, eps_2, final_time, valid_solution))

    df.to_csv("benchmark/benchmark_{}.csv".format(graph.path_file[6:-4]), index=False)


if __name__ == "__main__":
    main()
