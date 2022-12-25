# MWVC_ILS

The aim of the project was to study and implement the Iterated Local Search to solve the problem of the Minimum Weight Vertex Cover belonging to the group of problems NP-Complete, therefore unsolvable with deterministic algorithms. A local search starts from a valid solution and iteratively builds a new solution trying to improve it until reaching a global optimal. During the execution of the algorithm the solution may end up in a very good local. Thanks to the perturbation operators and the acceptance criteria, the solution of local optimal is then removed by exploring new areas of the research space.


# Packages required
pip install -r requiriments.txt

# Usage
python main.py
