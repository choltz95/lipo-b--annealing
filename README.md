# lipo-b--annealing

Pile of code for IC layout

B*-Tree Annealer partially based on Prof. Yao-Wen Chang's implementation and materials

Includes Python interface supporting multi-start and parameter tuning via Lipschitz Optimization (LIPO)

Wirelength (HPWL)-driven Mixed Integer Lienar Program (MILP) detailed placer

Parallelism handled with Python through joblib

Remember to set the root path variable in python scripts.
Multistart relies on a './tmp' directory which should initially contain all design
files (.nets, .block, etc) and will store the resulting .block and .rprt files
on completion.

The log dir \& plots.ipynb is for lipo iterations, not annealing iterations.

deps for Python:
- joblib (parallism)
- tqdm   (nice progress bars)
- dlib   (for lcb-lipo tuning framework)
- CVXPY & Coin-OR for detailed placement


### Build
>./clean

>cmake .

>make
