from copy import deepcopy
from functools import reduce
import operator
from joblib import Parallel, delayed
from subprocess import PIPE, run
from tqdm import tqdm
import tempfile

from tree import MultiStart

import sys, time

"""
Pickleable partial wrapper
the canonical partial'd function object is not pickleable
"""
def _partial(func, *part_args):
    def wrapper(*extra_args):
        args = list(part_args)
        args.extend(extra_args)
        return func(*args)

    return wrapper

"""
Process _worker
takes list of commands
and passes to subprocess
"""
def _process_worker(cmd):
    cmd = [str(cm) for cm in cmd]
    print(cmd)
    return run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)

"""
place worker
calls process worker and parses annealer feedback
"""
def _place_worker(cmd_path, cmd_params):
    cmd = ('sh',) + (cmd_path,) + cmd_params
    res = _process_worker(cmd)
    returncode = res.returncode
    stdout = res.stdout
    stderr = res.stderr
    # parses annealer report (example given at the bottom of this file)
    parse = [[string.lower().strip() for string in line.split()] for line in stdout.splitlines()]
    parse = [['_'.join(x[:-1])] + [x[-1]] if (len(x) > 1 and ':' in x[-2]) else '_'.join(x) for x in parse]
    parse = parse[1:-1]
    return parse

"""
Given a level of the tree (or other info)
samples a set of parameters for annealer
See Yao-wen Chang's ISPD paper on Fast-Annealing
"""
def sample_params(level):
    #P, alpha, beta, k, rnd, c
    # P should depend on the level and determines the inital temperature
    # T_init = d / log(P) where delta is the mean change in cost between transitions.
    return (0.9**level, 0.0, 1.0, 5.0, 2, 2.0,)

"""
Abstract annealer worker
calls place worker and parses feedback
into floats
"""
def _sample_worker(cmd_path, cmd_params, idx, level):
    sp = sample_params(level)
    result = _place_worker(cmd_path, cmd_params + sp)
    assert len(result) > 0
    cost = float(result[-4][-1])
    area = float(result[-8][-1])
    hpwl = float(result[-6][-1])
    return hpwl

"""
multistart worker
individual multistart process
"""
def _multistart_worker(ms, cmd_path, design_name, og_design_name, pid, idx):
    outfname = '{}'.format(idx)

    #_fn = _partial(_sample_worker, cmd_path, design_name, idx)
    level = ms.tree.find(pid)
    start_time = time.time()
    cost = _sample_worker(cmd_path, (design_name, og_design_name, outfname,), idx, level+1)
    opt_time = time.time() - start_time

    #tqdm.write('design: {} params: {} cost: {} in {} sec'.format(design_name,
    #                                                              ' '.join([str(x) for x in p]), cost, opt_time))
    return cost, pid, outfname

"""
top-level multistart fn
handles parallelism and
interfaces with the MultiStart object
"""
def multistart(cmd_path, design_name, max_iterations, k, ncores=1):

    # instantiate the MultiStart object
    ms = MultiStart(k, max_iterations)
    _worker = _partial(_multistart_worker, ms, cmd_path)

    #Do initial set of placements on the given design
    curids = [0]*k
    curfnames = [design_name]*k
    res = Parallel(n_jobs=ncores)(delayed(_worker)(fname, design_name, pid, idx)
                                  for idx, (fname, pid) in enumerate(tqdm(zip(curfnames,curids))))
    for r in res:
        c, pid, fname = r
        ms.add(pid, c, fname)

    # main loop
    for _ in range(max_iterations):
        # get top-k results and generate new ids
        curccosts, curids, curfnames = ms.get_topk()
        new_indices = ms.get_k_new_ids()
        # spawn an annealer for each top-k instance
        res = Parallel(n_jobs=ncores)(delayed(_worker)(fname, design_name, pid, idx)
                                      for fname, pid, idx in tqdm(zip(curfnames,curids, new_indices)))
        for r in res:
            c, pid, fname = r
            ms.add(pid, c, fname)

    return ms

def main():
    #TODO: collect all of this into a params object/dict
    cmd_path = '/home/orange3xchicken/milp/lipo-b--annealing/run' # path to root directory
    design_name = 'ami33'
    ncores = 1
    parratype = 'threaded' # determines joblib backend - either multiple threads or multiple cores
    max_iterations = 2
    k = 5 # split on top 5 results

    ms = multistart(cmd_path, design_name, max_iterations, k, ncores)
    ms.tree.print_tree()
    print()
    print('hpwl: {} fname: ./tmp/{}.rpt'.format(ms.get_topk()[0][0],
                                                ms.get_topk()[-1][0]))

if __name__ == "__main__":
    main()

"""
######################################################################
           input: ami33
   num of blocks: 33
num of terminals: 40
     num of nets: 121
            area: 1314768
 area difference: 0.0
            hpwl: 135828.0
 hpwl difference: 0.0
      total cost: 725298.0
            SAME
           LEGAL
        IN BOUND
######################################################################
"""
