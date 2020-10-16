from copy import deepcopy
from functools import reduce
import operator
from joblib import Parallel, delayed
from subprocess import PIPE, run
from tqdm import tqdm
import tempfile

from tree import MultiStart

import sys, time

def _partial(func, *part_args):
    def wrapper(*extra_args):
        args = list(part_args)
        args.extend(extra_args)
        return func(*args)

    return wrapper

def _process_worker(cmd):
    cmd = [str(cm) for cm in cmd]
    return run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)

def _place_worker(cmd_path, cmd_params):
    cmd = ('sh',) + (cmd_path,) + cmd_params
    res = _process_worker(cmd)
    returncode = res.returncode
    stdout = res.stdout
    stderr = res.stderr
    parse = [[string.lower().strip() for string in line.split()] for line in stdout.splitlines()]
    parse = [['_'.join(x[:-1])] + [x[-1]] if (len(x) > 1 and ':' in x[-2]) else '_'.join(x) for x in parse]
    # remove header and footer
    parse = parse[1:-1]

    # keys:
    return parse

def sample_params():
    #P, alpha, beta, k, rnd, c
    return (0.9, 0.0, 1.0, 5.0, 2, 2.0,)

def _sample_worker(cmd_path, cmd_params, idx):
    sp = sample_params()
    result = _place_worker(cmd_path, cmd_params + sp)
    assert len(result) > 0
    cost = float(result[-4][-1])
    area = float(result[-8][-1])
    hpwl = float(result[-6][-1])

    #log = ','.join([str(sp) for sp in sp]+[str(cost), str(area),str(hpwl)])

    return hpwl

def _multistart_worker(cmd_path, design_name, pid, idx):
    outfname = 'tmp/{}_{}'.format(pid, idx)

    #_fn = _partial(_sample_worker, cmd_path, design_name, idx)

    start_time = time.time()
    cost = _sample_worker(cmd_path, (design_name,outfname,), idx)
    opt_time = time.time() - start_time

    #tqdm.write('design: {} params: {} cost: {} in {} sec'.format(design_name, ' '.join([str(x) for x in p]), cost, opt_time))
    return cost, pid, outfname

def multistart(cmd_path, design_name, max_iterations, k, ncores=1):
    ms = MultiStart(k, max_iterations)
    _worker = _partial(_multistart_worker, cmd_path)

    curids = list(range(k))
    curfnames = [design_name]*k
    # spawn an annealer for each instance
    res = Parallel(n_jobs=ncores)(delayed(_worker)(fname, pid, idx) for idx, (fname, pid) in enumerate(tqdm(zip(curfnames,curids))))
    for r in res:
        c, pid, fname = r
        ms.add(pid, c, fname)

    # main loop
    for _ in range(max_iterations):
        # get top-k results
        curccosts, curids, curfnames = ms.get_topk()
        # spawn an annealer for each instance
        res = Parallel(n_jobs=nccores)(delayed(_worker)(fname, pid, idx) for idx, (fname, pid) in enumerate(tqdm(zip(curfnames,curids))))
        for r in res:
            c, pid, fname = r
            ms.add(self, pid, c, fname)

    return ms

def main():
    cmd_path = '/Users/orange3xchicken/lipo-b--annealing/run'
    design_name = 'ami33'
    ncores = 1
    max_iterations = 0
    k = 5 # split on top 5 results

    # multistart object
    ms = multistart(cmd_path, design_name, max_iterations, k, ncores)

    print('hpwl: {} fname: {}'.format(ms.get_topk()[0][0], ms.get_topk()[-1][0]))


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
