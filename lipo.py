from copy import deepcopy
from functools import reduce
import operator
from joblib import Parallel, delayed
from subprocess import PIPE, run
#from functools import partial, wraps
from tqdm import tqdm

import dlib
import sys, time

def _process_worker(cmd):
    cmd = [str(cm) for cm in cmd]
    print(cmd)
    return run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)

def _place_worker(cmd_path, cmd_params):
    cmd = ['sh', cmd_path, *cmd_params]
    cmd.insert(-1,2)
    cmd.insert(5,0.0)
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

def _sample_worker(cmd_path, cmd_params, *sample_params):
    result = _place_worker(cmd_path, (cmd_params,) + sample_params)
    assert len(result) > 0
    cost = float(result[-4][-1])
    area = float(result[-8][-1])
    hpwl = float(result[-6][-1])

    log = ','.join([str(sp) for sp in sample_params]+[str(cost), str(area),str(hpwl)])

    with open('./log/'+cmd_params+'_log.txt','a+') as f:
        f.write( log + '\n')
    return hpwl

def _partial(func, *part_args):
    def wrapper(*extra_args):
        args = list(part_args)
        args.extend(extra_args)
        return func(*args)

    return wrapper

def _lipo_worker(cmd_path, design_name):
    #float P = 0.9, alpha_base = 0.5, beta = 0.1, R = float(H)/W;
    #int k = max(2, Nblcks/11), rnd = 2*Nblcks+20;
    #float c = max(100-int(Nblcks), 10), costs[2];

    Ps = [0.5,0.985]
    #alphas = [0.25, 0.75]
    betas = [0.0, 2.0]
    ks = [5,15]
    rnds = [1,5]
    cs = [1,5]

    #params = [Ps, alphas, betas, ks, rnds, cs]
    params = [Ps, betas, ks, cs]

    lbs = [p[0] for p in params]
    ubs = [p[1] for p in params]

    is_integer_variable = [False, False, True, False]


    _fn = _partial(_sample_worker, cmd_path, design_name)

    start_time = time.time()
    p,cost = dlib.find_min_global(_fn, lbs,
                                  ubs,
                                  is_integer_variable, 100)
    opt_time = time.time() - start_time
    tqdm.write('design: {} params: {} cost: {} in {} sec'.format(design_name, ' '.join([str(x) for x in p]), cost, opt_time))

def main():
    cmd_path = '/home/.../tuned-bstar-annealing/run'
    design_names = ['ami33', 'apte', 'hp','xerox','ami49']
    design_names=['ami33']
    design_name = 'ami33'
    cores = 1

    _worker = _partial(_lipo_worker, cmd_path)

    Parallel(n_jobs=cores)(delayed(_worker)(design_name) for design_name in tqdm(design_names))


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
