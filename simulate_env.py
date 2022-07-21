import os
from pathlib import Path
import pickle
from typing import Mapping, Sequence

import acme.utils.loggers as log_utils
import numpy as np

from environment.New_Env import TradingEnv, DeltaTradingEnv
from environment.utils import Utils

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_float('vov', 0.0, 'Vol of vol, zero means BSM; non-zero means SABR (Default 0.0)')
flags.DEFINE_integer('hed_ttm', 20, 'Hedging option maturity in days (Default 20)')


def main(argv):
    work_folder = f'v={FLAGS.vov}_hedttm={FLAGS.hed_ttm}'
    work_folder =  "./logs/sim_env/" + work_folder
    if not os.path.exists(work_folder):
        os.makedirs(work_folder)
    # Create an environment, grab the spec, and use it to create networks.
    utils = Utils(init_ttm=60, np_seed=1234, num_sim=5_000, volvol=FLAGS.vov, sabr=FLAGS.vov!=0.0, hed_ttm=FLAGS.hed_ttm)
    if FLAGS.delta == True:
        training_env = DeltaTradingEnv(utils=utils)
    else:
        training_env = TradingEnv(utils=utils)
    eval_utils = Utils(init_ttm=60, np_seed=4321, num_sim=5_000, volvol=FLAGS.vov, sabr=FLAGS.vov!=0.0, hed_ttm=FLAGS.hed_ttm)
    if FLAGS.delta == True:
        eval_env = DeltaTradingEnv(utils=eval_utils)
    else:
        eval_env = TradingEnv(utils=eval_utils)
    
    with open(f'{work_folder}/training_env.pickle', 'wb') as handle:
            pickle.dump(training_env, handle, protocol=pickle.HIGHEST_PROTOCOL)  
    
    with open(f'{work_folder}/eval_env.pickle', 'wb') as handle:
            pickle.dump(eval_env, handle, protocol=pickle.HIGHEST_PROTOCOL)  

if __name__ == '__main__':
    app.run(main)
