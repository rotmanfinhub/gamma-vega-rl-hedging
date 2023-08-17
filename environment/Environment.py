"""A trading environment"""
from typing import Optional
import dataclasses

import gym
from gym import spaces
from acme.utils import loggers
from absl import flags
FLAGS = flags.FLAGS

from acme.utils import loggers

import numpy as np

from environment.Trading import MainPortfolio

@dataclasses.dataclass
class StepResult:
    """Logging step metrics for analysis
    """
    episode: int = 0
    t: int = 0
    hed_action: float = 0.
    hed_share: float = 0.
    # clip_hed_action: float = 0.
    stock_price: float = 0.
    stock_position: float = 0.
    stock_pnl: float = 0.
    liab_port_gamma: float = 0.
    liab_port_vega: float = 0.
    liab_port_pnl: float = 0.
    hed_cost: float = 0.
    hed_port_gamma: float = 0.
    hed_port_vega: float = 0.
    hed_port_pnl: float = 0.
    gamma_before_hedge: float = 0.
    gamma_after_hedge: float = 0.
    vega_before_hedge: float = 0.
    vega_after_hedge: float = 0.
    step_pnl: float = 0.
    state_price: float = 0.
    state_gamma: float = 0.
    state_vega: float = 0.
    state_hed_gamma: float = 0.
    state_hed_vega: float = 0.

class TradingEnv(gym.Env):
    """
    This is the Gamma & Vega Trading Environment.
    """

    # trade_freq in unit of day, e.g 2: every 2 day; 0.5 twice a day;
    def __init__(self, utils, logger: Optional[loggers.Logger] = None):

        super(TradingEnv, self).__init__()
        self.logger = logger
        # seed and start
        self.seed(utils.seed)

        # simulated data: array of asset price, option price and delta paths (num_path x num_period)
        # generate data now
        self.portfolio = MainPortfolio(utils)
        self.utils = utils

        # other attributes
        self.num_path = self.portfolio.a_price.shape[0]

        # set num_period: initial time to maturity * daily trading freq + 1 (see get_sim_path() in utils.py)
        self.num_period = self.portfolio.a_price.shape[1]

        self.sim_episode = -1

        # track time step within an episode (it's step)
        self.t = None

        # time to maturity array
        self.ttm_array = np.arange(self.utils.init_ttm, -self.utils.frq, -self.utils.frq)

        # Action space: HIGH value has to be adjusted with respect to the option used for hedging
        self.action_space = spaces.Box(low=np.array([0]), 
                                       high=np.array([1.0]), dtype=np.float32)

        # Observation space
        max_gamma = self.portfolio.liab_port.max_gamma
        max_vega = self.portfolio.liab_port.max_vega
        obs_lowbound = np.array([self.portfolio.a_price.min(), 
                                 -1 * max_gamma * self.utils.contract_size, 
                                 -np.inf])
        obs_highbound = np.array([self.portfolio.a_price.max(), 
                                  max_gamma * self.utils.contract_size,
                                  np.inf])
        if FLAGS.vega_obs:
            obs_lowbound = np.concatenate([obs_lowbound, [-1 * max_vega * self.utils.contract_size,
                                                          -np.inf]])
            obs_highbound = np.concatenate([obs_highbound, [max_vega * self.utils.contract_size,
                                                            np.inf]])
        self.observation_space = spaces.Box(low=obs_lowbound,high=obs_highbound, dtype=np.float32)
            
        # Initializing the state values
        self.num_state = 5 if FLAGS.vega_obs else 3
        self.state = []

        # self.reset()

    def seed(self, seed):
        # set the np random seed
        np.random.seed(seed)

    def reset(self):
        """
        reset function which is used for each episode (spread is not considered at this moment)
        """

        # repeatedly go through available simulated paths (if needed)
        self.sim_episode = (self.sim_episode + 1) % self.num_path
        self.portfolio.reset(self.sim_episode)

        self.t = 0

        self.portfolio.liab_port.add(self.sim_episode, self.t, self.utils.num_conts_to_add)

        return self.portfolio.get_state(self.t)

    def step(self, action):
        """
        profit and loss period reward
        """
        result = StepResult(
            episode=self.sim_episode,
            t=self.t,
            hed_action=action[0],
        )
        # action constraints
        gamma_action_bound = -self.portfolio.get_gamma(self.t)/self.portfolio.hed_port.options[self.sim_episode, self.t].gamma_path[self.t]/self.utils.contract_size
        action_low = [0, gamma_action_bound]
        action_high = [0, gamma_action_bound]
        
        if FLAGS.vega_obs:
            # vega bounds
            vega_action_bound = -self.portfolio.get_vega(self.t)/self.portfolio.hed_port.options[self.sim_episode, self.t].vega_path[self.t]/self.utils.contract_size
            action_low.append(vega_action_bound)
            action_high.append(vega_action_bound)

        low_val = np.min(action_low)
        high_val = np.max(action_high)

        hed_share = low_val + action[0] * (high_val - low_val)
        result.hed_share = hed_share


        # current prices at t
        result.gamma_before_hedge = self.portfolio.get_gamma(self.t)
        result.vega_before_hedge = self.portfolio.get_vega(self.t)
        result.step_pnl = reward = self.portfolio.step(hed_share, self.t, result)
        result.gamma_after_hedge = self.portfolio.get_gamma(self.t)
        result.vega_after_hedge = self.portfolio.get_vega(self.t)
        
        self.t = self.t + 1

        state = self.portfolio.get_state(self.t)
        if self.t == self.num_period - 1:
            done = True
            state[1:] = 0
        else:
            done = False
        
        result.state_price, result.state_gamma, result.state_hed_gamma = state[:3]
        if FLAGS.vega_obs:
            result.state_vega, result.state_hed_vega = state[3:]
        
        # for other info later
        info = {"path_row": self.sim_episode}
        if self.logger:
            self.logger.write(dataclasses.asdict(result))
        return state, reward, done, info
