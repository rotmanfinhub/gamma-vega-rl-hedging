""" Utility Functions & Imports"""
import random
import math
import numpy
import numpy as np
import decimal
import scipy.linalg
import numpy.random as nrand
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch import det
from tqdm import tqdm
from scipy.stats import norm
import pandas as pd
from scipy.integrate import quad
from scipy import optimize
import scipy.stats as si
from abc import ABC, abstractmethod
from environment.Trading import Option, SyntheticOption
from absl import flags
FLAGS = flags.FLAGS
import numpy as np
random.seed(1)


class Utils:
    def __init__(self, init_ttm, np_seed, num_sim, mu=0.0, init_vol=0.2, 
                 s=10, k=10, r=0, q=0, t=252, frq=1, spread=0,
                 hed_ttm=60, beta=1, rho=-0.7, volvol=0.6, ds=0.001, 
                 poisson_rate=1, moneyness_mean=1.0, moneyness_std=0.0, ttms=None, 
                 num_conts_to_add = -1, contract_size = 100,
                 action_low=0, action_high=3, kappa = 0.0, svj_rho = -0.1, mu_s=0.2, sigma_sq_s=0.1, lambda_d=0.2, gbm = False, sabr=False):
        if ttms is None:
            # ttms = [60, 120, 240, 480]
            ttms = [120]   # single ttm
        # Annual Return
        self.mu = mu
        # Annual Volatility
        self.init_vol = init_vol
        # Initial Asset Value
        self.S = s
        # Option Strike Price
        self.K = k
        # Annual Risk Free Rate
        self.r = r
        # Annual Dividend
        self.q = q
        # Annual Trading Day
        self.T = t
        # frequency of trading
        self.frq = frq
        self.dt = self.frq / self.T
        # Number of simulations
        self.num_sim = num_sim
        # Initial time to maturity
        self.init_ttm = init_ttm
        # Number of periods
        self.num_period = int(self.init_ttm / self.frq)
        # Time to maturity for hedging options
        self.hed_ttm = hed_ttm
        # Add Option Moneyness mean, std
        self.moneyness_mean = moneyness_mean
        self.moneyness_std = moneyness_std
        # Spread of buying and selling options
        self.spread = spread
        # Action space
        self.action_low = action_low
        self.action_high = action_high
        # SABR parameters
        self.beta = beta
        self.rho = rho
        self.volvol = volvol
        self.ds = ds
        self.implied_vol = None
        self.poisson_rate = poisson_rate
        self.ttms = ttms
        self.num_conts_to_add = num_conts_to_add
        # contract size, each option contract corresponds to how many underlying shares
        self.contract_size = contract_size
        # Stochastic Processes Indocators
        self.gbm = gbm
        self.sabr = sabr

        # set the np random seed
        np.random.seed(np_seed)
        self.seed = np_seed

    def _brownian_sim(self):
        z = np.random.normal(size=(self.num_sim, self.num_period + 1))

        a_price = np.zeros((self.num_sim, self.num_period + 1))
        a_price[:, 0] = self.S

        for t in range(self.num_period):
            a_price[:, t + 1] = a_price[:, t] * np.exp(
                (self.mu - (self.init_vol ** 2) / 2) * self.dt + self.init_vol * np.sqrt(self.dt) * z[:, t]
            )
        return a_price

    # BSM Call Option Pricing Formula & BS Delta, Gamma, Vega formula
    # T here is time to maturity
    @staticmethod
    def bs_call(iv, ttm: np.ndarray, S: np.ndarray, K, r, q, T=250):
        """Black Scholes Formula

        Args:
            iv (np.ndarray): implied volatility. it can be a scalar for constant vol; 
                             or it has the same shape of stock price S for stochastic vol.
            ttm (np.ndarray): time to maturity, it should have the same channel dim as stock price S.
                              If batch dim is not presented, ttm batch dim will be expanded to match S.
            S (np.ndarray): stock price.
            K (np.ndarray): option strike, it should have the same dim as stock price S.
            r (float): risk free rate
            q (float): dividend yield.
            T (int, optional): business days in a year. Defaults to 250.

        Returns:
            np.ndarray: option price in the same shape of stock price S.
            np.ndarray: option delta in the same shape of stock price S.
            np.ndarray: option gamma in the same shape of stock price S.
            np.ndarray: option vega in the same shape of stock price S.
        """
        if (ttm.ndim + 1) == S.ndim:
            # expand batch dim
            ttm = np.tile(np.expand_dims(ttm, 0), (S.shape[0],) + tuple([1]*ttm.ndim))
        assert (ttm.ndim == S.ndim), 'Maturity dim does not match spot dim'
        for i in range(ttm.ndim):
            assert ttm.shape[i] == S.shape[i], f'Maturity dim {i} size does not match spot dim {i} size.'
        # active option
        active_option = (ttm > 0).astype(np.uint0)
        matured_option = (ttm == 0).astype(np.uint0)

        # active option
        fT = np.maximum(ttm, 1)/T
        d1 = (np.log(S / K) + (r - q + iv * iv / 2) * np.abs(fT)) / (iv * np.sqrt(fT))
        d2 = d1 - iv * np.sqrt(fT)
        n_prime = np.exp(-1 * d1 * d1 / 2) / np.sqrt(2 * np.pi)

        active_bs_price = S * np.exp(-q * fT) * norm.cdf(d1) - K * np.exp(-r * fT) * norm.cdf(d2)
        active_bs_delta = np.exp(-q * fT) * norm.cdf(d1)
        active_bs_gamma = (n_prime * np.exp(-q * fT)) / (S * iv * np.sqrt(fT))
        active_bs_vega = (1/100) * S * np.exp(-q * fT) * np.sqrt(fT) * n_prime
        
        # matured option
        payoff = np.maximum(S - K, 0)

        # consolidate
        price = active_option*active_bs_price + matured_option*payoff
        delta = active_option*active_bs_delta
        gamma = active_option*active_bs_gamma
        vega = active_option*active_bs_vega

        return price, delta, gamma, vega

    def get_sim_path(self):
        """ Simulate BSM underlying dynamic 
        
        Returns:
            np.ndarray: underlying asset price in shape (num_path, num_period)
            np.ndarray: scalar constant vol
        """

        # asset price 2-d array
        print("Generating asset price paths")
        a_price = self._brownian_sim()

        return a_price, np.array(self.init_vol)

    def _sabr_sim(self):
        """Simulate SABR model
        1). stock price
        2). instantaneous vol

        Returns:
            np.ndarray: stock price in shape (num_path, num_period)
            np.ndarray: instantaneous vol in shape (num_path, num_period)
        """
        qs = np.random.normal(size=(self.num_sim, self.num_period + 1))
        qi = np.random.normal(size=(self.num_sim, self.num_period + 1))
        qv = self.rho * qs + np.sqrt(1 - self.rho * self.rho) * qi

        vol = np.zeros((self.num_sim, self.num_period + 1))
        vol[:, 0] = self.init_vol

        a_price = np.zeros((self.num_sim, self.num_period + 1))
        a_price[:, 0] = self.S

        for t in range(self.num_period):
            gvol = vol[:, t] * (a_price[:, t] ** (self.beta - 1))
            a_price[:, t + 1] = a_price[:, t] * np.exp(
                (self.mu - (gvol ** 2) / 2) * self.dt + gvol * np.sqrt(self.dt) * qs[:, t]
            )
            vol[:, t + 1] = vol[:, t] * np.exp(
                -self.volvol * self.volvol * 0.5 * self.dt + self.volvol * qv[:, t] * np.sqrt(self.dt)
            )

        return a_price, vol

    def _sabr_implied_vol(self, vol, tt, price):
        """Convert SABR instantaneous vol to option implied vol

        Args:
            vol (np.ndarray): SABR instantaneous vol in shape (num_path, num_period)
            tt (np.ndarray): time to maturity in shape (num_period,)
            price (np.ndarray): underlying stock price in shape (num_path, num_period)

        Returns:
            np.ndarray: implied vol in shape (num_path, num_period)
        """
        F = price * np.exp((self.r - self.q) * tt)
        x = (F * self.K) ** ((1 - self.beta) / 2)
        y = (1 - self.beta) * np.log(F / self.K)
        A = vol / (x * (1 + y * y / 24 + y * y * y * y / 1920))
        B = 1 + tt * (
                ((1 - self.beta) ** 2) * (vol * vol) / (24 * x * x)
                + self.rho * self.beta * self.volvol * vol / (4 * x)
                + self.volvol * self.volvol * (2 - 3 * self.rho * self.rho) / 24
        )
        Phi = (self.volvol * x / vol) * np.log(F / self.K)
        Chi = np.log((np.sqrt(1 - 2 * self.rho * Phi + Phi * Phi) + Phi - self.rho) / (1 - self.rho))

        SABRIV = np.where(F == self.K, vol * B / (F ** (1 - self.beta)), A * B * Phi / Chi)

        return SABRIV

    def get_sim_path_sabr(self):
        """ Simulate SABR underlying dynamic and implied volatility dynamic 
        
        Returns:
            np.ndarray: underlying asset price in shape (num_path, num_period)
            np.ndarray: implied volatility in shape (num_path, num_period)
        """

        # asset price 2-d array; sabr_vol
        print("Generating asset price paths (SABR)")
        a_price, sabr_vol = self._sabr_sim()

        # time to maturity "rank 1" array: e.g. [M, M-1, ..., 0]
        ttm = np.arange(self.init_ttm, -self.frq, -self.frq, dtype=np.int16)

        # BS price 2-d array and bs delta 2-d array
        print("Generating implied vol")

        # SABR implied vol
        implied_vol = self._sabr_implied_vol(sabr_vol, ttm / self.T, a_price)

        self.implied_vol = implied_vol
        return a_price, implied_vol

    def init_env(self):
        """Initialize environment
        Entrypoint to simulate market dynamics: 
        1). stock prices 
        2). implied volatilities

        If it is constant vol environment, then BSM model is used.
        If it is stochastic vol environment, then SABR model is used.

        Returns:
            np.ndarray: underlying asset price in shape (num_path, num_period)
            np.ndarray: implied volatility in shape (num_path, num_period)
        """
        if self.sabr:
            return self.get_sim_path_sabr()
        elif self.gbm:
            return self.get_sim_path()

    def atm_hedges(self, a_prices, vol):
        """Generate ATM hedging options' prices & risk profiles

        Args:
            a_prices (np.ndarray): simulated underlying asset prices in shape (num_episodes, num_steps)
            vol (np.ndarray): simulated volatilities. it is either a constant vol for BSM model,
                              or an (num_episodes, num_steps) array for SABR model

        Returns:
            np.ndarray: a list of Options in shape (num_episodes, num_hedges), 
                        here num_hedges = num_steps as a new ATM option is initiated at every time step 
        """
        print("Generate hedging portfolio option prices and risk profiles")
        hedge_ttm = self.hed_ttm 
        num_episode = a_prices.shape[0]
        num_hedge = num_step = a_prices.shape[1]
        price = np.empty((num_episode, num_step, num_hedge), dtype=float)
        delta = np.empty_like(price, dtype=float)
        gamma = np.empty_like(price, dtype=float)
        vega = np.empty_like(price, dtype=float)
        inactive_option = np.empty_like(price, dtype=np.bool8)

        # generate portfolio price & risk profiles for each step 
        # step dimension is the smallest size for a loop
        all_option_ttms = None         # all options ttms
        all_option_strikes = None       # all options strikes
        for step_i in tqdm(range(num_step)):
            if all_option_ttms is not None:
                # decrease ttm of accumulated options
                all_option_ttms -= 1
            # new options shape for calculation is (num_episode, )
            step_a_prices = a_prices[:, step_i]
            option_ttms = hedge_ttm*np.ones((num_episode,))
            option_strikes = step_a_prices
            # add this step's new options
            if all_option_ttms is not None:
                all_option_ttms = np.c_[all_option_ttms, option_ttms[:,None]]
                all_option_strikes = np.c_[all_option_strikes, option_strikes[:,None]]
            else:
                all_option_ttms = option_ttms[:,None]
                all_option_strikes = option_strikes[:,None]
            # vol
            if vol.ndim == 0:
                step_vol = vol
            else:
                step_vol = np.tile(np.expand_dims(vol[:, step_i], -1), (1,all_option_ttms.shape[1]))
            # expand stock price to (num_episode, step_num_opts)
            step_a_prices = np.tile(np.expand_dims(step_a_prices, -1), (1,all_option_ttms.shape[1]))
            # bs price and risk profiles
            step_option_price, step_option_delta, step_option_gamma, step_option_vega = \
                self.bs_call(step_vol,all_option_ttms,step_a_prices, all_option_strikes,self.r,self.q,self.T)
            
            for option_i in range(num_hedge):
                if (option_i > step_i) or (all_option_ttms[:, option_i].mean() < 0):
                    # option is not initiated or option expired
                    inactive_option[:, step_i, option_i] = True
                    price[:, step_i, option_i] = 0
                    delta[:, step_i, option_i] = 0
                    gamma[:, step_i, option_i]  = 0
                    vega[:, step_i, option_i] = 0
                else:
                    # option expired
                    inactive_option[:, step_i, option_i] = False
                    price[:, step_i, option_i] = step_option_price[:, option_i]
                    delta[:, step_i, option_i] = step_option_delta[:, option_i]
                    gamma[:, step_i, option_i] = step_option_gamma[:, option_i]
                    vega[:, step_i, option_i] = step_option_vega[:, option_i]

        # construct options in shape (num_sim, num_hedge)
        print("Initialize hedging portfolio options.")
        options = []
        for ep_i in tqdm(range(num_episode)):
            options.append([])
            for option_i in range(num_hedge):
                options[-1].append(Option(price[ep_i,:,option_i], delta[ep_i,:,option_i], 
                                          gamma[ep_i,:,option_i], vega[ep_i,:,option_i], 
                                          inactive_option[ep_i,:,option_i], 0, self.contract_size))              
        return np.array(options)

    def agg_poisson_dist(self, a_prices, vol):
        """Generate Poisson arrival options' prices & risk profiles
        
        One option is added at initial day (time step 0)
        Generate liability options and aggregate their prices and risk profiles 
        to represent liability portfolio price and risk profiles

        Args:
            a_prices (np.ndarray): simulated underlying asset prices in shape (num_episodes, num_steps)
            vol (np.ndarray): simulated volatilities. it is either a constant vol for BSM model,
                              or an (num_episodes, num_steps) array for SABR model

        Returns:
            np.ndarray: a list of synthetic Options in shape (num_episodes,), one for each episode.  
        """
        print("Genrate Poisson arrival portfolio option prices and risk profiles")
        options = []
        num_opts_per_day = np.random.poisson(self.poisson_rate, a_prices.shape)
        num_episode = a_prices.shape[0]
        num_step = a_prices.shape[1]
        max_num_opts = num_opts_per_day.max(axis=0) # maximum number of options for each time step

        # generate portfolio price & risk profiles for each step 
        # step dimension is the smallest size for a loop
        all_option_ttms = None         # all options ttms
        all_option_strikes = None       # all options strikes
        all_option_buysell = None       # all options buy/sell position
        port_price = next_port_price = port_delta = port_gamma = port_vega = None
        for step_i in tqdm(range(num_step)):
            if all_option_ttms is not None:
                # decrease ttm of accumulated options
                all_option_ttms -= 1
            # options shape for calculation is (num_episode, max_num_opts at step_i)
            if step_i == 0:
                ep_num_opts = np.ones(num_episode)
                step_num_opts = 1
            else:
                ep_num_opts = num_opts_per_day[:, step_i]
                step_num_opts = max_num_opts[step_i]
            step_a_prices = a_prices[:, step_i]
            # randomize option time to maturities by selecting from self.ttms
            option_ttms = np.random.choice(self.ttms, (num_episode, step_num_opts))
            # flush non-existing option's ttm to -1, 
            # so option's price and risk profile will be calculated as 0 from bs_call
            option_ttms[ep_num_opts[:,None]<=np.arange(option_ttms.shape[1])] = -1
            # randomize option strikes
            moneyness = np.random.normal(self.moneyness_mean, self.moneyness_std, (num_episode, step_num_opts))
            option_strikes = step_a_prices[:,None]*moneyness # ATM
            # randomize buy or sell equal likely
            if step_i == 0:
                # step 0 - always underwrite one option
                option_buysell = np.ones((num_episode, step_num_opts), dtype=a_prices.dtype)
            else:
                option_buysell = np.random.choice([-1.0,1.0], (num_episode, step_num_opts))
            # add this step's new options
            if all_option_ttms is not None:
                all_option_ttms = np.c_[all_option_ttms, option_ttms]
                all_option_strikes = np.c_[all_option_strikes, option_strikes]
                all_option_buysell = np.c_[all_option_buysell, option_buysell]
            else:
                all_option_ttms = option_ttms
                all_option_strikes = option_strikes
                all_option_buysell = option_buysell
            # vol
            if vol.ndim == 0:
                step_vol = vol
            else:
                step_vol = np.tile(np.expand_dims(vol[:, step_i], -1), (1,all_option_ttms.shape[1]))
            # expand stock price to (num_episode, step_num_opts)
            step_a_prices = np.tile(np.expand_dims(step_a_prices, -1), (1,all_option_ttms.shape[1]))
            # bs price and risk profiles
            step_port_price, step_port_delta, step_port_gamma, step_port_vega = \
                self.bs_call(step_vol,all_option_ttms,step_a_prices, all_option_strikes,self.r,self.q,self.T)
            step_port_price *= all_option_buysell
            step_port_delta *= all_option_buysell
            step_port_gamma *= all_option_buysell
            step_port_vega *= all_option_buysell
            if step_i > 0:
                if step_num_opts > 0:
                    step_next_port_price = step_port_price[:,:(-step_num_opts)].sum(axis=1) # only consider the positions from last step
                else:
                    step_next_port_price = step_port_price.sum(axis=1)
            step_port_price = step_port_price.sum(axis=1)
            step_port_delta = step_port_delta.sum(axis=1)
            step_port_gamma = step_port_gamma.sum(axis=1)
            step_port_vega = step_port_vega.sum(axis=1)
            if step_i > 0:
                if step_i > 1:
                    # greater than step 2, concatenate 
                    next_port_price = np.c_[next_port_price, step_next_port_price[:, None]]
                else:
                    # at step 2, initialize next port price
                    next_port_price = step_next_port_price[:, None]
                port_price = np.c_[port_price, step_port_price[:, None]]
                port_delta = np.c_[port_delta, step_port_delta[:, None]]
                port_gamma = np.c_[port_gamma, step_port_gamma[:, None]]
                port_vega = np.c_[port_vega, step_port_vega[:, None]]
            else:
                port_price = step_port_price[:, None]
                port_delta = step_port_delta[:, None]
                port_gamma = step_port_gamma[:, None]
                port_vega = step_port_vega[:, None]
        
        print("Initialize Poisson arrival liability portfolio options.")
        for ep_i in tqdm(range(num_episode)):
            options.append(
                SyntheticOption(port_price[ep_i,:], next_port_price[ep_i,:], port_delta[ep_i,:], port_gamma[ep_i,:], port_vega[ep_i,:], 
                                np.zeros((num_step,)).astype(np.bool8), self.num_conts_to_add, self.contract_size))
        return np.array(options)
