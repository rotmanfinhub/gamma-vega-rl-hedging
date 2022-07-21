from abc import ABC, abstractmethod
from absl import flags
FLAGS = flags.FLAGS
import numpy as np

class AssetInterface(ABC):
    """Asset Interface
    Asset contains its price, and risk profiles.
    Any kind of asset can inherit this interface, i.e. single option, portfolio constructed by multi-assets and etc. 
    """

    @abstractmethod
    def step(self, t):
        """Step on time t and generate asset's P&L as reward"""
        pass

    @abstractmethod
    def get_value(self, t):
        """asset value at time t"""
        pass

    @abstractmethod
    def get_delta(self, t):
        """asset delta at time t"""
        pass
    
    @abstractmethod
    def get_gamma(self, t):
        """asset gamma at time t"""
        pass
    
    @abstractmethod
    def get_vega(self, t):
        """asset vega at time t"""
        pass

class Stock(AssetInterface):
    """Stock
    Stock delta is 1, other greeks are 0.
    Price path is in shape (num_episode, num_step).
    Active path stores the current episode stock trajectory, 
    and it is reset in MainPortfolio at the beginning of a new episode

    """
    def __init__(self, price_path) -> None:
        """Constructor

        Args:
            price_path (np.ndarray): simulated stock prices in shape (num_episode, num_step)
        """
        super().__init__()
        self.price_path = price_path
        self.active_path = []
        self.position = 0

    def set_path(self, sim_episode):
        self.active_path = self.price_path[sim_episode, :]
        self.position = 0

    def step(self, t):
        """Step on time t and generate stock P&L

        Args:
            t (int): step time t

        Returns:
            float: stock P&L from time t to t+1
        """
        return (self.active_path[t + 1] - self.active_path[t]) * self.position

    def get_value(self, t):
        """stock value at time t"""
        return self.position * self.active_path[t]
    
    def get_delta(self, t):
        """stock delta at time t"""
        return self.position * 1

    def get_gamma(self, t):
        """stock gamma at time t"""
        return 0

    def get_vega(self, t):
        """stock vega at time t"""
        return 0

class Option(AssetInterface):
    """Option storing its position information, price and risk profiles within its simulation path

        Price and risk profiles are in shape of (num_steps,)
        The option is active only when current time step passes its initialization time, and also
        time step is before its maturity time along the simulation trajectory.
        Inactive is a vector with same shape as (num_steps,), which tracks whether the
        option is inactive or not.
        If option is inactive, its price and risk profile will be zero. 
    """
    def __init__(self, 
                 price_path, 
                 delta_path, 
                 gamma_path, 
                 vega_path, 
                 inactive, 
                 num_contract,
                 contract_size=100):
        """Constructor
        
        Args:
            price_path (np.ndarray): option price
            delta_path (np.ndarray): option delta 
            gamma_path (np.ndarray): option gamma
            vega_path (np.ndarray): option vega
            inactive (np.ndarray): boolean array indicates option active status (True if it is inactive, o.w. False)
            num_contract (np.ndarray): number of contracts that is purchased (positive value) or sold (negative value).
            contract_size (int): one contract corresponds to how many underlying shares. (Default: 100)
        """
        super().__init__()
        self.price_path = price_path
        self.delta_path = delta_path
        self.gamma_path = gamma_path
        self.vega_path = vega_path
        self.num_contract = num_contract
        self.contract_size = contract_size
        self.inactive = inactive

    def step(self, t):
        """Step on time t and generate option P&L

        Args:
            t (int): step time t

        Returns:
            float: option P&L from time t to t+1
        """
        reward = 0
        if not self.inactive[t+1]:
            cur_option_price = self.price_path[t]
            next_option_price = self.price_path[t+1]
            reward = (next_option_price - cur_option_price) * self.num_contract * self.contract_size

        return reward

    def _get_profile(self, t, profile_path):
        """Get price or risk profile at time t

        Args:
            t (int): step time
            profile_path (np.ndarray): option price path or risk profile path

        Returns:
            float: if option is active at time t, return its value; o.w. return 0
        """
        if not self.inactive[t]:
            return profile_path[t] * self.num_contract * self.contract_size
        else:
            return 0

    def get_value(self, t):
        """price at time t"""
        return self._get_profile(t, self.price_path)

    def get_delta(self, t):
        """delta at time t"""
        return self._get_profile(t, self.delta_path)

    def get_gamma(self, t):
        """gamma at time t"""
        return self._get_profile(t, self.gamma_path)

    def get_vega(self, t):
        """vega at time t"""
        return self._get_profile(t, self.vega_path)


class SyntheticOption(Option):
    """Synthetic Option for Poisson arrival liability portfolio 
        
        This asset stores portfolio's current step prices, next step prices and risk profiles within its simulation path

        Current step prices are the portfolio price with positions at current time step t, including new arrival options at t.
        Next step prices are the portfolio price at time t+1 with option holdings from current time step t, without new arrival options at t+1.
        Step P&L considers the portfolio components are not changing from time t to t+1, so we maintain two portfolio prices  
    """
    def __init__(self, 
                 cur_price_path,
                 next_price_path, 
                 delta_path, 
                 gamma_path, 
                 vega_path, 
                 inactive, 
                 num_contract,
                 contract_size=100):
        """Constructor
        
        Args:
            cur_price_path (np.ndarray): option price at time t, portfolio positions includes new arrival options at t. shape (num_episode, num_step).
            next_price_path (np.ndarray): option price at time t+1, portfolio positions excludes new arrival options at t+1. shape (num_episode, num_step).
            delta_path (np.ndarray): option delta 
            gamma_path (np.ndarray): option gamma
            vega_path (np.ndarray): option vega
            inactive (np.ndarray): boolean array indicates option active status (True if it is inactive, o.w. False)
            num_contract (np.ndarray): number of contracts that is purchased (positive value) or sold (negative value).
            contract_size (int): one contract corresponds to how many underlying shares. (Default: 100)
        """
        super().__init__(cur_price_path, delta_path, gamma_path, vega_path, inactive, num_contract,contract_size)
        self.next_price_path = next_price_path
        
    def step(self, t):
        """Step on time t and generate option P&L

        Args:
            t (int): step time t

        Returns:
            float: option P&L from time t to t+1
        """
        reward = 0
        if not self.inactive[t+1]:
            cur_option_price = self.price_path[t]
            next_option_price = self.next_price_path[t]
            reward = (next_option_price - cur_option_price) * self.num_contract * self.contract_size

        return reward


class Portfolio(AssetInterface):
    """Hedging option portfolio 

    Hedging portfolio contains a list of ATM options for hedging purpose.
    At each step of a new episode, agent's hedging action determines number of contracts that is added into this portfolio.
    Each hedging action incurs a proportional transaction cost.
    After each episode finishes, this portfolio will be cleared through wiping out all active options in MainPortfolio.
    """
    def __init__(self, utils, option_generator, stock_prices, vol):
        """Constructor
        
        Generate hedging options' prices and risk profile by using util function utils.atm_hedges

        Args:
            utils (utils.Utils): environment configurations & util functions
            stock_prices (np.ndarray): simulated stock prices in shape (num_episodes, num_steps).
            vol (np.ndarray): simulated volatilities. it is either a constant vol for BSM model, 
                              or an (num_episodes, num_steps) array for SABR model
        """
        super().__init__()
        self.options = option_generator(stock_prices, vol)
        self.active_options = []
        self.utils = utils

    def reset(self):
        """Reset portfolio by clearing the active options.
        """
        self.active_options = []

    def step(self, t):
        """Step on time t and generate hedging option portfolio P&L

        Aggregate each option's P&L from time t to t+1, which are currently in the hedging portfolio. 

        Args:
            t (int): time step t

        Returns:
            float: hedging portfolio P&L
        """
        reward = 0
        for option in self.active_options:
            reward += option.step(t)
        if len(self.active_options) > 0 and self.active_options[0].inactive[t]:
            del self.active_options[0]
        return reward

    def add(self, sim_episode, t, num_contracts):
        """add option
        It is for hedging portfolio, so adding any new option incurs transaction cost 

        Args:
            sim_episode (int): current simulation episode
            t (int): current time step
            num_contracts (float): number of contracts to add

        Returns:
            float: transaction cost for adding hedging option (negative value)
        """
        opt_to_add = self.options[sim_episode, t]
        opt_to_add.num_contract = num_contracts
        self.active_options.append(opt_to_add)
        return -1 * np.abs(self.utils.spread * opt_to_add.get_value(t))
        
    def get_value(self, t):
        """portfolio value at time t"""
        value = 0
        for option in self.active_options:
            value += option.get_value(t)

        return value

    def get_delta(self, t):
        """portfolio delta at time t"""
        delta = 0
        for option in self.active_options:
            delta += option.get_delta(t)

        return delta

    def get_gamma(self, t):
        """portfolio gamma at time t"""
        gamma = 0
        for option in self.active_options:
            gamma += option.get_gamma(t)

        return gamma

    def get_vega(self, t):
        """portfolio vega at time t"""
        vega = 0
        for option in self.active_options:
            vega += option.get_vega(t)

        return vega


class LiabilityPortfolio(Portfolio):
    """Poisson arrival liability portfolio

    Liability portfolio price and risk profiles are aggregated after the initial simulation 
    and stored as a single synthetic option.
    So there is only one active option added into liability portfolio at the begining of a new episode.
    After an episode finishes, the portfolio is cleared in MainPortfolio.
    """
    def __init__(self, option_generator, stock_prices, vol):
        """Constructor

        Args:
            utils (utils.Utils): environment configurations & util functions
            stock_prices (np.ndarray): simulated stock prices in shape (num_episodes, num_steps).
            vol (np.ndarray): simulated volatilities. it is either a constant vol for BSM model, 
                              or an (num_episodes, num_steps) array for SABR model
        """
        super().__init__(0.0, option_generator, stock_prices, vol)
        self.max_gamma = 0
        self.max_vega = 0
        for option in self.options:
            option_max_gamma = np.abs(option.gamma_path * option.num_contract * option.contract_size).max()
            option_max_vega = np.abs(option.vega_path * option.num_contract * option.contract_size).max()
            if option_max_gamma > self.max_gamma:
                self.max_gamma = option_max_gamma
            if option_max_vega > self.max_vega:
                self.max_vega = option_max_vega
            
    def add(self, sim_episode, t, num_contracts):
        """This function is only effective at the beginning of a new episode 

        Args:
            sim_episode (int): episode
            t (int): time step
            num_contracts (float): number of contract to add, it is a constant value setup in configuration
        """
        if t == 0:
            opt_to_add = self.options[sim_episode]
            opt_to_add.num_contract = num_contracts
            self.active_options = [opt_to_add]


class MainPortfolio(AssetInterface):
    """Main Portfolio
    This is the total portfolio contains three components:
    1. Liability portfolio: Poisson arrival underwritten options
    2. Hedging option portfolio: ATM options that are purchased by agent for hedging purpose
    3. Underlying stock: Automatic delta neutral hedging position
    """
    def __init__(self, utils):
        """Constructor

        Args:
            utils (utils.Utils): contains environment configurations and util functions
        """
        super().__init__()
        self.utils = utils
        self.a_price, self.vol = utils.init_env()

        self.liab_port = LiabilityPortfolio(utils.agg_poisson_dist, self.a_price, self.vol)
        self.hed_port = Portfolio(utils, utils.atm_hedges, self.a_price, self.vol)
        self.underlying = Stock(self.a_price)
        self.sim_episode = -1
    
    def get_value(self, t):
        """portfolio value at time t"""
        return self.hed_port.get_value(t) + self.liab_port.get_value(t) + self.underlying.get_value(t)

    def get_delta(self, t):
        """portfolio delta at time t"""
        return self.hed_port.get_delta(t) + self.liab_port.get_delta(t) + self.underlying.get_delta(t)
    
    def get_gamma(self, t):
        """portfolio gamma at time t"""
        return self.hed_port.get_gamma(t) + self.liab_port.get_gamma(t)
    
    def get_vega(self, t):
        """portfolio vega at time t"""
        return self.hed_port.get_vega(t) + self.liab_port.get_vega(t)

    def get_state(self, t):
        """Environment States at time t
        
        1. Underlying price
        2. Total portfolio gamma
        3. heding option's gamma
        4. Total portfolio vega
        5. heding option's vega
        """
        price = self.underlying.active_path[t]
        gamma = self.get_gamma(t)
        hed_gamma = self.hed_port.options[self.sim_episode, t].gamma_path[t]*self.utils.contract_size
        states = np.array([price, gamma, hed_gamma])
        if FLAGS.vega_obs:
            vega = self.get_vega(t)
            hed_vega = self.hed_port.options[self.sim_episode, t].vega_path[t]*self.utils.contract_size
            states = np.concatenate([states, [vega, hed_vega]])
        return states

    def reset(self, sim_episode):
        """Reset portfolio at the begining of a new episode

        1. Clear hedging option portfolio
        2. Clear liability portfolio
        3. Set underlying stock to new episode and clear position
        """
        self.hed_port.active_options = []
        self.liab_port.active_options = []
        self.underlying.set_path(sim_episode)
        self.sim_episode = sim_episode

    def step(self, action, t,  result):
        """Step on time t and generate reward

        Args:
            action (float): hedging action
            t (int): time step
            result (StepResult): logging step metrics

        Returns:
            float: P&L as step reward
        """
        result.stock_price = self.a_price[self.sim_episode, t]
        result.hed_cost = reward = self.hed_port.add(self.sim_episode, t, action)
        result.stock_position = self.underlying.position = -1 * (self.hed_port.get_delta(t) + self.liab_port.get_delta(t))
        result.liab_port_gamma = self.liab_port.get_gamma(t)
        result.liab_port_vega = self.liab_port.get_vega(t)
        result.hed_port_gamma = self.hed_port.get_gamma(t)
        result.hed_port_vega = self.hed_port.get_vega(t)
        result.liab_port_pnl = self.liab_port.step(t)
        result.hed_port_pnl = self.hed_port.step(t)
        result.stock_pnl = self.underlying.step(t)
        reward += (result.liab_port_pnl + result.hed_port_pnl + result.stock_pnl)

        return reward
