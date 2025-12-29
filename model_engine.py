import numpy as np
import pandas as pd
from arch import arch_model
import warnings

# Suppress convergence warnings to keep output clean for the user
warnings.filterwarnings("ignore")

class ForecastModel:
    def __init__(self):
        self.num_paths = 1000
        self.forecast_horizon = 288 # 24 hours * 12 (5-min intervals)

    def fit_and_simulate(self, prices_df, current_price=None):
        """
        Fits GARCH model and generates simulated price paths.
        
        Args:
            prices_df (pd.DataFrame): must contain 'close' column.
            current_price (float): Optional override for start price.
            
        Returns:
            list of lists: [ [price_path_1], [price_path_2], ... ]
        """
        # Calculate log returns for better statistical properties
        # r_t = ln(P_t / P_{t-1})
        prices = prices_df['close'].values
        returns = 100 * np.diff(np.log(prices)) # Percentage log returns
        
        # Fit GARCH(1,1) with Student's t-distribution
        # Volatility model: GARCH
        # Mean model: Zero (assuming short term random walk behavior dominantly) or Constant
        # Dist: Student T ('t')
        am = arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='t', mean='Zero')
        res = am.fit(disp='off')
        
        # Simulate
        # We need to simulate future returns.
        # arch library has a simulation method, but often it's for simulating *from* parameters from scratch.
        # We want to forecast *from the end of the series*.
        
        # forecast() gives analytic variances, but for full path simulation we need to manually simulate 
        # using the fitted parameters and the last state.
        
        params = res.params
        
        # Extract params
        omega = params['omega']
        alpha = params['alpha[1]']
        beta = params['beta[1]']
        nu = params['nu'] # Degrees of freedom for Student's T
        
        # Last variance and return from the data to seed the simulation
        last_vol = res.conditional_volatility[-1]
        last_return = returns[-1]
        
        simulated_paths = []
        
        # Vectorized simulation for speed?
        # Simulating conditional volatility is recursive, so hard to fully simplify without loop.
        # But we can loop over time steps and vectorize over paths.
        
        # Initialize arrays
        # shape: (num_paths, horizon)
        sim_returns = np.zeros((self.num_paths, self.forecast_horizon))
        
        # Initial volatility (variance) for all paths matches the last observed state
        # We project variance forward
        current_vol_sq = np.full(self.num_paths, last_vol**2)
        current_ret_sq = np.full(self.num_paths, last_return**2)
        
        # Random innovations: Student's t distribution
        # t.rvs(df, size)
        # We need standard t innovations, then scale by sigma
        dt = 1/288 # If we wanted to scale drift, but GARCH is usually unitless in steps
        
        for t in range(self.forecast_horizon):
            # GARCH(1,1) update: sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
            # Here epsilon_{t-1}^2 is basically the squared residual from previous step.
            # In GARCH notation: r_t = sigma_t * z_t
            # sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
            
            # Update volatility (variance)
            next_vol_sq = omega + alpha * current_ret_sq + beta * current_vol_sq
            next_vol = np.sqrt(next_vol_sq)
            
            # Generate innovations z_t ~ StudentT(nu)
            # numpy/scipy t distribution is standard (mean 0, var = nu/(nu-2))
            # arch model 't' assumes standardized innovation with var=1? 
            # Actually arch's 't' distribution is standardized so variance is 1.
            # So we typically need to scale standardized T random vars if using numpy's standard T which has var > 1.
            # Numpy: var = df / (df-2). To standardize: divide by sqrt(df/(df-2))
            std_correction = np.sqrt(nu / (nu - 2)) if nu > 2 else 1.0
            z_t = np.random.standard_t(nu, size=self.num_paths) / std_correction
            
            # Calculate return r_t = sigma_t * z_t
            r_t = next_vol * z_t
            
            # Store
            sim_returns[:, t] = r_t
            
            # Update state for next step
            current_ret_sq = r_t**2
            current_vol_sq = next_vol_sq

        # Reconstruct Prices
        # P_t = P_0 * exp(cumsum(r_t / 100))
        # (Since we used 100 * log returns)
        
        start_p = current_price if current_price else prices[-1]
        
        # Cumulative sum of returns along the time axis
        cum_returns = np.cumsum(sim_returns, axis=1)
        
        # Calculate price paths
        price_paths = start_p * np.exp(cum_returns / 100.0)
        
        # Filter negative prices (highly unlikely with log returns, but just in case of weird overflow)
        price_paths = np.maximum(price_paths, 0.00000001)
        
        return price_paths.tolist()
