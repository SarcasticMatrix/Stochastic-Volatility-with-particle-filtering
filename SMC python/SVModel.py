import numpy as np
from random import gauss
from scipy.stats import norm
from scipy.optimize import minimize

from typing import Optional
from tqdm import tqdm

from Model import Model

class SVModel(Model):


    def __init__(
        self,

        # Time and particles parameters
        T: Optional[int] = 100,
        N: Optional[int] = 100,

        # Hidden States parameters
        x0: Optional[float] = 1,
        x: Optional[np.ndarray] = np.empty((100)),

        # Observations parameters
        y: Optional[np.ndarray] = np.empty((100)),

        # SV Model parameters 
        alpha: Optional[float] = 0,
        beta: Optional[float] = 0.9702,
        mu: Optional[float] = 0,
        W: Optional[float] = 1,

        seed: Optional[int] = None
    ):
        super().__init__(T=T, N=N, seed=seed, model="SV")

        self.x0 = x0
        self.x = x

        self.y = y

        self.alpha = alpha
        self.beta = beta
        self.mu = mu 
        self.W = W

        self.f = lambda x, alpha, beta, W: np.random.normal(alpha+beta*x,W,self.N) 
        self.g = lambda x, y, mu: norm.pdf(y,mu,np.exp(x/2))
    
    
    def generate_trajectory(self): 
        self.x = np.zeros((self.T))
        self.y = np.zeros((self.T))

        self.x[0] = self.x0
        self.y[0] = self.mu + np.exp(self.x[0]/2)*gauss(0,1)

        for k in range(1,self.T):
      
            self.x[k] = self.alpha + self.beta*self.x[k-1] + gauss(0,self.W)
            self.y[k] = self.mu + np.exp(self.x[k]/2) * gauss(0,1)

        return self.x,self.y        
    
    def SIR(self):
        
        self.x0 = np.random.randn(self.N)
        X = np.zeros((self.T,self.N))

        weights0 = norm.pdf(self.y[0],0,np.exp(self.x0/2))
        
        weights0 = weights0/(weights0.sum())
        weights = np.zeros((self.T,self.N))
        weights[0,:] = weights0
        
        # for t in range(1,T-1):
        for t in tqdm(range(1,self.T)):
            A = np.random.choice(range(self.N),self.N,p=weights[t-1,:])
            
            #X[t,:] = np.random.normal(self.alpha+self.beta*X[t-1][A],self.W,self.N) 
            X[t,:] = self.f(X[t-1][A], self.alpha, self.beta, self.W)

            #weights[t,:] = norm.pdf(self.y[t],self.mu,np.exp(X[t,:]/2))
            weights[t,:] = self.g(X[t,:], self.y[t], self.mu)
            weights[t,:] = weights[t,:]/(weights[t,:].sum()) 

        self.weights = weights
        self.hiddenStates_simulated = X
        return weights,X

    def cdf_t(self, t, values):
        if 0 <= t <= self.T:
            sorted_indices = np.argsort(self.hiddenStates_simulated[t, :])
            sorted_hiddenStates_simulated_t = self.hiddenStates_simulated[t, sorted_indices]
            sorted_weights_t = self.weights[t, sorted_indices]

            def cdf(value):
                mask = sorted_hiddenStates_simulated_t < value
                return np.sum(sorted_weights_t[mask])

            cdf_values = np.vectorize(cdf)(values)
            return cdf_values,sorted_hiddenStates_simulated_t

        else:
            print("t should be < ", self.T, "and positive")

    def pdf_t(self, t, values, epsilon=1e-9):
        cdf_values,_ = self.cdf_t(t, values + epsilon)
        cdf_values_minus_epsilon,_ = self.cdf_t(t, values)
        pdf_values = (cdf_values - cdf_values_minus_epsilon) / epsilon
        return pdf_values

    def quantile_t(self,t,u):

        x_t = self.hiddenStates_simulated[t,:]

        cdf_value, _ = self.cdf_t(t,x_t)

        print(cdf_value)
        
        cdf_value = cdf_value[cdf_value > u]
        x_t = x_t[cdf_value > u]
        sorted_indices = np.argsort(cdf_value)

        return x_t[sorted_indices]