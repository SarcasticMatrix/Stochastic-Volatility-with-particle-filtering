import random
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

class Model:
    def __init__(
            self, 
            T: Optional[int] = 100, 
            N: Optional[int] = 100, 
            seed: Optional[int] = None,
            model: Optional[str] = "Not defined"
        ):

        random.seed(seed) if seed is not None else None
        self.T = T
        self.N = N
        self.time = np.arange(stop=self.T)
        self.model = model

    def compute_estimation(self):

        if not hasattr(self,"weights") or not hasattr(self,"hiddenStates_simulated"):
            self.SIR()
        
        self.estimation = np.sum(self.weights * self.hiddenStates_simulated, axis = 1)
        return self.estimation

    def compute_mse(self):
        if not hasattr(self,"estimation"):
            self.compute_estimation()
        
        self.mse = round(np.mean((self.estimation - self.x)**2),3)
        return self.mse
    
    def plot_trajectory(
            self,
            booleanObersvations: Optional[bool] = False
        ):

        plt.figure(figsize=(20,8))
        plt.plot(self.time,self.x,color='red',label='Hidden State', linewidth=1)
        plt.plot(self.time,self.y,color='green',label='Observations', linewidth=1)

        plt.xlabel('Time')
        plt.ylabel("Value")

        plt.grid(True, which="major", linestyle="--", linewidth=0.7, color="gray")
        
        plt.title('Hidden states and observations')
        plt.legend()
        plt.show()
    
    def plot_estimation(
        self,
        booleanObersvations: Optional[bool] = False
        ):

        plt.figure(figsize=(20,8))
        plt.plot(self.time,self.x,color='red',label='Hidden State', linewidth=1)
        plt.plot(self.time,self.estimation,color='blue',label='Hidden State Estimation', linewidth=1)
        if booleanObersvations:
            plt.plot(self.time,self.y,color='green',label='Observations', linewidth=1)

        plt.xlabel('Time')
        plt.ylabel("Value")

        plt.grid(True, which="major", linestyle="--", linewidth=0.7, color="gray")

        plt.title(
            f'SMC estimation with {self.N} particles - mse: {self.compute_mse()} \n {self.model} model',
            pad=15,
        )
        plt.legend()
        plt.show()
    
