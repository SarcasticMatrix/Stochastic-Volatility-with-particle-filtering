*the README is still being written*

# Stochastic Volatility with particle filtering

## Project Description

Nonlinear non-Gaussian state-space models are ubiquitous in statistics, econometrics, information engineering and signal processing. Particle methods, also known as Sequential Monte Carlo (SMC) methods, provide reliable numerical approximations to the associated state inference problems. However, in most applications, the state-space model of interest also depends on unknown static parameters that need to be estimated from the data. In this context, standard particle methods fail and it is necessary to rely on more sophisticated algorithms. The aim of this paper is to present a comprehensive review of particle methods (with a focus on the Storvik's filter) that have been proposed to perform static parameter estimation in state-space models applied to financial volatility. 

We don't explain further more the sereval SMC methods, you can read the full paper if needed. 

**Table of contents**
1. [Technologies](#technologies)
2. [Sequential Monte Carlo Methods](#SMC)
3. [Examples](#examples)
4. [Sources](#sources)


## 1. Programming language 

Simple but not very effective : Python, we would should have used C++, R or Julia.

## 2. Sequential Monte Carlo Methods

State-space models, also known as hidden Markov models (HMMs), are a very popular class of time series models : HMM is a bivariate stochastic processes $(X_n)_n$ and $(Y_n)_n$, where $(X_n)$  is the hidden component and $(Y_n)$  is the sequence of observations. HMM models may contain unkonwn parameters therefore, SMC methods will at the same time estimate the parameters of our models and the hidden component $X_n$. Applications of state-space models include stochastic volatility models where $X_n$ is the volatility of an asset and $Y_n$ its observed log-return. SMC methods consist in building an estimator $\left( Y_1,...,Y_T \right) \mapsto h \left( Y_1,...,Y_T \right)$. For a computationnal worry, the estimator should be easy to update at time after $T$.


## 3. Benchmark model : the Stochastic Volatility model 

We have discussed about three benchmark models : a linear model, the Kitagawa's model and the stochastic volatility model (SV). Here, we will only present the SV model. Let $T \in \mathbb{R}$, for instance $T=100$, and $X_{1:T} = \left( X_1,...,X_T \right)$ the hidden process, and $Y_{1:T} = \left( Y_1,...,Y_T \right)$ the observations, we write the Hidden Markov model as :

$$x_{t} = \alpha + \beta x_{t-1}+\omega_{t}$$
Avec, $\omega_t \sim \mathcal{N}(0,W)$

$$y_t = \exp\left(\frac{x_{t}}{2}\right) \nu_{t}$$
Avec, $\nu_t \sim \mathcal{N}(0,V)$

### 2.1. First approach, assuming the parameters are known

We apply a simple SIR with $\alpha$=0, $\beta$=0.97 and $W=1$ supposed known.

![stocha-param_connus](https://github.com/SarcasticMatrix/Stochastic-Volatility-with-particle-filtering/assets/94806199/c61e594c-379f-4cf4-9906-5412fec14a56)

### 2.2. Then, assuming the parameters are unknown

However, in real life, model parameters are usually unkonwn. That is why, whe should first, estimate them. We use the Storvik filter which estimates at the same time the hidden states and the model parameters.

![SV_storvik](https://github.com/SarcasticMatrix/Stochastic-Volatility-with-particle-filtering/assets/94806199/172931f6-f8e7-438b-a0da-80f0916e6774)

In order to be more accurate, we can generate many random trajectories, and then run the Storvik filter. Then, we juste have to mean the estimated parameters.

![estimation_par_SV](https://github.com/SarcasticMatrix/Stochastic-Volatility-with-particle-filtering/assets/94806199/feb1269e-76c9-4c9c-bf2f-efcce9a7175f)

### 2.3. Comparaison between PLS, SIR and Storvik's filter on SV benchmark model

![comparaison-PLS-SIR-STORVIK](https://github.com/SarcasticMatrix/Stochastic-Volatility-with-particle-filtering/assets/94806199/6e573136-db0a-439f-9645-98915e31b394)

As you can see, the PLS and Storvik's filter don't provide accurate estimations of the hidden state, but, the SIR filter with the estimated parameters (from the Storvik's filter) gives nice results. Therefore, the Double Forward Filter (SIR filter with the parameters estimated from the Storvik's filter) is a nice method for estimating the stochastic volatility. 

![comparaison-STORVIK-and-DOUBLE-FILTER](https://github.com/SarcasticMatrix/Stochastic-Volatility-with-particle-filtering/assets/94806199/64112e2d-1e1a-477e-8efb-301b74c1f7d1)


### 2.4. Comparaison between PLS, SIR and Storvik's filter on real life problem

We use the S&P500 index from January 2008 to March 2009 : 
![Capture](https://github.com/SarcasticMatrix/Stochastic-Volatility-with-particle-filtering/assets/94806199/79136f30-3d84-4607-b4a6-8830c75d5453)




### 3. Sources

* [1] S. GODSILL A. DOUCET et C. ANDRIEU. “On sequential Monte Carlo sampling methods for Bayesian filtering”. In : Statistics and Computing 10 (1999), p. 197-208.

* [2] Lopes Carvalho Johannes et Polson. “Particle learning and smoothing”. In : Statistical Science 25 (2010), p. 88-106.

* [3] D. Crisan. “Exact rates of convergence for a branching particle approxi- mation to the solution of the zakai equation”. In : Annals of Probability 32 (2003), p. 819-838.

* [4] D. Crisan et A. Doucet. “A survey of convergence results on particle filte- ring methods for practitioners”. In : IEEE Transactions on Signal Processing 50 3 (2002), p. 736-746.

* [5] P. Del Moral D. Crisan et T. J. Lyons. “Interacting particle systems ap- proximations of the Kushner Stratonovitch equation, Advances in Applied Probability”. In : International Journal of Control (1999), p. 819-838.

* [6] J. E. Handschin et D. Q. Mayne. “Monte Carlo techniques to estimate the conditional expectation in multi-stage non-linear filtering”. In : International Journal of Control 9.5 (1969), p. 547-559.

* [7] J. S. Liu et R. Chen. “Sequential Monte Carlo methods for dynamic sys- tems”. In : J. Amer. Statist. Assoc. 93.443 (1998), p. 1032-1044.

* [8] P. Del Moral et L. Miclo. “Branching and interacting particle systems ap- proximations of Feynman-Kac formulae with applications to non-linear filte- ring”. In : Séminaire de Probabilités, Lecture Notes in Mathematics 1729.XXXIV (2000), p. 1-145.

* [9] D.J. Salmond N.J. Gordon et A.F.M. Smith. “Novel approach to nonlinear/non- Gaussian Bayesian state estimation”. In : IEE Proceedings F Radar and Si- gnal Processing 140 2 (1993).

* [10] D. B. Rubin. “MUsing the SIR algorithm to simulate posterior distributions, Bayesian Statistics”. In : Oxford University Press 3 (1988).

* [11] G. Storvik. “Particle Filters for State-Space Models With the Presence of Unknown Static Parameters”. In : IEEE 50.2 (2002).
