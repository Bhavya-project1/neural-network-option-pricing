# Neural-Network Surrogates for Accelerating Option Pricing

## Accelerating Option Pricing with Neural Networks: From Black–Scholes to Heston Stochastic Volatility

This repository investigates whether a neural-network surrogate can reduce the computational cost of repeated option valuation while maintaining controlled approximation error.

The research question is deliberately conditional:

> When does the upfront cost of training a neural-network pricing surrogate become worthwhile relative to repeatedly evaluating a numerical pricing method?

The study uses **Black–Scholes as an analytical benchmark** and **Heston stochastic volatility as the main computational use case**. The conclusion is not that neural networks should replace established pricing methods in general, but that a learned approximation may be useful in selected high-query numerical-pricing workflows.

## Research Questions

1. Can a feed-forward neural network learn the Black–Scholes European call-price surface?
2. How does neural-network inference compare with Monte Carlo valuation?
3. Does the computational case become stronger for Heston, where reference valuation is numerical?
4. How many repeated pricing queries are needed to amortise the one-time training cost?

## Methodology

### Black–Scholes benchmark

The notebook compares closed-form Black–Scholes pricing, Monte Carlo simulation, and a neural-network approximation.

The input-to-price mapping is:

~~~text
(S,K,T,r,σ) → C
~~~

The synthetic experiment generates 50,000 parameter observations over explicit ranges for spot, strike, maturity, interest rate, and volatility. The network is evaluated on a held-out test sample.

### Heston stochastic volatility

The Heston input vector is:

~~~text
(S₀,V₀,K,T,r,κ,θ,ξ,ρ)
~~~

The notebook generates 8,000 synthetic parameter vectors and obtains reference labels from a compiled Monte Carlo engine using antithetic paths. The training-label experiment uses 5,000 paths and 100 time steps per option.

The Heston surrogate uses a feed-forward architecture:

~~~text
9 → 128 → 128 → 64 → 32 → 1
~~~

The training procedure uses feature scaling fitted on the training sample, Huber loss, early stopping, and three independent random seeds.

### Computational benchmarking

The notebook compares the trained surrogate with a compiled antithetic Heston Monte Carlo engine on the same held-out parameter vectors. It warms up the compiled and neural-network paths, repeats timing runs, estimates uncertainty, and calculates the number of future pricing queries required to offset the neural-network training cost.

A separate high-accuracy check reprices a held-out subset using 20,000 antithetic paths and 200 time steps.

## Evaluation

The notebook reports MAE, RMSE, normalized MAE, relative-error diagnostics with a clearly stated low-price exclusion rule, timing distributions, bootstrap uncertainty for speed-up, and an amortised break-even query estimate. It also analyses errors by moneyness and maturity.

The numerical outputs are generated when the notebook is run and are not embedded here as fixed performance claims. Benchmark results are implementation- and hardware-specific.

## Repository Contents

- Accelerating_Option_Pricing.ipynb — reproducible computational experiments.
- Accelerating_Option_Pricing_Research_Paper.pdf — accompanying research paper.

## Technology

Python · NumPy · pandas · SciPy · scikit-learn · PyTorch · Numba · Matplotlib

## Run

~~~bash
pip install -r requirements.txt
jupyter notebook
~~~

Open the notebook and execute the cells in sequence. Record the Python, PyTorch, NumPy, Numba, and hardware versions when publishing timing comparisons.

## Interpretation and Limitations

Black–Scholes already has an inexpensive analytical solution, so a neural surrogate is not automatically computationally justified for that model. Heston is a more relevant setting for an amortisation analysis, but surrogate validity is limited to the training domain and numerical reference engine.

The Heston Monte Carlo implementation and discretisation are modelling choices; a speed advantage does not establish that the surrogate is more accurate than the true model price. Validation should include parameter-domain coverage, boundary behaviour, no-arbitrage checks, error concentration by moneyness/maturity, and independent reference pricing.

## Disclaimer

Research and educational purposes only. A neural-network estimate is not automatically arbitrage-free or suitable for production valuation. Not investment advice.
