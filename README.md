# Neural Network Option Pricing

## Accelerating Option Pricing with Neural Networks: From Black–Scholes to Stochastic Volatility

This repository contains the research code and reproducibility materials for:

**"Accelerating Option Pricing with Neural Networks: From Black–Scholes to Stochastic Volatility"**

The study investigates whether neural-network surrogate models can provide a computational advantage for repeated derivative pricing.

Rather than assuming that machine learning should replace classical pricing models, the research asks a narrower question:

> **When does the upfront cost of training a neural-network pricing surrogate become worthwhile relative to repeatedly evaluating a numerical pricing method?**

The analysis begins with the Black–Scholes model as a controlled analytical benchmark and then extends the framework to the Heston stochastic-volatility model, where repeated Monte Carlo valuation provides a more meaningful computational setting for a learned surrogate.

---

## Research Objectives

The study addresses three main questions:

1. Can a feed-forward neural network accurately learn the Black–Scholes pricing function?
2. How does the computational cost of neural-network inference compare with Monte Carlo valuation?
3. Does the computational case for a neural-network surrogate become stronger under Heston stochastic volatility?

---

## Methodology

### Black–Scholes

The Black–Scholes experiment compares:

- Closed-form analytical pricing
- Monte Carlo simulation
- Neural-network pricing

Synthetic option parameters are generated and the neural network learns the mapping

\[
(S,K,T,r,\sigma) \rightarrow C
\]

where \(C\) is the European call price.

### Heston

The Heston experiment uses the nine-dimensional parameter vector

\[
(S_0,V_0,K,T,r,\kappa,\theta,\xi,\rho)
\]

and generates synthetic pricing labels using Monte Carlo simulation.

The neural network then learns the resulting pricing operator:

\[
x_H \rightarrow C_H
\]

---

## Neural Network

The final Heston surrogate uses the architecture:

```text
9 → 128 → 128 → 64 → 32 → 1
