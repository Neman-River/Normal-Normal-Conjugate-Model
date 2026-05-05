# Normal-Normal Bayesian Inference — Streamlit App

Interactive simulator for Bayesian inference of an unknown mean using the **Normal-Normal conjugate model**.

## What it does

Given continuous measurements from a Normal distribution with **known standard deviation σ** and **unknown mean μ**, the app updates a Normal prior on μ to produce an analytical posterior — no MCMC required.

### Bayesian workflow

```
Prior N(μ₀, σ₀)  →  Observed data x₁…xₙ  →  Posterior N(μ*, σ*)  →  Predictive N(μ*, √(σ²+σ*²))
```

**Conjugate update formulas:**

$$\mu^* = \frac{\sigma^2 \mu_0 + N\bar{x}\,\sigma_0^2}{\sigma^2 + N\sigma_0^2}$$

$$\sigma^{*2} = \frac{\sigma^2 \sigma_0^2}{\sigma^2 + N\sigma_0^2}$$

The posterior mean is a precision-weighted average of the prior mean and the sample mean. As N → ∞, the data dominates and the posterior collapses around x̄.

## App sections

| # | Section | Description |
|---|---------|-------------|
| 0 | Process Overview | Graphviz DAG of the full Bayesian workflow |
| 1 | Prior Distribution | PDF of Normal(μ₀, σ₀) with 95% credible interval |
| 2 | Observed Data | Histogram of observations with Normal overlay |
| 3 | Bayesian Update | Prior vs posterior overlay, weight breakdown |
| 4 | Posterior Predictive | Predictive distribution vs likelihood-only comparison |

## Sidebar controls

| Control | Description |
|---------|-------------|
| **μ₀** | Prior mean |
| **σ₀** | Prior standard deviation (width of prior belief) |
| **σ** | Known data standard deviation |
| **Data source** | Simulate from Normal(true μ, σ) or enter values manually |

## Running the app

```bash
uv run streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

## Dependencies

- `streamlit`
- `plotly`
- `scipy`
- `numpy`

Install with:

```bash
uv add streamlit plotly scipy numpy
```

## Related notebooks

- `NormalNormal.ipynb` — analytical derivation and Stan MCMC version
- `AB-testing-normalnormal.ipynb` — A/B testing application of the same model
