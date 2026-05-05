import numpy as np
import streamlit as st
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(page_title="Normal-Normal Bayesian Inference Simulator", layout="wide")

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.step-banner {
    padding: 14px 22px;
    border-radius: 8px;
    margin: 56px 0 4px 0;
    color: white;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.step-rule {
    border: none;
    border-top: 3px solid;
    margin: 0 0 20px 0;
    border-radius: 2px;
}
</style>
""", unsafe_allow_html=True)


def section(number: str, title: str, color: str) -> None:
    st.markdown(
        f'<div class="step-banner" style="background:{color};">'
        f'{number} &nbsp;·&nbsp; {title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<hr class="step-rule" style="border-color:{color};">',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Prior Hyperparameters")
mu_prior = st.sidebar.slider("μ₀ (prior mean)", -200.0, 200.0, 100.0, 1.0)
sigma_prior = st.sidebar.slider("σ₀ (prior std)", 1.0, 100.0, 50.0, 1.0)

prior_ci_lo = norm.ppf(0.025, mu_prior, sigma_prior)
prior_ci_hi = norm.ppf(0.975, mu_prior, sigma_prior)
st.sidebar.markdown(f"**E[μ]** = {mu_prior:.1f}")
st.sidebar.markdown(f"**SD[μ]** = {sigma_prior:.1f}")
st.sidebar.markdown(f"**95% CI**: [{prior_ci_lo:.1f}, {prior_ci_hi:.1f}]")

st.sidebar.divider()
st.sidebar.header("Known Likelihood Std")
sigma = st.sidebar.slider("σ (known std of data)", 0.1, 20.0, 2.0, 0.1)

st.sidebar.divider()
st.sidebar.header("Data")
data_mode = st.sidebar.radio("Data source", ["Simulate", "Enter manually"])

if data_mode == "Simulate":
    n_obs = st.sidebar.slider("N (observations)", 1, 100, 10)
    true_mu = st.sidebar.slider("True μ", -100.0, 200.0, 50.0, 1.0)
    seed = st.sidebar.number_input("Random seed", value=42, step=1)
    rng = np.random.default_rng(int(seed))
    observations = rng.normal(true_mu, sigma, size=n_obs).tolist()
else:
    raw = st.sidebar.text_area(
        "Observations (comma-separated floats)",
        "48.5, 50.1, 49.8, 51.2, 50.5, 49.3, 50.8, 48.9, 51.0, 49.6",
    )
    try:
        observations = [float(v.strip()) for v in raw.split(",") if v.strip()]
    except ValueError:
        st.sidebar.error("Enter numbers separated by commas.")
        observations = []

# ── Derived quantities ─────────────────────────────────────────────────────────
x = np.array(observations, dtype=float)
N = len(x)
x_bar = x.mean() if N > 0 else 0.0

# Analytical conjugate update
# μ* = (σ²·μ₀ + N·x̄·σ₀²) / (σ² + N·σ₀²)
# σ*² = (σ²·σ₀²) / (σ² + N·σ₀²)
sigma2 = sigma ** 2
sigma_prior2 = sigma_prior ** 2

mu_post = (sigma2 * mu_prior + N * x_bar * sigma_prior2) / (sigma2 + N * sigma_prior2)
sigma_post2 = (sigma2 * sigma_prior2) / (sigma2 + N * sigma_prior2)
sigma_post = np.sqrt(sigma_post2)

# ── Page title ────────────────────────────────────────────────────────────────
st.title("Normal-Normal Conjugate Model")
st.markdown(
    "Bayesian inference for an unknown mean μ using a Normal prior, "
    "when the data standard deviation σ is known. "
    "Adjust the sliders on the left to explore how prior beliefs and observed data "
    "shape the posterior."
)

# ── Section 0 – Process overview ──────────────────────────────────────────────
section("0", "Process Overview", "#555E6B")
st.markdown(
    "The diagram below shows the full Bayesian workflow — from prior belief about μ, "
    "through observed data, to the posterior and predictive distribution."
)
st.graphviz_chart("""
digraph {
    rankdir=LR
    graph [bgcolor=transparent splines=ortho nodesep=0.6]
    node  [fontname="Helvetica" fontsize=13 style="filled,rounded" shape=box
           margin="0.25,0.15"]
    edge  [fontname="Helvetica" fontsize=11 color="#555555"]

    Prior [label="Prior\\nNormal(μ₀, σ₀)"
           fillcolor="#AED6F1" color="#2E86C1"]

    Mu [label="Latent mean  μ"
        fillcolor="#D5F5E3" color="#1E8449" shape=ellipse]

    Data [label="Observed values\\nx₁, x₂, …, xₙ"
          fillcolor="#E8DAEF" color="#7D3C98"]

    Likelihood [label="Likelihood\\nNormal(μ, σ)"
                fillcolor="#FDEBD0" color="#CA6F1E"]

    Update [label="Conjugate update\\nμ* = weighted average\\nσ*² = harmonic shrinkage"
            fillcolor="#FDFEFE" color="#717D7E"]

    Posterior [label="Posterior\\nNormal(μ*, σ*)"
               fillcolor="#FADBD8" color="#C0392B"]

    Predictive [label="Posterior predictive\\nNormal(μ*, √(σ²+σ*²))"
                fillcolor="#FDEBD0" color="#D35400"]

    Prior    -> Mu         [label=" encodes belief\\n about μ"]
    Mu       -> Likelihood
    Likelihood -> Data     [label=" generates"]
    Prior    -> Update
    Data     -> Update
    Update   -> Posterior  [label=" yields"]
    Posterior -> Predictive [label=" marginalise\\n over μ"]
}
""")

# ── Section 1 – Prior ─────────────────────────────────────────────────────────
section("1", "Prior Distribution", "#2E86C1")
st.markdown("We encode our belief about the unknown mean μ before seeing any data:")
st.latex(r"\mu \sim \mathcal{N}(\mu_0,\; \sigma_0^2)")
st.markdown(
    "A wide σ₀ expresses vague prior knowledge (the mean could be almost anything); "
    "a narrow σ₀ expresses strong prior belief concentrated near μ₀."
)

mu_grid = np.linspace(
    norm.ppf(0.0005, mu_prior, sigma_prior),
    norm.ppf(0.9995, mu_prior, sigma_prior),
    400,
)
prior_pdf = norm.pdf(mu_grid, mu_prior, sigma_prior)

fig_prior = go.Figure()
fig_prior.add_trace(
    go.Scatter(
        x=mu_grid, y=prior_pdf, mode="lines",
        line=dict(color="steelblue", width=2),
        fill="tozeroy", fillcolor="rgba(70,130,180,0.15)",
        name="Prior PDF",
    )
)
fig_prior.update_layout(
    xaxis_title="μ", yaxis_title="Density",
    title=f"Normal({mu_prior:.1f}, {sigma_prior:.1f}) Prior",
    height=300, margin=dict(t=40, b=40),
)
st.plotly_chart(fig_prior, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Mean (μ₀)", f"{mu_prior:.2f}")
c2.metric("Std (σ₀)", f"{sigma_prior:.2f}")
c3.metric("95% CI", f"[{prior_ci_lo:.2f}, {prior_ci_hi:.2f}]")

# ── Section 2 – Observed Data ─────────────────────────────────────────────────
section("2", "Observed Data", "#7D3C98")
st.markdown(
    "Each observation xᵢ is a **continuous measurement** drawn from a Normal distribution "
    "with unknown mean μ and **known** standard deviation σ:"
)
st.latex(r"x_i \mid \mu \;\sim\; \mathcal{N}(\mu,\; \sigma^2), \quad i = 1, \dots, N")
st.markdown(
    "The likelihood function for all N observations is:"
)
st.latex(
    r"p(\mathbf{x} \mid \mu) = \prod_{i=1}^{N} \frac{1}{\sqrt{2\pi}\,\sigma}"
    r"\exp\!\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)"
)

if data_mode == "Simulate":
    st.markdown(
        f"**Simulated data:** {N} draws from Normal(μ = {true_mu}, σ = {sigma}) "
        f"using seed {int(seed)}."
    )
else:
    st.markdown(
        "**Manual data:** values you entered, treated as N independent "
        "Normal observations with the same unknown mean μ and known σ."
    )

if N == 0:
    st.warning("No data yet — enter or simulate some observations.")
else:
    fig_data = go.Figure()
    fig_data.add_trace(
        go.Histogram(
            x=x, nbinsx=max(5, N // 2),
            marker_color="mediumpurple", opacity=0.8,
            name="Observations",
        )
    )
    # Overlay normal fit
    x_lo = x.min() - 3 * sigma
    x_hi = x.max() + 3 * sigma
    xs = np.linspace(x_lo, x_hi, 300)
    fig_data.add_trace(
        go.Scatter(
            x=xs,
            y=norm.pdf(xs, x_bar, sigma) * N * (x.max() - x.min()) / max(5, N // 2),
            mode="lines",
            line=dict(color="black", width=1.5, dash="dot"),
            name=f"Normal(x̄={x_bar:.2f}, σ={sigma})",
        )
    )
    fig_data.update_layout(
        xaxis_title="Value", yaxis_title="Count",
        title="Distribution of Observed Values",
        height=280, margin=dict(t=40, b=40),
        showlegend=True,
    )
    st.plotly_chart(fig_data, use_container_width=True)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("N (observations)", N)
    d2.metric("Sample mean (x̄)", f"{x_bar:.3f}")
    d3.metric("Sample std", f"{x.std():.3f}")
    d4.metric("Known σ", f"{sigma:.2f}")

# ── Section 3 – Bayesian Update ───────────────────────────────────────────────
section("3", "Bayesian Update", "#C0392B")
st.markdown(
    "The Normal prior is **conjugate** to the Normal likelihood (with known σ), "
    "so the posterior is also Normal. The update is a precision-weighted average:"
)
st.latex(
    r"\mu^* = \frac{\sigma^2 \mu_0 + N\bar{x}\,\sigma_0^2}{\sigma^2 + N\sigma_0^2}"
)
st.latex(
    r"\sigma^{*2} = \frac{\sigma^2 \sigma_0^2}{\sigma^2 + N\sigma_0^2}"
)
st.latex(r"\mu \mid \mathbf{x} \sim \mathcal{N}(\mu^*,\; \sigma^{*2})")
st.markdown(
    "As N grows, the posterior mean is pulled toward the sample mean x̄ and the "
    "posterior variance shrinks — more data means a tighter belief about μ."
)

# Build a shared x-axis that covers both prior and posterior
lo = min(norm.ppf(0.0005, mu_prior, sigma_prior), norm.ppf(0.0005, mu_post, sigma_post))
hi = max(norm.ppf(0.9995, mu_prior, sigma_prior), norm.ppf(0.9995, mu_post, sigma_post))
mu_grid2 = np.linspace(lo, hi, 500)

prior_pdf2 = norm.pdf(mu_grid2, mu_prior, sigma_prior)
post_pdf = norm.pdf(mu_grid2, mu_post, sigma_post)

fig_update = go.Figure()
fig_update.add_trace(
    go.Scatter(
        x=mu_grid2, y=prior_pdf2, mode="lines",
        line=dict(color="steelblue", width=2, dash="dash"),
        name=f"Prior  N({mu_prior:.1f}, {sigma_prior:.1f})",
    )
)
fig_update.add_trace(
    go.Scatter(
        x=mu_grid2, y=post_pdf, mode="lines",
        line=dict(color="crimson", width=2),
        fill="tozeroy", fillcolor="rgba(220,20,60,0.10)",
        name=f"Posterior  N({mu_post:.2f}, {sigma_post:.2f})",
    )
)
if N > 0:
    fig_update.add_vline(
        x=x_bar, line_dash="dot", line_color="mediumpurple",
        annotation_text=f"x̄ = {x_bar:.2f}",
        annotation_position="top right",
    )
fig_update.update_layout(
    xaxis_title="μ", yaxis_title="Density",
    title="Prior vs Posterior for μ",
    height=340, margin=dict(t=40, b=40),
    legend=dict(x=0.02, y=0.95),
)
st.plotly_chart(fig_update, use_container_width=True)

post_ci_lo = norm.ppf(0.025, mu_post, sigma_post)
post_ci_hi = norm.ppf(0.975, mu_post, sigma_post)

p1, p2, p3 = st.columns(3)
p1.metric("Posterior mean (μ*)", f"{mu_post:.3f}")
p2.metric("Posterior std (σ*)", f"{sigma_post:.3f}")
p3.metric("95% credible interval", f"[{post_ci_lo:.2f}, {post_ci_hi:.2f}]")

st.markdown("---")
st.markdown("**How the posterior hyperparameters changed:**")
t1, t2 = st.columns(2)
with t1:
    st.markdown(
        f"| | Prior | Posterior |\n"
        f"|---|---|---|\n"
        f"| Mean | {mu_prior:.2f} | {mu_post:.2f} |\n"
        f"| Std | {sigma_prior:.2f} | {sigma_post:.3f} |"
    )
with t2:
    # Shrinkage factor: how much weight goes to the data vs prior
    if N > 0:
        data_weight = N * sigma_prior2 / (sigma2 + N * sigma_prior2)
        prior_weight = sigma2 / (sigma2 + N * sigma_prior2)
        st.markdown("**Posterior mean as weighted average:**")
        st.latex(
            rf"\mu^* = {prior_weight:.3f} \cdot \mu_0 + {data_weight:.3f} \cdot \bar{{x}}"
        )
        st.markdown(
            f"Data weight = **{data_weight:.1%}**, Prior weight = **{prior_weight:.1%}**"
        )

# ── Section 4 – Posterior Predictive ─────────────────────────────────────────
section("4", "Posterior Predictive Distribution", "#D35400")
st.markdown(
    "Integrating out the uncertainty in μ over the posterior gives the "
    "**predictive distribution** for a new observation x̃:"
)
st.latex(
    r"\tilde{x} \mid \mathbf{x} \;\sim\; \mathcal{N}\!\left(\mu^*,\; \sigma^2 + \sigma^{*2}\right)"
)
st.markdown(
    "The predictive variance has **two sources**: the inherent data noise σ² "
    "and the remaining uncertainty in μ given by σ*²."
)

sigma_pred = np.sqrt(sigma2 + sigma_post2)

pred_lo = norm.ppf(0.0005, mu_post, sigma_pred)
pred_hi = norm.ppf(0.9995, mu_post, sigma_pred)
pred_grid = np.linspace(pred_lo, pred_hi, 400)
pred_pdf = norm.pdf(pred_grid, mu_post, sigma_pred)

# Also show the likelihood component (data noise alone) for comparison
like_pdf = norm.pdf(pred_grid, mu_post, sigma)

fig_pred = go.Figure()
fig_pred.add_trace(
    go.Scatter(
        x=pred_grid, y=pred_pdf, mode="lines",
        line=dict(color="darkorange", width=2),
        fill="tozeroy", fillcolor="rgba(255,140,0,0.15)",
        name=f"Predictive  N({mu_post:.2f}, {sigma_pred:.2f}²)",
    )
)
fig_pred.add_trace(
    go.Scatter(
        x=pred_grid, y=like_pdf, mode="lines",
        line=dict(color="gray", width=1.5, dash="dash"),
        name=f"Likelihood only  N({mu_post:.2f}, {sigma:.2f}²)",
    )
)
fig_pred.update_layout(
    xaxis_title="Future observation x̃", yaxis_title="Density",
    title="Posterior Predictive Distribution",
    height=320, margin=dict(t=40, b=40),
    legend=dict(x=0.02, y=0.95),
)
st.plotly_chart(fig_pred, use_container_width=True)

pred_ci_lo = norm.ppf(0.025, mu_post, sigma_pred)
pred_ci_hi = norm.ppf(0.975, mu_post, sigma_pred)

q1, q2, q3 = st.columns(3)
q1.metric("Predictive mean", f"{mu_post:.3f}")
q2.metric("Predictive std", f"{sigma_pred:.3f}")
q3.metric("95% predictive interval", f"[{pred_ci_lo:.2f}, {pred_ci_hi:.2f}]")
