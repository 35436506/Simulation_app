# 🎲 Chapter 12 — Monte Carlo Simulation App

> **Spreadsheet Modeling & Decision Analysis** · Ragsdale · Chapter 12  
> *Introduction to Simulation Using Analytic Solver*

A Streamlit app replicating all simulation examples from Chapter 12, powered entirely by Python (NumPy Monte Carlo) — no Analytic Solver required.

---

## 📦 Modules

| Module | Figure | Technique |
|--------|--------|-----------|
| 🏥 Health Insurance | Fig 12-2, 12-6, 12-17 | Normal + Uniform RNG, VaR, CI |
| ✈️ Airline Overbooking | Fig 12-18 | Discrete + Binomial RNG, parameterized scan |
| 📦 Inventory Control | Fig 12-23, 12-27, 12-36, 12-38 | (s,Q) policy, service level, cost optimization |
| 📂 Project Selection | Fig 12-39 | Triangular RNG, Binomial success, budget constraint |
| 📈 Portfolio Optimization | Fig 12-43 | Correlated returns (Cholesky), efficient frontier |
| 🏘️ Real Estate NPV | Fig 12-47, 12-48 | Multi-year DCF simulation |
| 🎰 RNG Playground | — | Normal, Uniform, Triangular, Discrete, Binomial, Lognormal |

---

## 🚀 Deploy on Streamlit Cloud

1. Fork / clone this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo, set `app.py` as the main file
4. Click **Deploy** — done!

---

## 💻 Run Locally

```bash
git clone https://github.com/<your-username>/ch12-simulation-app.git
cd ch12-simulation-app
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔑 Key Concepts Implemented

### Random Number Generators (RNG)
```python
# PsiNormal equivalent
np.random.normal(mean, std, n_sims)

# PsiUniform equivalent
np.random.uniform(min, max, n_sims)

# PsiTriangular equivalent
np.random.triangular(min, most_likely, max, n_sims)

# PsiDiscrete equivalent
np.random.choice(values, p=probabilities, size=n_sims)

# PsiBinomial equivalent
np.random.binomial(n, p, n_sims)
```

### Correlated Returns (Portfolio)
```python
# Cholesky decomposition for correlated multivariate normal
cov = np.outer(std_devs, std_devs) * corr_matrix
L = np.linalg.cholesky(cov)
z = np.random.standard_normal((n_sims, n_assets))
correlated_returns = z @ L.T + means
```

### Value at Risk (VaR)
```python
# PsiTarget equivalent: P(Y <= target)
prob_exceed = np.mean(results > target)

# PsiPercentile equivalent
p90 = np.percentile(results, 90)
```

### Confidence Interval for Mean
```python
from scipy import stats
se = np.std(data, ddof=1) / np.sqrt(len(data))
ci_lo, ci_hi = stats.t.interval(0.95, df=len(data)-1, loc=np.mean(data), scale=se)
```

---

## 📁 Project Structure

```
ch12-simulation-app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🛠️ Requirements

```
streamlit>=1.32.0
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.8.0
scipy>=1.12.0
plotly>=5.20.0
```

---

*Built for educational purposes · Based on Ragsdale Chapter 12 dataset files (Fig12-2 through Fig12-48)*
