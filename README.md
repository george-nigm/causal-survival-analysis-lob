## Causal Inference For Portfolio Optimization And Trading

In the fast-paced financial world, traditional tools, while useful for asset allocation and risk control, often fall short in determining cause-and-effect relationships, especially when influenced by macro-economic confounding factors, spurious correlations and non-linearity. Causal inference emerges as a powerful and promising solution to this challenge. This research proposal presents a shift from conventional correlation analyses to causal inference techniques for analyzing financial time-series data. The approach employs causality methods to elucidate intricate market relationships, emphasizing the potential to predict price shifts, optimize portfolios, and simulate responses to economic changes. The proposal aims to: uncover directional relationships between financial instruments and macro-economic variables; identify lead-lag interdependencies for control and pairs trading; transition from covariance matrices in portfolio optimization methods to causal matrices; enable counterfactual analysis of potential market scenarios; enhance robustness and insightful of asset allocation decisions; and integrate causal insights with cutting-edge machine learning paradigms, notably Causal Reinforcement Learning. This research holds potential to redefine investment strategies, portfolio optimization and risk management by integrating causal inference and artificial intelligence in financial analytics.

![tg_image_1105072616](https://github.com/george-nigm/causal-portfolio-and-trading/assets/48650320/78a8408f-306d-49d3-bed2-6094a3b4ec23)

Find full research proposal: causal-portfolio-and-trading.pdf

### Feasability study

Dataset: Pinnacle ('2000-01-01' - '2021-12-13');

Causal discovery method: PCMCI (max time lag = 5);
Causal matrix construction: Summation of all time lag slices;

Portfolio optimization method: Mean Risk Portfolio Optimization (Maximizing Sharp ratio)

Expected returns (mu) & covariance matrix (cov) estimation: Historical.
Portfolio rebalancing: 22 days
Window size: 756 days

Backtesting: Backtrader library
cash: 100000
commission: 0.005
slippage_perc: 0.005
riskfreerate: 0.0
