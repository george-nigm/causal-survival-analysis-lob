## Causal Inference For Portfolio Optimization And Trading

In the fast-paced financial world, traditional tools, while useful for asset allocation and risk control, often fall short in determining cause-and-effect relationships, especially when influenced by macro-economic confounding factors, spurious correlations and non-linearity. Causal inference emerges as a powerful and promising solution to this challenge. This research proposal presents a shift from conventional correlation analyses to causal inference techniques for analyzing financial time-series data. The approach employs causality methods to elucidate intricate market relationships, emphasizing the potential to predict price shifts, optimize portfolios, and simulate responses to economic changes. The proposal aims to: uncover directional relationships between financial instruments and macro-economic variables; identify lead-lag interdependencies for control and pairs trading; transition from covariance matrices in portfolio optimization methods to causal matrices; enable counterfactual analysis of potential market scenarios; enhance robustness and insightful of asset allocation decisions; and integrate causal insights with cutting-edge machine learning paradigms, notably Causal Reinforcement Learning. This research holds potential to redefine investment strategies, portfolio optimization and risk management by integrating causal inference and artificial intelligence in financial analytics.

![tg_image_1105072616](https://github.com/george-nigm/causal-portfolio-and-trading/assets/48650320/78a8408f-306d-49d3-bed2-6094a3b4ec23)

Find full research proposal: causal-portfolio-and-trading.pdf

### Feasability study / Current results

Dataset: Pinnacle dataset ('2000-01-01' - '2021-12-13');

Method 1. Equally Weighted Portfolio;

Method 2. Portfolio optimization method: Mean Risk Portfolio Optimization (Maximizing Sharp ratio);

Method 3. Causal discovery method: Granger causality (max time lag = 1);<br>
Causal matrix construction: Summation of all time lag slices; <br>

Method 4. Causal discovery method: PCMCI (max time lag = 5);<br>
Causal matrix construction: Summation of all time lag slices; <br>

Backtesting: Backtrader library; <br>
- Cash: 100000; <br>
- Commission: 0.005; <br>
- Riskfreerate: 0.0;
- Portfolio rebalancing: 22 days; <br>
- Window size: 756 days; <br>
- Expected returns (mu) & covariance matrix (cov) estimation: Historical; <br>

|                                                   | E(R) | Std(R) | Sharpe | DD(R) | Sortino | MDD     |
|---------------------------------------------------|------|--------|--------|-------|---------|---------|
| Equally Weighted Portfolio                        |      |        |        |       |         |         |
| Mean Risk Portfolio Optimization                  |      |        |        |       |         |         |
| Causal Portfolio Optimization (Granger causality) |      |        |        |       |         |         |
| Causal Portfolio Optimization (PCMCI)             |      |        | 0.52   |       | 0.71    | -33.81% |

Causal Portfolio Optimization (PCMCI) offers the most effective allocation strategy that demonstrates superiority both in absolute values of the sharpe metric, and relatively the most robust strategy that guarantees a positive change in capital under management in comparison with other portfolio optimization methods.

#### Plans 
- Analysis of selected stocks included in the portfolio;
- Increase max time lag (e.g. max time lag = 5);
- Introduction of macro-variables as confounders for excluding spurious relationships;
- Analysis on Pinnacle dataset with non-linear methods of causal discovery;
- Counterfactual analysis of fed rate changes;
- Research of lead-lag dependencies for pairs trading;
