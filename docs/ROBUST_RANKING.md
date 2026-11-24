# EAS, Multi-Objective Alpha Score, and PRRA

PowerBench reports three complementary multi-objective summaries:

1. the alpha-conditioned Energy-Accuracy Score (EAS), which exposes a declared
   trade-off between forecasting error and energy per forecast; and
2. the Multi-Objective Alpha Score (MOAS), which extends fixed declared
   priorities to error, energy, latency, and memory; and
3. Preference-Robust Rank Acceptability (PRRA), which evaluates ranking
   stability under uncertain preferences over error, energy, latency, and
   memory.

Both are secondary analyses. Raw measurements, uncertainty intervals, and
Pareto status remain the primary evidence.

## Alpha-conditioned Energy-Accuracy Score

For a frozen study population, normalize forecasting error \(E\) as

\[
\widetilde E=\frac{E-E_{\min}}{E_{\max}-E_{\min}},
\]

and positive energy per forecast \(J\) in log space as

\[
\widetilde J=
\frac{\log_{10}J-\log_{10}J_{\min}}
{\log_{10}J_{\max}-\log_{10}J_{\min}}.
\]

The higher-is-better score is

\[
\mathrm{EAS}_{\alpha}
=1-\left[\alpha\widetilde E+(1-\alpha)\widetilde J\right],
\qquad 0\leq\alpha\leq1.
\]

The endpoints have direct meanings: \(\alpha=1\) is accuracy-only and
\(\alpha=0\) is energy-only. PowerBench predeclares \(\alpha=0.5\) for the
grouped comparison and also reports the complete alpha sensitivity curve and
pairwise crossover points. This prevents choosing alpha after observing which
model wins.

EAS is intentionally simple and interpretable, but it uses only two objectives
and its min-max normalization depends on the included alternatives. The study
population and normalization bounds must therefore be frozen and retained with
the reported results.

## Multi-Objective Alpha Score

MOAS is the fixed-preference extension of EAS. Let the independently declared,
non-negative priorities be
\(\boldsymbol\lambda=(\lambda_E,\lambda_J,\lambda_T,\lambda_M)\) for error,
energy, latency, and memory. At least one priority must be positive. Normalize
them as

\[
\bar\lambda_k=\frac{\lambda_k}{\sum_j\lambda_j},
\qquad
\sum_k\bar\lambda_k=1.
\]

For hardware \(h\), criterion \(k\), and model \(m\), the lower-is-better
component is normalized within hardware:

\[
\widetilde{x}_{hmk}=
\frac{g_k(x_{hmk})-\min_{m'}g_k(x_{hm'k})}
{\max_{m'}g_k(x_{hm'k})-\min_{m'}g_k(x_{hm'k})}.
\]

Here, \(g_E(x)=x\), while energy, latency, and memory use
\(g_k(x)=\log_{10}(x)\) for positive \(x\). The higher-is-better score is

\[
\mathrm{MOAS}_{hm}(\boldsymbol\lambda)
=1-\sum_{k\in\{E,J,T,M\}}\bar\lambda_k\widetilde{x}_{hmk}.
\]

MOAS lies in \([0,1]\) for complete rows. A common rescaling of all priorities
does not change it. The publication notebook predeclares a balanced
\((1,1,1,1)\) scenario and four one-criterion-priority scenarios
\((4,1,1,1)\) and its permutations. These scenarios expose sensitivity without
fitting weights to the observed winner.

Normalization is performed within hardware because board energy, CPU package
energy, Jetson rail energy, and relevant memory scopes are not physically
interchangeable. MOAS therefore supports within-hardware model comparison; an
across-hardware summary must aggregate hardware with equal declared weight and
complete model coverage.

MOAS is a PowerBench fixed-priority sensitivity metric, not an established
standard metric. Like any weighted sum, it permits compensation: a poor value
on one criterion may be offset by strong values on others. Raw values and
Pareto status must remain visible.

## PRRA status and name

Preference-Robust Rank Acceptability (PRRA) is a secondary multi-objective
analysis developed for PowerBench. It is not a standard metric copied verbatim
from another benchmark. Its preference-sampling and rank-acceptability concepts
are adapted from Stochastic Multicriteria Acceptability Analysis (SMAA), while
its non-dominance report follows standard Pareto analysis.

PRRA must be reported alongside raw error, energy, latency, and memory. It does
not replace those measurements.

## Motivation

PowerBench compares models using objectives that conflict:

- forecasting error should be minimized;
- energy per forecast should be minimized;
- inference latency should be minimized;
- relevant peak memory should be minimized.

EAS and MOAS are useful when criterion priorities are known. They are not
preference-neutral: a fixed priority vector must be selected, and min-max
normalization can be sensitive to extreme alternatives.

PRRA addresses these limitations as follows:

1. It uses all four deployment criteria.
2. It transforms values to within-hardware ranks, reducing sensitivity to units,
   scale, and extreme magnitudes.
3. It samples the full criterion-weight simplex instead of selecting one weight
   vector after observing results.
4. It reports a rank distribution rather than hiding preference sensitivity in
   one deterministic result.
5. It compares criteria within hardware, avoiding direct normalization of
   incompatible GPU-board, Jetson-rail, and CPU-package measurement scopes.
6. It requires complete hardware coverage, preventing missing results from
   improving a model's aggregate rank.

## Notation

Let:

- \(h\in\mathcal H\) identify hardware;
- \(m\in\mathcal M\) identify a model;
- \(k\in\{1,\ldots,K\}\) identify a criterion, with \(K=4\);
- \(b\in\{1,\ldots,B\}\) identify a preference draw;
- \(M_h\) be the number of eligible models on hardware \(h\);
- \(x_{hmk}\) be the raw lower-is-better value for criterion \(k\).

Each model is represented by one operating point selected using a rule declared
before ranking: the successful batch size with maximum measured throughput.
Accuracy is joined from the independent fixed-batch prediction phase.

## Step 1: within-hardware rank costs

For each hardware and criterion, ascending average ranks are assigned. Tied
values receive their average rank. The fractional rank cost is

\[
c_{hmk} =
\frac{\operatorname{rank}_{h,k}(x_{hmk})-1}{M_h-1}.
\]

Thus \(c=0\) is best and \(c=1\) is worst for that criterion on that hardware.
The transformation is invariant to common positive rescaling and monotone unit
changes, but it intentionally discards effect magnitude.

## Step 2: preference distribution

When no criterion preference is justified, PowerBench samples

\[
\mathbf w^{(b)}
\sim \operatorname{Dirichlet}(\mathbf 1_K),
\qquad
\sum_{k=1}^{K}w_k^{(b)}=1,\quad w_k^{(b)}\geq0.
\]

`Dirichlet(1,1,1,1)` is uniform over the four-dimensional weight simplex and is
symmetric among criteria:

\[
\mathbb E[w_k]=\frac{1}{K}.
\]

This is not equivalent to using fixed equal weights. Fixed equal weights test
one preference; symmetric sampling evaluates all non-negative linear
preferences over the simplex.

PowerBench uses \(B=20{,}000\) draws and seed 12345.

## Step 3: sampled cost and rank

For each draw, the scalar cost is

\[
L_{hm}^{(b)}
=
\sum_{k=1}^{K}w_k^{(b)}c_{hmk}.
\]

The sampled model rank is

\[
r_{hm}^{(b)}
=
1+\sum_{j\neq m}
\mathbb I\!\left[L_{hj}^{(b)}<L_{hm}^{(b)}\right].
\]

Lower sampled cost is better. Exact sampled-cost ties receive reproducible
infinitesimal random tie-breaking so that model-name or input-row order cannot
systematically win a tie.

## Step 4: rank acceptability

The empirical probability that model \(m\) receives rank \(r\) is

\[
\widehat b_{hmr}
=
\frac{1}{B}
\sum_{b=1}^{B}
\mathbb I\!\left[r_{hm}^{(b)}=r\right].
\]

Its expected rank is

\[
\overline r_{hm}
=
\sum_{r=1}^{M_h}r\,\widehat b_{hmr}.
\]

PRRA maps expected rank to the interval \([0,1]\):

\[
A_{hm}
=
1-\frac{\overline r_{hm}-1}{M_h-1}
=
\sum_{r=1}^{M_h}
\widehat b_{hmr}\frac{M_h-r}{M_h-1}.
\]

An acceptability of 1 means the model is first under every sampled preference;
0 means it is last under every sampled preference.

The top-\(q\) preference acceptability is

\[
P_{hm}^{(q)}
=
\sum_{r=1}^{q}\widehat b_{hmr}.
\]

PowerBench reports \(q=1\) and \(q=3\).

## Step 5: Pareto-efficient status

Model \(m\) is dominated on hardware \(h\) if another eligible model \(j\)
satisfies

\[
x_{hjk}\leq x_{hmk}\quad\forall k
\]

and the inequality is strict for at least one criterion. A model is marked
Pareto-efficient when no such model exists. Pareto status requires no criterion
weights and remains primary evidence next to raw trade-off plots.

## Step 6: hardware aggregation

Only models with finite values for every criterion on every included hardware
are eligible. Each hardware receives equal weight:

\[
\overline A_m
=
\frac{1}{|\mathcal H|}
\sum_{h\in\mathcal H}A_{hm}.
\]

PowerBench also reports

\[
A_m^{\mathrm{worst}}
=
\min_{h\in\mathcal H}A_{hm},
\]

hardware-mean top-1/top-3 acceptability, and the fraction of hardware platforms
on which the model is Pareto-efficient. Models are ordered first by
\(\overline A_m\) and then by \(A_m^{\mathrm{worst}}\).

## Interpretation

- `rank_acceptability` is higher-is-better and lies in \([0,1]\).
- `top1_probability` is the fraction of sampled preferences for which a model
  ranks first.
- `top3_probability` is the corresponding top-three fraction.
- These are probabilities over unknown criterion preferences.
- They are not probabilities that a model is truly superior.
- Repeat uncertainty and source-level accuracy uncertainty are not sampled by
  the present implementation and must be reported separately.

## Limitations

PRRA is more robust to scale and outlier magnitude than global min-max
scalarization, but it is not universally better:

- ranks discard the magnitude of practical improvements;
- results depend on the eligible candidate set;
- a symmetric Dirichlet distribution is an explicit assumption, not an absence
  of assumptions;
- a weighted sum permits compensation among criteria;
- selecting the maximum-throughput point is a declared operating-point policy;
- missing models are excluded through complete-case analysis;
- measurement uncertainty is not yet propagated through the rank simulation.

For these reasons, PRRA is a secondary preference-sensitivity summary. Raw
metrics, confidence intervals, Pareto fronts, and alpha sensitivity remain
necessary.

## Joint reporting rule

The publication tables report EAS, MOAS, and PRRA together, without averaging
them into another composite score:

- EAS answers: which model is preferred for a declared accuracy--energy
  trade-off?
- MOAS answers: which model is preferred for one declared four-objective
  deployment profile?
- PRRA answers: how stable is each model's rank when preferences over accuracy,
  energy, latency, and memory are uncertain?

A model is not described as universally best solely because it has the highest
EAS, MOAS, or PRRA. Conclusions must also inspect raw effect sizes, uncertainty,
and Pareto dominance.

## Intellectual provenance

SMAA explores weight space when preferences are incomplete and reports rank
acceptability indices. PRRA adopts that motivation, then adds PowerBench-specific
within-hardware rank costs, complete-case hardware aggregation, fixed
best-throughput operating-point selection, and the normalized expected-rank
summary \(A_{hm}\).

The supporting references are:

- Tervonen and Lahdelma, "Implementing stochastic multicriteria acceptability
  analysis," *European Journal of Operational Research*, 2007,
  https://doi.org/10.1016/j.ejor.2005.12.037.
- Zhang and Golovin, "Random Hypervolume Scalarizations for Provable
  Multi-Objective Black Box Optimization," ICML 2020,
  https://proceedings.mlr.press/v119/zhang20i.html.
- Jansen et al., "Statistical Comparisons of Classifiers by Generalized
  Stochastic Dominance," *Journal of Machine Learning Research*, 2023,
  https://jmlr.org/papers/v24/22-0902.html.

BibTeX entries are provided in `docs/ROBUST_RANKING_REFERENCES.bib`.
