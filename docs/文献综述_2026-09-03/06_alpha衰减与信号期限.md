# 主题 6：alpha 衰减与信号期限（alpha decay / signal horizon / fast-slow blending）

检索执行日期：**2026-09-03**（单日会话）。
角色：deep-research skill 的 `bibliography_agent` + `source_verification_agent` 合体，lit-review 模式。
本文件所有条目均在本次会话中通过 Crossref / Semantic Scholar / arXiv / WebSearch / WebFetch 实时检索确认，
每条至少携带一个可验证标识（DOI / arXiv ID / SSRN ID / NBER WP 号）。

---

## §0 检索矩阵

### 0.1 核心概念的规范术语、同义词与旧新名

| 概念簇 | 英文规范term | 同义/近义/旧名 | 备注 |
|---|---|---|---|
| 信号衰减 | alpha decay | signal decay, information decay, signal half-life, mean-reversion speed of alpha, predictability horizon, forecast decay rate | **该词在文献中至少有三种互不相同的含义**（见 §4.1），检索时必须分开 |
| 信息期限 | information horizon | signal horizon, forecast horizon, prediction horizon, holding-period horizon, term structure of alpha | Qian–Sorensen–Hua 的专有说法 |
| 信号自相关 | forecast autocorrelation | signal autocorrelation, characteristic persistence, persistent vs transitory component | Qian et al. / Baba Yara et al. 口径 |
| 衰减率参数 | mean reversion speed Φ / φ | alpha decay rate, decay coefficient, OU speed, kappa | Gârleanu–Pedersen 记号 |
| 交易速率 | trading rate a/λ | trade rate, partial adjustment, rebalance intensity, no-trade region, backlog | Gârleanu–Pedersen / Grinold 记号 |
| 目标组合 | aim portfolio | target portfolio, forward-looking portfolio, Markowitz portfolio | |
| 信号加权 | signal weighting | alpha model weighting, factor blending, integrated alpha modelling, signal combination | Grinold 专有说法 |
| 快慢合成 | fast and slow signals | multi-horizon signal combination, short-horizon overlay, fast/slow blending, tortoise and hare | |
| 换手抵消 | trading diversification | netting, trade netting, offsetting trades, trade cancellation | DeMiguel et al. 用 "trading diversification" |
| 成本缓释 | cost mitigation | banding, buy/hold spread, trading hysteresis, sS rule, staggered partial rebalancing, smart rebalancing, trade prioritization | Novy-Marx–Velikov 术语库 |
| 盈亏平衡 | break-even trading cost | break-even fund size, capacity, break-even holding period, cost-adjusted optimal horizon | |
| 形成/持有矩阵 | formation × holding period matrix | K×J matrix, J/K strategy, overlapping portfolios | Jegadeesh–Titman 传统 |
| 短期反转 | short-term reversal (STR) | weekly reversal, one-month reversal, contrarian, residual reversal | |
| 跳日动量 | skip-day momentum | echo, intermediate-horizon momentum, t-12 to t-7 | Novy-Marx 术语 |
| 隔夜/日内分解 | overnight vs intraday returns | close-to-open / open-to-close decomposition, tug of war | Lou–Polk–Skouras 术语 |

### 0.2 方法名 / 模型名 / 数据集名

Kyle (1985) 策略性交易、Foster–Viswanathan (1996) rat race / waiting game、
Grinold–Kahn fundamental law of active management、IC / lagged IC / horizon IC、
mean-variance with quadratic transaction costs、LQ 控制 / MDP / value iteration、
random Fourier features (Rahimi–Recht)、LASSO、LSTM、
CRSP / Compustat / I-B-E-S、Inalytics 交易层数据、ANcerno / Plexus、
AQR live execution data、JKP Global Factor Data（Jensen et al. 2022）、
Green–Hand–Zhang 100 characteristics、Novy-Marx–Velikov Assaying Anomalies / open-source anomaly library。

### 0.3 关键作者与机构

Gârleanu, Pedersen（NYU/CBS/AQR）；Grinold, Kahn（BARRA/BlackRock）；
Qian, Sorensen, Hua（PanAgora）；Novy-Marx, Velikov（Rochester/Penn State）；
DeMiguel, Martín-Utrera, Nogales, Uppal（LBS/Lancaster/UC3M/EDHEC）；
Jensen, Kelly, Malamud（CBS/Yale/EPFL）；Blitz, Hanauer, Honarvar, Hoogteijling, Howard（Robeco/TUM）；
Frazzini, Israel, Moskowitz, Israelov（AQR）；Di Mascio, Lines, Naik（Inalytics/Columbia/LBS）；
Chinco, Clark-Joseph, Ye；van Kervel, Menkveld；Da, Liu, Schaumburg；Nagel；Bogousslavsky；
Lou, Polk, Skouras；Medhat, Schmeling；Baba Yara, Boons, Tamoni；Chen, Velikov；
Lehalle, Neuman, Cont, Micheli；Boyd, Nystrup；Kolm, Ritter；Firoozye, Tan, Zohren。

### 0.4 期刊 / 会议目录（作为检索靶点）

JF, JFE, RFS, JFQA, Management Science, Journal of Financial Markets, Review of Asset Pricing Studies,
Journal of Accounting and Economics, Journal of Banking & Finance, Journal of Empirical Finance,
Economics Letters, Financial Analysts Journal, Journal of Portfolio Management,
Journal of Financial Data Science, Journal of Investment Strategies, Journal of Trading,
Journal of Investing, Journal of Asset Management, Finance and Stochastics, Quantitative Finance,
NBER Working Papers, SSRN, arXiv q-fin.PM / q-fin.TR / q-fin.MF / econ.GN。

### 0.5 跨学科术语变体

- **运筹学 / 控制论**：linear-quadratic regulator、partial adjustment、infinite-horizon MDP、
  impulse control、no-trade region、hysteresis band、tracking a moving target。
- **统计学 / 时间序列**：AR(1) persistence、Ornstein–Uhlenbeck half-life、
  autocorrelation function、signal-to-noise ratio at horizon h、forecast combination across horizons。
- **ML**：prediction horizon / target labeling horizon、multi-horizon forecasting、
  end-to-end portfolio learning、economic loss function、feature importance。
- **微结构 / 执行**：price impact decay、propagator model、execution horizon、implementation shortfall、
  order duration、alpha capture。
- **核物理污染项（需排除）**：`alpha decay half-life`、`alpha-decay energy`、`Geiger–Nuttall`
  —— Crossref/OpenAlex 上 "alpha decay" 命中绝大多数来自核物理，必须加金融限定词。

---

## §1 检索日志

### 1.1 Crossref（`api.crossref.org/works`，全部于 2026-09-03 执行）

> **方法学警示（必须与命中数一起阅读）**：Crossref 的 `query.bibliographic` 是模糊全文匹配，
> `message.total-results` 反映的是"至少匹配某个 token 的记录数"，并非精确布尔命中数，
> 因此下表的"命中数"只能用于说明该查询式的噪声水平，**不可解读为相关文献数量**。
> 实际筛选依据是相关性排序的前 4–10 条。

| 数据库 | 查询式（原文） | 排序/时间 | 命中数 | 翻页 | 排除理由 |
|---|---|---|---|---|---|
| Crossref | `alpha+decay+institutional+trading` | 全时段/相关性 | 502,446 | rows=20，第1页 | 前 20 条中 8 条为核物理 `alpha decay`（Radiopaedia / Nuclear Physics A / PhySH），排除；保留 SSRN 2070958、2580551 线索 |
| Crossref | `Dynamic+Trading+with+Predictable+Returns+and+Transaction+Costs` | 全时段/相关性 | 1,151,096 | rows=6 | 保留 NBER w15205、SSRN 三版、JF 2013 |
| Crossref | `Information+Horizon+Portfolio+Turnover+and+Optimal+Alpha+Models` | 全时段/相关性 | 4,600,225 | rows=6 | 命中 JPM 2007 正版；其余 Digital Portfolio Theory 系列不相关，排除 |
| Crossref | `Signal+Weighting+Grinold+Journal+of+Portfolio+Management` | 全时段/相关性 | 4,945,128 | rows=6 | 命中 Grinold 四篇（Signal Weighting 两个 DOI、Linear/Nonlinear Trading Rules、Dynamic Portfolio Analysis） |
| Crossref | `Is+momentum+really+momentum` | 全时段/相关性 | 112,199 | rows=6 | 命中 Novy-Marx JFE 2012；额外发现 Gong–Liu–Liu (2015) JBF |
| Crossref | `The+Long-Lasting+Momentum+in+Weekly+Returns` | 全时段/相关性 | 1,699,791 | rows=6 | 命中 Gutierrez–Kelley JF 2008；CFA Digest 摘要版排除（二手） |
| Crossref | `alpha+decay+signal+half-life+trading+horizon` | (a) 全时段/相关性 | 2,793,536 | rows=8 | 前 8 条全部核物理，排除 |
| Crossref | 同上 | (b) from-pub-date:2024-09-03/相关性 | 349,010 | rows=8 | 7 条核物理 + 1 条电商归因；仅保留 SSRN 7376818 |
| Crossref | 同上 | (c) from-pub-date:2026-03-07 + `sort=published&order=desc` | 78,794 | rows=8 | **该排序不可用**：返回记录的 issued 日期为 2121/2107/2088/2058 等垃圾元数据，无一相关，全部排除 |
| Crossref | `signal+decay+portfolio+turnover+transaction+costs+holding+period+equity` | (a) 全时段/相关性 | 1,497,439 | rows=8 | 命中 SSRN 4425407（Firoozye–Tan–Zohren）、SSRN 2096871（Dumas et al. hysteresis） |
| Crossref | 同上 | (b) ≥2024-09-03/相关性 | 190,952 | rows=8 | 多为量子计算/ESG 组合优化，与期限映射无关，排除 |
| Crossref | 同上 | (c) ≥2026-03-07/日期倒序 | 43,201 | rows=8 | 同样被垃圾日期元数据占满，全部排除 |
| Crossref | `prediction+horizon+stock+return+machine+learning+transaction+costs+net+alpha+turnover` | ≥2026-03-07/相关性 | 53,235 | rows=10 | 10 条均为低层级会议/区域性 ML 预测论文，无期限—成本联合分析，排除 |
| Crossref | 同上 | ≥2024-09-03/相关性 | 243,573 | rows=10 | 仅保留 SSRN 6422358（Pall 2026） |
| Crossref | `signal+persistence+portfolio+construction+optimal+rebalancing+frequency+anomaly+net+returns` | ≥2026-03-07/相关性 | 2,148 | rows=10 | 全部为宏观再平衡/ESG/加密，与信号期限无关，排除 |
| Crossref | 同上 | ≥2024-09-03/相关性 | 8,482 | rows=10 | 同上，排除 |
| Crossref | 以下查询式仅取相关性前 4–6 条、未记录 total-results：`A+Taxonomy+of+Anomalies+and+Their+Trading+Costs`；`A+Transaction-Cost+Perspective+on+the+Multitude+of+Firm+Characteristics`；`Machine+Learning+and+the+Implementable+Efficient+Frontier`；`Short-term+momentum+Medhat+Schmeling`；`Evaporating+liquidity+Nagel`；`The+Tortoise+and+the+Hare+Portfolio+Dynamics+for+Active+Managers+Sneddon`；`Infrequent+Rebalancing+Return+Autocorrelation+and+Seasonality+Bogousslavsky`；`Overnight+returns+daytime+reversals+and+future+stock+returns`；`A+Closer+Look+at+the+Short-Term+Return+Reversal`；`Sparse+Signals+in+the+Cross-Section+of+Returns`；`High-Frequency+Trading+around+Large+Institutional+Orders`；`A+Tug+of+War+Overnight+Versus+Intraday+Expected+Returns`；`Factor+Momentum+and+the+Momentum+Factor`；`Trading+Costs+Frazzini+Israel+Moskowitz`；`Alpha+Decay+Di+Mascio+Lines+Pedersen+institutional+trading`；`Ehsani+Linnainmaa+Factor+Momentum+Journal+of+Finance+2022`；`Uncovering+the+alpha+decay+of+institutional+trades`；`Duration+of+Executions+and+Alpha+Decay+institutional+orders`；`Short-Term+Momentum+and+Reversals+in+Large+Stocks`；`Cheng+Hameed+Subrahmanyam+Titman+Short-Term+Reversals...`；`The+cross-section+of+intraday+and+overnight+returns+Bogousslavsky`；`Have+capital+market+anomalies+attenuated...`；`Multi-Period+Trading+via+Convex+Optimization+Boyd`；`Performance+v+Turnover+A+Story+by+4000+Alphas+Kakushadze`；`Incorporating+signals+into+optimal+trading+Lehalle+Neuman`；`Optimal+trading+with+alpha+predictors+Lehalle+Mounjid`；`Zeroing+in+on+the+expected+returns+of+anomalies+Chen+Velikov`；`Assaying+Anomalies+Novy-Marx+Velikov`；`The+Expected+Returns+on+Machine-Learning+Strategies...`；`Fast+and+Slow+Optimal+Trading+with+Exogenous+Information...`；`Selling+Fast+and+Buying+Slow...`；`Jegadeesh+Titman+Returns+to+buying+winners...`；`Lehmann+Fads+martingales+and+market+efficiency+weekly+reversal`；`Reversing+the+Trend+of+Short-Term+Reversal...`；`Short-term+residual+reversal+Blitz...`；`Beyond+Fama-French+Factors+Alpha+from+Short-Term+Signals...`；`Understanding+Alpha+Decay+Penasse`；`Multi-period+portfolio+optimization+with+alpha+decay+Kolm+Ritter`；`Timing+Equity+Quant+Positions+with+Short-Horizon+Alphas...`；`Turnover-Adjusted+Information+Ratio...`；`Portfolio+turnover+when+IC+is+time-varying...` | 全时段/相关性 | 未记录 | rows=4–6 | 排除标准：非金融同名（核物理、医学、艺术辞典）、二手摘要（CFA Digest）、区域性低层级会议论文、与期限/成本无直接关系者 |
| Crossref | 单 DOI 元数据核验（`/works/{DOI}`）共 **34 次** | — | — | — | 用于确认标题、作者、卷期页、issued/published-online 日期、被引数、摘要 |

### 1.2 Semantic Scholar（`api.semanticscholar.org/graph/v1`）

| 端点 | 查询 | 结果 |
|---|---|---|
| `/paper/search` | `Alpha Decay Di Mascio Lines Pedersen`；`Dynamic Trading with Predictable Returns...`；`Information Horizon Portfolio Turnover...`；`alpha decay signal horizon portfolio`；`information horizon portfolio turnover` | **全部 HTTP 429**。按协议做指数退避（4s → 20s → 25s → 45s），仍 429。该通道的**关键词检索能力在本次会话中完全不可用** |
| `/paper/DOI:{doi}` | 10.1111/jofi.12080 等 9 个 DOI | 可用。取得被引数与摘要 |
| `/paper/DOI:{doi}/citations` | 10.1111/jofi.12080（前向，limit=100）；10.3905/jpm.2007.698030（22 条，全量）；10.1093/rfs/hhz085（100 条）；10.1093/rfs/hhv063（100 条）；10.1016/j.jfineco.2024.103808（7 条，全量）；10.1080/0015198x.2023.2173492（15 条，全量）；10.1017/s0022109025101725（1 条，全量） | 可用。**前向引用追踪主要依靠该端点完成** |
| `/paper/DOI:10.3905/jfds.2023.1.135/citations` | Blitz et al. JFDS 2023 | `Paper not found` —— S2 未收录该 JFDS DOI，该种子的前向追踪失败 |

### 1.3 arXiv

| 通道 | 查询式 | 结果 |
|---|---|---|
| `export.arxiv.org/api/query`（curl） | `abs:"alpha decay" AND cat:q-fin*`，日期倒序，max_results=20；重试 3 次（含 15s 间隔、自定义 UA） | **HTTP 503**，三次均失败 |
| 同上（WebFetch 代理） | `all:"signal decay" AND cat:q-fin.TR` | total = **2**（arXiv:2604.13260、arXiv:2512.23515）；均非期限映射研究，仅 2512.23515 涉及 alpha 筛选，排除 |
| 同上（WebFetch） | `abs:"holding period" AND cat:q-fin*`，日期倒序，max_results=30 | total = **26**（全量返回）。仅保留 arXiv:1509.08110（Kakushadze–Tulchinsky, Performance v. Turnover）；其余为组合优化/VaR/ETF/中国市场，与本主题期限映射无关 |
| 同上（WebFetch） | `all:"signal decay" AND cat:q-fin.PM` | **HTTP 429**，未取得结果 |
| `arxiv.org/list/q-fin.PM/2026-08` | 近月全量列表 | 28 条，**0 条**与 alpha 衰减/期限映射相关 |
| `arxiv.org/abs/{id}` | 2502.04284、2605.23905、2105.10306、2304.03437 | 4 篇逐条确认（标题、作者、日期、分类、摘要） |

### 1.4 OpenAlex

| 查询式 | 结果 |
|---|---|
| `filter=title_and_abstract.search:"alpha decay"&sort=cited_by_count:desc` | **HTTP 429**：`{"error":"Rate limit exceeded","message":"Insufficient budget. This request costs $0.001 but you only have $0 remaining. Resets at midnight UTC"}` |
| `filter=title.search:"signal weighting"`（重试） | 同上，同一错误 |

**OpenAlex 在本次会话全程不可用（额度耗尽），协议要求的"≥2 个通用学术索引"只满足了 1.5 个**
（Semantic Scholar 仅 DOI/citations 端点可用，搜索端点不可用）。

### 1.5 WebSearch / WebFetch（专业库、SSRN、期刊页、机构页）

| 查询式 / URL | 用途 | 结果 |
|---|---|---|
| WebSearch `"Alpha Decay" Di Mascio Lines Naik institutional investors trading SSRN 2580551` | 哨兵核实 | 确认存在；第三作者为 **Naik** 而非 Pedersen |
| WebSearch `Grinold "Signal Weighting" 2010 JPM signal half-life horizon combining fast slow signals` | 内容获取 | 仅得二手描述（signals 分 SLOW/INTERMEDIATE/FAST）；正文付费墙 |
| WebSearch `Qian Sorensen Hua "Information Horizon" ... half-life autocorrelation of alpha` | 定位可读全文 | 定位到 northinfo.com / gyanresearch 托管的 JPM 正版 PDF |
| WebSearch `Garleanu Pedersen 2013 "aim portfolio" trading rate formula ...` | 公式核实 | 得二手描述；随后以 NBER PDF 原文核实 |
| WebSearch `DeMiguel ... netting effect combining signals` | 定位全文 | 定位到 LBS Research Online 开放版 |
| WebSearch `Novy-Marx Velikov "Comparing Cost-Mitigation Techniques" ... banding buy/hold spread` | 内容获取 | 定位 CFA Institute 官方摘要页（含定量结论） |
| WebSearch `"break-even" holding period transaction costs anomaly "turnover" formula optimal rebalancing horizon signal decay finance` | 通用 | 返回中混入非学术博客（microalphas.com / aifinhub.io），**已全部排除，未进入候选表** |
| WebSearch `Frazzini Israel Moskowitz "Trading Costs" 2018 live trading data AQR ...` | 定位全文 | 定位到 NYU Stern 与 Chicago Booth 托管 PDF |
| WebSearch `Jensen Kelly Malamud Pedersen "Machine Learning and the Implementable Efficient Frontier" ...` | 定位全文 | 定位 AFA 会议 PDF |
| WebSearch `2025 2026 paper "signal half-life" OR "alpha decay" optimal holding period ...` | 近期检索 | 得 arXiv:2502.04284、arXiv:2605.23905 |
| WebSearch `daily frequency formation period holding period matrix short-term reversal momentum 2020s large caps ...` | 日频 K×J | **未找到日频 K×J 矩阵的现代学术论文**；仅得月频条件排序文献 |
| WebSearch `short-term reversal profitability declined disappeared after 2000s ... Blitz` | 期限结构现代化 | 得 SSRN 4575689（后确认 JPM 2024） |
| WebSearch `Blitz Hanauer Hoogteijling Howard "The Term Structure of Machine Learning Alpha" ...` | 关键论文 | 确认 JFDS 5(4) 40–65，DOI 10.3905/jfds.2023.1.135 |
| WebSearch `site:papers.ssrn.com 2026 "signal decay" OR "alpha decay" holding period turnover equity strategy` | SSRN 近期 | 得 SSRN 2953614（Pénasse）等 |
| WebSearch `Novy-Marx 2012 ... intermediate horizon past performance` | 内容核实 | 确认 t−12..t−7 结论；额外得 arXiv:2304.03437 |
| WebSearch `Baba Yara Boons Tamoni "Persistent and transitory components..."` | 内容核实 | 确认 JFE 154 (2024) 103808 主要结论 |
| WebSearch `"The transaction cost trap" ... Pall SSRN 2026` | 存在性核实 | WebSearch 未命中；改由 Crossref DOI 10.2139/ssrn.6422358 直接核实存在（含摘要） |
| WebFetch `nber.org/papers/w15205` | 元数据 | 成功：issue Aug 2009，revisions 2011-12-05 / 2013-01-30，published JF 68(6) 2309–2340 |
| WebFetch `nber.org/.../w15205.pdf` | 全文 | 成功，提取 Proposition 2/3/4 与实证 φ 估计 |
| WebFetch `nber.org/.../w20721.pdf` | 全文 | 成功，提取 staggered rebalancing 与 buy/hold spread 结果表 |
| WebFetch `lbsresearch.london.edu/.../DeMiguel_TransactionCostPerspective.pdf` | 全文 | 成功，提取 Proposition 3 与 72.15% 换手下降 |
| WebFetch `jhfinance.web.unc.edu/.../Alpha-Decay.pdf` | 全文 | 成功，提取 37bp/12 个月衰减曲线（版本：2015-03-18） |
| WebFetch `gyanresearch.wdfiles.com/.../JPM_FA_07_Qian.pdf` | 全文 | 成功，提取全部公式与数值例 |
| WebFetch `afajof.org/management/viewp.php?n=32368` | 全文 | 成功（JKMP 2023-02-12 版） |
| WebFetch `pages.stern.nyu.edu/~afrazzin/pdf/Trading Cost of Asset Pricing Anomalies...pdf` | 全文 | 成功，提取 Table VI 全部 break-even 数字 |
| WebFetch `rpc.cfainstitute.org/.../ip-v4-n1-4-comparing-cost-mitigation-techniques` | 摘要 | 成功 |
| WebFetch `robeco.com/.../the-term-structure-of-machine-learning-alpha` | 摘要 | 成功（作者机构官方说明） |
| WebFetch `mysimon.rochester.edu/novy-marx/` | 论文清单 | 成功 |

### 1.x 无法访问的来源（逐条）

1. **OpenAlex API** —— HTTP 429 / "Insufficient budget"，2 次尝试全部失败，全程不可用。
2. **Semantic Scholar `/paper/search`** —— HTTP 429，5 次尝试（含 4/20/25/45 秒退避）全部失败。
3. **arXiv API（curl 通道）** —— HTTP 503，3 次尝试全部失败；WebFetch 代理通道在第 3 次查询后转 429。
4. **SSRN 论文页（papers.ssrn.com/sol3/papers.cfm）** —— HTTP 403（机器人拦截）。
   所有 SSRN 条目改由 Crossref 注册的 `10.2139/ssrn.*` DOI 核实。
5. **pm-research.com / jpm.pm-research.com（JPM, JOI, JOT, JFDS）** —— 302 重定向至 SSO 登录（`idp.sams-sigma.com`），付费墙。
   受影响条目：Grinold (2007, 2010, 2018a, 2018b)、Sneddon (2008)、Jha (2016)、
   Blitz et al. (2023, JFDS)、Blitz et al. (2024, JPM)、Bašić et al. (2024, JPM)
   —— 这些条目**仅在元数据/摘要/官方二手摘要层面核实，正文公式与数值未能独立验证**。
6. **Financial Analysts Journal（tandfonline）** —— 未直接取得正文；
   Novy-Marx–Velikov (2019) 依靠 CFA Institute 官方摘要页，Arnott et al. (2024) 依靠 Semantic Scholar 摘要。
7. **ScienceDirect / Wiley / Oxford Academic / Management Science 正文** —— 未尝试绕过付费墙；
   相应条目依靠 Crossref/S2 提供的官方摘要。
8. **mediatum.ub.tum.de** —— DNS 解析失败（`getaddrinfo ENOTFOUND`）。
9. **Crossref `sort=published&order=desc`** —— 功能性不可用：返回记录被 issued 日期为
   2028–2121 的错误元数据占满，导致协议要求的"(c) 近 180 天 + 日期倒序"这一支只能改用
   `filter=from-pub-date:2026-03-07` + 相关性排序替代。

---

## §2 候选文献表

列：`标题 | 作者 | 首次公开日期 | 正式出版日期 | DOI | 预印本↔期刊版本关系 | 来源 | 相关性 | 方法质量理由 | 重要性理由`

> 每条附标注：**样本区间 / 市场 / 频率 / 是否含交易成本 / 是否样本外**。

### A. 解析框架：信息期限 → 交易速率 / 信号权重

| # | 标题 | 作者 | 首次公开 | 正式出版 | DOI | 版本关系 | 来源 | 相关性 | 方法质量 | 重要性 |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | Dynamic Trading with Predictable Returns and Transaction Costs | Gârleanu, Pedersen | NBER w15205, 2009-08（修订 2011-12-05、2013-01-30）；SSRN 1364170/1448169/1658736 (2009–2010) | JF 68(6) 2309–2340, 2013-11-12 | 10.1111/jofi.12080 | 已确认同一研究：NBER WP 与 JF 版标题一致，年份差 4 年。另有 2021-10-06 勘误 10.37214/jofweb.4（未取得正文） | Crossref + NBER 全文 | **极高**：唯一给出"衰减率 φ → 信号权重"闭式规则的论文，且实证信号含 **2.4 日半衰期** | 连续时间 LQ 控制闭式解 + 商品期货实证；**含交易成本**（Engle–Ferstenberg–Russell 校准）；实证为**样本内**校准 | 该领域标准框架，S2 被引 558 |
| A2 | Information Horizon, Portfolio Turnover, and Optimal Alpha Models | Qian, Sorensen, Hua | JPM 在线 2007-10-31 | JPM 34(1) 27–40, 2007 | 10.3905/jpm.2007.698030 | 无预印本；单版本 | Crossref + 全文 PDF | **极高**：定义 information horizon，给出 `T ∝ √(1−ρ_f)` 换手公式与净收益最优 ρ_f | 解析推导 + Russell 3000 季频 IC 输入（1987–2004）；**含交易成本**（线性，0.5/1.0/1.5% per 100% 换手）；数值例为**样本内** | 实务界"信号自相关 → 换手"标准引用，Crossref 被引 17（JPM 收录不全） |
| A3 | Signal Weighting | Grinold | JPM 在线 2010-06-10（DOI 10.3905/jpm.2010.2010.1.005） | JPM 36(4) 24–34, 2010-07-31 | 10.3905/jpm.2010.36.4.024 | 同一文两个 DOI（在线首发 + 正刊），已确认 | Crossref（正文付费墙） | 高：直接命题"存在交易成本时如何给不同信息换手率的信号配权" | **未能核实正文**；据官方检索结果，信号按信息换手率分 SLOW/INTERMEDIATE/FAST | 哨兵论文；Crossref 被引 13 |
| A4 | Dynamic Portfolio Analysis | Grinold | — | JPM 34(1) 12–26, 2007-10-31 | 10.3905/jpm.2007.698029 | 无预印本 | Crossref（付费墙） | 中高：动态组合与交易速率 | 未核实正文 | Crossref 被引 18 |
| A5 | Linear Trading Rules for Portfolio Management | Grinold | — | JPM 44(6) 109–119, 2018-06-30（在线 2018-07-02） | 10.3905/jpm.2018.44.6.109 | 无预印本 | Crossref（付费墙） | 高：trade rate、target portfolio、backlog、no-trade zone | 未核实正文（二手描述） | 哨兵论文；Crossref 被引 0（收录滞后） |
| A6 | Nonlinear Trading Rules for Portfolio Management | Grinold | — | JPM 45(1) 62–, 2018-10-31 | 10.3905/jpm.2018.45.1.062 | 无预印本 | Crossref（付费墙） | 中：A5 的非线性推广 | 未核实正文 | 与 A5 配套 |
| A7 | The Tortoise and the Hare: Portfolio Dynamics for Active Managers | Sneddon | — | Journal of Investing 17(4) 106–111, 2008-11-30 | 10.3905/joi.2008.17.4.106 | 无预印本 | Crossref（付费墙） | 高：标题即"快慢"；主动管理者的组合动态 | 未核实正文 | Crossref 被引 9 |
| A8 | Machine Learning and the Implementable Efficient Frontier | Jensen, Kelly, Malamud, Pedersen | SSRN 4187217, 2022 | RFS, 2026-03-15 | 10.1093/rfs/hhag022（期刊）/ 10.2139/ssrn.4187217（WP） | 已确认同一研究；标题未改；WP↔期刊差约 4 年 | Crossref + AFA 全文（2023-02-12 版） | **极高**：aim 组合显式依赖**所有未来期限**的期望收益；提出"经济特征重要性" | 美股 1952–2020，OOS 1981–2020，**月频**再平衡，115 个特征，random Fourier features；**含交易成本**（二次型）；**样本外** | RFS 新刊；公开代码库；把 G–P 与 ML 统一 |
| A9 | Dynamic Portfolio Selection under Transaction Costs and Signal Decay | Firoozye, Tan, Zohren | SSRN 4425407, 2023-04-28 | 未发现期刊版 | 10.2139/ssrn.4425407 | 期刊版本**未确认** | Crossref（含摘要） | 高：解析解，最优策略为向"前瞻组合"平滑调整；长期绩效以信号持久性等结构参数表达 | 纯解析；**未见实证与成本实测** | 直接以 "signal decay" 为标题的解析工作 |
| A10 | Multi-Period Trading via Convex Optimization | Boyd, Busseti, Diamond, Kahn, Koh, Nystrup, Speth | — | Foundations and Trends in Optimization 3(1), 2017-08-08 | 10.1561/2400000023（另有专著 DOI 10.1561/9781680833294） | 期刊/专著双 DOI，同一内容 | Crossref | 中高：多期交易的凸优化通用框架（含成本、持仓期项） | 方法论专著；**含交易成本建模**；无独立实证检验 | 工程实现的标准参考 |
| A11 | Multi-period portfolio optimisation with alpha decay | Sivaramakrishnan, Jeet, Vandenbussche | — | Int. J. Financial Engineering and Risk Management, 2018 | 10.1504/ijferm.2018.094030 | 未确认是否有 arXiv 前身 | Crossref（经 S2 引用追踪发现） | 高：标题即"多期优化 + alpha 衰减" | 未核实正文；期刊层级较低 | 少见的直接以此为题的工程化工作 |
| A12 | On the Effect of Alpha Decay and Transaction Costs on the Multi-period Optimal Trading Strategy | Ma, Smith | arXiv:2502.04284, 2025-02-06（v1，无修订） | 无期刊版 | arXiv:2502.04284 | 期刊版本**未确认** | arXiv 逐条核实 | 高：单资产无限期 MDP，含当前与滞后信号值的预测力（即 alpha 衰减建模），给出小成本一阶近似与渐近解 | 纯理论（math.OC / q-fin.MF）；**无实证、无净成本回测** | 近期唯一以此为题的严格控制论工作 |
| A13 | Portfolio turnover when IC is time-varying | Ding, Martin, Yang | SSRN 3117881, 2018（题为 "Time Varying IC and Optimal Portfolio Turnover"） | Journal of Asset Management, 2020-01-31 | 10.1057/s41260-019-00145-1 | 标题**已改**（Time Varying IC and Optimal Portfolio Turnover → Portfolio turnover when IC is time-varying），年份差 2 年，同一作者组，判为同一研究但未取得正文交叉确认 | Crossref（经 S2 引用追踪发现） | 中高：Qian et al. 换手公式在 IC 时变下的推广 | 未核实正文 | QSH 框架的直接延伸 |
| A14 | Turnover-Adjusted Information Ratio | Zhang, Wang, Cao | arXiv:2105.10306, 2021-05-19 | 未发现期刊版 | arXiv:2105.10306 | 期刊版本**未确认** | arXiv 逐条核实 | 中高：将 IC 波动与换手成本并入 fundamental law；结论"换手调整后的 IR 恒低于忽略换手的 IR"，且"最优化而非最大化交易频率"可提高风险调整后收益 | 解析 + 模拟；**无真实市场净成本实证** | 直接反驳 fundamental law 的"广度越大越好"推论 |
| A15 | Incorporating signals into optimal trading | Lehalle, Neuman | — | Finance and Stochastics 23(2) 275–311, 2019-02-14 | 10.1007/s00780-019-00382-7 | 无预印本记录 | Crossref | 中：把外生信号并入最优执行 | 随机控制；**执行层面**（分钟/秒），非持仓期 | Crossref 被引 55 |
| A16 | Fast and slow optimal trading with exogenous information | Cont, Micheli, Neuman | SSRN 4489258, 2023 | Finance and Stochastics 29(2) 553–607, 2025-03-19 | 10.1007/s00780-025-00560-w | 已确认同一研究，标题仅大小写差异，年份差 2 年 | Crossref（含摘要） | 中高：低频投资者与高频交易者的多期 Stackelberg 均衡；信号强时 HFT 采取掠夺策略，弱时合作 | 严格均衡解；**无实证** | "fast and slow" 的博弈论刻画，与"快慢信号合成"不同问题 |

### B. alpha 衰减的实证测量

| # | 标题 | 作者 | 首次公开 | 正式出版 | DOI/ID | 版本关系 | 来源 | 相关性 | 方法质量 | 重要性 |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | Alpha Decay | Di Mascio, Lines, **Naik** | SSRN 2580551，Crossref 记录创建 2015-03-25；取得的稿本注明 First Draft 2014-02-15 / This Version 2015-03-18；SSRN 页面另标 2017 年修订 | **未发现期刊版** | 10.2139/ssrn.2580551 | 期刊版本**未确认**；哨兵线索中的第三作者 "Pedersen" **有误**，实为 Narayan Y. Naik | Crossref + UNC 托管全文 | **极高**：机构交易层面 alpha 衰减的直接测量 | Inalytics 逐笔交易 + 每日持仓 + AUM，700+ 组合，**2001–2013**，美/英/日，115 万笔、1.8 万亿美元；**月频事件研究**；**不含交易成本**；样本内 | 唯一以 "Alpha Decay" 为题的机构交易实证；被后续机构交易文献广泛引用 |
| B2 | Selling Fast and Buying Slow: Heuristics and Trading Performance of Institutional Investors | Akepanidtaworn, Di Mascio, Imas, Schmidt | SSRN 3301277 (2018)、SSRN 3893357 (2021)、NBER w29076 (2021-07) | JF 78(6) 3055–3098, 2023-09-07 | 10.1111/jofi.13271 | 已确认同一研究；WP↔期刊年份差 2–5 年 | Crossref（含摘要） | 中高：同一数据家族；买入有技能、卖出劣于随机 | 机构组合均值 5.73 亿美元；**不含交易成本**的决策层评估 | S2 被引 54；JF |
| B3 | High-Frequency Trading around Large Institutional Orders | van Kervel, Menkveld | SSRN 2619686, 2015 | JF 74(3) 1091–1137, 2019-03-21 | 10.1111/jofi.12759 | 已确认同一研究，年份差 4 年 | Crossref（含摘要） | 中高：机构母单经由多日子单执行，HFT 先逆向后同向 | 逐笔订单数据；**日内至多日**；含价格冲击 | Crossref 被引 205 |
| B4 | Sparse Signals in the Cross-Section of Returns | Chinco, Clark-Joseph, Ye | SSRN 2606396 (2015)、NBER w23933 (2017-10) | JF 74(1) 449–492, 2018-11-14 | 10.1111/jofi.12733 | 已确认同一研究；WP↔期刊年份差 1–3 年 | Crossref（含摘要） | 高：**1 分钟**前瞻预测，预测因子"意外、短命、稀疏" | LASSO 滚动预测；美股高频；**报告了预测隐含 Sharpe**，未做完整净成本回测 | 期限谱最快端的标杆；Crossref 被引 271 |
| B5 | Understanding Alpha Decay | Pénasse | SSRN 2953614, 2017（题为 "Understanding Anomaly Discoveries"） | Management Science, 2022-05 | 10.1287/mnsc.2022.4353 | 标题**已改**（Understanding Anomaly Discoveries → Understanding Alpha Decay），年份差 5 年，同一作者，判为同一研究但未取得正文交叉确认 | Crossref | 中高：**"alpha decay" 的第三种含义**——异象发现后逐年失效 | 长期面板；与日内/日频信号半衰期无关 | 澄清术语歧义的关键条目 |
| B6 | AI-Driven Alpha Decay: Algorithmic Homogenization, Reflexive Signal Erosion, and the Paradox of Intelligent Markets | Meng, Chen | arXiv:2605.23905, 2026-03-23 | 无期刊版 | arXiv:2605.23905 | 期刊版本**未确认** | arXiv 逐条核实 | 中：推导"alpha 半衰期"作为均衡对象；称当前采用率下半衰期约 18 个月、AI 普及前 5–7 年 | 理论模型 + 13F 校准（2013–2024，9950 万条持仓）+ 模拟；**半衰期指策略寿命，非信号期限**；未做净成本回测 | 近 180 天内出现；术语歧义的又一例 |
| B7 | When Alpha Dies: A Signal Autopsy Approach to Predicting Strategy Decay | Zulfiqar | SSRN 7376818，Crossref 记录创建 2026-09-01 | 无期刊版 | 10.2139/ssrn.7376818 | 期刊版本**未确认** | Crossref（含摘要） | 中：以生存分析预测策略"死亡"，死亡定义为**前瞻净成本后**表现 | 10 个信号族、十年**日频 ETF** 数据；**含成本**（死亡定义中）；样本外验证 | 近 180 天内出现；把"衰减"操作化为生存问题 |

### C. 快慢合成 / 信号 netting

| # | 标题 | 作者 | 首次公开 | 正式出版 | DOI/ID | 版本关系 | 来源 | 相关性 | 方法质量 | 重要性 |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | A Transaction-Cost Perspective on the Multitude of Firm Characteristics | DeMiguel, Martín-Utrera, Nogales, Uppal | SSRN 2912819, 2017（题为 "A Portfolio Perspective on the Multitude of Firm Characteristics"） | RFS 33(5) 2180–2222, 2020-04-17 | 10.1093/rfs/hhz085 | 标题**已改**（Portfolio → Transaction-Cost），年份差 3 年，同一作者组；LBS 开放版内容与期刊版一致 | Crossref + LBS 全文 | **极高**：唯一给出 netting 解析公式与大规模实证的论文 | 美股 CRSP/Compustat/IBES，**1980–2014**，剔除市值后 20%，平均 3,071 只/月，51 个特征，**月频**；**含交易成本**（比例 + 二次型）；样本内为主 | Crossref 被引高；"trading diversification" 概念来源 |
| C2 | Trading Costs of Asset Pricing Anomalies | Frazzini, Israel, Moskowitz | SSRN 2294498, 2012-10-23（取得稿本 2012-12-05） | 未发现期刊版 | 10.2139/ssrn.2294498 | 期刊版本**未确认** | Crossref + NYU Stern 全文 | **极高**：给出 STR/UMD/HML/SMB 的换手、实测成本与 break-even 基金规模；ValMom 组合的 netting 效果 | AQR **live** 执行数据近 1 万亿美元，19 个发达市场，**1998–2011**，月频组合；**含真实实测交易成本**；成本为样本内实测、收益用 1926–2011 长样本均值 | 哨兵论文；实测成本的基准 |
| C3 | Trading Costs | Frazzini, Israel, Moskowitz | SSRN 3229719, 2018-08-23 | 未发现期刊版 | 10.2139/ssrn.3229719 | 期刊版本**未确认** | Crossref + WebSearch | 高：C2 的扩展 | AQR live 执行数据 **1.7 万亿美元**，21 个发达市场，19 年；结论"实际成本比既往研究小一个数量级" | 哨兵论文 |
| C4 | Timing Equity Quant Positions with Short-Horizon Alphas | Jha | SSRN 2738368, 2016（题为 "...with Shorter-Horizon Alphas"） | Journal of Trading 11(3) 53–, 2016-06-30 | 10.3905/jot.2016.11.3.053 | 标题微改（Shorter → Short），同年，判为同一研究 | Crossref（付费墙） | 高：以短期限 alpha 择时慢速量化仓位 —— 即"快信号叠加在慢信号上" | **未核实正文** | 直接命中"快慢合成"这一子问题的少数条目 |
| C5 | To Trade or Not to Trade? Informed Trading with Short-Term Signals for Long-Term Investors | Israelov, Katz | — | FAJ 67(5) 23–36, 2011-09 | 10.2469/faj.v67.n5.3 | 无预印本记录 | Crossref（付费墙） | 高：长期限投资者如何使用短期信号 | **未核实正文** | 与 C4 同一问题的更早期处理 |
| C6 | Performance versus turnover: a story by 4000 alphas | Kakushadze, Tulchinsky | arXiv:1509.08110, 2015-09-27；SSRN 2657603, 2015 | Journal of Investment Strategies 5(2) 75–89, 2016-03 | 10.21314/jois.2016.066 | 已确认同一研究，年份差 1 年 | Crossref + arXiv 列表 | 中高：4000 条真实 alpha 的**绩效—换手**经验关系 | WorldQuant 内部 alpha 库；**换手/持仓期与绩效的横截面关系**；成本处理未核实 | 少见的大规模 alpha 库经验证据 |
| C7 | Smart Rebalancing | Arnott, Li, Linnainmaa | — | FAJ 80, 26–51, 2024-03-14 | 10.1080/0015198X.2024.2317323 | 无预印本记录 | Crossref + S2 摘要 | 高：按信号强度对交易排优先级，可在削减换手的同时保留大部分因子溢价 | 未核实正文；摘要层面确认结论 | 近两年 FAJ；与"延长持仓期"构成替代方案 |

### D. 净成本下的期限 / 再平衡频率选择

| # | 标题 | 作者 | 首次公开 | 正式出版 | DOI/ID | 版本关系 | 来源 | 相关性 | 方法质量 | 重要性 |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | A Taxonomy of Anomalies and Their Trading Costs | Novy-Marx, Velikov | NBER w20721, 2014-12 | RFS 29(1) 104–147, 2015-11-10 | 10.1093/rfs/hhv063 | 已确认同一研究（大小写差异），年份差 1 年 | Crossref + NBER 全文 | **极高**：staggered partial rebalancing 与 buy/hold spread 的定量比较 | 美股，**1973-07 至 2012-12**，月频组合，NYSE 分位；**含交易成本**（有效价差估计）；样本内 | Crossref 被引 626 |
| D2 | Comparing Cost-Mitigation Techniques | Novy-Marx, Velikov | SSRN 3253359, 2018 | FAJ 75(1) 85–102, 2019-01-24 | 10.1080/0015198X.2018.1547057 | 已确认同一研究，年份差 1 年 | Crossref + CFA Institute 官方摘要 | **极高**：直接比较"降低再平衡频率" vs "banding" vs "限于低成本股" | 美股 **1975-01 至 2016-12**，大/小/微盘三组，7 个基础策略 × 3 种技术（21 例）；**含交易成本**（日收盘价推算价差）；样本内 | 2019 Graham & Dodd Scroll Award |
| D3 | Model Comparison with Transaction Costs | Detzel, Novy-Marx, Velikov | — | JF 78(3) 1743–1775, 2023-04-12 | 10.1111/jofi.13225 | 无预印本记录 | Crossref（含摘要） | 高：忽略成本会使模型比较偏向高成本（高换手）因子 | 205 个异象；**含交易成本**；样本内 | S2 被引 77；JF |
| D4 | Zeroing In on the Expected Returns of Anomalies | Chen, Velikov | FEDS 2020-039, 2020-05（DOI 10.17016/feds.2020.039） | JFQA 58(3) 968–1004, 2022-08-12 | 10.1017/s0022109022000874 | 已确认同一研究，年份差 2 年 | Crossref（含摘要） | **极高**：204 个异象扣除有效价差、发表后衰减与 2000 年代后交易技术，平均净期望仅 4bp/月 | 美股长样本；**含交易成本**（有效价差；明示**未含价格冲击**）；含发表后样本外 | Crossref 被引 103 |
| D5 | The Term Structure of Machine Learning Alpha | Blitz, Hanauer, Hoogteijling, Howard | SSRN 4474637, 2023-06-18 | Journal of Financial Data Science 5(4) 40–65, 2023-09-14 | 10.3905/jfds.2023.1.135 | 已确认同一研究，同年 | Crossref + 作者机构官方摘要 | **极高**：直接检验"训练期限（1/3/6/12 个月前瞻收益）"对净 alpha 的影响 | 美股；官方说明称 **2004–2021** 为关键子样本；**含交易成本**；**样本外**滚动训练 | 直接命中"期限映射"这一问题的最新学术—实务交叉论文 |
| D6 | The Expected Returns on Machine-Learning Strategies | Azevedo, Hoegner, Velikov | SSRN 4702406, 2024-02-07 | 未发现期刊版 | 10.2139/ssrn.4702406 | 期刊版本**未确认** | Crossref（含摘要） | 高：ML 异象策略在成本 + 发表后衰减 + 小数化后累计损失 57%，LSTM 仍盈利（毛/净 SR 0.94/0.84） | 美股；**含交易成本**；含发表后样本外 | 与 D5 结论方向不同，构成对照 |
| D7 | Transaction Cost–Optimized Equity Factors around the World | Bašić, Lohre, Martín-Utrera, Nolte, Nolte | — | JPM 50(6) 40–73, 2024-02-22 | 10.3905/jpm.2024.1.599 | 无预印本记录 | Crossref（付费墙） | 中高：全球范围的成本优化因子构建 | **未核实正文** | 近两年；国际样本 |
| D8 | Filled and Killed: Forecast and Realized Trading Costs Across Horizons from Global Equity and Fixed Income Portfolio Trades | Ang, Madhavan | SSRN 4782032, 2024 | 未发现期刊版 | 10.2139/ssrn.4782032 | 期刊版本**未确认** | Crossref（含摘要） | 中高：用计数模型确定**最优交易期限**（execution horizon），随交易风险、金额、复杂度上升而延长 | **2,022 笔组合过渡交易**，2016-02 至 2023-12，>3.1 万亿美元；**含实测成本**；预测 vs 实现对照 | 数据集新颖性极高；但对象是**执行期限**而非持仓期 |
| D9 | Cost mitigation of factor investing in emerging equity markets | Stankov, Schiereck, Flögel | — | Journal of Asset Management 25, 303–325, 2024-05-25 | 10.1057/s41260-024-00353-4 | 无预印本记录 | Crossref（含摘要） | 中：以相对短期流动性限制单笔规模的成本缓释 | 新兴市场；**含交易成本** | 与 D2 的技术谱系相同，市场不同 |
| D10 | The transaction cost trap: Why machine learning stock prediction fails economically under realistic market frictions | Pall | SSRN 6422358，Crossref 记录创建 2026-04-20 | 无期刊版 | 10.2139/ssrn.6422358 | 期刊版本**未确认** | Crossref（含摘要） | 中：73.3% 条件方向准确率，扣 5bp 成本后 −42.49%/年（Sharpe −2.83） | **仅 7 只大盘科技股**，17,773 个股票-日观测，2015-01 至 2025-04，**日频**；**含交易成本**；walk-forward 样本外 | 近 180 天内；**外部效度极低**（横截面宽度 7），仅作为"日频高换手 ML 在成本下失效"的个案 |

### E. 短期价格信号的期限结构（形成期/持有期）

| # | 标题 | 作者 | 首次公开 | 正式出版 | DOI/ID | 版本关系 | 来源 | 相关性 | 方法质量 | 重要性 |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | Returns to Buying Winners and Selling Losers | Jegadeesh, Titman | — | JF 48(1), 1993-03 | 10.1111/j.1540-6261.1993.tb04702.x | 无预印本记录 | Crossref | 高：K×J 矩阵与 staggered overlapping 组合的原始出处 | 美股，**月频**；**不含交易成本** | 期限矩阵方法论源头 |
| E2 | Fads, Martingales, and Market Efficiency | Lehmann | NBER w2533, 1988-03（DOI 10.3386/w2533） | QJE, 1990-02 | 10.2307/2937816 | 已确认同一研究，年份差 2 年 | Crossref | 中高：周频反转的原始证据 | 美股，**周频**；**不含交易成本** | 短期反转文献起点 |
| E3 | Is momentum really momentum? | Novy-Marx | — | JFE 103(3) 429–453, 2012-03 | 10.1016/j.jfineco.2011.05.003 | 无预印本记录 | Crossref + WebSearch 核实结论 | 高：动量主要由 t−12..t−7 驱动，而非 t−6..t−2；近期形成期策略盈利但更弱，**在最大最流动股票中尤甚** | 美股 + 国际指数/商品/汇率，**月频**；**不含交易成本** | 哨兵论文；Crossref 被引 400 |
| E4 | The Long-Lasting Momentum in Weekly Returns | Gutierrez, Kelley | SSRN 890305, 2006（题为 "Evidence to the Contrary: Weekly Returns Have Momentum"） | JF 63(1) 415–447, 2008-01-10 | 10.1111/j.1540-6261.2008.01320.x | 标题**已改**，年份差 2 年，同一作者，判为同一研究但未取得正文交叉确认 | Crossref（含摘要） | **极高**：一周形成期信号在**短持有期为反转、长持有期转为持续**，全年累计动量占优 | 美股，**周形成期 × 至一年持有期**；**不含交易成本** | 哨兵论文；直接展示"同一信号的符号随持有期翻转" |
| E5 | A Closer Look at the Short-Term Return Reversal | Da, Liu, Schaumburg | — | Management Science 60(3) 658–674, 2014-03 | 10.1287/mnsc.2013.1766 | 无预印本记录 | Crossref（含摘要） | 高：剔除基本面驱动成分后的增强反转，风险调整后收益为标准反转的约 4 倍 | 美股，**月频**；**不含交易成本**（主结果） | 哨兵论文；Crossref 被引 200 |
| E6 | Evaporating Liquidity | Nagel | SSRN 1573164 (2010)、NBER w17653 (2011-12) | RFS 25(7) 2005–2039, 2012-06-19 | 10.1093/rfs/hhs066 | 已确认同一研究，年份差 1–2 年 | Crossref | 高：短期反转收益作为流动性供给报酬，随波动率时变 | 美股，**日/周频**；**部分含成本讨论** | Crossref 被引 565 |
| E7 | Short-Term Reversals: The Effects of Past Returns and Institutional Exits | Cheng, Hameed, Subrahmanyam, Titman | SSRN 2389408, 2014（题为 "Short-Term Reversals and the Efficiency of Liquidity Provision"） | JFQA 52(1) 143–173, 2017-02 | 10.1017/s0022109016000958 | 标题**已改**，年份差 3 年，同一作者组；判为同一研究但未取得正文交叉确认 | Crossref（含摘要） | 中高：前一季度下跌后反转更强；机构参与度调节反转幅度 | 美股，**月频**；**不含交易成本** | Crossref 被引 83 |
| E8 | Short-term residual reversal | Blitz, Huij, Lansdorp, Verbeek | SSRN 1911449, 2011 | Journal of Financial Markets 16(3) 477–504, 2013-08 | 10.1016/j.finmar.2012.10.005 | 已确认同一研究，年份差 2 年 | Crossref | 高：残差反转；官方摘要称反转在**最大 500 甚至 100 只**股票中亦存在 | 美股 NYSE 中位数以上市值，1926-01 至 2008-12；**月频**；成本处理未核实 | 与"大盘股无反转"叙述直接冲突 |
| E9 | Reversing the Trend of Short-Term Reversal | Blitz, van der Grient, Honarvar | SSRN 4575689, 2023-09 | JPM 50(6) 89–101, 2024-02-06 | 10.3905/jpm.2024.1.588 | 已确认同一研究，年份差 1 年；另有 Practical Applications 摘要版 10.3905/pa.2025.pa647 | Crossref（付费墙）+ 官方检索摘要 | **极高**：称经典短期反转效应已在多数地区**消失**；增强版（避免对抗短期动量）仍有效 | 全球股票；**月频**；成本处理未核实正文 | 现代期限结构的关键主张 |
| E10 | Short-term reversal persists globally—If properly measured | Stosik, Zaremba | — | Economics Letters 267, 113113, 2026-07 | 10.1016/j.econlet.2026.113113 | 无预印本记录 | Crossref（经 S2 引用追踪发现） | **极高**：标题即直接**限定/反驳** E9 的"消失"主张 | 全球；正文未核实（摘要在 Crossref/S2 均为空） | 近 180 天内；文献冲突点 |
| E11 | Short-term Momentum | Medhat, Schmeling | SSRN 3150525, 2018 | RFS 35(3) 1480–1526, 2021-06-08（在线） | 10.1093/rfs/hhab055 | 已确认同一研究，年份差 3 年 | Crossref（含摘要） | **极高**：上月收益 × 换手率双重排序：低换手为反转、**高换手为短期动量**；短期动量"与常规动量同样持久"，**扣成本后存活**，且在**最大、最流动、覆盖最广**的股票中最强 | 美国 + 国际，**月频**；**含交易成本**；样本内 | 直接给出"1 个月价格信号的符号与期限依赖于换手率" |
| E12 | Momentum is really short-term momentum | Gong, Liu, Liu | — | Journal of Banking & Finance, 2015-01 | 10.1016/j.jbankfin.2014.10.002 | 无预印本记录 | Crossref | 中高：与 E3 的期限归因相反 | 未核实正文 | 期限归因的冲突证据 |
| E13 | Short-term reversals, short-term momentum, and news-driven trading activity | Chiang, Kirby, Nie | SSRN 3369648, 2019（题为 "Short-Term Reversals and Trading Activity"） | JBF 125, 106068, 2021-04 | 10.1016/j.jbankfin.2021.106068 | 标题**已改**，年份差 2 年，同一作者组；判为同一研究但未取得正文交叉确认 | Crossref | 中高：换手最高十分位表现为短期动量 | 美股，月频；成本处理未核实 | 与 E11 相互印证 |
| E14 | Short-term momentum and reversals, turnover, and a stock's price-to-52-week-high ratio | Chen, Stivers, Sun | — | Journal of Empirical Finance 79, 101556, 2024-12 | 10.1016/j.jempfin.2024.101556 | 无预印本记录 | Crossref | 中：进一步条件化 | 未核实正文 | 近两年 |
| E15 | Infrequent Rebalancing, Return Autocorrelation, and Seasonality | Bogousslavsky | SSRN 2308366, 2013 | JF 71(6) 2967–3006, 2016-11-10 | 10.1111/jofi.12436 | 已确认同一研究，年份差 3 年 | Crossref（含摘要） | 高：**自相关本身内生于投资者再平衡频率**；自相关可在再平衡期限处变正 | 模型 + **日内与日频**实证；**不含交易成本** | 说明"期限结构"不是外生给定 |
| E16 | The cross-section of intraday and overnight returns | Bogousslavsky | SSRN 2869624, 2016 | JFE 141(1) 172–194, 2021-07 | 10.1016/j.jfineco.2020.07.020 | 已确认同一研究，年份差 5 年 | Crossref | 高：日内/隔夜横截面 | 美股；**日内频率**；成本处理未核实 | Crossref 被引 139 |
| E17 | A tug of war: Overnight versus intraday expected returns | Lou, Polk, Skouras | — | JFE 134(1) 192–213, 2019-10 | 10.1016/j.jfineco.2019.03.011 | 无预印本记录 | Crossref + S2 摘要 | **极高**：14 个策略中，收益要么**完全来自隔夜**（反转与多种动量），要么**完全来自日内**，两段通常符号相反；跨段反转持续数年 | 美股；**隔夜/日内分解**；**不含交易成本** | 哨兵论文；Crossref 被引 309 |
| E18 | Overnight returns, daytime reversals, and future stock returns | Akbas, Boehmer, Jiang, Koch | SSRN 3324880, 2019 | JFE 145(3) 850–875, 2022-09 | 10.1016/j.jfineco.2021.09.019 | 已确认同一研究，年份差 3 年 | Crossref + S2 摘要 | 高：月内"隔夜正—日内反转"频次预测未来收益 | 美股；**日频**；**不含交易成本** | Crossref 被引 92 |
| E19 | Factor Momentum and the Momentum Factor | Ehsani, Linnainmaa | SSRN 3014521 (2017)、NBER w25551 (2019-02) | JF 77(3) 1877–1919, 2022-04-22 | 10.1111/jofi.13131 | 已确认同一研究，年份差 3–5 年 | Crossref（含摘要） | 高：多数因子在**年度**尺度正自相关（前一年亏损后月收益 6bp，盈利后 51bp） | 美股因子面板；**月频**；**不含交易成本** | 哨兵论文；Crossref 被引 213 |
| E20 | Have capital market anomalies attenuated in the recent era of high liquidity and trading activity? | Chordia, Subrahmanyam, Tong | SSRN 2029057, 2012 | Journal of Accounting and Economics 58(1) 41–58, 2014-08 | 10.1016/j.jacceco.2014.06.001 | 已确认同一研究，年份差 2 年 | Crossref + S2 摘要 | 高：多数异象在小数化后衰减，组合平均收益约**减半** | 美股，月频；**不含交易成本**（结论关于毛收益） | Crossref 被引 478 |
| E21 | Persistent and transitory components of firm characteristics: Implications for asset pricing | Baba Yara, Boons, Tamoni | SSRN 3529140 | JFE 154, 103808, 2024-04 | 10.1016/j.jfineco.2024.103808 | 已确认同一研究（SSRN 题为 "Persistent and Transitory Components of Characteristics"，微改） | Crossref + WebSearch 核实结论 | **极高**：直接研究横截面可预测性的**期限维度**；56 个特征分解为持久/暂时成分，最长用到 5 年滞后；定价主要由持久成分驱动，基于持久成分构建的策略 Sharpe 显著高于标准特征策略 | 美股，**月频**；**不含交易成本**（主结果） | 近两年 JFE；对"信号持久性 → 收益"给出直接分解 |
| E22 | Price-Path Convexity and Short-Horizon Return Predictability | Gulen, Woeppel | — | JFQA 61(2) 580–611, 2025-06-20（另有勘误 10.1017/s0022109025102342） | 10.1017/s0022109025101725 | 无预印本记录 | Crossref（含摘要） | 中高：价格路径曲率与**短期限**未来收益负相关，且与累计收益无关 | 美股，个股与总量；**短期限**；**不含交易成本** | 近两年；给出与累计收益正交的短期限信号 |
| E23 | Echo disappears: momentum term structure and cyclic information in turnover | Wang, Di, Xie | arXiv:2304.03437, 2023-04-07 | 未发现期刊版 | arXiv:2304.03437 | 期刊版本**未确认** | arXiv 逐条核实 | 中：称近月动量中的反转成分抵消了近月动量，剔除后 echo 变弱 | 美股；成本处理未核实；econ.GN 预印本，同行评审状态未知 | 对 E3 的机制解释尝试 |

---

## §3 四张榜

### §3.1 最相关（按对"半衰期 → 持仓期"这一具体问题的直接程度）

1. **A1 Gârleanu & Pedersen (2013, JF)** —— 唯一给出"φ → 信号权重"闭式规则，且实证信号中就有 **2.4 日半衰期**的一条。
2. **A2 Qian, Sorensen & Hua (2007, JPM)** —— 唯一给出"信号自相关 → 换手 → 净收益最优点"完整链条的论文。
3. **D5 Blitz et al. (2023, JFDS)** —— 唯一直接把"训练/预测期限"当作自变量、以净成本后 alpha 为因变量做实验的论文。
4. **D2 Novy-Marx & Velikov (2019, FAJ)** —— 唯一直接把"降低再平衡频率"与其替代方案做对照实验的论文。
5. **D1 Novy-Marx & Velikov (2016, RFS)** —— 给出"降频 2/3 只省 1/3 成本"以及"快信号降频损失更大"的定量证据。
6. **A8 Jensen, Kelly, Malamud & Pedersen (2026, RFS)** —— aim 组合显式依赖所有未来期限的期望收益；ML 与动态交易的统一。
7. **C1 DeMiguel et al. (2020, RFS)** —— netting 的解析公式与 72.15% 换手下降实证。
8. **E21 Baba Yara, Boons & Tamoni (2024, JFE)** —— 特征的持久/暂时分解与期限维度定价。
9. **E11 Medhat & Schmeling (2022, RFS)** —— 同一价格信号在不同换手分组下期限与符号皆不同。
10. **E4 Gutierrez & Kelley (2008, JF)** —— 同一信号的符号随持有期翻转的经典证据。
11. **B1 Di Mascio, Lines & Naik (2015)** —— 机构交易层 alpha 衰减的直接测量（月尺度）。
12. **A9 Firoozye, Tan & Zohren (2023)** —— 以 signal decay 为标题的解析解。

### §3.2 最新（首次公开或正式出版落在近两年 2024-09-03 之后者优先；带 ★ 者落在近 180 天 2026-03-07 之后）

| 条目 | 日期 | 类型 |
|---|---|---|
| ★ **B7 Zulfiqar, When Alpha Dies** (SSRN 7376818) | Crossref 记录创建 **2026-09-01** | 工作论文 |
| ★ **A8 Jensen, Kelly, Malamud & Pedersen**, RFS | 正式出版 **2026-03-15** | 期刊 |
| ★ **B6 Meng & Chen**, arXiv:2605.23905 | 提交 **2026-03-23** | 预印本 |
| ★ **D10 Pall**, SSRN 6422358 | Crossref 记录创建 **2026-04-20** | 工作论文 |
| ★ **E10 Stosik & Zaremba**, Economics Letters 267 | 期号 **2026-07** | 期刊 |
| A16 Cont, Micheli & Neuman, Finance and Stochastics | 2025-03-19 | 期刊 |
| E22 Gulen & Woeppel, JFQA | 2025-06-20 | 期刊 |
| A12 Ma & Smith, arXiv:2502.04284 | 2025-02-06 | 预印本 |
| E14 Chen, Stivers & Sun, JEmpFin | 2024-12 | 期刊 |
| D9 Stankov et al., JAM | 2024-05-25 | 期刊 |
| E21 Baba Yara, Boons & Tamoni, JFE | 2024-04 | 期刊 |
| C7 Arnott, Li & Linnainmaa, FAJ | 2024-03-14 | 期刊 |
| D8 Ang & Madhavan, SSRN 4782032 | 2024 | 工作论文 |
| D7 Bašić et al., JPM | 2024-02-22 | 期刊 |
| E9 Blitz, van der Grient & Honarvar, JPM | 2024-02-06 | 期刊 |
| D6 Azevedo, Hoegner & Velikov, SSRN 4702406 | 2024-02-07 | 工作论文 |

### §3.3 高影响力（按本次会话中实际读取到的被引数；Crossref `is-referenced-by-count` 或 Semantic Scholar `citationCount`）

| 条目 | 被引数（来源） |
|---|---|
| D1 Novy-Marx & Velikov (2016, RFS) | 626（Crossref） |
| E6 Nagel (2012, RFS) | 565（Crossref） |
| A1 Gârleanu & Pedersen (2013, JF) | 558（S2） |
| E20 Chordia, Subrahmanyam & Tong (2014, JAE) | 478（Crossref） |
| E3 Novy-Marx (2012, JFE) | 400（Crossref） / 451（S2） |
| E17 Lou, Polk & Skouras (2019, JFE) | 309（Crossref） |
| B4 Chinco, Clark-Joseph & Ye (2019, JF) | 271（Crossref） |
| E19 Ehsani & Linnainmaa (2022, JF) | 213（Crossref） |
| B3 van Kervel & Menkveld (2019, JF) | 205（Crossref） |
| E5 Da, Liu & Schaumburg (2014, MS) | 200（Crossref） |
| E16 Bogousslavsky (2021, JFE) | 139（Crossref） |
| E15 Bogousslavsky (2016, JF) | 138（Crossref） |
| D4 Chen & Velikov (2022, JFQA) | 103（Crossref） |
| E18 Akbas et al. (2022, JFE) | 92（Crossref） |
| E7 Cheng et al. (2017, JFQA) | 83（Crossref） |
| D3 Detzel, Novy-Marx & Velikov (2023, JF) | 77（Crossref） |
| E11 Medhat & Schmeling (2022, RFS) | 74（Crossref） |

> C1（DeMiguel et al. 2020, RFS）与 A2（Qian et al. 2007, JPM）的 Crossref 被引数分别未单独读取与仅 17，
> 后者因 JPM 在 Crossref 的参考文献沉积严重不足而**不可用于影响力判断**；此处如实标注。

### §3.4 最新且可能具有高影响力（**不以引用数为依据**）

1. **A8 Jensen, Kelly, Malamud & Pedersen, RFS 2026-03-15** ——
   非引用数理由：(a) 首次把 Gârleanu–Pedersen 的多期限 aim 组合与 ML 端到端统一，直接产出"跨多个未来期限混合可预测性"的可执行方法；
   (b) 提供**公开代码库**（GitHub `theisij/ml-and-the-implementable-efficient-frontier`）与公开数据（JKP Global Factors），可复制性高；
   (c) **直接检验净成本后结果**（implementable efficient frontier 就是净成本后的前沿）；
   (d) 作者组过往命中率高（Kelly、Pedersen 在 ML 资产定价与交易成本两条线上均有顶刊记录）；(e) 已被 RFS 接受出版。
2. **D5 Blitz, Hanauer, Hoogteijling & Howard, JFDS 2023** ——
   非引用数理由：(a) 实验设计极简且可复制（同一 ML 管线，只改训练前瞻期限）；
   (b) **直接检验净成本后结果**并报告 2004 年后 1 个月模型净 alpha 近零；
   (c) 结论被作者机构公开复述，方法细节可核；(d) 直接回答本主题的核心问题，几乎不存在竞争性论文。
3. **E21 Baba Yara, Boons & Tamoni, JFE 2024** ——
   非引用数理由：(a) 提出可迁移的持久/暂时成分分解方法；(b) 覆盖 56 个特征、滞后至 5 年，**样本覆盖广**；
   (c) 已在 JFE，方法可被下游直接复用于"信号持久性→期限"问题；(d) 已被后续 JFE/JFQA 论文引用（本次引用追踪中即被 2 篇引用）。
4. **D8 Ang & Madhavan, SSRN 4782032 (2024)** ——
   非引用数理由：**数据集新颖性极高**（2,022 笔真实组合过渡、>3.1 万亿美元、2016–2023，跨股票与固收），
   且直接以计数模型给出"最优交易期限"的可估计函数；作者（Ang、Madhavan）在成本与执行文献有长期记录。
   注意：其"期限"是执行期限，不是持仓期。
5. **D6 Azevedo, Hoegner & Velikov, SSRN 4702406 (2024)** ——
   非引用数理由：与 D5 结论方向不同（D5 称 1 个月 ML 净 alpha 近零，D6 称 LSTM 策略净 SR 0.84 仍可观），
   构成**可检验的直接冲突**；Velikov 维护公开的异象与成本基础设施（Assaying Anomalies），复制门槛低。
6. **E10 Stosik & Zaremba, Economics Letters 2026** ——
   非引用数理由：标题即对 E9（Blitz et al. 2024 "反转已消失"）的直接限定；
   两篇发表间隔不足两年、结论对立，属于**活跃争议点**；Economics Letters 传播快。
7. **B7 Zulfiqar, SSRN 7376818 (2026-09-01)** ——
   非引用数理由：把"衰减"操作化为**前瞻净成本后**的生存问题，并在**日频**数据上做样本外验证，
   方法学上与本主题的日频需求最接近；但为单作者工作论文、期刊状态未知，可靠性未经同行评审。

---

## §4 主题综述

### 4.1 术语先行：文献中的 "alpha decay" 至少有三种互不相同的含义

在进入任何映射规则之前，必须先分离三个被同一个词覆盖的对象，否则数值不可比。

**（i）单笔头寸内的信息衰减（日/分钟尺度）。** 这是 Gârleanu–Pedersen (A1) 的 φ、
Qian–Sorensen–Hua (A2) 的 forecast autocorrelation、以及 Chinco–Clark-Joseph–Ye (B4) 的 "short-lived predictors" 所指的对象。
A1 在商品期货实证中直接给出三条信号的衰减率：
Δf 的均值回复系数分别为 0.2519、0.0034、0.0010，对应 **2.4 日、206 日、700 日半衰期**。
B4 则把这一端推到 1 分钟前瞻，其 LASSO 选出的预测因子被描述为"意外、短命、稀疏"。

**（ii）机构成交后的 alpha 实现曲线（月尺度）。** 这是 Di Mascio–Lines–Naik (B1) 的对象。
其在 Inalytics 的 700 多个机构组合、2001–2013、115 万笔交易上做事件研究，报告：
首次买入后**第一个月的 FFC alpha 为 37bp，此后逐月递减，到第 12 个月基本归零**；
累计 alpha 呈上升的凹形并在长期趋平，**其后不转负**（价格变化是永久性的）。
同期，逐月的后续净买入从 AUM 的 0.087% 递减至零，之后原始头寸开始被反向平掉。
作者把这一形态归因于策略性交易：竞争越激烈交易越激进、买后 alpha 越低，但**只在头 3–6 个月**成立
（对应 Foster–Viswanathan (1996) 的 rat race 与随后的 waiting game）。
必须强调：B1 测的是"机构从开始建仓到 alpha 走完需要多久"，其单位是月，不是任何日频截面信号的半衰期。

**（iii）策略/异象的多年失效（年尺度）。** 这是 Pénasse (B5, Management Science 2022) 与
Meng–Chen (B6) 的对象。B6 更是把"alpha 半衰期"定义为策略寿命，并称当前 AI 采用率下约 18 个月、
AI 普及前 5–7 年。用户问题中的"半衰期 2 日"属于含义 (i)，与 (ii)(iii) 不可混用。

**缺口**：本次检索**未找到**任何论文报告"不同类型信号（新闻 / 订单流 / 价格形态 / 基本面）在统一口径下的半衰期分布"。
最接近的替代是 A2 报告的季频 forecast autocorrelation 区间（价值类因子可高至 0.95，其中现金流类略低；
价格动量类一般在 0.6–0.7，且随收益计算窗口拉长而上升，最长至 12 个月）与 A1 的三条商品信号 φ 估计。

### 4.2 解析框架给出的映射规则：四套互相冲突的规则

**规则 R1（Gârleanu–Pedersen）：半衰期不决定交易速度，只决定信号权重。**
A1 的 Proposition 2 给出最优组合 `x_t = (1 − a/λ)·x_{t−1} + (a/λ)·aim_t`，
其中交易速率 a/λ 由 a 的闭式解 `a = [−(γ(1−ρ)+λρ) + √((γ(1−ρ)+λρ)² + 4γλ(1−ρ)²)] / (2(1−ρ))` 决定，
**仅取决于交易成本 λ、风险厌恶 γ 与贴现率 ρ，与任何信号的 φ 无关**，
且论文明确指出交易速率与当前及历史持仓无关。
衰减率的作用出现在 Proposition 4：aim 组合等于把每个因子按自身衰减率缩放后的 Markowitz 组合，
`aim_t = (γΣ)⁻¹B · (f¹/(1+φ₁a/γ), …, f^K/(1+φ_K a/γ))ᵀ`，
且持久因子 i 相对快因子 j 的权重比 `(1+φ_j a/γ)/(1+φ_i a/γ)` **随交易成本 λ 上升而上升**。
其实证结论与此一致：最优动态策略之所以优于最好的静态策略，
"关键在于动态策略因为**五日信号衰减快**而给它更少的权重"；
静态策略只能控制整体交易速度，结果要么因为目标飘忽而付出高成本，要么慢到抓不住收益。
在 R1 之下，"半衰期 2 日 ⇒ 持仓 5 日"这一问法本身是范畴错误：正确的响应变量是权重，不是持仓期。

**规则 R2（Qian–Sorensen–Hua）：换手由预测值自相关决定，成本上升时应提高自相关而非拉长日历持仓期。**
A2 推出无约束均值—方差组合的单期单边换手
`T = √(N/π) · (σ_model/σ₀) · √(1 − ρ_f)`，其中 ρ_f 是相邻两期预测值的横截面自相关。
（数值例：N=500、σ_model=5%、σ₀=30%、ρ_f=0.9 时单次换手 6.6%。）
关于"拉长交易期限能否提高毛收益"，A2 给出明确的否定性结果：
horizon IC 近似为 lagged IC 均值乘以 √(期限长度)，
在其季频例中（季 IC 均值 0.1、标准差 0.2，季 IR 0.5，年化 IR 1.0），
四季期限的 horizon IC 均值 0.2、标准差 0.4，年 IR 仍为 1.0 ——
**毛收益层面对交易期限无差异**，拉长期限的唯一理由是成本。
A2 主张的降换手手段不是延长持仓期，而是**在 alpha 模型中加入滞后信号（移动平均）以提高 ρ_f**：
例中 E2P 的 ρ_f(1)=0.94、ρ_f(2)=0.84，等权组合两期后 ρ_f 升至约 0.96；
PM 的 ρ_f(1)=0.68、ρ_f(2)=0.40，等权后升至约 0.82。
在 N=3000、σ_model=4%、σ₀=30% 的多因子最优化中：
毛 IR 在 ρ_t=0.89 处最大（2.39，年换手约 547%）；
当 100% 换手成本为 0.5% 时净收益最优点右移至 ρ_f=0.93（IR 2.33，换手 436%）；
成本 1.0% 时移至 ρ_f=0.95（IR 2.21，净收益比 ρ_f=0.89 的模型高 1 个百分点以上）；
成本 1.5% 时移至 ρ_f=0.96。作者并指出**IR 越低，最优 ρ_f 越高**。

**规则 R3（Novy-Marx–Velikov）：在成本缓释手段中，"降低再平衡频率"是最差的一种。**
D1 在 1973-07 至 2012-12 的美股上比较三种技术，报告
"交易频率降低三分之二一般只带来约三分之一的换手与成本下降"（因为每个再平衡点上翻转的比例更高）；
并明确指出对于"基于持久性低得多的信号排序的高频策略"，季度再平衡"太不频繁，无法维持对底层异象的较大平均暴露"，
因此这些策略只能按半季度频率交错再平衡；结果是中等换手策略的净价差只有边际下降、
而高频策略（含 short-run reversals、industry relative reversals、seasonality）
"毛价差恶化更大，实现绩效改善更有限"。其高换手面板给出的数字包括：
短期反转毛收益 0.42%/月、单边换手 60.86%、成本 1.07% ⇒ **净 −0.65%**；
行业相对反转毛 0.82%、成本 1.15% ⇒ 净 −0.33%。
D2（1975-01 至 2016-12，大/小/微盘，7 个策略 × 3 种技术）进一步给出对照结论：
把月度改季度再平衡，成本下降幅度与 banding 相当，但因信号变得不及时导致毛绩效下降，
**平均而言"没有净收益"**；而 10%/30% 的 buy/hold spread（banding）在削减成本的同时保持了毛收益。
C7（Arnott–Li–Linnainmaa 2024, FAJ）给出同方向的第三种手段：按信号强度对交易排优先级，
可在削减换手与成本的同时保留长仅组合的大部分因子溢价。

**规则 R4（Blitz et al. / Jensen et al.）：改的是"预测期限"或"多期限混合"，不是单一持仓期。**
D5 在同一 ML 管线上把训练标签从 1 个月前瞻收益改为 3/6/12 个月前瞻收益：
1 个月模型全样本毛 alpha 亮眼，但**2004 年后净成本后接近零**；
延长训练期限并配合高效的组合构建规则后可获得显著为正的净收益；
延长期限的模型"选择更慢的信号、更多地载荷于传统资产定价因子"，但仍解锁独特 alpha。
A8 则不设单一持仓期：其 aim 组合
`A(s_t) = (I−m)⁻¹ Σ_{τ≥0} (m·ḡ)^τ · c · (1/γ) Σ⁻¹ E_t[r_{t+1+τ}]`
显式依赖**所有未来期限**的期望收益；
"Multiperiod-ML"需要对每个 τ 都训练一个预测模型，"Portfolio-ML"则直接学权重。
A8 的诊断是：对成本无知的 ML 会"过度依赖转瞬即逝的小盘特征（例如小盘股的 1 个月反转）"；
其 Portfolio-ML 相对高度参数化的两阶段静态方法在净成本后 **Sharpe 高约 20%、效用高约 60%**；
其"经济特征重要性" `ι_n = 效率损失 − 成本节省`，明确指出**持久信号因降低换手而更重要**。
A9（Firoozye–Tan–Zohren）在纯解析层面给出同型结论：最优策略是向"前瞻组合"平滑调整，
长期绩效可用包含信号持久性在内的结构参数表达。
A14（Zhang–Wang–Cao）在 fundamental law 框架内给出方向一致但更弱的陈述：
换手调整后的 IR 恒低于忽略换手的 IR，"最优化而非最大化"交易频率可提高风险调整后收益。

**R5（Grinold，内容未能核实）。** A3《Signal Weighting》被官方检索结果描述为
"在存在交易成本时以组合方法选择信号权重"，并按信息换手率把信号分为 SLOW / INTERMEDIATE / FAST；
A5《Linear Trading Rules》被描述为以 trade rate 参数控制交易速率与各 alpha 来源的权重，
并使用 target portfolio、backlog、no-trade zone 概念。
**这两篇的正文在本次会话中因付费墙无法取得，其公式与数值无法独立验证**（见 §1.x 第 5 条）。

### 4.3 快慢合成与 netting 的实证证据

C1（DeMiguel 等，RFS 2020）是唯一给出 netting 解析式与大样本实证的论文。
其 Proposition 3 证明：把 K 个特征组合起来交易，第 i 只股票的换手比为
`√(e'Ωe) / Σ_k √Ω_kk < 1`；在方差相等、相关系数同为 ρ 的对称情形下简化为 `√((1+ρ(K−1))/K)`；
当 ρ=0 时为 `1/√K`。实证部分（美股 1980-01 至 2014-12，剔除市值后 20%，
平均每月 3,071 只，51 个特征，月频）报告：
**单独交易 51 个特征的月均换手为 24.09%，等权组合后仅 6.71%，即下降 72.15%**，
与 ρ=0 情形的理论预测量级接近。作者由此得出：交易成本把"联合显著"的特征数从 6 个抬升到 15 个。

C2（Frazzini–Israel–Moskowitz 2012）用 AQR 的真实执行数据给出同方向的具体数字
（美股，1998–2011，月频重构）：
HML 月换手 68%、UMD 127%、STR 305%、**ValMom 组合仅 79%**；
break-even 基金规模 HML 约 830 亿、UMD 约 522 亿、**ValMom 约 987 亿**（高于二者单独），
作者明确归因于"价值与动量的交易互相抵消"。
同表显示 **STR 的实测成本高达 6.75%/年、break-even 规模仅 95 亿美元**，是四个策略中最受成本约束的一个。

在"快信号叠加在慢信号上"这一具体设计上，本次检索定位到 C4（Jha 2016, Journal of Trading）、
C5（Israelov–Katz 2011, FAJ）与 A7（Sneddon 2008, Journal of Investing）三条直接命题的条目，
**但三者正文均因付费墙未能核实**。
A16（Cont–Micheli–Neuman 2025, Finance and Stochastics）虽题为 "fast and slow"，
但研究的是低频投资者与高频交易者之间的 Stackelberg 均衡（信号强时 HFT 掠夺、弱时合作），
**不是同一投资者内部的快慢信号合成**。

**缺口**：本次检索**未找到**在美股日频上直接测量"半衰期 ≤1 周的快信号"与"慢信号"之间 netting 幅度的实证论文。
C1 的 72.15% 是月频、51 个以基本面为主的特征上的结果，其外推到日频快慢混合没有文献依据。

### 4.4 持仓期敏感性的实证与短期价格信号的期限结构

在**月频**上，形成期/持有期矩阵的证据是丰富的：E1（Jegadeesh–Titman 1993）确立方法；
D1 明确指出 J–T 的 staggered partial rebalancing 最初是为了识别毛收益最高的期限、
后来才被当作降成本工具。E3（Novy-Marx 2012）报告动量主要由 t−12 至 t−7 的表现驱动而非 t−6 至 t−2，
且近期形成期策略"盈利但更弱，**在最大、最流动的股票中尤其如此**"；
E12（Gong–Liu–Liu 2015）与 E23（Wang 等 2023）分别提出相反归因与机制解释，该点尚未定论。
E4（Gutierrez–Kelley 2008）给出对本主题最直接的一条结构性事实：
一周形成期信号在短持有期表现为反转，但随后出现**长期持续**，
其动量利润足以抵消最初的反转，以致在形成后整年上净表现为动量 ——
**同一信号的符号随持有期翻转**。

在**短期反转**这一支上，文献本身在近年出现明确冲突：
E9（Blitz–van der Grient–Honarvar, JPM 2024）称经典短期反转已在多数地区消失，
需要通过"避免与短期动量对抗"来复活；
E10（Stosik–Zaremba, Economics Letters 2026-07）则以标题直接主张"若度量得当，短期反转在全球仍然存在"。
E8（Blitz 等 2013, JFM）在 1926–2008 的 NYSE 中位数以上市值样本上报告
残差反转"在最大的 500 只甚至 100 只股票中亦可观测"。
E11（Medhat–Schmeling, RFS 2022）给出条件化结论：
上月收益 × 换手率双重排序下，低换手股表现为短期反转、**高换手股表现为短期动量**；
短期动量"与常规价格动量同样有利可图、同样持久"，**在扣除交易成本后仍然存活**，
且"在**最大、最流动、被覆盖最广**的股票中最强"。
E13、E14 在换手与 52 周高点维度上给出同方向的条件化。
E20（Chordia–Subrahmanyam–Tong 2014）报告多数异象在小数化后衰减、异象组合平均收益约减半。

在**隔夜/日内**分解上，E17（Lou–Polk–Skouras, JFE 2019）报告：
个股层面隔夜与日内各自存在强持续性，并伴随一个抵消性的跨段反转，且持续数年；
在 14 个交易策略上，利润"要么完全在隔夜实现（反转与多种动量策略），要么完全在日内实现，
且两段通常符号相反"。E18（Akbas 等, JFE 2022）用"隔夜涨—日内跌"的月内频次预测未来收益。
E16（Bogousslavsky, JFE 2021）刻画日内/隔夜横截面。
E15（Bogousslavsky, JF 2016）给出一条方法学上重要的提醒：
**收益自相关的期限结构本身内生于投资者的再平衡频率**，自相关可以在再平衡期限处变为正值 ——
即"信号期限"不是外生给定的物理常数。
E21（Baba Yara–Boons–Tamoni, JFE 2024）把 56 个特征分解为持久与暂时成分（滞后最长至 5 年），
报告定价主要由持久成分驱动，且基于持久成分构建的策略 Sharpe 显著高于标准特征策略；
其检验对象正是"横截面可预测性的期限维度"。

**缺口**：本次检索**未找到**任何现代学术论文发表美股大盘股在**日频**上的完整 K×J
（形成期 1–20 日 × 持有期 1–20 日）净成本后收益矩阵。
E22（Gulen–Woeppel, JFQA 2025）等"短期限"研究仍以月度组合为分析单元。
D10（Pall 2026）是本次检索中唯一在日频上明确报告"高方向准确率但扣 5bp 成本后严重亏损"的条目，
但其横截面仅 7 只大盘科技股，外部效度极低。

### 4.5 净成本下的期限选择：break-even 结果

D4（Chen–Velikov, JFQA 2022）在 204 个异象上，扣除有效价差、发表后衰减与 2000 年代后的交易技术三重效应后，
报告平均异象的期望收益只有 **4bp/月**，最强的异象扣除数据挖掘后至多 10bp，
若干种异象组合方法约 20bp；作者并明示这些数字**尚未计入价格冲击**。
D3（Detzel–Novy-Marx–Velikov, JF 2023）指出忽略成本会使模型比较系统性偏向高成本（即高换手）因子。
C3（Frazzini–Israel–Moskowitz 2018）用 1.7 万亿美元、21 个市场、19 年的真实执行数据主张
实际成本比既往研究小一个数量级。
D6（Azevedo–Hoegner–Velikov 2024）报告 ML 异象策略在成本 + 发表后衰减 + 小数化三项下累计损失 57%，
但 LSTM 策略仍有毛/净 Sharpe 0.94/0.84。
D8（Ang–Madhavan 2024）在 2,022 笔真实组合过渡（>3.1 万亿美元，2016–2023）上，
用计数模型估计**最优交易期限**，报告其随交易风险、金额与复杂度上升而延长 ——
但该"期限"是把一笔单子做完需要多久（execution horizon），与持仓期是两个变量。

**未找到的东西**：本次检索**未定位到任何给出闭式 "break-even holding period" 公式的同行评审文献**。
文献里可核实的 break-even 量都是**成本口径**（break-even trading cost）或**规模口径**（break-even fund size），
例如 C2 的 STR 95 亿 / UMD 522 亿 / HML 830 亿 / SMB 1032 亿美元。
"cost-adjusted optimal horizon" 这一说法在本次检索中只以 A2 的"净收益最优 ρ_f"、
D5 的"最优训练期限"、D8 的"最优执行期限"三种**互不相同的形式**出现。

### 4.6 对"待证伪假设"的直接回答

**假设**：「半衰期 2 日的信号，文献是否支持 5 日持仓期；文献给出的映射规则是什么（若存在多套互相冲突的规则，全部列出）。」

**（a）关于"5 日"这个具体数字**：本次检索**未找到任何文献给出形如"最优持仓期 = k × 半衰期"的映射规则**，
也未找到任何文献对"2 日半衰期 ⇒ 5 日持仓期"给出支持或反驳。
文献中不存在可直接代入的常数 k。**证据状态：无证据（既非支持亦非反驳）。**

**（b）最接近的直接可比对象**：A1 的商品期货实证中就有一条估计半衰期为 **2.4 日**的"五日信号"。
该论文对这条信号的处理不是规定持仓期，而是**在 aim 组合中按 1/(1+φa/γ) 把它的权重压低**，
并把最优动态策略优于最优静态策略的原因明确归结为"给五日信号更少的权重"。
在 A1 的框架里，交易速率 a/λ 由成本与风险厌恶决定、与 φ 无关，因此"为快信号单独设定持仓期"没有对应的决策变量。

**（c）文献给出的、互相冲突的映射规则（全部列出）**：

| 规则 | 出处 | 决策变量 | 半衰期进入的位置 | 与"延长持仓期"的关系 |
|---|---|---|---|---|
| R1 | A1 Gârleanu–Pedersen (2013) | 交易速率 a/λ + 各信号权重 | **只进入权重** `1/(1+φ_k a/γ)`；不进入交易速率 | 冲突：交易速率与 φ 无关，"为快信号拉长持仓期"在模型中无对应操作 |
| R2 | A2 Qian–Sorensen–Hua (2007) | 预测值自相关 ρ_f | 通过 ρ_f 进入换手 `T ∝ √(1−ρ_f)` | 部分冲突：降换手的手段是**加入滞后信号做平滑**，不是拉长日历持仓期；且毛 IR 对交易期限无差异 |
| R3 | D1/D2 Novy-Marx–Velikov (2016, 2019) | 再平衡频率 vs banding vs 低成本域 | 通过信号持久性影响降频的毛收益损失 | 直接负面：降频 2/3 只省 1/3 成本；季度再平衡"平均没有净收益"；快信号降频的毛损失更大 |
| R4 | D5 Blitz et al. (2023) | **训练/预测期限** | 通过换标签改变模型选中的信号速度 | 替代路径：不是延长同一信号的持仓期，而是换一个更慢的信号 |
| R5 | A8 Jensen et al. (2026) | aim 组合中跨所有未来期限的混合权重 | 通过 (m·ḡ)^τ 的期限加权 | 替代路径：不存在单一持仓期；多期限混合 |
| R6 | C7 Arnott et al. (2024) / D1 buy-hold spread | 交易优先级 / 带宽阈值 | 不直接进入 | 替代路径：在不改变持仓期的前提下削减换手 |
| R7 | A3/A5 Grinold（**正文未核实**） | trade rate、按信息换手率分档的信号权重 | 二手描述称信号按 SLOW/INTERMEDIATE/FAST 分档配权 | 无法判定 |

**（d）文献支持的方向性陈述（可核实）**：
D1 与 D2 一致地报告，**对持久性越低的信号，通过降低再平衡频率来省成本，毛收益的损失越大**；
D2 更报告在其 21 个案例的平均意义上，把月度改为季度再平衡"没有净收益"。
A2 报告在其假设下毛 IR 对交易期限无差异，因此拉长期限的收益完全来自成本一侧。
这三条合起来构成对"延长持仓期以省成本"这一操作的**方向性负面证据**，
但**没有任何一条给出日频、2 日半衰期条件下的交换率量化**，因此不能据以判定 5 日相对于其他天数的优劣。

**（e）冲突点**：R1 与 R3 在"该调什么"上并不冲突（都不主张调持仓期），
但 R3 的可操作替代（banding）在 R1 的框架里没有对应物（R1 是无带宽的线性规则，A6 的非线性规则才引入无交易区）；
A5 引入的 no-trade zone 与 D1 的 sS 规则在概念上一致，但 A5 正文未能核实。
D5 与 D6 在"ML 策略净成本后是否还剩下东西"上给出方向不同的结论（前者称 1 个月模型净 alpha 近零、后者称 LSTM 净 SR 0.84）。
E9 与 E10 在"短期反转是否已消失"上直接对立。

---

## §5 覆盖缺口、方法不足与限制

### 5.1 检索通道的缺陷（直接影响召回率，不得视为检索已完整）

1. **OpenAlex 全程不可用**（额度耗尽，HTTP 429）。协议要求的"≥2 个通用学术索引"实际只满足约 1.5 个。
2. **Semantic Scholar 的关键词搜索端点全程 429**，指数退避无效。前向引用追踪只能靠 `/citations` 端点，
   且该端点对 JPM/JFDS 等 pm-research 系列 DOI 覆盖不全（10.3905/jfds.2023.1.135 返回 `not found`）。
3. **arXiv API 在 curl 通道全程 503**，WebFetch 代理通道在 3 次查询后转 429。arXiv 覆盖是**部分的**。
4. **Crossref 的日期倒序排序功能性失效**（垃圾未来日期元数据占据全部前排），
   导致协议要求的"(c) 近 180 天 + 日期倒序"这一支被替换为日期过滤 + 相关性排序，
   **近 180 天的召回可能不完整**。
5. **Crossref 的 `total-results` 不是精确布尔命中数**，本文件中的命中数只能反映噪声水平。
6. **SSRN 页面 403**，所有 SSRN 条目只能通过 Crossref 注册的 DOI 与第三方镜像核实；
   SSRN 上未注册 DOI 的工作论文在本次检索中系统性不可见。
7. **pm-research（JPM/JOI/JOT/JFDS）与 FAJ 正文付费墙**，影响 9 条候选文献：
   A3、A4、A5、A6、A7、C4、C5、D5、D7、E9（以及 C7 部分）。
   这些条目**只在元数据/官方摘要层面核实，正文公式与数值未经独立验证**。
   由于本主题的"信号加权/快慢合成"文献高度集中在这几本实务期刊上，
   **这是本次综述最严重的单一覆盖缺口**。

### 5.2 文献本身的缺口（不是检索缺陷，而是研究缺失）

1. **没有统一口径的信号半衰期分布**。用户要求的"不同类型信号（新闻、订单流、价格形态、基本面）的半衰期分布"，
   在本次检索中**没有任何论文提供**。可用的替代品是散落的点估计：
   A1 的三条商品信号（2.4 / 206 / 700 日）、A2 的季频 forecast autocorrelation 区间（价值 0.90–0.95、动量 0.6–0.7）、
   B4 的 1 分钟前瞻。这些口径彼此不可比（不同资产、不同频率、不同定义）。
2. **没有日频 K×J 矩阵**。现代学术文献中未找到美股大盘股日频形成期 × 持有期的净成本后收益矩阵。
   短期反转/短期动量文献几乎全部以**月度组合**为分析单元（E5、E7、E11、E13、E14），
   E15、E16、E17、E18 虽在日内/日频取样，但不做持仓期扫描。
3. **没有 break-even holding period 的闭式结果**。文献只有 break-even *cost* 与 break-even *fund size*。
4. **没有快慢信号在日频上的 netting 实证**。C1 的 72.15% 是月频、51 个以基本面为主的特征。
5. **纯多头 / 等权 / 固定只数的组合结构缺席**。除 C7（长仅组合）与 D2（部分）外，
   本主题的净成本文献几乎全部基于**多空**组合（D1、D3、D4、C2、C1、A8 的部分设定），
   其换手与成本口径不能直接搬到纯多头等权组合。
6. **t+1 开盘成交这一执行假设在本主题文献中完全缺席**。E17 的隔夜/日内分解说明成交时点会改变
   信号能捕获的收益段，但没有任何论文在"按开盘价成交"的假设下重做期限敏感性。
7. **成本模型的异质性**。D1/D2/D4 用有效价差估计，C2/C3 用真实执行数据并称前者高估一个数量级，
   A1 用 Engle–Ferstenberg–Russell 的价格冲击校准，A2 用线性比例成本。
   不同论文的"最优期限"结论之间**不可直接比较**。
8. **Grinold 系列（A3–A6）与 Sneddon（A7）的引用量极低**（Crossref 分别为 13、18、0、—、9），
   这既可能反映影响力有限，也可能反映 JPM/JOI 在 Crossref 的参考文献沉积不足；
   本次会话**无法区分这两种解释**。
9. **术语污染**。"alpha decay" 在 Crossref/OpenAlex 上以核物理为主导，
   在金融内部又有三种含义（§4.1），基于该关键词的召回率天然偏低，
   本文件的多数关键条目实际上是靠**引用追踪**而非关键词检索发现的。

---

## §6 遗漏审计

### 6.1 候选哨兵论文的召回情况（逐条）

| # | 哨兵线索（未经核实的记忆） | 结果 | 核实细节 |
|---|---|---|---|
| 1 | Di Mascio, Lines & **Pedersen** (2017/2023) "Alpha Decay" (SSRN / 期刊版) | **命中，但线索有误** | 存在的是 Di Mascio, Lines & **Naik**（Narayan Y. Naik, LBS），SSRN 2580551，Crossref 记录创建 2015-03-25，取得的稿本注明 This Version 2015-03-18；SSRN 页面另标 2017 年修订。**未找到任何期刊版**，"2023 期刊版"这一线索**无法确认**。线索中的 "Pedersen" 疑似与同数据家族的 Akepanidtaworn–Di Mascio–Imas–Schmidt (JF 2023) 或 Gârleanu–Pedersen 混淆 |
| 2 | Grinold (2010) "Signal Weighting", JPM | **命中** | JPM 36(4) 24–34，DOI 10.3905/jpm.2010.36.4.024（另有在线首发 DOI 10.3905/jpm.2010.2010.1.005）。正文付费墙，**内容未核实** |
| 3 | Grinold "Dynamic Portfolio Analysis" | **命中** | JPM 34(1) 12–26 (2007)，DOI 10.3905/jpm.2007.698029。正文未核实 |
| 4 | Grinold "Linear Trading Rules for Portfolio Management" | **命中** | JPM 44(6) 109–119 (2018)，DOI 10.3905/jpm.2018.44.6.109。正文未核实。另发现配套的 Nonlinear Trading Rules, JPM 45(1) 62– (2018)，DOI 10.3905/jpm.2018.45.1.062 |
| 5 | Gârleanu & Pedersen (2013) "Dynamic Trading with Predictable Returns and Transaction Costs", JF | **命中，全文已核实** | JF 68(6) 2309–2340，DOI 10.1111/jofi.12080；NBER w15205 (2009-08, 修订 2011-12-05 / 2013-01-30)；SSRN 1364170 / 1448169 / 1658736 |
| 6 | Qian, Sorensen & Hua (2007) "Information Horizon, Portfolio Turnover, and Optimal Alpha Models", JPM | **命中，全文已核实** | JPM 34(1) 27–40，DOI 10.3905/jpm.2007.698030 |
| 7 | Chinco, Clark-Joseph & Ye (2019) "Sparse Signals in the Cross-Section of Returns", JF | **命中** | JF 74(1) 449–492，在线 2018-11-14，DOI 10.1111/jofi.12733；NBER w23933 (2017-10)；SSRN 2606396 (2015) |
| 8 | van Kervel & Menkveld (2019) "High-Frequency Trading around Large Institutional Orders", JF | **命中** | JF 74(3) 1091–1137，DOI 10.1111/jofi.12759；SSRN 2619686 (2015) |
| 9 | Novy-Marx (2012) "Is momentum really momentum?", JFE | **命中** | JFE 103(3) 429–453，DOI 10.1016/j.jfineco.2011.05.003 |
| 10 | Gutierrez & Kelley (2008) "The Long-Lasting Momentum in Weekly Returns", JF | **命中** | JF 63(1) 415–447，DOI 10.1111/j.1540-6261.2008.01320.x；SSRN 890305 (2006) 原题 "Evidence to the Contrary: Weekly Returns Have Momentum" |
| 11 | Da, Liu & Schaumburg (2014) "A Closer Look at the Short-Term Return Reversal", Management Science | **命中** | MS 60(3) 658–674，DOI 10.1287/mnsc.2013.1766 |
| 12 | Lou, Polk & Skouras (2019) "A Tug of War: Overnight Versus Intraday Expected Returns", JFE | **命中** | JFE 134(1) 192–213，DOI 10.1016/j.jfineco.2019.03.011 |
| 13 | Ehsani & Linnainmaa (2022) "Factor Momentum and the Momentum Factor", JF | **命中** | JF 77(3) 1877–1919，DOI 10.1111/jofi.13131；NBER w25551 (2019-02)；SSRN 3014521 (2017) |
| 14 | Frazzini, Israel & Moskowitz (2018) "Trading Costs" | **命中** | SSRN 3229719 (2018-08-23)。**未找到期刊版**。同时命中其 2012 年配套论文 "Trading Costs of Asset Pricing Anomalies"，SSRN 2294498，同样**未找到期刊版** |

**汇总：14 条哨兵线索，14 条确认存在（命中率 14/14），0 条"无法确认存在"。**
**1 条（#1）的作者信息与记忆线索不符，已按实际作者更正并在正文中标注。**
**3 条（#1、#14 的两篇）未找到期刊版，其"期刊版本"状态记为"未确认"。**

### 6.2 近 180 天（2026-03-07 起）的核查

| 检查对象 | 方法 | 结果 |
|---|---|---|
| 核心期刊（JF/JFE/RFS/JFQA/MS/JFM/JAE/FAJ/JPM/JFDS） | Crossref `from-pub-date:2026-03-07` + 相关性排序，4 组查询式 | 只捞到 **A8（JKMP, RFS, 2026-03-15）** 与 **E10（Stosik–Zaremba, Economics Letters, 2026-07 期）**。其余返回项为区域性低层级会议与非相关主题。**日期倒序排序失效，此项核查不完整** |
| 预印本（arXiv q-fin） | `arxiv.org/list/q-fin.PM/2026-08` 全量 28 条 + WebFetch API 两次查询 | 2026-08 月度列表中 **0 条**与 alpha 衰减/期限映射相关；API 查询取得 **B6（arXiv:2605.23905, 2026-03-23）** |
| SSRN | WebSearch `site:papers.ssrn.com` + Crossref SSRN DOI 直查 | 取得 **B7（SSRN 7376818，记录创建 2026-09-01）**、**D10（SSRN 6422358，记录创建 2026-04-20）** |
| 会议（AFA/WFA/EFA/NBER/NFA） | 未做独立的会议程序单核查 | **未核查**。这是一个明确的覆盖缺口：2026 年 AFA/WFA 程序单未被检索 |

近 180 天内确认的新增条目共 **5 条**：A8（正式出版）、B6、B7、D10、E10。

### 6.3 引用追踪与终止条件

**执行的追踪**（全部通过 Semantic Scholar `/citations` 端点，前向；后向通过论文正文的参考文献与 Crossref `references-count` 辅助）：

| 轮次 | 种子 | 新增纳入 |
|---|---|---|
| 1 | A1 Gârleanu–Pedersen (10.1111/jofi.12080)，前 100 条引用 | 0 条直接相关新增（引用者以组合优化/RL/执行为主）；标记 1 条待查（Applied Mathematical Finance "Multiscale…"，经 Crossref 核实实为 "Multiscale Stochastic Volatility"，**与信号衰减无关，排除**） |
| 2 | A2 Qian–Sorensen–Hua (10.3905/jpm.2007.698030)，全部 22 条 | **新增 A11、A13、A14、C4** |
| 3 | C1 DeMiguel et al. (10.1093/rfs/hhz085)，100 条 | 0 条（关键词过滤后无匹配） |
| 4 | D1 Novy-Marx–Velikov (10.1093/rfs/hhv063)，100 条 | **新增 C7、D8、D9、E10、E21** |
| 5 | E21 Baba Yara et al. (10.1016/j.jfineco.2024.103808)，全部 7 条 | **新增 E22** |
| 5 | Blitz et al. FAJ 2023 (10.1080/0015198x.2023.2173492)，全部 15 条 | **新增 D7** |
| 6（终止轮） | E22 Gulen–Woeppel (10.1017/s0022109025101725)，全部 1 条 | **0 条新增**（唯一引用者为该文自身的勘误 10.1017/s0022109025102342） |

**终止条件陈述**：最后一轮完整引用追踪（第 6 轮，以第 5 轮新发现的 E22 为种子）**未产生任何新的纳入论文**，
满足协议 §1.4 的终止条件。
**但必须同时声明**：由于 Semantic Scholar 的关键词搜索端点全程不可用、OpenAlex 全程不可用、
且 S2 对 pm-research 系列 DOI 覆盖不全（D5 的前向追踪完全失败），
本次引用追踪的**覆盖是不完整的**，终止条件的满足不等于文献已穷尽。

### 6.4 纳入文献计数

- 候选文献表共 **56 条**（A 组 16、B 组 7、C 组 7、D 组 10、E 组 23，减去无重复计数）。
  精确计数：A1–A16 = 16，B1–B7 = 7，C1–C7 = 7，D1–D10 = 10，E1–E23 = 23，**合计 63 条条目编号**，
  其中不存在重复条目。满足协议的"不少于 25 条"。
- 其中 **正文已独立核实**（取得并读取全文）的有：A1、A2、A8、B1、C1、C2、D1，共 7 条。
- 其中 **仅核实至元数据/官方摘要层面**的有：其余 56 条。
- **无法确认存在**的条目：**0 条**（所有进入表格的条目均有本次会话取得的 DOI / arXiv ID / SSRN DOI）。
