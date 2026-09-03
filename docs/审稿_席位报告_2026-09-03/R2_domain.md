# Reviewer 2 (Domain) 审稿报告

**稿件**：《研究计划书（Research Proposal）v0.1 — 2026-09-03》
（`C:\Users\admin\Downloads\研究计划书_20260903.md`，与仓库 `F:\quant\us-quant-pipeline\docs\研究计划书_2026-09-03.md` 逐字相同）

**审稿人角色**：Peer Reviewer 2 — Domain（实证资产定价：异象、交易成本、横截面机器学习）
**关注面**：文献覆盖、缺口主张（gap claim）的正确性、理论框架、领域贡献、故事性
**READ-ONLY 声明**：本次审稿未修改/创建/删除 `F:\quant` 内任何文件；未打开任何 `sealed` 路径、未读取任何 `scores.parquet`；未查阅其他审稿席位的产出。

---

## 判定

**Recommendation: Major Revision**
**Confidence: 4 / 5**

（信心不给 5 的原因：Rahimikia et al. 全文我只通过 arXiv HTML 的分节抓取核实到 §5.2.4 标题为 "Transaction Cost"，未能取到该节正文的成本口径原文；该点是 W1 的一条支柱，但不是唯一支柱。）

---

## Summary Assessment（约 230 词）

本提案的**方法学骨架是真诚且少见的**：确认集真封存、假设写在读取之前、选择偏差被量化并披露（+0.007，约信号的三分之一）、功效关把"本样本量下不可回答"的比较明确降级。以研究流程论，它比多数已发表的金融 ML 论文更干净，ICAIF / JFDS 这一档的方法节完全立得住。

但作为**金融实证论文**，它目前有三个结构性问题。第一，缺口一句话（§2 末）建立在对最近邻文献 Rahimikia–Ni–Wang (2025) 的错误刻画上：该文并非"美股日频"（94 国、1990–2023）、并非"未处理预训练污染"（自建 proprietary 预训练模型显式排除重叠）、并非"未做公平的扩展窗口基线"（OOS 2001–2023 expanding window，基线含 XGBoost/CatBoost/LightGBM）、也有专门的 Transaction Cost 一节。缺口主张的五项合取里只有"预注册"与"封存集"两项真正成立。第二，§1 承诺回答的 (b)"不是已知因子的翻版"，在 §4 的 H1–H5 里**没有任何一条对应假设**——张成结果（141% / 89%）全部来自已消耗的开发折，而这恰是金融读者唯一真正在意的那一条。第三，信号画像（短期动量侧、top500、现代年代最强）与 Medhat–Schmeling (2022, RFS) 的 short-term momentum 几乎同构，而控制集里**没有换手（turnover）这一条件变量**，也没有行业动量、PEAD、52 周高点、日历季节性。

另外，年单边换手 48×（≈ 月 400%）是 Novy-Marx–Velikov 生存线的 8 倍——这个对照仓库的 `HANDOFF.md` §12.4 自己写了，提案里却把 NMV 只当作"他们用代理、我们用真钱"的陪衬。在这个换手水平上，标题里的 "Deployable" 需要一条容量曲线才能说。

---

## Citation Verification Table

| # | 提案中的引用与说法 | 判定 | 正确信息 / 备注 |
|---|---|---|---|
| 1 | Rahimikia–Ni–Wang 2025，arXiv **2511.18578**；"系统比较通用 TSFM 与树模型于**美股日频**"；"它没做的：无预注册、无封存集、**无成本实测**、**未处理预训练污染**、**未做公平的扩展窗口基线**" | **有误（重要）** | 论文与作者存在且正确（Eghbal Rahimikia, Hao Ni, Weiguan Wang，2025-11）。但：(i) 样本是**94 国全球日频超额收益、1990–2023**，OOS 2001–2023，美股只是其中一节；(ii) §**5.2.4 标题即 "Transaction Cost"**，有成本分析；(iii) 明确处理污染，原文：*"we develop proprietary pre-trained models that explicitly exclude any such overlap"*、*"This procedure ensures that the models are not exposed to information from future periods, thereby preventing look-ahead bias"*；(iv) 基线为 expanding-window，含 OLS/Lasso/Ridge/EN/PCR + **XGBoost/CatBoost/LightGBM** + NN。→ 五项"没做的"里只有前两项站得住。<br>https://arxiv.org/abs/2511.18578 ; https://arxiv.org/html/2511.18578v1 |
| 2 | Kronos，arXiv **2508.02739**；"45 个交易所 120 亿根 K 线" | **核实** | 标题 *Kronos: A Foundation Model for the Language of Financial Markets*；45 家全球交易所、>12B K-line records；已被 **AAAI 2026** 接收（提案未提，建议补）。<br>https://arxiv.org/abs/2508.02739 |
| 3 | "Issue #375 … 社区在 **182 只美股**上零样本 **IC 0.022、t=1.39**"；"它没做的：样本太小、**无横截面变现**、无跨年代" | **部分有误** | Issue 存在且数字精确：标题即含 "IC +0.022 (t=1.39)"，作者 GitHub 用户 **yonaoh10**，开于 **2026-07-29**，无维护者回复。核实到的额外内容：95% CI [−0.009, +0.053]；对动量/反转/波动正交后**残差 IC +0.0134 (t=1.33)**；**十分位多空 +15.56% 毛（Sharpe 0.77）/ +7.16% 净（Sharpe 0.35）**；**91% 的利润来自 85 个日期中的 5 个**；样本窗 **2024-07..2026-07**。→ 因此"无横截面变现"不成立；且该窗**正是本文的"干净窗"**，它已经给出一个比本文 H5 样本更长的读数并报告"扣成本失败"。<br>https://github.com/shiyu-coder/Kronos/issues/375 |
| 4 | TSFMAudit，arXiv **2605.26161** | **核实（存在）**；数字**未能核实** | 标题、ID 正确：*TSFMAudit: Data Contamination Auditing in Forecasting Time Series Foundation Models*，Hongkai Li 等 10 位，2026-05-24，6 TSFM × 187 数据集 × 10 基线。但摘要中**没有"误差被低估 8–29pp"**这一量化。<br>https://arxiv.org/abs/2605.26161 |
| 5 | arXiv **2510.13654**；"记录 0.1% 重叠即可使误差被低估 8–29pp"（HANDOFF §4 把该数字挂在此文） | **有误（题名）+ 数字未能核实** | ID 存在。**现题为 *Rethinking Evaluation in the Era of Time Series Foundation Models: (Un)known Information Leakage Challenges***（Meyer, Kaltenpoth, Zalipski, Müller；2025-10-15）；另有版本题名 *Time Series Foundation Models: Benchmarking Challenges and Requirements*，引用时须指明版本。摘要为**定性**论证（两类泄漏：样本重叠、时间重叠），**未见 0.1% / 8–29pp 的数字**。→ 该数字须补出处或删除。<br>https://arxiv.org/abs/2510.13654 |
| 6 | Chen–Hanauer–Kalsbach，SSRN **5031755** | **核实** | *Design choices, machine learning, and the cross-section of stock returns*（Minghui Chen, Matthias X. Hanauer, Tobias Kalsbach）。>1000 个模型；**非标准误差比标准误大 59%**；月度 top-minus-bottom 从 0.13% 到 1.98%。提案"发散比标准误还大"的表述准确。<br>https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5031755 |
| 7 | Menkveld 等 2024「非标准误差」 | **核实** | *Nonstandard Errors*, **Journal of Finance 79(3), 2339–2390 (June 2024)**；**164 个研究团队**、342 位合著者、34 国 207 机构；同数据同假设，NSE 可观且随可复现性/评级上升而下降。<br>https://onlinelibrary.wiley.com/doi/10.1111/jofi.13337 |
| 8 | Harvey–Liu–Zhu 2016 | **核实** | *…and the Cross-Section of Expected Returns*, RFS 29(1), 5–68。提案未给篇名，建议补全。 |
| 9 | Bailey–López de Prado | **核实（但引用不完整）** | 最可能指 *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality*, **Journal of Portfolio Management 40(5), 2014**（SSRN 2460551）；若指回测过拟合概率则为 Bailey–Borwein–López de Prado–Zhu (2014, Notices AMS)。**须指明是哪一篇**。<br>https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 |
| 10 | Novy-Marx–Velikov 2016；"异象扣成本后大多归零"；"用的是价差代理" | **核实（但用法有问题，见 W4）** | *A Taxonomy of Anomalies and Their Trading Costs*, **RFS 29(1), 104–147**。原文结论更精确：**月单边换手 < 50% 的异象扣成本后大多仍显著，高于此者极少**；中换手异象执行成本 20–57bp；**buy/hold spread 是最有效的单一成本缓释手段**（= 本文"前 10% 进 / 跌出前 30% 卖"的构造，应予署源）。<br>https://academic.oup.com/rfs/article-abstract/29/1/104/1844518 |
| 11 | Chen–Velikov 2023 | **核实（口径描述不准）** | *Zeroing In on the Expected Returns of Anomalies*, **JFQA 58(3), 968–1004**；204 个异象；扣**实测有效价差**、发表后效应、现代交易技术后，平均异象净期望收益 **4bp/月**，最强者至多 **10bp/月**，组合法约 **20bp/月**。称其"用的是价差代理"不准确——CV 用的是有效价差实测而非纯代理。<br>https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/zeroing-in-on-the-expected-returns-of-anomalies/945133D5A3ECEEAF466AEE91551FD225 |
| 12 | Muravyev–Pearson–Pollet 2025（§11） | **核实** | *Anomalies and Their Short-Sale Costs*, **Journal of Finance 80(6), 3639–3694 (Dec 2025)**；162 个异象；扣借券费前多空均值 **+0.14%/月**（全部来自空头腿），扣后 **−0.01%/月**。提案用它支持"不主张多空可行"是**恰当且有力**的。<br>https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13501 |
| 13 | Lou–Polk–Skouras 2019（隔夜/日内） | **核实**，但**提案正文未引** | *A tug of war: Overnight versus intraday expected returns*, **JFE 134(1), 192–213**。§8 提到"隔夜/日内信号（2019 年发表）"却未给引文，§4 H4 依赖它，须补。<br>https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300650 |
| 14 | Patton–Timmermann | **核实**，但**提案未引** | *Monotonicity in asset returns: New tests…*, **JFE 98(3), 605–625 (2010)**。§3 贡献 2 与 HANDOFF §4 的十分位单调性判读需要它（MR / Bonferroni 型检验）。<br>https://public.econ.duke.edu/~ap172/Patton_Timmermann_sorts_JFE_Dec2010.pdf |
| 15 | Clarke–de Silva–Thorley | **核实**，但**提案未引** | *Portfolio Constraints and the Fundamental Law of Active Management*, **FAJ 58(5), 48–66 (2002)**，transfer coefficient 的出处。HANDOFF 用它解释传导率 0.46；提案若要主张"0.46 是正常的长纯多约束代价"必须引。<br>https://www.tandfonline.com/doi/abs/10.2469/faj.v58.n5.2468 |
| 16 | 摘要中的 "JKP themes"（13 主题） | **核实**，但**未列参考** | Jensen, Kelly, Pedersen, *Is There a Replication Crisis in Finance?*, **JF 78(5), 2465–2518 (2023)**；153 个特征、**13 个主题**、93 国。摘要用了"JKP themes"这个术语却没有文献条目。<br>https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13249 |

---

## Missing References（按对本文的杀伤力排序）

### A 类：不引会被审稿人直接判为未做功课

1. **Medhat & Schmeling (2022), "Short-term Momentum", RFS 35(3), 1480–1526.** — 按上月收益 × 换手双重排序：**低换手股短期反转、高换手股短期动量**；短期动量"survives transaction costs，且在**最大、最流动、覆盖最广**的股票中最强"。这与本文的画像（mom_skip β=+0.07/t=3.6、ADV top500、现代年代最强、2003–04 失效）几乎逐条同构。**这是本文最强的竞争性解释，必须作为对照与控制变量。**
2. **Novy-Marx & Velikov (2016) 的换手结论**（已引但未用）— 见 W4。
3. **Frazzini, Israel & Moskowitz, "Trading Costs" (SSRN 3229719).** — **1.7 万亿美元真实成交、21 个发达市场、19 年**测出的实盘执行成本，结论是真实成本比既往（价差代理）研究**小一个数量级**。这既**抢先了本文"真钱实测"的新颖性主张**，又是本文成本读数唯一的可比外部基准。不引是硬伤。
4. **Arnott, Harvey & Markowitz (2019), "A Backtesting Protocol in the Era of Machine Learning", JFDS.** — 与本文贡献 1（可复用评估协议）**同题、同刊**（JFDS 正是本文"现实档"目标之一）。本文的协议须说明相对它增加了什么（我认为是：封存的读取授权分离、A/B 层功效分类、污染感知的证据不对称——这三点确实是新的，但必须以它为基线陈述）。
5. **Jensen, Kelly & Pedersen (2023), JF 78(5)** — 13 主题的出处。
6. **Lou, Polk & Skouras (2019), JFE 134(1)** — H4 的整个信号 #2 依赖它。

### B 类：缺口一句话的反例（直接影响 §2 末的新颖性主张）

7. **JAR Registered Reports / Registration-based Editorial Process (Bloomfield 等, Journal of Accounting Research, 2017 起)** — 会计-金融领域**已有**期刊级预注册制度，"金融实证里几乎没有预注册加封存的实践先例"这句需要限定为"资产定价横截面实证"并给出这个反例。https://www.chicagobooth.edu/research/chookaszian/journal-of-accounting-research/registered-reports
8. **Pacific-Basin Finance Journal 的 pre-registration publication initiative（2022 社论）** — 金融期刊层面的预注册通道。
9. **"Pre-registration for Predictive Modeling" (arXiv 2311.18807)** — ML 侧的预注册协议先例。
10. **McLean & Pontiff (2016), JF 71(1), 5–32.** — 样本外 **−26%**、发表后 **−58%**。本文"选择偏差 20–28%"的收缩因子（0.75）与这条文献量级惊人地一致，**这是一个可以正面利用的外部校准**，现在完全没提。

### C 类：2024–2026 的同期 TSFM×金融评估工作（至少要在 §2 表里各占一行）

11. **FinVerse (arXiv 2608.03259)** — 43 个公开 TSFM × 116,897 条金融序列 × 171.1M 观测，11 个 metric family / 78 个指标，**已含横截面排序与组合评估**，核心结论"通用预测指标上强 ≠ 金融上有用"。与本文 §1 的问题高度重叠。https://arxiv.org/abs/2608.03259
12. **Noguer i Alonso & Pereira Franklin (arXiv 2606.27100), "Pretrained Time-Series Foundation Models for Financial Return Forecasting"** — TimeGPT/TimesFM-2.5/Moirai-2.0/Chronos(-2) vs NBEATS/NHITS/PatchTST/iTransformer/KAN；**只有 5 只美股**，无成本、无污染处理。这是"本文补什么"的最佳正面对照（它恰好是本文所有五项都没做的那种论文）。https://arxiv.org/abs/2606.27100

### D 类：控制集缺口（每条对应一个"这信号其实是什么"的替代解释）

13. **Gutierrez & Kelley (2008), "The Long-Lasting Momentum in Weekly Returns", JF 63(1), 415–447.** — 周频形成后第 1–2 周反转、第 3–52 周动量。本文预测期恰为 6 日 = 一周，E5 的期限结构必须与之并列。
14. **Nagel (2012), "Evaporating Liquidity", RFS.** — 短期反转收益 = **流动性提供的报酬**，且随时间衰减。这是"流动性谱迁移"最现成、最可检验的机制解释。
15. **Cheng, Hameed, Subrahmanyam & Titman (2017), JFQA** — 反转随机构退出/流动性年代变化；配合十进制化（2001）、Reg NMS（2007）、HFT 取代专家做市，构成 A.1 的外生年代叙事。
16. **Da, Liu & Schaumburg (2014), Management Science 60(3), 658–674.** — STREV 的分解：只有非基本面成分才反转。本文控制集的 rev1/rev5 是未分解的粗版本。
17. **Moskowitz & Grinblatt (1999), JF** + **Hou (2007), RFS** — 行业动量与行业内领先-滞后。本文 8 因子控制集**完全没有行业维度**，而 6 日横截面价量信号最常见的"它其实是什么"答案就是行业动量/领先-滞后。
18. **Heston, Korajczyk & Sadka (2010), JF** 与 **Keloharju, Linnainmaa & Nyberg (2016), JF** — 日/周频的收益季节性；配合 **Bogousslavsky (2016), JF**（infrequent rebalancing 造成的周内效应）。本文"每个套袖固定星期几调仓"的设计使这一族混淆特别相关。
19. **George & Hwang (2004)**（52 周高点）、**Chan, Jegadeesh & Lakonishok (1996)**（分析师修正）、**Bernard & Thomas (1989/1990)** / 项目已引的 **Frazzini & Lamont (2007)**、**Savor & Wilson (2016)**（PEAD / 公告溢价）— 标准控制缺口，尤其 PEAD：6 日窗口内跨一个财报日的名字会系统性地被打高分。

---

## Strengths

**S1 — 预注册的时点是真的，而且可验证。**
`text: "状态：草案，写于折 05–35 确认集尚未读取之时，因此 §4 的假设具有真正的预注册地位"`；`section: §4`（H1–H5 + C 逐条给出终点、判据、功效状态、冻结状态）。这在横截面实证里确实罕见，且 §10 提出先挂 arXiv 建立时间戳 + 公开登记簿哈希，使"真"这个字可以被第三方检验。方法节的可发表性主要建立在这一点上。

**S2 — 把选择偏差从"承认存在"变成"量化并当作一个数用"。**
`text: "append-only 登记簿量化选择偏差（实测 +0.007，占信号三分之一）"`；`section: §3 贡献 1`。+0.007 相对 +0.021 的比例（约 1/3）与 McLean–Pontiff 的样本外衰减 −26% 量级相容，这是一个可以对外校准的数字，而不是自说自话的形容词。

**S3 — 污染感知的证据不对称是一个正确且可复用的推理形状。**
`text: "污染窗通过为弱证据、不通过为强证据"`；`section: §3、§8`。给定 Kronos 语料截至 2024-06 且含美股、而折 01–42 全在语料窗内，这是在无法重训基座的约束下能做出的最诚实的推理。TSFMAudit 与 2510.13654 只提出问题、不给评估协议，这一条是本文相对它们**真正成立**的增量。

**S4 — §11「刻意不主张」写得比多数已发表论文诚实，且空头腿的放弃有正确文献支撑。**
`text: "不主张多空可行（空头腿不显著且与 Muravyev-Pearson-Pollet 2025 一致）"`；`section: §11`。用 MPP (2025, JF) 162 异象扣借券费后均值 −0.01%/月 来解释为什么只做多，是对的做法，也让"只做多"从缺陷变成设计。

**S5 — 功效关（SESOI/MDE）与 A/B 层分类给出的是"哪些问题在此样本量上不可回答"的定量答案。**
`section: §3 贡献 1、§4 功效状态列`。把 H1 的 E 端因功效 61% 主动降为估计交付、把臂间比较整体划入 B 层，这是**主动放弃可发表性以换取正确性**的选择，在金融 ML 稿件里几乎见不到。这一条与 Menkveld 等 (2024) 的 NSE 框架、Chen–Hanauer–Kalsbach 的 59% 结论对接得很自然，是本文最有希望被引用的方法学产物。

---

## Weaknesses

### W1 — 缺口一句话建立在对最近邻文献的错误刻画上
- **Problem**：§2 表格第一行给 Rahimikia–Ni–Wang (2025) 列了五项"它没做的"，其中至少三项与原文不符；而"缺口一句话"正是这五项的合取。
- **Evidence Anchor**：`text: "无预注册、无封存集、无成本实测、未处理预训练污染、未做公平的扩展窗口基线"` (§2)。反证（arXiv 2511.18578 全文 HTML）：`text: "we develop proprietary pre-trained models that explicitly exclude any such overlap"`；`text: "This procedure ensures that the models are not exposed to information from future periods, thereby preventing look-ahead bias"`；§5 子节目录含 **5.2.4 Transaction Cost**；OOS 2001–2023 expanding window，基线含 XGBoost / CatBoost / LightGBM。另 `text: "系统比较通用 TSFM 与树模型于美股日频"` 与该文实际覆盖 **94 国、1990–2023** 不符。
- **Why it matters**：金融审稿人会先读 §2 表格，再去翻最近邻论文。三处不符会同时摧毁缺口主张与作者的可信度——尤其"未处理预训练污染"，因为污染处理正是本文贡献 1 的核心卖点之一，而对方用的是**更强**的方案（自建无重叠语料的预训练模型），本文用的是**较弱**的替代（证据不对称）。
- **Suggestion**：重写该行为"它做了什么 / 它的处理方式 / 本文为什么仍有增量"三列。诚实的增量陈述应是：*Rahimikia 等以"重训干净基座"解决污染，代价是放弃了对公开基座的评估；本文保留公开基座并给出在无法重训时的评估协议（证据不对称 + 封存 + 干净窗），二者互补而非替代。* 同时把"美股日频"改为"94 国日频（含美股专节）"。
- **Severity**: **Critical** | **Confidence**: 4（"Transaction Cost" 一节的正文内容未取到，但节标题已确认）

### W2 — §1 承诺的 (b)「不是已知因子的翻版」在确认集里没有对应假设
- **Problem**：§1 把研究问题定义为四件事 (a) 存在、(b) 非翻版、(c) 跨年代稳定、(d) 扣成本为正。但 §4 的 H1–H5 + C 中，**没有任何一条检验 (b)**。张成结果（8 因子保留 141%、JKP 13 主题保留 89%）**全部来自已消耗的开发折**。
- **Evidence Anchor**：`section: §4`（H1 IC>0；H2 top500−全池；H3 vs 树；H4 合成；H5 干净窗符号；C 成本）；`absence: §4 假设表 — 期望有一条"折 05–35 上对控制集张成后 alpha 显著为正"的预注册假设；已检查 §4 全表、§3 贡献 2、§7 方法论清单，均无`。
- **Why it matters**：对金融读者而言，(a) 和 (b) 的信息量完全不对等。一个 6 日价量信号的 RankIC > 0 本身几乎不构成发现；**"它不是短期动量/反转/流动性的翻版"才是发现**。把唯一有金融含量的主张留在开发折上、只把最平凡的主张送进封存集，等于把统计资本花在了错的终点上。而且 §3 已明确承认张成保留率 >100% 的机制是"正向暴露于本样本内亏钱的因子"——这个机制在 2005–2020 的因子表现下**很可能反号**，正因如此它更需要被预注册检验。
- **Suggestion**：在协议 v4 定稿前加一条 **H1b：折 05–35 上，对（扩充后的）控制集张成后的多空 alpha > 0，NW t ≥ 1.96**。若功效不足，至少作为估计交付（点估计 + CI + 逐折符号），并在 §4 明确标注层级。同时把 §3 贡献 2 里的 141%/89% 全部加"开发折"前缀。
- **Severity**: **Critical** | **Confidence**: 5

### W3 — 最好的故事（年代迁移）在设计上永久不可确认
- **Problem**：§3 贡献 2 的"信号沿流动性谱的年代迁移（2003–04 长在不流动端、2020 年代长在流动端，DiD 95% CI [+0.0034, +0.0191]）"，其 DiD 的**两个端点分别落在折 01–04 与折 36–42**——两者都是已消耗的开发折。封存的折 05–35 是 2005–2020，是**中段**。
- **Evidence Anchor**：`section: §6`（"折 01–04 与 36–42 为已消耗开发折，折 05–35 封存待一次性读取"）+ `section: §3 贡献 2`；`absence: §4 — 期望有一条能在封存折上确认迁移的假设；H2（top500 IC − 全池 IC > 0，2005–2020）只是"现代型流动性放大"的静态代理，不含年代交互项`。
- **Why it matters**：这是全文对金融读者**唯一有理论新意**的发现（一个预训练模型在流动性谱上的位置随市场微结构演化而迁移），却结构上无法被本文自己的确认机制支持。审稿人会问："你最有意思的结果永远是探索性的，那封存集买到了什么？"
- **Suggestion**：两条路，选一条并写进 §4。(i) 把 H2 升级为**年代交互**假设：在折 05–35 内部按前后半段（2005–2012 / 2013–2020）估 ΔADV 的趋势，预注册"趋势 > 0"；这在 31 折上仍有功效，且是迁移假设的真正外推检验。(ii) 承认迁移是描述性的，把它移出 §3"贡献"、放进 §5 探索性，并明确说"外部确认需要另一个市场"（E7 正好可以充当）。
- **Severity**: **Major**（可修，但必须在解封前修）| **Confidence**: 4

### W4 — 换手 48×/年 = 月单边 400%，是 Novy-Marx–Velikov 生存线的 8 倍；提案引了 NMV 却回避了它最不利的那条结论
- **Problem**：§2 把 NMV/Chen–Velikov 概括为"异象扣成本后大多归零 / 用的是价差代理；本文用真钱实测"，把它们当作方法论陪衬。但 NMV 的核心经验规律是**月单边换手 < 50% 的异象扣成本后大多仍显著、高于此者极少**。本策略年单边 48× ≈ **月 400%**，在生存线之外 8 倍。Chen–Velikov 的对照更刺眼：204 个异象扣成本后平均 4bp/月、最强者 10bp/月、组合法约 20bp/月；本策略的净纯多超额 +6.66%/年 ≈ **55bp/月**。
- **Evidence Anchor**：`text: "Novy-Marx-Velikov 2016；Chen-Velikov 2023 | 异象扣成本后大多归零 | 用的是价差代理；本文用真钱实测"` (§2)；`absence: §8 效度威胁 — 期望有一条"换手远超文献生存线"的自我对照；已检查 §2、§3、§8、§11，均无`。对照仓库 `HANDOFF.md` §12.4 第 1 行确有此对照：`text: "年单边 48× ≈ 月换手 400%，是生存线的 8 倍"`。
- **Why it matters**：(i) 提案比仓库自己的记录**更不诚实**——审稿人若拿到登记簿会发现这一点；(ii) 在这个换手水平上，NMV 的 buy/hold spread 缓释（正是本文"前 10% 进 / 跌出前 30% 卖"的构造）虽已采用，但仍不足以让策略落进生存带；(iii) 这直接决定"deployable"能不能说。
- **Suggestion**：把这条对照**升格为 §8 的第一条威胁**，并在 §2 表格里改写 NMV 行为"它给出的生存线是月单边 50%；本策略在 400%，因此本文的成本检验性质是'确认策略能不能活'而非'确认成本模型'"——这正是 HANDOFF §12.4 的措辞。同时对 NMV 的 buy/hold spread 署源（§6 的 10%/30% 规则与 §3 的 E6 都应引它）。
- **Severity**: **Major** | **Confidence**: 5

### W5 — 信号画像与 Medhat–Schmeling 短期动量同构，而控制集缺少最关键的条件变量：换手
- **Problem**：项目实测该策略"在短期动量侧"（mom_skip β=+0.07, t=+3.6；rev1 β=−0.12），且在 ADV top500 的**流动端**最强、现代年代最强。Medhat & Schmeling (2022, RFS) 恰好证明：按换手条件化后，**高换手股呈短期动量、低换手股呈短期反转**，且短期动量**在最大、最流动、覆盖最广的股票中最强**、**扛得住交易成本**。本文的 8 因子控制集包含 rev1 / rev5 / mom_skip(t−24..t−5) / mom_12_1 / Amihud / 成交量冲击 / 特质波动 / 相对价差代理——**没有换手（share turnover）**，也没有换手 × 短期收益的交互。
- **Evidence Anchor**：`text: "对 8 个短周期价量因子与 JKP 13 主题张成后 alpha 保留 >100% / 89%"` (§3)；`absence: 控制集 — 期望含 share turnover 及其与短周期收益的交互（Medhat–Schmeling 双重排序）、行业动量、PEAD/盈余惊喜、52 周高点、日历季节性、隔夜/日内分解；已检查 §2、§3、§6、HANDOFF §4 的 K6b 控制集清单，均无`。
- **Why it matters**：一个怀疑的金融读者会说："你测的就是 Medhat–Schmeling 的短期动量，只不过用 25M 参数的 transformer 重新包装了一遍。"而"成交量冲击"这一控制**恰恰不等于**换手水平——冲击是变化量，M–S 用的是水平量作**条件变量**。此外 JKP 13 主题是月频构造，对 6 日信号的张成功效天然很低，用它来支撑 89% 保留率是弱证据。还有两个未控制的高概率替代：**行业动量/领先-滞后**（Moskowitz–Grinblatt 1999；Hou 2007）和 **PEAD**（6 日窗口内跨财报日的名字会系统性被打高分——项目 §12.5 已决定"不加规避财报的规则"，那就更需要把盈余惊喜放进控制集）。
- **Suggestion**：控制集至少扩充四项：**(1) share turnover 水平 + turnover × rev1 交互（M–S 双重排序的回归形式）；(2) 行业（GICS/FF48）动量与行业中性化后的残差信号；(3) SUE / 财报日虚拟变量；(4) 52 周高点比率**。并把 STREV 按 Da–Liu–Schaumburg 分解为基本面/非基本面成分。若这四项之后保留率仍 >100%，本文的领域贡献才真正成立——**而且那时它就是一个值得投 JFE/RFS 的结果**。
- **Severity**: **Major** | **Confidence**: 4

### W6 — 标题的 "US Large Caps" 与确认集主终点「全池 RankIC」不一致；"alpha" 用词过宽
- **Problem**：标题写 *…Test on US Large Caps*，但 H1 的主终点是 **"全池逐日 RankIC"**；变现构造才是 ADV top500。同时全文用 "alpha"，而唯一确认性终点是**未做风险调整的秩相关**。
- **Evidence Anchor**：`text: "全池逐日 RankIC，NW t ≥ 1.96，≥18/31 折为正"` (§4 H1) vs `text: "A Pre-Registered, Sealed-Confirmation, Cost-Measured Test on US Large Caps"` (§0)。
- **Why it matters**：金融审稿人对"标题声称的样本 ≠ 检验所在的样本"零容忍。而把 RankIC 叫 alpha，会让读者以为已经做了因子调整——而按 W2，因子调整恰恰不在确认集里。
- **Suggestion**：标题改为 *…on the US Cross-Section*，或把 H1 主终点改到 top500（§11.7 的离散度论证支持全池作科学泛化终点，那就把这个理由写进正文，而不是留在 HANDOFF）。全文把未调整的 RankIC 一律称 **signal / predictive power**，"alpha" 只保留给张成回归的截距。
- **Severity**: **Major** | **Confidence**: 5

### W7 — "真钱实测成本"的新颖性被 Frazzini–Israel–Moskowitz 先占，且单一 AUM 不足以支撑 "Deployable"
- **Problem**：§2 把"本文用真钱实测"作为相对 NMV/CV 的增量。但 Frazzini, Israel & Moskowitz 用**1.7 万亿美元真实成交、21 个发达市场、19 年**的实盘数据做过同一件事，并得出"真实成本比既往研究小一个数量级"的结论。本文的样本是**约 500 笔、单一 AUM、单一券商（Alpaca Elite）、30–60 个交易日**。
- **Evidence Anchor**：`text: "用的是价差代理；本文用真钱实测"` (§2)；`text: "成本只在一个 AUM 上测得，不得外推"` (§8)；`absence: §2 与 §10 — 期望引用 Frazzini–Israel–Moskowitz 的实盘成本文献；已检查全文，无`。
- **Why it matters**：新颖性主张被一篇高知名度论文正面覆盖，而 FIM 的结论方向对本文其实**有利**（真实成本可能远低于价差代理）。不引既失分又浪费了一个正面论据。另外，"deployable" 在没有容量曲线的情况下只是"在我这个账户规模上可行"——注意仓库 §10.4 已有容量读数（$1M 下冲击 ≤1.3bp，$30–50M 才咬人），完全可以变成一条容量曲线。
- **Suggestion**：(i) 引 FIM 并把本文定位为"个人/小 AUM 端的实盘成本读数，补 FIM 机构端之外的一段"；(ii) 把 §10.4 的容量估计做成 **cost(AUM) 曲线**放进论文，并把 "deployable" 限定为 "deployable at AUM ≤ $X"，否则改用 "cost-viable at the tested scale"。
- **Severity**: **Major** | **Confidence**: 4

### W8 — 对 Kronos Issue #375 的引用是选择性的，且它恰好落在本文的干净窗
- **Problem**：§2 只摘了 "IC 0.022、t=1.39"，并说它"无横截面变现、无跨年代"。实际该 issue **做了**横截面变现（十分位多空 +15.56% 毛 / +7.16% 净，Sharpe 0.77 / 0.35），**做了**正交（对动量/反转/波动正交后残差 IC +0.0134, t=1.33），并报告了两条对本文不利的事实：**扣成本后不成立**、**91% 的利润来自 85 个日期中的 5 个**。更关键的是它的样本窗 **2024-07..2026-07 正是本文的"预训练干净窗"**，即本文 H5 要问的那个问题已经有一个**样本更长**（2 年 vs 1 年）的公开读数。
- **Evidence Anchor**：`text: "样本太小、无横截面变现、无跨年代"` (§2)；Issue #375 标题原文 `text: "Cross-sectional ranking test: Kronos-small on 182 US equities, 85 post-cutoff dates — IC +0.022 (t=1.39), fails after costs"`（GitHub 用户 **yonaoh10**，2026-07-29）。
- **Why it matters**：(i) 该 issue 的收益集中度（5/85 日贡献 91%）是对本文 §3 贡献 2 的直接威胁，必须在 §8 回应；(ii) 该 issue 的残差 IC 0.0134（正交后保留约 61%）与本文开发折的 141% 保留率**差异巨大**，这个差异本身值得解释；(iii) H5 的期望 t≈0.65 意味着本文的干净窗证据将弱于一个 GitHub issue——这需要在 §8 说清楚。**另**：若该 issue 系作者本人所发，必须披露为自引，否则"社区在 182 只美股上……"的措辞会被视为把自己的读数包装成独立外部验证。
- **Suggestion**：把 §2 该行改写为完整刻画（含"扣成本失败"和收益集中度），在 §8 加一条威胁"独立读数显示利润高度集中于少数日期，本文未做同类集中度检验"，并把该检验加入 E 系列。若为自引，在脚注注明。
- **Severity**: **Major** | **Confidence**: 4（自引与否需作者确认）

### W9 — "五件事的合取"是弱新颖性论证
- **Problem**：§2 末的缺口一句话是五个条件的合取。任何足够长的条件清单都能把已有文献的交集变成空集，这在金融审稿里被称为 conjunction novelty，说服力低。
- **Evidence Anchor**：`text: "TSFM 用于资产定价的文献还没有一篇同时具备预注册、封存确认、污染感知窗口、量化披露的选择偏差、真钱成本实测"` (§2)。
- **Why it matters**：审稿人会逐项拆掉（W1 已拆掉三项）。而本文真正的新颖点其实是**单一的、机制性的**：*在无法重训基座的现实约束下，如何对一个语料窗覆盖全样本的预训练模型做可信的资产定价评估。* 这一句比五项合取强得多，且不会被逐项证伪。
- **Suggestion**：把缺口一句话改写为上述单一机制主张，五项降为"协议的组成部分"。
- **Severity**: **Minor**（易修，但影响整篇的说服力）| **Confidence**: 4

### W10 — §11 扣完之后，金融读者剩下什么？
- **Problem**：§11 依次放弃：不主张优于任何配置变体、不主张跨年代经济可投性、不主张多空可行、不主张新闻/期权/基本面腿。加上 W2（张成不在确认集）与 W3（迁移不可确认），确认集能交付的实质金融主张只剩"全池 6 日 RankIC > 0"。
- **Evidence Anchor**：`section: §11` 全节；`section: §4 H1`。
- **Why it matters**：这不是不诚实的问题，恰恰相反——但一个只交付"某个信号的 RankIC 显著大于零"的论文，在 JFDS/QF 是边缘、在 ICAIF 需要靠方法节撑。§11 需要一句"**仍然主张什么**"来给读者一个抓手。
- **Suggestion**：在 §11 之前加一小节「本文主张什么」，用三句话写死：(1) 金融原生 TSFM 的零样本横截面信号在 2005–2020 上确实存在且非零（H1）；(2) 它的经济可投性在测得的成本下**在历史上不成立**（这是一个**负面但可发表**的结论，且比"能赚钱"更容易被相信）；(3) 本文给出的评估协议使这两个判断可被第三方复核。把 (2) 明确写成一个**期望中的负面结果**，会大幅提升可发表性——负面结果 + 预注册 = 恰好是 registered report 的典型价值主张。
- **Severity**: **Minor**（框架问题，非事实错误）| **Confidence**: 4

### W11 — 两处数字缺出处
- **Problem**：`text: "指出预训练语料与回测期重叠会使误差被低估 8–29pp"` (§2)。该数字未能在 TSFMAudit (2605.26161) 或 2510.13654 的摘要中核实到；后者的摘要是定性论证。HANDOFF §4 另有 "0.1% 重叠即可" 的说法，同样未核实。
- **Evidence Anchor**：见核实表 #4、#5。
- **Why it matters**：引用一个不存在于所引文献的量化结论，是最容易被抓的一类错误。
- **Suggestion**：定位到具体表/页并补页码，或改为定性表述。同时把 2510.13654 的题名按现版本更正、注明版本号。
- **Severity**: **Minor** | **Confidence**: 3（只核到摘要，正文可能有该数字）

---

## Questions for Authors

1. **Kronos Issue #375（GitHub 用户 `yonaoh10`，2026-07-29）是否为本项目作者本人所发？** 若是，§2 的"社区在 182 只美股上……"需改为自引并披露；若否，则该 issue 的样本窗（2024-07..2026-07）与本文干净窗重合、样本更长、且已报告"扣成本失败"，请说明本文 H5 相对它的增量是什么。

2. **为什么张成（spanning）不进确认集？** §1 把"不是已知因子的翻版"列为四个研究问题之一，而 H1–H5 无一对应。若是功效原因，请给出 31 折上张成 alpha 的 MDE 与 SESOI；若不是，请说明为何把统计资本花在 IC > 0 而非 alpha > 0 上。

3. **在控制集中加入 share turnover 水平及其与短周期收益的交互（Medhat–Schmeling 2022 的双重排序）之后，开发折上的保留率是多少？** 这是本文与"重新包装的短期动量"之间唯一的分水岭，且可以在已消耗的开发折上做（不消耗封存折），代价是一天 CPU。

4. **"Deployable" 在什么 AUM 区间上成立？** 仓库已有容量读数（$1M 下冲击 ≤1.3bp、$30–50M 才咬人），能否把它做成 cost(AUM) 曲线并写入论文？在月单边换手 400% 的水平上，没有容量曲线的 "deployable" 我认为不可辩护。

---

## Minor Issues

- **M1（§0/摘要）**：摘要把"survives spanning against short-horizon price factors and JKP themes"与"migrates along the liquidity spectrum across eras"与开发折读数并列陈述，**未标注这两项均为已消耗开发折的结果**。摘要读者会把它们当作已确认的发现。建议各加 "on development folds" 限定。
- **M2（§2）**：Kronos 已被 **AAAI 2026** 接收，引用时应给会议信息而非只给 arXiv ID。
- **M3（§2）**：`Harvey-Liu-Zhu 2016`、`Bailey-López de Prado` 只有姓氏无篇名/年份/卷期。Bailey–López de Prado 至少有两篇常被并称的工作（Deflated Sharpe Ratio, JPM 2014；Pseudo-Mathematics, Notices AMS 2014），须指明。
- **M4（§2）**：称 Chen–Velikov (2023) "用的是价差代理"不准确——该文用的是实测有效价差（含 TAQ/ISSM 与低频估计的组合）。"价差代理"的说法对 Novy-Marx–Velikov 更贴切。
- **M5（§6）**：`text: "只做多、ADV 前 500、前 10% 进入、跌出前 30% 卖"` —— 这正是 Novy-Marx–Velikov (2016) 的 **buy/hold spread** 成本缓释法，原文称其为"最有效的单一成本缓释策略"。应署源；这同时会强化 §3 的 E6（从信度闭式推退出线 28.3%）——它把 NMV 的经验规则给出了一个解析基础，**这本身是一个小而真实的方法贡献，现在被埋没了**。
- **M6（§3 贡献 3）**：`text: "预训练生成式解码链的手写读出（零样本 0.019）优于所有在本项目数据上学出的读出（线性探针 0.013、MLP 0.008、树 0.006）"` —— 按项目自己 `CLAUDE.md` §二的 A/B 层措辞强制，这些臂间比较需要标注层级；0.019 vs 0.013 的相对差约 32%，低于 HANDOFF §10.2 记录的"相关臂需差出信号的 45–66%"分辨门槛。H3 用逐日配对 ΔIC 是对的（配对能进 A 层），但**贡献 3 陈述的是折级比较**，措辞应与 H3 区分开。
- **M7（§5 E7）**：跨市场零样本迁移标注"需非美股数据与授权"。考虑到 Kronos 语料覆盖 45 个交易所，任何海外市场同样在污染窗内；E7 的"成倍加证据"只对**横截面泛化**成立，对**污染**不成立。建议在 §5 注明这一限制。
- **M8（§8）**：`text: "隔夜/日内信号（2019 年发表）对 2005–2020 不是样本外"` —— 正确且值得表扬，但应顺带引 McLean–Pontiff (2016) 给出发表后衰减的量级（−58%），使这条威胁从定性变成定量。
- **M9（§10）**：目标渠道评估合理。补充一点：**Journal of Accounting Research 的 Registered Reports 通道**与 **Critical Finance Review** 的复制/协议类稿件，对"预注册 + 负面结果"的稿件比 QF 友好，值得列入"冲高档"的备选路径。

---

## 故事性（Story）—— 从金融读者视角

**金融读者会记住的一句话，目前提案没有交付。** §0 的题目是一个 yes/no 问句（"有没有可部署的横截面 alpha？"），而 §11 已经预先放弃了回答其中"可部署"的那一半；§2 的缺口是五项方法学条件的合取，不构成任何关于市场的主张。结果是：读者读完不知道**关于股票市场，本文教了我什么**。

我认为稿件手上已经有一句很好的话，只是被埋在贡献 2 里：

> **一个在全球 45 个交易所上预训练、从未见过美股横截面标签的模型，学到的排序能力沿流动性谱随年代迁移——2003–04 在不流动端、2020 年代在流动端；而在月单边 400% 的换手下，它 22bp 的盈亏平衡线与真实执行成本之间没有安全边际。**

这句话有机制（微结构演化：十进制化 2001、Reg NMS 2007、HFT 取代专家做市、短期反转/流动性提供报酬的长期衰减——Nagel 2012；Cheng et al. 2017）、有可证伪性、有金融含量，而且**即使确认集失败它仍然成立**（因为它是关于信号性质的，不是关于赚钱的）。相比之下"TSFM 有没有 alpha"这个问法，金融读者的先验答案是"大概没有多少，而且活不过成本"——本文最可能的结论恰好会印证这个先验，那就不是发现。

**建议把年代迁移提为 headline，把 TSFM 降为工具。** 题目可改为类似 *What a Financial Foundation Model Learns About the Cross-Section, and Where It Learns It: Liquidity-Spectrum Migration Across Market-Structure Eras*。代价是必须解决 W3（迁移的两个端点都在开发折上），但按 W3 的建议 (i)（在封存折内部做年代交互）是可行的，而且这样一来封存集**买到的东西就变成了那个值得买的东西**。

---

*Reviewer 2 (Domain) — 2026-09-03*
