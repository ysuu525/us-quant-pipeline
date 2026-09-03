# Peer Reviewer 3 — Perspective / Cross-disciplinary Report

**稿件**：`C:\Users\admin\Downloads\研究计划书_20260903.md`（Research Proposal v0.1, 2026-09-03）
**审稿人身份**：ML researcher，方向为 time-series foundation models（Chronos / TimesFM / Moirai / Kronos）、evaluation contamination & memorization、probing / readouts、adaptive data analysis（Dwork et al.）；此前有数年 quant practitioner 经历。
**视角定位**：cross-disciplinary、practical impact、assumption-challenging。我不重复统计细节的复核（那是别的席位的事），我问的是：**这套证据在 ML 社区会被怎么读、在交易台会被怎么读，以及作者有没有用尽他们手上最便宜的武器。**
**已读材料**：计划书全文；`HANDOFF.md` §2/§4/§5/§10/§11/§12；`CLAUDE.md`；`experiments/ledger.md`；`docs/思路整理_2026-09-03.md`、`docs/审视_原始规则_2026-09-03.md`；`src/signals/kronos_adapter.py`、`src/kronos_ft/infer.py`、`scripts/ridge_probe_folds.py`、`third_party/kronos/model/kronos.py`。外部核对：Kronos 论文（arXiv 2508.02739）、TSFMAudit（arXiv 2605.26161）、arXiv 2510.13654、MOMENT（arXiv 2402.03885）。
**未读**：任何 `outputs/**/sealed*` 路径、任何 `scores.parquet`、任何其他审稿席位的产物。

---

## 推荐

**Major revision**（作为 conference paper 计划书评审）
**Confidence: 4 / 5**

置信度不是 5 的原因：确认集与干净窗未读，论文的实证骨架有 40% 还是空的；我对 §4 之外的判断因此是对**设计**的判断，不是对**结论**的判断。

---

## 总体评述（Summary assessment）

这份计划书的真正贡献不在 alpha，而在**认识论**：它是我见过的少数几个把「评估数据是消耗品」当作预算来管理、并且把自己犯的错也计入账本的实证金融项目。append-only ledger、SESOI/MDE 功效关、A/B 比较分层、封存计算与读取分离——这四件里任何一件单独拿出来，在 ICAIF/JFDS 都够写一节；合起来是一个可复用的协议。

但计划书目前把这些资产**错配**了。它想讲的故事是「TSFM 有没有可部署的 alpha」，而它手上最强的东西是「在语料覆盖评估期、开发折已被消耗数十次的条件下，一个人还能做出什么可辩护的主张」。前一个问题它答不了——干净证据只有 1 年、期望 t 0.65、成本未测、§4 六个假设里四个的冻结状态写着「待定」；后一个问题它已经答了一大半。

三个必须修的结构性问题：（1）**§4 自称「具有真正的预注册地位」，但同一张表的「冻结状态」栏显示 4/6 未冻结**，这是全文最致命的自相矛盾；（2）**污染论证只有「证据不对称 + 1 年干净窗」这一套修辞，而作者手上有零成本的、正面的论证与探针没用**（模型输入逐窗 z-score、无年份、无 ticker；24.7M 参数 vs 120 亿 K 线；autoregressive ⇒ 可算无标签 NLL）；（3）**贡献 3「手写读出胜过所有学出的读出」是关于 effective sample size 的陈述，却被写成关于 inductive bias 的陈述**，且四个对照数来自 1/5/7 三种不同折数。

修好这三点，这是一篇好论文。修不好，ML 审稿人会在污染那一条上直接给 reject，而 finance 审稿人会在 §4 的冻结状态上给 reject。

---

## Strengths

**S1｜把「评估数据是预算」真正制度化，且实现是可机器核对的，不是宣言。**
- `section: §7`；`file: F:\quant\us-quant-pipeline\HANDOFF.md:637-729`（`sealed.py` 哨兵 + `SEALED_MANIFEST.json` 记录快照 ID / 代码 SHA-256 / 折边界 / scores 哈希与行数；`assert_readable()` 守卫；纪律测试 `test_no_ordinary_script_references_sealed_outputs`）
- `file: F:\quant\us-quant-pipeline\src\signals\kronos_adapter.py:50-53`（折号白名单 `{1–4, 36–42}` 硬编码，越界即 `FoldNotAllowedError`，**不提供绕过开关**）
在 ML 的 benchmark-contamination 文献里，大家反复呼吁「holdout 要有访问控制」，但几乎没人给出实现。这份工程是可以直接当 artifact 发布的，我认为它比 alpha 更有引用价值。

**S2｜自我否证的密度是这个项目最稀缺的资产，而且被完整记录。**
- `file: experiments/ledger.md:140`（三折「合成较最优臂 +11%」→ 七折 +3.1%，方案否决）
- `file: experiments/ledger.md:127`（三折「lb200 领先 41%」→ 七折反转为落后 56%）
- `file: HANDOFF.md:567-579`（K15 推翻 §4 自己写的「σ_cs −7.2%/年」，判为窗口伪迹，真值 −1.1~−1.7%/年）
- `file: HANDOFF.md:626-630`（**自记违规**：K15-B 运行前没写判据/SESOI/MDE，因此不得据其得出否定结论）
- `text: "本会话两次把噪声读成结构……共同机制是在 n=7 上做推断而手上有 832 天"`（`HANDOFF.md:122-125`）
四次小样本反转 + 一次自记违规，这种记录在 published finance empirics 里近乎不存在。

**S3｜污染问题被正确地识别为 first-order，且细节处理比多数 TSFM 论文严谨。**
- `file: HANDOFF.md:696`（fold43 因验证窗 2024-01..2024-07 落在语料窗内而被**机械移除**，折边界由 `scripts/emit_folds.py` 生成、不手写窗口）
我外部核对了 Kronos 论文：`text: "The pre-training data for Kronos extends up to June 2024"`，且其自身评估自 2024-07 起。作者对语料截止的判断是对的。多数 TSFM 应用论文连这一步都不做。

**S4｜A/B 比较分层是一个真正的、可迁移到 ML 的方法学贡献。**
- `file: CLAUDE.md §二「比较分层：A 层做推断，B 层按先验决定」`（正交臂 99% 淘汰门槛 = 信号本身的 1.10–1.51 倍，**永不触发**；分辨 50%/30%/20% 相对臂间差分别需 20/55/124 年）
- 措辞强制条款（「不得写『不可分』，必须写『在本样本量下该问题不可回答，按预先规则取 X』」）解决的是 ML 社区一个非常普遍的病：把低功效的 ablation 写成 null result。这一条我建议单独提到论文正文的显著位置，它对 NeurIPS/ICLR 读者的价值可能高于对金融读者的价值。

**S5｜成本侧的机械审计诚实到罕见。**
- `file: HANDOFF.md:397-402`（**AUM 是阻塞字段**：`fill − DlyOpen` 在原理上看不见自己对开盘印价的冲击，放大 AUM 只让基准被自己污染 → 因此 AUM 只能由真实部署规模决定）
- `file: HANDOFF.md:463-474`（Alpaca 反向单 403 落在 87.6% 的交易日上，2.7 笔/日）
- `file: HANDOFF.md:420-424`（「Alpaca 零佣金对本策略不成立」——OPG 只对 Elite 开放）
这一节比论文里任何 alpha 结果都更能说服 practitioner 读者。

**S6｜E6 是个漂亮的零参数结果，应当前置。**
- `section: §3 贡献 3 附带`；`file: experiments/ledger.md:406`（信度 0.75 → 噪声 SD 0.5 → z_exit = Φ⁻¹(0.9) − 0.707 = 0.5745 → 退出分位 28.3%，与实际使用的 30% 一致）
从一个可测的量（打分信度）闭式推出一个交易参数，且事后发现与经验值吻合——这正是「先验固定优于数据搜索」这一主张的最好例证。

---

## Weaknesses

### W1｜§4 自称的「预注册地位」与同表「冻结状态」栏直接冲突：6 个假设里 4 个未冻结
- **Problem**：文首写「§4 的假设具有真正的预注册地位」，而 §4 表最后一栏显示 H1「协议 v4 待定稿」、H3「**待写入 v4**」、H4「**待用户裁定处置规则**」、C「协议 v1 草稿，待 Alpaca 答复与 AUM」。只有 H2、H5 标「已在 v3」。
- **Evidence Anchor**：`text: "状态：草案，写于折 05–35 确认集尚未读取之时，因此 §4 的假设具有真正的预注册地位"`（`研究计划书_20260903.md:3`）；`section: §4` 表第 5 栏（`:42-47`）；`file: HANDOFF.md:771-774`（信号#2 的处置规则**尚待裁定**，且预注册 §4 与 CLAUDE.md §五 相抵）
- **Why it matters**：这是论文的核心卖点。「未读结果」≠「已冻结」——adaptive data analysis 的偏差来自**设计空间在读取前仍可移动**，而不只是来自读取次数。H3/H4 的估计量今天还没定义，它们不享有预注册地位；把它们和 H2/H5 并列放在「预注册假设」表里，会被审稿人当作 preregistration inflation。
- **Suggestion**：表里增加一列 `preregistration tier`：Tier-1 = 解封前有时间戳的冻结文本（H2、H5）；Tier-2 = 解封前冻结但晚于本稿（H1 协议 v4、H3）；Tier-3 = 估计量尚未定义（H4、C）。论文正文只对 Tier-1/2 主张预注册，Tier-3 明确标为 exploratory。同时把「冻结完成的时间戳」作为可核对字段（见借用 B3）。
- **Severity: Critical**（对论文主张而言）；**Confidence: 5**

### W2｜功效关被用「预期效应」冒充了 SESOI，而项目自己的 DECISION 说全池 IC 的外部锚不确定
- **Problem**：§4 H1 写「IC 端功效 >99.9%（预期 0.0155，MDE80 0.0083）」。这是拿 MDE 对**预期效应**比，不是对 SESOI 比。CLAUDE.md 的规则是 `MDE ≤ SESOI` 才可做确认性检验。全池 IC 的 SESOI 至今没定。
- **Evidence Anchor**：`file: CLAUDE.md §二「判据还必须先过功效关」`（要求同时写下 SESOI / MDE / 目标功效）；`file: HANDOFF.md:343`（`text: "同模态树基线 0.00628 < k=2 的 MDE80 0.00834，过不了功效关；Simonsohn small telescopes d33%=0.00865 是唯一非循环的先验锚但需另算……暂不立项"`）；`file: experiments/ledger.md`（2026-09-02 DECISION：**全池 IC 维持为带 CI 的区间估计，不做二元 PASS**）
- **Why it matters**：两个候选 SESOI 骑在 MDE80=0.00834 两侧（0.00628 不过关 / 0.00865 险过）。也就是说 H1 能不能作为确认性检验，**取决于一个尚未做出的判断**。更严重的是，§4 的 H1 把全池 IC 写成二元判据（`NW t ≥ 1.96，≥18/31 折为正`），这与登记簿里已生效的 DECISION 直接矛盾。论文若这样写，等于自己违反自己的旗舰规则——而 §7 正是要拿这套规则当贡献卖。
- **Suggestion**：（a）在解封前把全池 IC 的 SESOI 定死并登记，我建议用 Simonsohn `d_33%` = 0.00865 这个非循环锚（并按非零零假设重算功效，作者自己已识别这是需要的）；（b）若定 0.00865，则 MDE80 0.00834 < SESOI，H1 合法，论文里写清「SESOI 的来源是 small-telescopes 而非我们自己的点估计」；（c）若不定，H1 降级为区间估计并**同步修改摘要**（现摘要暗示确认性）。
- **Severity: Critical**；**Confidence: 5**

### W3｜H2 在约 45–51% 功效下仍写成二元判据
- **Problem**：§4 的 H2 判据是「ΔADV，95% CI 下界 > 0，≥18/31 折」——一个二元 pass/fail；功效栏自己写「半强度约 45%」。
- **Evidence Anchor**：`section: §4` H2 行（`:43`）；`file: HANDOFF.md:342`（`text: "真值 0.0026 时功效 FT 51.5% / ZS 45.5%（含折数与符号门约 49%），仍不过功效关"`）
- **Why it matters**：CLAUDE.md 明文「MDE > SESOI 时不得设计成确认性检验，可降级为探索性但不得据其未过门槛得出否定结论」。H2 现在的写法既是确认性的（有二元判据），又注定要在约一半的世界里失败。它会污染 H1 的解读（读者会把 H2 的失败读成机制被否）。
- **Suggestion**：H2 与 H5 同格式：只给效应估计 + CI + 折数一致性，**不给判据**，并在正文写明「本样本量下该问题不可回答」。ΔADV 的跨年代 DiD（`HANDOFF.md:279`，95% CI [+0.0034,+0.0191]）已经是 A 层且已成立——那个才是可以拿来讲机制的结果，把 H2 让给它。
- **Severity: Major**；**Confidence: 5**

### W4｜摘要与贡献 1 的「selection bias +0.007，占信号三分之一」用错了估计量
- **Problem**：`+0.0070` 是**岭探针 alpha 内层选参 vs 外层选参**的差，是另一个实验里另一个量的选择偏差；它被搬来当作**本项目信号级的选择偏差**。§8 自己写的是 20–28%（对 0.0207 而言 ≈ +0.004~0.006）。
- **Evidence Anchor**：`text: "We disclose a measured selection bias of +0.007 (one third of the signal)"`（`研究计划书_20260903.md:108`）；`section: §3 贡献 1`（`:34`，同一个 +0.007 在同一句里既当「登记簿量化的选择偏差」又当「评估窗选参」的错误损失）；`section: §8`（`:78`，`text: "选择偏差估 20–28%"`）；`file: experiments/ledger.md:164`（`text: "内层选参均值 +0.01256 vs 事后最优上界 +0.01951，差 +0.0070……可作为同类偏差的参照尺"`——原文明说是**参照尺**）
- **Why it matters**：这是摘要里的一个数字，而且是往「我们很诚实」的方向夸大。审稿人一旦发现「量化披露的选择偏差」本身是从另一个估计量借来的，全篇的可信度会被反噬——尤其是一篇以「披露」为卖点的论文。
- **Suggestion**：摘要改成两句话：「我们在开发折上直接测得一次选参偏差 +0.0070（=信号的三分之一），并据此把项目级选择偏差按对抗性复核估为 20–28%（+0.004~0.006）。」把「实测量」与「外推量」分开，并给出后者的推导方法（见借用 B1，可以把它变成一个有定理支撑的数）。
- **Severity: Major**；**Confidence: 5**

### W5｜贡献 3 的四个读数来自 1/5/7 三种不同折数，且比较本身不是同一层面的比较
- **Problem**：`零样本 0.019`（7 折）、`线性探针 0.013`（5 折 36–40）、`MLP 0.008`（**fold40 单折**）、`树 0.006`（7 折）。CLAUDE.md §三 明文「少于七折的结论一律不算数」。
- **Evidence Anchor**：`section: §3 贡献 3`（`:36`）；`file: experiments/ledger.md:141`（MLP `+0.00753` 来自 fold40 单折、验证窗 127 天）；`file: experiments/ledger.md:164`（线性探针五折 36–40 均值 `+0.01256`）；`file: experiments/ledger.md:186`（XGBoost 七折合并 `+0.006280`）；`file: HANDOFF.md:39`（零样本七折 `+0.0192`）；`file: CLAUDE.md §三`
- **Why it matters**：贡献 3 是三个贡献里唯一面向 ML 社区的一个，而它的证据表混了 n=1。ML 审稿人第一眼就会问「MLP 那个数跑了几折」。更深一层：这四个数**不测同一件事**（下详 W6）。
- **Suggestion**：（a）MLP 补齐七折或从表里删掉（它在 CPU 上很便宜）；（b）表里为每一行标注 folds / days / 是否内层选参 / 是否确定性输出 / scoring_config；（c）同时报诚实口径与 oracle 上界（探针 oracle `+0.01951`，`ledger.md:178`）——「即使给探针一个事后最优的正则，它仍输给零样本」比诚实口径单独一个数强得多，而这个数你们已经有了。
- **Severity: Major**；**Confidence: 5**

### W6｜「手写读出胜过所有学出的读出」是一个关于 effective sample size 的陈述，被写成了关于 inductive bias 的陈述——而且三个 confound 全都指向同一方向
- **Problem**：现有对照里，「手写读出」与「学出的读出」在三个维度上不可比：
  1. **decoder access**：生成式读出用的是「隐层 → 输出头 → token 概率 → 采样 → 反解价格 → 6 步自回归 → 5 条路径取均值」这一整条预训练解码链；线性探针只拿 anchor 位一个 pooled 隐向量（`file: scripts/ridge_probe_folds.py`，`pool` 参数 = mean/last）。这是「用了多少网络计算」的差，不是「手写 vs 学习」的差。作者自己在 `ledger.md:178` 已经写出了这个候选解释，但计划书 §3 没有把它写进主张的限定里。
  2. **effective sample size**：探针的训练数据 = 3 年窗（内层还要切掉尾部 6 个月并 purge 7 天，`scripts/ridge_probe_folds.py:211-218`），项目自估约 750 个有效独立日；「手写读出」的训练数据 = Kronos 的 12B K 线。所以正确的命名是 **transfer vs in-domain estimation**，不是 hand-written vs learned——那条解码链是**学出来的**，只是在别人的数据上。
  3. **contamination**：这是最要命的一条。零样本臂的「训练集」覆盖了折 01–42 的**全部评估期**；探针的训练集是评估窗之前的 3 年且带 purge。**污染会精确地产生现在观察到的这个方向的差**。
- **Evidence Anchor**：`file: third_party/kronos/model/kronos.py:544-547`（逐窗 z-score + clip）与 `:467`（多采样路径取均值）；`file: src/kronos_ft/infer.py:48-56`（`score = open(t+6)/open(t+1) − 1`，即一个**固定的线性泛函**作用在解码出的路径上）；`file: scripts/ridge_probe_folds.py:211-218`（内层选参窗 + purge）；`file: experiments/ledger.md:178`（`text: "轴一（读出手工写死 vs 学习）影响大……轴二（主干是否适配我们数据）影响小"`）
- **Why it matters**：ML 社区对这个结论有既有先验，而且是**相反**的：MOMENT（arXiv 2402.03885）报告 linear probing 在多数数据集与 horizon 上接近 SOTA。所以本文的反例是有新意的——但**只有在 confound 被拆开之后才是**。按现在的写法，reviewer 会说「你测的是 decoder access，不是 readout paradigm」，而这个反驳是对的。
- **Suggestion（这是我给全文最重要的一条实验建议）**：加一个**嵌套对照**——把解码出的 6 步路径（6×OHLC，5 条采样路径）落盘，然后在**同一批解码输出**上学一个读出（例如对 `{open_h}` 的线性泛函、或对 5 条路径的分位数/离差的线性组合），用与探针相同的 3 年窗 + 内层选参 + purge。手写读出 `o6/o1 − 1` 恰好是这个函数族里的一个特解。若学出来的版本在样本外仍输给这个特解，结论就变成一个**干净的 shrinkage 命题**：「在 6 日标签、约 750 个有效独立日的信噪比下，向一个手工指定的泛函收缩优于估计它」——这个命题可比现在的版本强得多，而且它自动免疫 decoder-access confound。成本：一次重打分把 path 落盘（2 折约 40 分钟 GPU）+ CPU 闭式解。
- **Severity: Major**；**Confidence: 4**

### W7｜污染论证只有「证据不对称 + 1 年干净窗」，而作者手上有零成本的正面论证没用
- **Problem**：§3 贡献 1 与 §8 把污染处理成一个**修辞规则**（「污染窗通过为弱证据、不通过为强证据」）加一个 1 年干净窗。对 ML 审稿人来说这不够：asymmetric evidence 是一个先验声明，它不产生任何关于污染**强度**的信息。
- **Evidence Anchor**：`section: §3 贡献 1`（`:34`）、`section: §8`（`:77`）；`absence: 研究计划书_20260903.md §5 探索性实验表 — 期望有至少一项 contamination / memorization probe；实测 E1–E7 七项中零项针对污染`（已检查 §3、§5、§7、§8 四处）
- **Why it matters**：这是本文与 ML 社区接口的唯一位置，也是最容易被 desk-reject 的位置。而且——作者手上有非常便宜的武器完全没拿出来（见下面「污染探针清单」）。特别是：**Kronos 的输入在结构上已经删掉了三样能用来「查表」的东西**：逐窗 z-score（`kronos.py:544-547`，绝对价格与成交量水平被抹掉）、时间戳只含 `minute/hour/weekday/day/month`（`src/kronos_ft/infer.py:35-43`，**没有年份**）、没有 ticker/PERMNO embedding。也就是说任何污染都必须通过**形状识别**发生，而不能通过「这只票这天」发生。这一段论证的成本是零，力度远大于「证据不对称」。
- **Suggestion**：把 §8 的一条 bullet 扩成一整节「Contamination: what channel is even open?」，包含（i）上面的输入结构论证 + 代码引用；（ii）容量算术（Kronos-small 24.7M 参数 vs 12B K 线 ⇒ 约 485 records/param；且 CRSP 日线全样本 2000–2024 约 5×10⁷ bar，是 1.2×10¹⁰ 语料的 **≲0.5%**，而语料含 7 种粒度、以分钟级为主）；（iii）至少两项正面探针（见下）。
- **Severity: Major**；**Confidence: 4**（(ii) 的 0.5% 是我的量级估算，需作者自行核 CRSP bar 数）

### W8｜干净窗的功效被「终点选择」人为压低了约一倍，且干净窗恰好是 Kronos 作者自己的测试期
- **Problem 一**：§4 H5 与 §8 都写「干净窗功效低（预期 t 约 0.65）」。这个 0.65 是**夏普口径**（SR 0.65 × √1 年）。而项目自己已经论证过：确认年代与开发年代的差异「机械进入毛、不进入 IC」，IC 终点不受 σ_cs 影响。用 IC 口径重算：由 `HANDOFF.md:604` 的 31 折 MDE80 = 0.00834（≈3906 天）反解 SE ≈ 0.00298 → 252 天（fold44–45）SE ≈ 0.0117 → 对预期 IC 0.0155 的 **t ≈ 1.3**；504 天（2 年）SE ≈ 0.0083 → **t ≈ 1.9**。也就是说换个终点，唯一的干净证据的期望 t 从 0.65 变成 1.3，再等一年变成 1.9。
- **Problem 二**：Kronos 论文自己的测试期就是 2024-07 起（`text: "Testing commenced in July 2024 onward to ensure strict temporal separation"`）。所以 fold44–45 对**梯度**是干净的，但对**开发者的模型选择**不是：哪个 checkpoint 被发布、哪套超参、哪个 tokenizer，都可能部分地由 2024-07 之后的表现决定。计划书没有讨论这一层。
- **Evidence Anchor**：`section: §4` H5 行（`:46`）、`section: §8`（`:81`）；`file: HANDOFF.md:702`（`text: "夏普 0.65 下预期 t（非中心参数）由约 0.88 降到约 0.65"`）；`file: HANDOFF.md:602-604`（IC 端 MDE80 0.00834、功效 >99.9% 的同一套推理）；Kronos arXiv 2508.02739 数据节
- **Why it matters**：干净窗是全文对抗污染质疑的唯一硬证据。用一个功效只有一半的终点去报它，等于自废武功；而不讨论 developer-selection contamination，等于把 ML 审稿人最熟悉的那条攻击线留给对方。
- **Suggestion**：（a）干净窗的主报告量改为**全池 RankIC**（并保留夏普作为经济口径的次要报告），把期望 t 从 0.65 改到约 1.3；（b）把「等一个覆盖到 2026-01 / 2026-07 的新 CRSP 快照」写进时间线——干净窗**每年白送 2 折**，这是全项目投入产出比最高的「实验」，成本只有日历时间（`HANDOFF.md:699-703` 已指出 fold46 需新快照）；（c）加一段讨论 checkpoint-level selection：Kronos 的发布决策部分依赖 2024-07 后的表现，因此干净窗对「模型家族的选择」不是完全外生的；（d）干净窗的判读改用 **non-inferiority / equivalence 框架**（见借用 B8），而不是符号检验——低功效下这是唯一能挤出信息的写法。
- **Severity: Major**；**Confidence: 4**（IC 口径的 t≈1.3/1.9 是我按作者自己报的 SE 做的换算，作者应自行精算）

### W9｜「非翻版」这个头条结论依赖一次事后修订的一致性门槛，摘要没有披露
- **Problem**：K6（预注册在先）三个规格保留率 104–137%，但一致性门槛 4/7 < 5/7 → 判**「不稳定、不可判定」**。随后 K6b 把一致性门槛从「逐折独立回归」改为「固定合并回归载荷后的逐折残差」，得 6/7 → 判**「非翻版」**。登记簿非常诚实地记了「此改动相对 K6 是事后的（K6 出 4/7 之后才发现原门槛缺陷）」。
- **Evidence Anchor**：`file: experiments/ledger.md:182`（K6 判「不稳定、不可判定」）；`file: experiments/ledger.md:191`（`text: "此改动相对 K6 是事后的（K6 出 4/7 之后才发现原门槛缺陷）"`）；`file: experiments/ledger.md:192`（K6b S1b 保留 141%、6/7 → 判「非翻版」）；`section: §3 贡献 2` 与摘要（`:35`、`:108`，只写「保留 >100%」「survives spanning」，未提门槛修订）
- **Why it matters**：修订本身可能是对的（原门槛确实是 7 个回归元配 119 天的低功效诊断），而且它写在 K6b 运行之前——这是教科书式的正确处理。但**摘要不写这件事**，就变成了本文正在批判的那类操作。一个对抗性审稿人会把 `ledger.md:191` 贴出来，说「作者自己记了这是 result-triggered 的门槛修订，却在摘要里当作一次干净的通过」。
- **Suggestion**：在正文与（一句话）摘要里披露：「张成检验的一致性门槛在看到第一版结果的 4/7 之后被修订为固定载荷残差版；修订理由是方法学的（逐折独立回归重度过拟合），修订文本先于 K6b 运行，原判读不追溯修改。」这句话会**提高**而不是降低论文的可信度——它正是 §7 想卖的那种披露。另外记得同时保留 `ledger.md:192` 里那条关键限定：保留率 >100% 的机制是策略正向暴露于本样本内亏钱的因子。
- **Severity: Major**；**Confidence: 5**

### W10｜封存是流程纪律不是隔离，且「预注册早于解封」外部不可验证
- **Problem**：作者自己已经更正过一处夸大：有 `scores.parquet` 与原始面板，随时可以重新生成标签并算出 IC；不写 labels 只防误操作。更进一步：ledger 是 append-only **按约定**，git 历史与仓库都由作者控制，外部读者无法验证任何一份预注册文本确实写在解封之前。
- **Evidence Anchor**：`file: HANDOFF.md:713-716`（`text: "封存本质上是流程纪律，不是密码学隔离"`）；`file: HANDOFF.md:191`、`:315`、`:558`（关键脚本与 ledger 修改**全部未提交**，长期处于工作区状态）
- **Why it matters**：论文的核心区别于同行的地方就是「我们真的封存了」。若这一点只能靠作者自述，它在审稿中的证据等级等于零。
- **Suggestion**：成本几小时、收益极高——（a）现在就把 `ledger.md` + 协议 v3/v4 + `SEALED_MANIFEST.json` 的 SHA-256 做**第三方时间戳**（OpenTimestamps / RFC-3161 / OSF 预注册），并把时间戳收据一并发布；（b）把仓库推到公开 remote（或至少一个不受作者控制的 mirror），让 commit 时间由第三方见证；（c）更强的版本：把标签生成所需的一个 key 交给导师托管（registered report 的 sealed-envelope 做法），使「无法提前算 IC」从约定变成事实；（d）论文里保留 `HANDOFF.md:713-716` 那句自我更正——它是可信度而不是弱点。
- **Severity: Major**；**Confidence: 5**

### W11｜确认集上的多重性没有统一方案
- **Problem**：折 05–35 上将同时跑 H1（全池 IC）、H2（ΔADV）、H3（vs 扩展窗口树），且 k=2 配置（FT/ZS）。§4 只对 H4 写了固定序贯，对 H1/H2/H3 之间没有 family-wise 方案。HANDOFF §5 曾提过「候选配置 ≤3 → Bonferroni t 门槛 2.39」，但那只覆盖配置维度，没覆盖终点维度。
- **Evidence Anchor**：`section: §4`（`:42-45`，H4 行写「固定序贯」，H1/H2/H3 行无多重性字样）；`file: HANDOFF.md:150`（`text: "候选配置 ≤3（Bonferroni t 门槛 2.39）"`）
- **Why it matters**：一篇以「我们比别人更严格地对待多重检验」为卖点的论文，在自己的确认集上没有写 FWER 方案，是显眼的漏洞；而且 H1 与 H3 高度相关（同一批日 IC），朴素 Bonferroni 会过度保守，需要明说用哪种（fixed-sequence / Holm / 相关性感知）。
- **Suggestion**：在 v4 里写死一条固定序贯链：H1（全池 IC，α=0.05）→ H3（vs 树，α=0.05）→ H4（合成，α=0.05）；H2、H5、E 端均为估计交付、不进 α 预算。k=2 的配置维度按已有的联合功效模拟处理并在论文中给出「至少一个配置成功」的定义。
- **Severity: Major**；**Confidence: 4**

### W12｜H3 的功效算术偏乐观（两处）
- **Problem**：§4 写「预期差 0.013、配对 SE 约 0.003 → t≈5，A 层」。两个问题：（i）0.013 = 0.0192 − 0.0063，其中 0.0192 是**未扣选择偏差**的开发折读数；按 §8 自己的 20–28% 折扣，预期差应为约 0.0144 − 0.0063 = 0.008 → t ≈ 2.7；（ii）H3 的基线是**扩展窗口**树模型，到 fold35 它有约 15 年训练数据，而 0.00628 那个数来自 3 年滚动窗口的树，扩展窗口基线只会**更强**。
- **Evidence Anchor**：`section: §4` H3 行（`:44`）；`file: experiments/ledger.md:186`（XGBoost 七折 `+0.006280`，t=0.688，4/7 正）；`file: HANDOFF.md:39`（零样本 `+0.0192`）；`section: §8`（`:78`，选择偏差 20–28%）
- **Why it matters**：t≈5 与 t≈2.7 是「稳过」与「险过」的区别；后者需要更认真的 SESOI 与多重性处理（W2、W11）。另外 §4 写「树模型超参须在开发折或按文献先定」——若改按文献固定，0.00628 就不再是该估计量的期望值，功效计算的输入也要换。
- **Suggestion**：H3 的功效重算用扣除选择偏差后的期望差，并明确扩展窗口基线的超参来源（文献固定优于开发折调参，与 CLAUDE.md §一.2 一致）。若重算后 t≈2.7，仍是 A 层，但要写清余量不大。
- **Severity: Major**；**Confidence: 4**

### W13｜E1–E7 集体不满足项目自己的功效关，且 E1 用了低功效的设计
- **Problem**：§5 写「各自预注册判据后再跑」，但表里没有任何一项给出 SESOI/MDE/目标功效；E2 直接被标为「A 层」而没有 MDE。更具体地：E1「IC 对采样数（5/10/20/40）曲线」本质上是一个**臂间比较**，期望效应 +5~11%，正落在项目实测的「B 层不可回答」区间（分辨 20% 需 124 年）。
- **Evidence Anchor**：`section: §5`（`:53-61`，7 行全无 SESOI/MDE 栏；E2 行写「A 层」）；`file: CLAUDE.md §二「判据还必须先过功效关」`；`file: CLAUDE.md §二 B 层表`（分辨 50%/30%/20% 需 20/55/124 年）；`file: experiments/ledger.md:195`（sample_count 5→20 四读数均值 +6.0%、3/4 正）
- **Why it matters**：§7 把功效关当成贡献卖，§5 却有七个不过功效关的实验。审稿人会直接引用作者自己的规则来打作者。
- **Suggestion**：
  - **E1 改设计**：不要用 IC 做终点。用**无标签的 test–retest 信度**——项目已经有现成做法：同模型同窗口只改 batch size 重打两次，秩相关 0.727（`ledger.md:82`）。在每个 sample_count 上重复两次打分算信度，Spearman-Brown 直接给出「信号 vs 采样噪声」的分解，样本量是全部 (股, 日) 对而不是 7 个折，功效高几个数量级，且**不用任何标签、不消耗任何折**。这才是「TSFM alpha 里信号 vs 采样噪声的首次拆分」应有的做法。
  - **E2 加终点定义**：「是否校准」要指定是 PIT-uniformity / CRPS / 还是「实现波动对预测离差的回归斜率」，并给 SESOI。注意 sample_count=5 时路径离差本身是 5 个样本的 SD，噪声极大，必须先做信度修正。**并且 E2 可以被改造成污染探针**（见下），那会让它从「稀释」变成「加强」。
  - **E3 有一个设计冲突必须先解决**：探针训练用 10/20 年，必然覆盖 2005–2020，即确认期的收益进入了一个会被报告的模型。这不违反「不读折 05–35 验证窗」这个不变量，但会让「折 05–35 完全未碰」这句话在字面上不再成立。请把不变量精确表述为「no validation-window read of folds 05–35」，并在论文里明说 E3 用到了该期的训练侧数据；或者把长窗限制在 2000–2004 + 2021–2023 之外不取（但那样就做不出 20 年曲线）。这是一扇单向门，值得在跑之前登记。
  - **E4/E5/E6 收进方法节或附录**；**E7 保留但重新定位**：日本/英国/港股更可能在 Kronos 语料内（45 交易所），所以 E7 买的是**样本**不是**洁净度**，论文不得暗示后者。
- **Severity: Major**；**Confidence: 4**

### W14｜「可部署」这个词现在承担不起，practitioner 不会买账
- **Problem**：标题与摘要用 deployable 作为主张的名词。手上的数：夏普 0.72、日收益 NW t=1.32、最大回撤 −12.3%、最长水下 359 个交易日、年单边换手 48×、breakeven 21.9bp（扣选择偏差后 16.4bp）、部署线 C ≤ 10.4bp（FT）/ 4.2bp（ZS）、成本一次都没实测过。
- **Evidence Anchor**：`file: HANDOFF.md:41-44`（夏普 0.72 / 日 t +1.32 / BE 21.9bp）；`file: HANDOFF.md:364`（每日一边换手 0.18969 ⇒ 年 ≈ 47.8×）；`file: experiments/ledger.md`（2026-09-03 design-revision：`go ⟺ C ≤ BE_dev×0.75 − 6bp` → FT ≤ 10.4bp、ZS ≤ 4.2bp）；`file: HANDOFF.md:787`（`text: "年单边 48× ≈ 月换手 400%，是生存线的 8 倍"`，对照 Novy-Marx–Velikov）
- **Why it matters**：交易台读者看到「月换手 400%、BE 22bp、3.3 年样本、成本未测」，第一反应是「这活不活得下来完全由执行决定」。用 deployable 作主张会让整篇被归到「又一个没上过线的回测」。而作者自己在 `docs/思路整理` 里已经写出了正确的定位：**「这是一个 alpha 分量，不是一个产品」**——这句话如果进摘要，practitioner 的信任度会**上升**。
- **Suggestion**：把 deployable 从主张降为**问题**（标题里保持问号），并在摘要里明确写出定位句。同时把 §3 贡献 2 的「跨年代稳定」措辞收紧：项目自己最强的发现之一恰恰是**不稳定**（A.1：2003–04 毛超额 +0.83%/年 vs 近代 +10.48%；流动性谱迁移 DiD CI 不含 0）。诚实的表述是「IC 层面可能跨年代存在，变现层面明确不稳定」。
- **Severity: Major**；**Confidence: 5**
- **哪一个结果会改变 practitioner 的判断（正反两向）**：不是确认集，是**成本小试实测的 C 与其按日聚类的 SE，对上扣掉选择偏差后的 breakeven（FT 16.4bp / ZS 10.2bp）**。理由三条：(i) 其余每一个量都已经知道是「正但边际」，只有 C 能一票否决；(ii) C 无法从历史推断——EDGE 在现代 top500 上失效（55.2% 的 s2 为负，`HANDOFF.md:285`），TAQ 拿不到；(iii) 它是**唯一产生统计资本而不是消耗它的实验**（`HANDOFF.md:176`）。若 C ≲ 3bp，「一个 alpha 分量」的定位可辩护；若 C ≳ 8bp，ZS 臂当场出局、FT 臂只剩很薄的余量，论文应改写成「TSFM 信号存在但在零售/小基金执行条件下不可变现」——那同样是一篇好论文，而且更少见。

### W15｜数据授权是论文级风险，不只是工程风险
- **Problem**：CRSP 经学校 WRDS 获取，用于「学术研究与学校实盘项目」，授权口径**待书面确认**；JKP 因子是 CC BY-NC（禁商用）。而论文最独特的实证部分——真钱成本实测——正是由 CRSP 派生的信号驱动下单的。
- **Evidence Anchor**：`section: §6`（`:65`）、`section: §8`（`:84`）；`file: HANDOFF.md:795`（`text: "WRDS 使用条款与各校政策写明数据不得用于非学术或商业用途……本项目的 CRSP 快照本身就是经 WRDS 下载的"`）；`file: HANDOFF.md:61`（JKP `CC BY-NC 禁商用`）
- **Why it matters**：（i）多数目标渠道要求 data availability / compliance statement；（ii）若授权不覆盖实盘，论文最核心的差异化结果可能必须撤下；（iii）即使覆盖，**外部读者无法复现**（CRSP 是付费数据，真钱执行更不可复现），审稿人会要求一个 fallback。
- **Suggestion**：（a）书面确认前不要写「real money」进摘要，写「a pre-registered real-money cost protocol」；（b）**把小试的下单信号改由一份交易许可的行情源（Alpaca 自身数据 / Polygon）重算，并报告 CRSP→vendor 的订单一致率**——这一步同时解决了 §11.4a 里那个悬着的口径问题「CRSP `DlyOpen` 是否等于主上市所官方开盘价」，一举两得；（c）论文附一份可复现的 vendor-only 复算路径，让不持有 CRSP 的读者能重跑至少信号层。
- **Severity: Major**；**Confidence: 4**

### W16｜§7 的「AI 辅助研究失误谱」：内容是资产，包装是负债
- **Problem**：§7 把「四类错误的实测损失」定位为「AI 辅助研究的失误谱」案例。四类错误本身（前视贡献 37% 表观增益、评估窗选参 +0.0070、无一致性门槛、执行时点早一日）是**有量化损失的方法学证据**，价值很高；但绑上「AI 辅助」这个标签有两个副作用：(i) 审稿人会转而质疑整篇结果的可靠性（「既然 AI 犯了四次错，还有多少没抓到？」）；(ii) 它把一篇金融/ML 论文变成半篇 meta-science 论文，稀释主线。
- **Evidence Anchor**：`section: §7`（`:73`）；`file: CLAUDE.md §一`（四类错误各附实例与损失）；`file: HANDOFF.md:626-630`（规则写下数日内即出现一次自记违规 K15-B）
- **Why it matters**：`HANDOFF.md:626` 那条自记违规是把双刃剑：它证明记录是真诚的，也证明**纪律本身失败过**。若 §7 的框架是「我们有一套能防住这些错误的纪律」，这条会被反将一军。
- **Suggestion**：改成 field-agnostic 的框架——「a measured taxonomy of low-SNR empirical errors, with the apparent-gain each contributed」，与 Menkveld 的 non-standard errors 并置（后者测的是**跨团队**发散，本文测的是**同一团队内部**的发散，这个对照点很漂亮）。AI 辅助只作一句话方法披露 + 一个脚注。并且**主动报告纪律的失败次数**（至少 1 次自记违规），因为那是 taxonomy 的一部分而不是它的反例。
- **Severity: Minor→Major（取决于包装）**；**Confidence: 3**

---

## 跨学科借用机会（每项含可行性 + 能定下什么）

### A. 污染 / 记忆探针清单（全部不消耗封存折）

| # | 探针 | 做法 | 可行性 | 能定下什么 |
|---|---|---|---|---|
| **P1** | **输入结构论证**（零算力，**最高性价比**） | 直接从代码论证：逐窗 z-score + clip（`third_party/kronos/model/kronos.py:544-547`）抹掉绝对价格/成交量水平；时间戳只含 minute/hour/weekday/day/month（`src/kronos_ft/infer.py:35-43`），**无年份**；无 ticker embedding。⇒ 污染只能走「形状识别」，不能走「查这只票这天」 | 立刻，零成本 | **排除最致命的朴素版本**（verbatim lookup）。这一段应该出现在污染节的第一段 |
| **P2** | **容量算术**（零算力） | 24.7M 参数 / 1.2×10¹⁰ K 线 ≈ 485 records/param；CRSP 日线全样本约 5×10⁷ bar ≲ 语料的 0.5%，而语料含 7 种粒度、以分钟为主 | 立刻（bar 数需作者核） | 给 episodic memorization 一个数量级上界 |
| **P3** | **无标签 NLL / loss-gap membership 探针** | Kronos 是自回归 token 模型 ⇒ 任意输入窗的 per-token NLL 一次前向即得，**不需要标签、不需要采样**。对比：in-corpus 窗（2005–2020 / 2020–2023）vs post-corpus 窗（2024-07 起），按已实现波动 / 截面离散度 / 价格水平做匹配；再按 TSFMAudit 的建议做 reference debiasing（同语料参考 = Kronos-base；异语料参考 = Chronos / Moirai / TimesFM） | 高（只有前向；25M/102M 参数；干净窗只用**输入**、不读收益，不消耗评估预算） | 给污染通道一个**数**而不是一句修辞。注意 TSFMAudit 自报 raw-loss 类基线很弱（MCC 0.04–0.06），所以只能当**上界**报 |
| **P4** | **TSFMAudit 式 probe-adaptation dynamics**（最贴文献） | 固定轮数微调，记录逐轮 loss 下降 d_t 与参数位移 ‖Δθ‖=w_t，算 a_t=d_t/w_t；in-corpus 训练窗 vs post-corpus 训练窗；参考模型去偏；FP-0 阈值校准 | 高——**你们本来就在每折跑这个微调**，新增产物只有 ‖Δθ‖ 与逐轮 loss（inner loss 已在 `outputs/sealed_confirm/_logs/`；若要用封存折的日志需另行取得读取授权，最干净的做法是在开发折 + 干净窗上重跑） | 一个与 2026 年文献对齐的污染分数 + 阈值判定。这是**性价比最高的正面证据** |
| **P5** | **时间戳消融** | 重打 2 个开发折，(a) 打乱 day/month（保 weekday），(b) 全部置常数。若 IC 不变，calendar 通道确认为空 | 高（约 1 GPU-hour；注意 RNG 口径要配对，改 batch/stamp 会重排采样路径，须与同口径对照比） | 直接证伪「模型靠日历定位历史」这条路径 |
| **P6** | **provenance / 存续性切分**（纯 CPU，只用已消耗的开发折） | 按「该 (名字, 年代) 是否可能出现在一份 2024 年拉取的现代 vendor 快照里」切分：2024-06 前已退市 / 换过 ticker 的名字 vs 存续名字。若污染走的是现代免费源（Yahoo/Polygon 型，通常不保留退市历史），优势应集中在存续组 | 中（CRSP 有退市标记；top500 内退市多为并购，功效中等；**必须先写判据**） | 检验 provenance 通道；且它是唯一能间接回答「Kronos 的美股切片到底来自哪里」的廉价实验 |
| **P7** | **扰动 / 时移稳健性** | 价格加小噪声、窗口前后平移 1–3 个交易日，测 IC 衰减曲线；与 post-corpus 窗上的同一条曲线对比 | 高，但**无干净零假设**，只能作为 P3/P4 的旁证 | 弱证据，放附录 |
| **P8** | ~~从零训练同架构模型作对照~~ | — | **不建议** | 24.7M 参数在 ~10⁷ 量级 bar 上训练 vs 1.2×10¹⁰，数据量差 3 个数量级；任何差距都无法归因于污染。这个对照会主动制造一个不可解释的结果 |
| **P9** | **前向真钱期同时记录 forward IC**（成本小试期间） | 小试的估计量是 fill 分布不是 P&L（这点是对的），但**同期的 forward IC / 组合超额可以额外记录为估计交付** | 极高——反正在交易 | 这段数据是**post-corpus + post-preregistration**，是全文最干净的证据，白拿不要浪费。须在小试协议里预注册为「估计交付、不作判据、不得用于停机」 |

**我对「1 年干净窗 + 证据不对称够不够」的直接回答**：作为 ML 审稿人——**不够**，但缺的不是数据量，是**正面探针**。P1+P2（零成本）+ P3 或 P4（一天量级）就能把「污染是否可能」从信仰变成一个带误差棒的量。加上 W8 建议的终点切换（t 从 0.65 到约 1.3）与 P9（前向白拿的干净样本），污染这一节就从全文最弱变成全文最有辨识度的一节。

### B. 从其他学科借的协议工具

| # | 借用 | 来源 | 可行性 | 能定下什么 |
|---|---|---|---|---|
| **B1** | **用「实测的 N」算 Deflated Sharpe / haircut** | Bailey & López de Prado 2014（DSR、minimum backtest length）；Harvey & Liu 2020 | **极高，CPU 几分钟**：`ledger.md` 里 `eval` 81 条 + `ablation-read` 51 条，试验次数是**可数的** | 这是本文能拿到的最独特的一句话：**几乎没有任何回测论文能给出真实的 N，因为没人记账**。把 §8 的「选择偏差估 20–28%」从粗估变成有定理支撑的 haircut，同时直接修掉 W4 |
| **B2** | **Thresholdout / Ladder：给开发折的继续使用一个预算** | Dwork et al. 2015 (Science, reusable holdout)；Blum & Hardt 2015 (The Ladder) | **极高，纯记账**：只需在 ledger 里加「查询计数 + 噪声阈值」两个字段 | 小试等待期还要在开发折上跑信号#2 读数、构造对拍、K9 补报夏普——这些查询目前不受任何界的约束。加一条 Thresholdout 规则（只有当开发折读数与训练折估计相差超过噪声阈值时才「花掉」一次预算），能把继续使用开发折的偏差**变成有界的**，而不是「方向性证据」这种定性说法 |
| **B3** | **第三方时间戳 + sealed envelope** | Registered Reports（Nosek 等）、OSF / AsPredicted、RFC-3161 / OpenTimestamps | **极高，几小时** | 修掉 W10：让「预注册早于解封」可被外部验证，而不是自述。这是本文最核心主张的**证据等级问题**，不修等于零 |
| **B4** | **Holdout access log 作为可发布 artifact** | ML benchmark governance（Kaggle 私榜、NeurIPS D&B track 的 dataset documentation） | 高——已建 80%（`SEALED_MANIFEST.json` + 纪律测试） | 把「谁在什么时候读了封存集的什么」变成论文附件里的一张表。这可能是本文最容易被后人复用的东西 |
| **B5** | **Registered Report Stage-1 投稿** | Nosek et al.；心理学/医学的成熟做法 | 中（取决于目标渠道是否有该 track；ICAIF/JFDS 需确认） | 若能拿到 Stage-1 in-principle acceptance，本文对 publication bias 的免疫力就从自述变成制度事实——这比任何统计修正都强 |
| **B6** | **Specification curve / multiverse（只在开发折上，回溯装配）** | Simonsohn-Simmons-Nelson；与 Menkveld 的 non-standard errors 对偶 | **高且几乎免费**——曲线上的点大部分已经在 ledger 里躺着（lookback 三档、e1/e30、ZS/FT、合成、universe、持有期、执行时点） | §7 现在的说法是「我们**不报**配置扫描」并给了正面论证。这个论证是对的（B 层），但读者会读成「藏了」。**回溯装配一条描述性的 specification curve** 能把拒绝变成结果：「所有配置的读数分布是这样的，而我们的 MDE 是这么大，所以这条曲线上的排序不可回答」——这既回答了 Menkveld，又不花确认集预算 |
| **B7** | **Small telescopes 作为 SESOI 的非循环锚** | Simonsohn 2015 | 高（作者已经算出 `d_33%`=0.00865，只差「非零零假设下的功效」这一步） | 直接修掉 W2：给全池 IC 一个不是从自己点估计推出来的 SESOI |
| **B8** | **干净窗改用 non-inferiority / equivalence（TOST）框架** | 生物统计的非劣效性检验 | 高，纯统计写法 | 在 t≈1.3 的窗口上，「IC > 0 显著吗」几乎注定失败且无信息；但「干净窗的 IC 与开发折效应**不相容**吗」是可以有功效地回答的。预先声明一个等效边界（例如 IC 的 50%），把干净窗从「无用的确认」变成「有用的证伪机会」 |
| **B9** | ~~canary / 构造性 held-out~~ | LLM 污染文献 | **不可行**——语料是别人的，事后无法植入 canary | 记录为「本文无法使用的方法」，本身也是一句有价值的方法学论述 |

---

## 给作者的问题

**Q1（污染的事实前提）**：Kronos 的预训练语料里到底**有没有**本项目 universe 里的 CRSP 名字？什么频率？来自哪个 vendor？我核了 Kronos 论文与 HuggingFace 卡片：语料截止 2024-06 是明写的，但「45 exchanges / 12B K-lines」下的美股切片、频率与 provenance 都**没有文档**（论文只在评估数据里提到 XNAS）。整个污染框架——包括 fold43 的移除、干净窗的定义、以及 §3 贡献 1 的「污染感知窗口」——都建立在这个未核事实上。如果美股日线其实**只在评估集、不在预训练集**，你的结论会怎么改？你打算怎样在论文里处理这个不可核实性（P6 是我能想到的唯一间接办法）？

**Q2（预注册的边界）**：§4 表里 4/6 假设的冻结状态是「待写入 / 待定稿 / 待裁定」。你打算在论文里如何界定「预注册」这个词？具体地：H3 与 H4 在解封之前才会冻结，它们是否会被明确标注为「晚于本稿冻结、时间戳为 X」，还是与 H2/H5 混在同一张表里？

**Q3（贡献 3 的性质）**：你的主张到底是关于 **inductive bias**（手写的泛函形式更好）还是关于 **effective sample size**（3 年 ≈ 750 个有效独立日估不出来，12B K 线能）？如果是后者——而 `ledger.md:178` 的「轴一/轴二」分解读起来就是后者——那么标题应该改，且 W6 建议的嵌套对照（在解码输出上学读出，手写读出是其特解）是唯一能干净支持它的实验。你愿意做这个对照吗？

**Q4（干净窗的终点与时机）**：为什么干净窗的功效用夏普口径（t≈0.65）而不是全池 IC 口径报？按你们自己 §11.7 的推理，IC 端不受 σ_cs 影响，我粗算 1 年约 t≈1.3、2 年约 t≈1.9。既然干净窗每年白送 2 折而其他所有证据都已封顶，把最终读取推迟到覆盖**两年**干净窗的 CRSP 快照，是不是本项目投入产出比最高的一步？

---

## 故事 / 标题（我的视角）

**我认为真正的 title question 应该是**：
> **当评估期整段落在预训练语料内、开发折已被消耗数十次时，我们还能对一个金融 TSFM 的横截面 alpha 做出什么样的可辩护主张？**

理由：这是本文**已经**回答了一大半的问题，也是它相对同行唯一不可替代的位置。「有没有可部署的 alpha」这个问题，在 C 未测、干净窗 1 年、§4 四项未冻结的状态下，本文答不了，而且很可能永远只能答「一个 alpha 分量，不是产品」。

**两个替代 framing（各一句）**：
- **ML-facing**：*"A label-free contamination audit for time-series foundation models: how to bound what a pretrained forecaster could have memorized about your evaluation period without spending any of your evaluation budget."*
- **Finance-facing**：*"Alpha component or product? A pre-registered, sealed-confirmation, real-cost test of a foundation-model price signal at 48× annual turnover and a 22bp breakeven."*

（若最终两条线都想要，我的建议是**拆成两篇**：污染审计 + 封存协议投 ML 的 evaluation/benchmark 场子，alpha + 成本实测投 ICAIF/JFDS。现在这一篇同时对两个读者群说话，结果是对哪一边都不够深。）

---

## Minor issues

1. **引文数字未核**：§2 表把「预训练语料与回测期重叠会使误差被低估 **8–29pp**」归给 TSFMAudit（arXiv 2605.26161）与 arXiv 2510.13654。我读了两篇的 HTML：2510.13654 报的是 COVID 冲击下时间重叠序列带来约 **37% 的 MAE 改善**、以及**只有 6% 的数据集从未被用于预训练/微调**；TSFMAudit 报的是检测性能（MCC 0.125 / macro-F1 0.521 / balanced acc 0.603），两篇都没有出现「0.1% 重叠 → 误差低估 8–29pp」。请回原文核对出处；若查不到，按计划书自己的规则标「未核」或删。`file: HANDOFF.md:131-134` 里同一句话也要一并修。
2. **TSFMAudit 的力度要如实报**：它自报的 MCC 只有 0.125。若论文引用它来支撑「污染很严重」，会被审稿人指出该文自己的检测器很弱。正确用法是引用它的**方法**（adaptation dynamics + reference debiasing + FP-0 校准），不是引用它作为「污染幅度」的证据。
3. **「打赢一个量级」措辞过强**：零样本 0.0192 vs XGBoost 0.00628 是 3.1×，不是一个量级（对 LightGBM 0.0024 才接近）。`file: HANDOFF.md:48-49`、`file: experiments/ledger.md:186`。论文里请用倍数。
4. **§4 主终点是 RankIC，而 CLAUDE.md §五 把 RankIC 降级为「诊断、不作决策依据」**（`file: CLAUDE.md §五`；`file: experiments/ledger.md:171`，corr(IC, 毛价差)=0.873，fold41 IC 为正而毛价差为零）。K15（`HANDOFF.md:606-610`）给了一个很好的解决：IC 是**科学泛化**终点、钱是**经济**终点，且两者的差被 σ_cs 机械地解释。这段论证必须进论文，否则 finance 审稿人会直接引用你们自己的 §五。
5. **seed 方差从未估计**（`file: experiments/ledger.md:405`；`section: §8` 已诚实写出）。建议至少给一个便宜的替代上界：现有的「同模型换 batch 重打分」信度 0.727–0.758（`ledger.md:82`）可以作为**打分随机性**的界，但它不覆盖**训练随机性**；论文里要把这两种方差分开讲，别让读者以为 0.75 已经涵盖了 seed。
6. **`sample_count` 口径分裂**：实盘打分拟用 20，封存确认队列固定 5，且两口径读数不得并列比较（`file: CLAUDE.md §八`、§二例外条款）。论文必须写清「报告的所有确认集读数是 sc=5 口径，而部署口径是 sc=20，后者按理论（纯方差削减）采纳、未做臂间检验」——否则会被读成偷换。
7. **§6 的持仓期写「6 → 5 日（解析驱动修订，待登记）」**，而 §4/§3 的所有读数都是 NT=6 口径。论文里必须交代清楚哪些数属于哪个构造，且按 `ledger.md`（2026-09-03）的规定，新构造在 36–42 的基准读数**不得**与 NT=6 并列比较（B 层）。现在计划书没有区分。
8. **§5 E7 的措辞**：日/英/港股很可能同样在 Kronos 的 45 交易所语料内，所以 E7「唯一能成倍加证据的实验」这句话只在**样本量**意义上成立，在**洁净度**意义上不成立。请加限定。
9. **§10 建议补一句关于 artifact 的计划**：`sealed.py` + manifest + 纪律测试 + ledger 是本文最可能被复用的东西，应明确承诺开源（并注意其中不得含 CRSP 数据）。
10. **摘要时态**：现摘要用现在时描述尚未发生的成本实测（"measure execution cost with real money"），末尾又写 `[Confirmation and cost results to be filled.]`。定稿前统一为「协议已预注册、结果待填」的措辞，避免读者误以为已完成。
11. **§3 贡献 2 的「跨年代稳定」与项目最强的负面证据直接冲突**（A.1：2003–04 毛 +0.83%/年 vs 近代 +10.48%，`HANDOFF.md:94-96`）。§11 的「机制 = 流动性谱迁移、远端成因未解释」（`HANDOFF.md:290-292`）应当进摘要——「未解释」比「稳定」更诚实也更有意思。
12. **一处措辞建议**：§3 贡献 1 里的「污染感知的证据不对称」若要作为方法贡献，需要给出**它的适用条件**（何时一次通过才算弱证据？弱到什么程度？）。否则它是一句无法证伪的先验声明。P3/P4 的量化探针恰好能给它一个刻度。

---

*本报告只读不写，未触碰 `F:\quant` 下任何文件；未打开任何路径含 `sealed` 的目录，未读取任何 `scores.parquet`。*
