# Reviewer 0 — Journal-Fit Seat 审稿报告

**稿件**：`研究计划书_20260903.md`（v0.1 草案，中文正文 + 150 词英文摘要）
**审稿席位**：Journal-Fit Reviewer（5 席模拟评审的第 0 席）
**身份设定**：ACM ICAIF 程序委员会资深委员 / area chair，ML + 实证金融双背景，长期处理 ICAIF 与 NeurIPS finance workshop 投稿，熟悉这些会议与 *Journal of Financial Data Science* / *Quantitative Finance* 的口味差异。
**审稿焦点**：故事性（一篇还是几篇）、对 ICAIF 的 fit 与创新性、题目/摘要与正文的精准性、投稿就绪度与档期、摘要质量。
**明确不做**：统计推断的正确性（属 Reviewer 1）。本报告中凡涉及功效/标准误的地方，只在「题目是否兑现」的意义上引用，不复核其算法。
**利益披露**：本席位可读仓库 `F:\quant\us-quant-pipeline` 的 `HANDOFF.md` / `CLAUDE.md` / `experiments/*` / `docs/*`，用于核对稿件断言是否有已落盘的证据；**未读取任何 `sealed` 路径、未计算任何 `scores.parquet`**。

---

## Recommendation

**Major Revision**（若按现状以 ICAIF 主会 8 页论文投出：**Reject**；本稿的问题不是质量，是**装配**——材料够两篇好论文，现在压在一篇里、并挂了一个设计明确拒绝回答的题目）

**Confidence: 4 / 5**

降一分的原因：确认集折 05–35 未读，稿件最终的实证形状我看不到；我评的是**设计与叙事**，不是结果。

---

## Summary Assessment（约 230 词）

这是我在 ICAIF 语境下**很少见到的一类稿件：方法学纪律远超投稿均值，而叙事装配远低于投稿均值**。作者手上有五件东西同时齐备——预注册、封存确认集、污染感知窗口、**量化并披露**的选择偏差、以及真钱成本口径——TSFM 用于资产定价的文献里确实没有第二篇这样做的（§2 的缺口陈述我逐行核过，站得住）。更值钱的是这些纪律**产出了硬数**而非姿态：选择偏差实测 +0.007（信号本身的三分之一）、臂间比较的淘汰门槛超过信号本身因而永不触发、分辨 20% 的臂间差需 124 年。这些是对 ICAIF 全场臂间比较泛滥的直接批评工具，我会想让 PC 看到。

但稿件把这些和另外两篇论文捆在了一起：一篇是「Kronos 零样本有没有横截面 alpha」的实证论文，一篇是「手写生成式读出 vs 学出读出」的 ML 论文，外加 7 项探索与 1 个成本小试。**ICAIF 读者会找不到脊柱。** 更要紧的是题目问的是 *deployable*，而 §11 与仓库登记簿（2026-09-02 DECISION）都明确把经济可投性降为**估计交付、不作确认**——题目问了一个设计已经决定不回答的问题。题目的 *zero-shot* 与 §4 的 k=2（含微调臂）也不一致，且摘要里的头条数字（RankIC 0.021 / t 3.6）经核对属于**微调臂**，零样本是 0.0192。

修法不是加实验，是**拆稿 + 改题**。拆开后，第一篇现在就能投，第二篇的自然档期是 ICAIF 2027。

---

## Strengths

**S1｜五项方法学要素的组合在本文献里确有空白，且缺口陈述可核。**
`section: §2` 的六行对照表我逐条核对了指向：Rahimikia 等 2025 确为「通用 TSFM 零样本不可用、金融原生预训练才有效」的三档结论（仓库 `HANDOFF.md` §2 独立复述并与本项目 XGBoost +0.00628 对齐）；污染文献（TSFMAudit 一类）确实只提问题不给协议；Chen-Hanauer-Kalsbach / Menkveld 非标准误差确实没人在 TSFM 上量化自己的选择偏差。**这不是常见的「我们是第一个」空话，缺口是具体的、可证伪的。**

**S2｜选择偏差 +0.007 是本稿最有转载价值的一个数，且它是硬结果不是态度。**
`text: "内层选参均值 +0.01256 vs 事后最优上界 +0.01951，差 +0.0070"`（`experiments/ledger.md` 第 164 行）。同一处还记录了它是**当天修完 bug 又重犯**的产物。ICAIF 每年有大量论文在评估窗上选超参而不自知；一个**同管线、同数据、同作者**测出的「这么做能白拿信号的三分之一」是我见过最有说服力的说明方式。这一条单独就能撑一节。

**S3｜A/B 层把「不可回答」变成了可算的量，这是对 ICAIF 的直接方法学供给。**
`section: §3 贡献 1 + §7`；原始表在 `CLAUDE.md` §二「比较分层」：正交臂 99% 淘汰门槛为信号本身的 1.10–1.51 倍（**永不触发**），分辨 50%/30%/20% 的相对臂间差需 **20 / 55 / 124 年**。ICAIF 投稿里「我们的方法比 baseline 高 8%」几乎从不附样本量论证；把这个换算表摆出来是有攻击性的、也是建设性的贡献。

**S4｜流动性迁移是真正的机制发现，而且是本项目少数功效充足的 A 层读数。**
`text: "2003–04 长在不流动端、2020 年代长在流动端，DiD 95% CI [+0.0034, +0.0191]"`（§3 贡献 2；仓库 `HANDOFF.md` §10.4 K7b、§10.5 硬结论 1 独立佐证）。它同时推翻了三个更平庸的解释（信号消失 / 因子倾斜 / 基座不适配早期）。**这是全稿唯一一个「读者会记住并复述」的发现**，而它现在被压在贡献 2 的后半句里。

**S5｜诚实边界写在明处，审稿人不需要挖。**
`section: §8 + §11`。seed 方差未估计、干净窗只有 1 年且预期 t≈0.65、成本只在一个 AUM 上测得不得外推、不主张多空可行、隔夜/日内信号对 2005–2020 不是样本外——这七条里有四条是我本来准备当作 weakness 提的。**作者先说了，这在评审经济学上是纯收益。**

**S6｜读出比较的「退位史」本身就是论文的方法学论据，故事自洽。**
`text: "线性探针 +0.0278 超过微调 +0.0265 的读数作废——该数是在外层验证窗上挑 alpha 得到的"`（`ledger.md` 第 164 行）。手写读出赢，**且赢的过程包含一次它先输后赢的纠错**——贡献 1（协议）与贡献 3（读出）在这里天然咬合。这是拆稿后第一篇的最佳收尾。

**S7｜成本侧的工程锚比 ICAIF 同类交易论文深一个层次。**
`section: §6 + 仓库 HANDOFF.md §11.4a`：Alpaca 的 `time_in_force=opg` **只对收费的 Elite Smart Router 开放**（零佣金对本策略不成立）、Elite All-In $0.0040/股、SEC Section 31 仅卖出 0.206bp、OPG 提交窗口 9:28am ET 截止、以及「按交易日聚类而非按笔数」的有效样本量论证。ICAIF 绝大多数交易论文在这一层是手挥（"we assume 10bps"）。**这是本稿对该会众最实用的一段，但它现在只占 §6 的两行。**

---

## Weaknesses

### W1｜题目问的是设计已经决定不回答的问题
- **Problem**：题目的操作性形容词是 *deployable*，但正文与登记簿都拒绝主张可投性。
- **Evidence Anchor**：`text: "Does a Financial Time-Series Foundation Model Carry Deployable Cross-Sectional Alpha?"`（§0）对 `text: "不主张跨年代经济可投性（E 端为估计）"`（§11）；外部佐证 `experiments/ledger.md` 2026-09-02 DECISION：`text: "历史确认集 E 端降为估计交付 / 跨年代压力测试，不作现代可投性的确认检验"`。
- **Why it matters**：审稿人读完题目后在 §11 撞上一句「本文不主张这个」，第一反应是 over-claiming，第二反应是不信任其余部分。这在 ICAIF 是常见的 reject 触发器：**题目与 limitation 节互相打脸比结果不显著更致命**。而且这里冤枉——作者的克制是优点，题目却把它写成了缺点。
- **Suggestion**：把 *deployable* 换成设计真正回答的问题。两个可用替代：(a) 若走协议论文——*How Much Alpha Is Design Freedom? A Pre-Registered, Sealed Protocol for Evaluating Financial Time-Series Foundation Models*；(b) 若走实证论文——*Cross-Sectional Alpha from a Financially-Pretrained Time-Series Model: A Pre-Registered Test and Its Liquidity-Regime Mechanism*。**"deployable" 只有在成本小试真的跑完、且确认集给出正号之后才能进题目。**
- **Severity: Critical｜Confidence: 5**

### W2｜"zero-shot" 是题目和摘要的主语，但头条数字是微调臂的
- **Problem**：题目/摘要以零样本立论；§4 的 H1 却是 `text: "零样本/微调 Kronos 的横截面信号 > 0"`（k=2 双臂）；而摘要引用的 `RankIC 0.021, NW t 3.6` 经核对属**微调 lb90 臂**，零样本七折均值是 **+0.01919**。
- **Evidence Anchor**：`ledger.md` 第 162 行 `text: "七折均值：零样本 +0.01919，微调 +0.02071，增量 +0.00152（+7.9%）"`；`HANDOFF.md` §2 表首行 `逐日 RankIC +0.0207（NW t=3.63）` 与次行 `零样本 RankIC +0.0192`。**稿件内部亦不自洽**：§3 贡献 2 把 0.021 归给零样本，§3 贡献 3 又把 0.019 归给零样本，同一页两个数。
- **Why it matters**：这是一个审稿人五分钟内就能查出的算术不一致，且方向对作者有利。更严重的是**经济数字的差距远大于 IC**：微调臂毛年化 +10.48% / BE 21.9bp，零样本臂 **+4.78% / BE 13.57bp**（`HANDOFF.md` §9）。一篇以 zero-shot 为题的论文，如果 abstract 的经济学是 fine-tuned 臂的，这不是笔误，是 framing 错误。
- **Suggestion**：三选一，**必须在投稿前定死**：(a) 题目改 zero-shot，则摘要与所有头条数字一律用 ZS 口径（0.0192 / 4.78% / BE 13.57bp），微调臂降为一节稳健性；(b) 题目去掉 zero-shot（如 *a financially-pretrained TSFM*），双臂并列报，明确 k=2 是预注册的双臂族；(c) 拆稿后第一篇（协议）根本不需要押一个臂，把 ZS/FT 的不可分辨性**当作 B 层的展示案例**。我推荐 (c)，其次 (a)。
- **Severity: Critical｜Confidence: 5**

### W3｜一份稿子里装着两到三篇论文，ICAIF 读者找不到脊柱
- **Problem**：§3 列三个贡献（协议方法 / 实证信号 + 流动性迁移 / 读出比较），§5 再列 7 项探索性研究，§4 另有成本小试 C。三个贡献的**读者不同、可交付时间不同、甚至最佳投稿渠道不同**。
- **Evidence Anchor**：`section: §3 + §5`；`absence: 全稿 — 期望有一句「本文的中心论断是 X，其余为支撑」；检查面：§0 题目、§1 一句话研究问题、§3 三个贡献、附录摘要 — 四处给出四个不同的重心（可部署性 / 四要素齐备 / 协议 / 零样本 alpha）。`
- **Why it matters**：ICAIF 主会 8–9 页（含参考文献）。三个贡献里任何一个写足都要 6 页。现状下最可能的结局是三个都写成两页、三个都不够深，**审稿人给「interesting but unfocused」然后 reject**。这是 ICAIF 最常见的拒稿理由之一，比方法缺陷常见得多。
- **Suggestion**：明确拆成两篇，脊柱各自单一：
  - **论文 A（现在就能写完，零依赖封存集）**：*协议 + 失误谱 + 不可回答性*。骨架 = 五要素协议（§7）→ 四类错误的实测损失（前视 37% / 选参 +0.007 / 无一致性门槛 / 执行早一日）→ A/B 层与 20/55/124 年换算 → 读出比较作为「选择偏差如何翻转结论」的完整案例（S6）→ Kronos 作 running example。**这一篇的每一个数都已落盘。**
  - **论文 B（2027）**：*实证 + 机制*。骨架 = 确认集结果 → 张成检验 → **流动性迁移 DiD 作为核心机制**（把 S4 提到 headline）→ 实测成本 → 可投性判定。
  - 7 项探索里，E1（采样噪声分解）、E2（校准）、E3（学习曲线）归论文 A 的附录或第三篇；E7（跨市场）需要没有的数据与授权，**不要写进任何投稿计划书**，它现在的作用只是让审稿人怀疑范围控制。
- **Severity: Major｜Confidence: 4**

### W4｜"Cost-Measured" 写在题目里是完成时，实际未测且被一个非技术条件阻塞
- **Problem**：题目断言 cost-measured；`section: §9` 阶段 2 才做小试，且依赖 `text: "Alpaca 书面答复、学校实盘基金、AUM"`——三个作者不完全控制的外部条件。
- **Evidence Anchor**：`section: §9 阶段 2`；`HANDOFF.md` §11.4 第 2 条 `text: "AUM 是阻塞字段，不是可选参数"`，且 `text: "只是拿来当测量工具就必须保持阻塞、小试不启动"`；§8 自认 `text: "成本只在一个 AUM 上测得，不得外推"`。
- **Why it matters**：**题目承诺了一个可能在投稿日仍不存在的测量**。若小试因学校基金未落地而不启动，论文 B 的第四支柱直接缺失，而题目已经把它写死。ICAIF 审稿人对「real money」四个字期待很高，交付不了的反噬也大。
- **Suggestion**：(1) 题目里的 cost 表述改为可兑现的强度，例如 *with an execution-cost budget derived from broker-verified fee schedules*；(2) **现在就设计一个不依赖真钱的降级交付**：BE 的点估计与 CI（21.9bp / [12.4, 31.5]，v4 修订 1 已有）+ Alpaca Elite 的确定项（全包 0.900bp，占余量 22%）+ 一个**显式标注为未测的缺口**（fill 相对 CRSP `DlyOpen` 的偏离，余量约 3.10bp）。这个三段式本身就是好内容，且完全在作者控制内；(3) 在论文里把 `CRSP DlyOpen 是否等于主上市所官方开盘价` 这个未解口径（`HANDOFF.md` §11.4a 末）写成公开问题——审稿人会欣赏。
- **Severity: Major｜Confidence: 4**

### W5｜时间线里没有 arXiv 时间戳，而读取排在成稿之前——预注册的可核性会在读取瞬间归零
- **Problem**：§10 说要先挂 arXiv 建时间戳，但 §9 的四个阶段里**没有这一步**；且阶段 3（读取折 05–35）在阶段 4（论文初稿）之前。
- **Evidence Anchor**：`text: "先挂 arXiv q-fin.ST 建立时间戳（预注册文本与登记簿哈希一并公开）"`（§10）对 `absence: §9 时间线 — 期望有「公开预注册文本 + 登记簿哈希」这一行；检查面：阶段 0/1/2/3/4 全部五行，均无。`
- **Why it matters**：**这是全稿最时间紧迫的一条，而且是纯行政成本。** 本稿的全部溢价来自「预注册」三个字。一旦折 05–35 被读取而此前没有公开、带时间戳的预注册文本，作者对审稿人就只剩「相信我，我们事先写好了」——而仓库里恰好有多条**事后修订**的记录（k=2 的 minimax-regret 修订、lookback 平局规则修订、退出线再归类），全部诚实登记，但**一个不信任的审稿人会把它们读成「预注册可以随便改」**。有外部时间戳时这些是加分（可核的修订史）；没有时间戳时它们是减分。
- **Suggestion**：把「arXiv q-fin.ST 挂预注册文本 + `ledger.md` 的 SHA-256」**移到阶段 1，硬性排在阶段 3 之前**，并在 §9 里写成一个带门的依赖（读取授权的前置条件之一 = 时间戳已公开）。同时考虑 OSF 或 AsPredicted 注册（免费、即时、金融/心理学审稿人都认），与 arXiv 并行成本近似为零。**如果只采纳本报告的一条建议，请采纳这一条。**
- **Severity: Critical（时间紧迫性），内容上 Major｜Confidence: 5**

### W6｜§4 的 H1 判据与仓库登记簿的现行裁定冲突
- **Problem**：§4 H1 写的是二元判据 `text: "全池逐日 RankIC，NW t ≥ 1.96，≥18/31 折为正"`；但登记簿 2026-09-02 的 DECISION 是 `text: "全池 IC 维持为带 CI 的区间估计，不做二元 PASS"`。
- **Evidence Anchor**：`section: §4 H1 行` 对 `experiments/ledger.md` 2026-09-02 DECISION 条（同条给出理由：树基线 0.00628 < k=2 的 MDE80 0.00834；Simonsohn d_33% = 0.00865 为唯一非循环先验锚，暂不立项）。
- **Why it matters**：计划书自称「§4 的假设具有真正的预注册地位」，那么它就是**预注册文本本身**。预注册文本与它所依据的登记簿在主终点的判据形式上不一致，是审稿人最容易抓住的自相矛盾——尤其在一篇以预注册为卖点的论文里。（判据孰是孰非属 Reviewer 1；我只指出**两份文件说的不是同一件事**。）
- **Suggestion**：投稿前做一次**单向核对**：以 `ledger.md` 的 DECISION 条为准，把 §4 六行逐行改写成与登记簿逐字一致的表述；凡登记簿已降级为估计交付的（E 端、全池 IC），§4 就不能再写 t≥1.96 的二元门。另建议在论文里附一张「预注册条目 → 登记簿条目 → 最终报告」的三列追溯表，**把修订史当作卖点而不是负债展示**。
- **Severity: Major｜Confidence: 4**（降一分：不排除有更晚的裁定我未检索到）

### W7｜对 ICAIF 读者，缺少同管线的通用 TSFM 对照与 ML 资产定价文献锚
- **Problem**：核心论断之一是「金融原生预训练才有效、通用 TSFM 不行」，但本文**没有自己跑过任何通用 TSFM**，这条完全借自 Rahimikia 等 2025。文献表也没有 ML 资产定价的标准锚。
- **Evidence Anchor**：`absence: §2 文献表与 §6 数据与设置 — 期望至少一个具名通用 TSFM（Chronos / TimesFM / Moirai / Lag-Llama / MOMENT 之一）在同折同口径上的对照读数；检查面：§2 六行对照表、§4 H1–H5、§5 E1–E7、§6 模型设置、附录摘要 — 五处均无。` 另 `absence: §2 — 期望 Gu-Kelly-Xiu 2020、Chen-Pelger-Zhu 一类 ML 资产定价基准；检查面：§2 全表 — 无。`
- **Why it matters**：ICAIF 的 PC 大约一半是 ML 背景，他们看 TSFM 论文的第一个问题就是「和 Chronos / TimesFM 比呢」；另一半是金融背景，他们的第一个问题是「和 Gu-Kelly-Xiu 的神经网络比呢」。**两个问题现在都只有一个引用作答。** 而本文已经有了强树基线（XGBoost +0.00628）——把它扩成「通用 TSFM / 树 / 金融原生 TSFM」三档，论断就从借来的变成本文的，且是 ICAIF 读者最想要的那张表。
- **Suggestion**：新增一项探索性实验（记为 E8）：在**同一 42/45 折切分、同一 26 特征、同一读出口径**下跑一个通用 TSFM 的零样本。选一个权重公开、推理便宜的（Chronos-bolt 或 TimesFM 的 small 档）。这不消耗任何确认折，GPU 成本与 E1/E2 同量级。文献侧补 Gu-Kelly-Xiu 2020 与 Chen-Pelger-Zhu 各一行到 §2 表。
- **Severity: Major（对 ICAIF fit）｜Confidence: 4**

### W8｜"US Large Caps" 既不精确，又恰好埋掉了本文最好的发现
- **Problem**：题目说 US large caps；但 §4 H1 的终点是**全池**逐日 RankIC，只有变现构造用 ADV 前 500。而 ADV top-500 是**流动性筛**不是**市值筛**。
- **Evidence Anchor**：`text: "H1 ... 全池逐日 RankIC"` 与 `text: "变现构造：只做多、ADV 前 500"`（§4、§6）；机制侧 `HANDOFF.md` §10.5 硬结论 1：信号在**流动性谱**上迁移。
- **Why it matters**：两重损失。(1) 精准性：主终点跑在全池上，题目却承诺 large caps，审稿人核对 §4 时会认为作者没读自己的设计。(2) **叙事损失更大**：本文最有意思的发现就是「信号在流动性谱上的位置随年代迁移」，而题目用 "large caps" 把流动性维度压成了一个静态的样本描述。
- **Suggestion**：题目改用 `liquid US equities (ADV top 500)` 或干脆把流动性提到题目里（见 W1 的替代题 (b)）。§4/§6 加一句显式说明：主终点 = 全池，经济终点 = ADV top 500，二者是**两个不同的宇宙**，且这个差本身是 H2 的对象。
- **Severity: Major｜Confidence: 5**

### W9｜读出比较跨口径并列，违反项目自己的规则
- **Problem**：§3 贡献 3 把 `零样本 0.019`（**七折**）与 `线性探针 0.013`（**五折 36–40**）、`MLP 0.008`（**单折 40**）并列。
- **Evidence Anchor**：`text: "手写读出（零样本 0.019）优于所有在本项目数据上学出的读出（线性探针 0.013、MLP 0.008、树 0.006）"`（§3）；对照 `ledger.md` 第 164 行：探针五折均值 +0.01256，**同五折**的生成式零样本是 **+0.02193**（不是 0.019）；第 141 行 MLP +0.00753 是 **fold40 单折**，同折零样本 +0.01499。项目规则 `CLAUDE.md` §八：`text: "不同口径的读数不得并列比较"`。
- **Why it matters**：数字方向不变（手写读出仍然赢，而且**用同口径赢得更多**：0.02193 vs 0.01256），所以这不是结论问题，是自伤问题——**一篇教别人怎么不骗自己的论文，在自己的贡献列表里犯了自己第 8 条规则**。审稿人抓到这一处会重新怀疑其余所有数字。
- **Suggestion**：全部改成同口径。建议统一到五折（36–40）：生成式零样本 0.0219 / 生成式微调 0.0254 / 线性探针 0.0126 / MLP 与树各自补齐同五折读数或明确标注折数与不可比。表格里加一列「折数 / 口径」。
- **Severity: Minor（数字）/ Major（可信度）｜Confidence: 5**

### W10｜"Sealed" 的强度大于设计实际提供的，且作者自己已经更正过
- **Problem**：题目与摘要用 *sealed*，读者会理解为技术性隔离；仓库自己的更正是这只是流程纪律。
- **Evidence Anchor**：`HANDOFF.md` §11.8：`text: "封存本质上是流程纪律，不是密码学隔离"`，以及 `text: "不写 labels 只能防止误操作，不构成隔离"`（并注明这是对此前一处夸大的必须更正）。稿件 `absence: §6/§7/§8 — 期望有一句说明封存的强度边界；检查面：§6 数据与设置、§7 方法论、§8 效度威胁 — 三处均无。`
- **Why it matters**：作者已经做了正确的自我更正，**却没把它写进论文**。如果审稿人（或后续复现者）读到公开的仓库，会发现论文声称的强度高于作者自己的记录——这比一开始就说清楚糟糕得多。
- **Suggestion**：在 §7 加一段两句话的诚实说明：封存 = 单向的流程纪律（哨兵文件 + `assert_readable` 守卫 + 禁止引用封存路径的纪律测试 + append-only 登记簿），**不是**密码学承诺；可核性来自公开的时间戳与哈希（见 W5），而非物理不可读。**这一段会加分不会减分**——它正是本文声称要示范的那种诚实。
- **Severity: Minor｜Confidence: 5**

### W11｜摘要不回答题目提的问题，且末尾带一个 `[to be filled]` 括号
- **Problem**：摘要以 "We ask whether…" 开头，全篇没有给出答案，最后是 `text: "[Confirmation and cost results to be filled.]"`。
- **Evidence Anchor**：`section: 附：英文摘要草稿`。
- **Why it matters**：任何渠道都不接受带 TBD 括号的摘要；更根本的是，**一个问了问题不回答的摘要读起来是 proposal 而不是 paper**——这恰好是本稿现在的真实状态，但一旦决定投稿，摘要必须有答案。若走论文 A（协议），答案是现成的且很强：「设计自由度值信号的三分之一」。
- **Suggestion**：见下方「摘要质量」小节，给出改写方向与一版可用骨架。
- **Severity: Major（就绪度）｜Confidence: 5**

### W12｜"beats every readout learned on our data" 把最好的故事讲小了
- **Problem**：摘要的这句是一个静态优越性断言；而实际发生的事更有意思——学出的读出**先赢后输**，赢是因为在外层验证窗上选了 alpha。
- **Evidence Anchor**：`text: "A hand-written generative readout beats every readout learned on our data"`（摘要）对 `ledger.md` 第 164 行的作废记录（见 S6）。
- **Why it matters**：现在这句话是一个所有人都会怀疑的强断言（"你调够了吗？"）。改成过程叙事后，它变成一个**没人能反驳、且论证了论文中心论点的**断言。这是本稿最容易的一处升级。
- **Suggestion**：改写为类似：*A learned linear probe appears to beat the hand-written generative readout (+0.0278 vs +0.0265 on one fold) — until hyper-parameters are selected outside the evaluation window, after which it collapses to +0.0126 against +0.0219. The +0.0070 gap is our measured price of design freedom.* 一句话同时交付贡献 1 与贡献 3。
- **Severity: Minor（但机会成本高）｜Confidence: 4**

### W13｜若干与仓库现状不符的小数字
- **Problem / Evidence Anchor**：
  - `text: "42 折滚动前进"`（§6）——机械生成给出 **45 折**，封存队列为 **33 折 = 05–35(31) + 44–45(2)**，fold43 已因落在 Kronos 语料窗内被移除（`HANDOFF.md` §11.8）。
  - `text: "判据 BE 的 95% CI 下界 > C + 预留"`（§4 行 C）——登记簿 2026-09-03 的现行部署判据是 `C ≤ BE_dev,disc − 6bp`（FT 上线线 C ≤ 10.4bp / ZS ≤ 4.2bp），且 C_stop=4bp 已由门控降为披露。两者不是同一个判据。
  - `text: "持仓期 6 → 5 日（解析驱动修订，待登记）"`（§6）——登记簿 2026-09-03 已有 `design-revision` 条完成登记，"待登记"已过期。
- **Why it matters**：单条都不致命，但一篇以「登记簿是唯一依据」为卖点的论文，正文与登记簿对不上是结构性尴尬。
- **Suggestion**：投稿前对 §4 与 §6 做一次与 `ledger.md` 的逐字核对（与 W6 合并为一次工作）。
- **Severity: Minor｜Confidence: 4**

---

## 摘要质量（150 词英文草稿）

**总评：写得密、诚实，但它是一份 proposal 的摘要，不是 paper 的摘要。** 结构上它做对了一件重要的事——七个断言全部可核，没有一句 "we propose a novel framework"。问题在三处：

**(1) 问了不答。** 以 `We ask whether…` 开头，全文没有答案，末尾是 `[Confirmation and cost results to be filled.]`。任何渠道都不收带 TBD 括号的摘要（W11）。

**(2) 三个断言与正文/仓库对不上**（逐条已在 W2 / W9 / W12 给出锚点）：
- `when used zero-shot … (RankIC 0.021, NW t 3.6)` —— 0.021/t3.6 是**微调臂**，零样本是 0.0192；
- `beats every readout learned on our data` —— 跨口径并列（7 折 vs 5 折 vs 单折），同口径下差距其实**更大**，改同口径是纯收益；
- `measure execution cost with real money under a sequential stopping rule` —— 现在时叙述一个**尚未启动、且被 AUM 阻塞**的测量。

**(3) 篇幅分配倒置。** 8 个词给别人的模型卡（`12B K-lines across 45 exchanges`），7 个词给本文最好的发现（流动性迁移）。缺 ICAIF 读者要的 artifact 钩子。

**建议方向**（若走论文 A，答案现成且很强）：把摘要的主语从「Kronos 有没有 alpha」换成「**设计自由度值多少 alpha**」。骨架示意——

> *Evaluating a foundation model as a stock-selection signal is mostly an exercise in not fooling yourself. We instrument that claim: on a single model (Kronos, financially pretrained) and a single US equity pipeline, we measure what four common evaluation choices are worth. Selecting a hyper-parameter on the evaluation window buys **+0.0070 RankIC — one third of the signal itself**. A look-ahead cross-sectional statistic buys 37%. … We further show that the arm-vs-arm comparisons this literature reports are **structurally unanswerable at this sample size**: separating a 20% relative difference would take 124 years. We therefore pre-register, seal a 31-fold confirmation set, treat the pretraining-overlapping period asymmetrically, and release the pre-registration, the append-only ledger, and the sealing harness.*

这个版本有答案、每个数都已落盘、不依赖封存集、且**恰好是 ICAIF workshop 会想要的稿子**。

---

## Venue Analysis

> 核对日期 2026-09-03。已核项均附来源 URL；未能从一手来源确认的一律标 **未核**，不做外推。
> **最重要的一条：作者自己想投的 ICAIF，其 2026 workshop 轨仍然开着，deadline 10 月 1 日 —— 距今 28 天。** 主会 8 月 2 日已截，但九个 workshop 里有八个收到 10/1，AI4F 收到 10/7。这正是「不追高」的路径：同一个会、同一座城、同一周。

### 候选渠道表

| 渠道 | 会期 / 出版 | 截稿 | 状态 | Fit（本席位判断） | 来源 |
|---|---|---|---|---|---|
| **ICAIF '26 workshop — MFMB**（Multimodal & Foundation Models in Banking） | 2026-11-14/15，Milan | **2026-10-01** | **开放** | ★★★★☆ 题目里就有 foundation models；TSFM 选股信号是正中靶心 | icaif2026.org/workshop.html |
| **ICAIF '26 workshop — RAIOps4Fin**（Responsible AI Ops for Finance），6 页 | 2026-11-14/15 | **2026-10-01**（notif 10/15） | **开放** | ★★★★☆ 「负责任 AI 运维」＝评估纪律 / 污染 / 可复现 / 部署治理，与**论文 A** 的脊柱几乎同构 | raiops4fin2026.github.io/ICAIF/ |
| ICAIF '26 workshop — AI-Driven Market Microstructure (CeFi & DeFi) | 2026-11-14/15 | 2026-10-01 | 开放 | ★★★☆☆ 执行成本现实主义（S7）是他们的语言，但本文主体不是微结构 | icaif2026.org/workshop.html |
| ICAIF '26 workshop — IAFM'26（Interpretability & Alignment of Financial Models） | 2026-11-14/15 | 2026-10-01（notif 10/15） | 开放 | ★★★☆☆ 「手写读出 vs 学出读出」可包装成可解释性问题 | iafm-icaif.netlify.app |
| ICAIF '26 workshop — AI4F（3rd, LLMs & GenAI for Finance），**不限页数** | 2026-11-14/15 | **2026-10-07 AoE**（notif 10/14） | **开放** | ★★★☆☆ Kronos 是生成式解码器，勉强在 GenAI 口径内；最宽松、最晚，但偏 LLM/agent | linqalpha.com（会议页） |
| ICAIF '26 主会 / tutorials / competitions | 2026-11-16/17 | 8/2（延 8/9）、8/22、8/9 | **已截** | — | icaif2026.org/important-dates.html |
| ICAIF '26 late-breaking / poster / demo | — | — | **不存在该轨** | 三个页面均已核，**不要指望有安全网** | icaif2026.org/call-for-papers.html |
| **AAAI-27 workshops** | 2027-02-22/23，Montréal | **论文 2026-11-20**（各 workshop CFP 自 10/2 起陆续发布） | **CFP 未出，10/2 起查** | ★★★★☆ **唯一一个「确认集结果有可能赶上」的 2026 年内截稿**；但金融/时序 workshop 是否入选 **未核**（组织方 9/25 才通知） | aaai.org/conference/aaai/aaai-27/workshops-call/ |
| AAAI-27 主会 | 2027-02 | 7/28 已截 | 已截 | — | 同上 |
| NeurIPS 2026 — FMTS（Foundation Models for Temporal Systems），4 页非存档 | 2026-12-11/12，Sydney | 2026-09-16 11:59 UTC | 开放 | ★★☆☆☆ **NeurIPS 2026 全部 102 个 workshop 中没有金融 workshop**；FMTS 自述范围列的是医疗/气候/运营，投横截面选股是 scope stretch | fmts-workshop.github.io |
| NeurIPS 2026 — TS-LIMITS | 2026-12-12/13，Paris | **未核**（第三方 tracker 称 9/6，官网 JS 渲染无法确认） | **未核** | ★★☆☆☆ 若 9/6 属实则仅剩 3 天，不现实 | ts-limits.github.io |
| ICLR 2027 | 未核 | **摘要 2026-09-18 / 正文 09-25 AoE** | 开放 | ★☆☆☆☆ **追高档**：结果尚封存的金融应用论文对 ICLR fit 差，且 22 天 | iclr.cc/Conferences/2027/CallForPapers |
| **ICAIF 2027（第 8 届）** | **未核** | **未核**（聚合站的「2027 年 8 月」是自动外推，非官方） | **未核** | ★★★★★ **论文 B 的天然归宿**；按 2026 的 8/2 模式推测约 2027 年 8 月，时间上从容 | myhuiban.com（聚合站，仅供参考） |
| **JFDS**（Journal of Financial Data Science） | rolling | — | 开放 | ★★★★☆ **论文 B 的最佳期刊档**：实务导向，实测成本与诚实评估正是其读者要的；自述审期 **12–16 周** | jfds.pm-research.com/authors |
| Quantitative Finance | rolling | — | 开放 | ★★★☆☆ 论文 B 的备选；编辑部**目标首轮六个月** | tandfonline.com/journals/rquf20 |
| Journal of Financial Econometrics | rolling | — | 开放 | ★★☆☆☆ 与作者 §10 自评一致（**冲高档**，需 Compustat 级基本面基线）；≤40 页双倍行距，**接收时强制交复现代码/数据**；审期未核 | academic.oup.com/jfec |
| Finance Research Letters | rolling | — | 开放 | ★☆☆☆☆ **<2,500 词**，装不下本文任何一条脊柱 | sciencedirect.com/journal/finance-research-letters |
| FinNLP 2026 (@EMNLP) | 2026-10-28 | 8/11、8/27 | 已截 | 本文非 NLP | sigfintech.github.io/finnlp2026 |
| KDD 2027 Cycle 2 | 2027-08 | **未核**（正文提及约 2027-02，未公布确切日） | 未核 | ★★☆☆☆ | kdd2027.kdd.org |
| Registered Report 轨 | — | — | JFDS/QF/JFEC/FRL **均无 RR 轨**（已核） | Pacific-Basin Finance Journal 有预注册倡议，但**亚太范围与本文美股宇宙冲突** | cos.io/initiatives/registered-reports |

### 关键档期冲突（必须先看这一条）

**所有 2026 年内还开着的截稿日，全部早于确认集结果。** ICAIF workshop 10/1、AI4F 10/7、ICLR 9/25、FMTS 9/16 —— 而按 §9，折 05–35 的读取排在 1–3 个月的成本小试之后，最早 2026-12 至 2027-02。

这个约束本身就把 W3 的拆稿建议从「建议」变成了「唯一可行解」：**2026 年内能投出去的，只可能是不依赖封存集的论文 A。** 好消息是论文 A 的每一个数都已落盘（选择偏差 +0.007、四类错误的实测损失、A/B 层的 20/55/124 年换算、读出比较的完整纠错史），**不需要跑任何新实验，只需要重写**。

### 排序建议

**第 1 位｜ICAIF '26 workshop（MFMB 或 RAIOps4Fin），10 月 1 日，投论文 A。**
理由：(a) 这是作者自己选的会，同城同周，**完全没有追高**；(b) 论文 A 零依赖封存集，28 天足够重写一份已有材料的稿子；(c) workshop 的评审预期是「有意思的方向 + 诚实的方法」，而不是「完整的实证闭环」，本稿的方法学密度在这个层级是**明显超配**的；(d) 它给论文 B 在 ICAIF 2027 主会留出了干净的升级路径。
两个 workshop 二选一：若把脊柱写成「foundation model 的评估协议」选 **MFMB**；若写成「部署前的负责任评估与失误谱」选 **RAIOps4Fin**（6 页，与论文 A 的体量正好）。
**⚠ 两条必须先查（本席位未核）**：① 选定 workshop 是否 **archival** —— 若存档，需确认它不阻断 2027 主会的完整版（多数 ICAIF workshop 非存档，但**请从该 workshop 的 CFP 原文确认**）；② 同一届不同 workshop **不要同时投**（10/1 的 notif 是 10/15，晚于 AI4F 的 10/7 截稿，无法先后排队，只能一次选一个）。

**第 2 位｜arXiv q-fin.ST + OSF 预注册时间戳 —— 不是渠道，是前置条件，现在就做。**
见 W5。这一步必须**排在折 05–35 读取之前**，成本近乎为零，而它决定了「预注册」三个字在论文 A 与论文 B 里是否可核。四个候选期刊都没有正式 RR 轨（已核），所以**公开时间戳是本项目唯一可用的预注册凭证**。

**第 3 位｜AAAI-27 workshops，11 月 20 日，作为论文 B 的早期出口备查。**
这是 2026 年内唯一一个**确认集结果有可能赶上**的截稿（若小试提前启动、读取在 10–11 月完成）。但金融/时序 workshop 是否入选目前**未核**，组织方 9/25 才收到通知、CFP 自 10/2 起发布。**建议：10 月 2 日在日历上设一个提醒去查一次**，成本一次点击。若没有合适 workshop，直接放弃，不要迁就。

**第 4 位｜ICAIF 2027 主会，投论文 B（完整实证 + 机制 + 实测成本）。**
按 2026 年 8/2 的模式推测约 2027 年 8 月（**日期未核**）。从 §9 的时间线倒推，这是**最从容**的档期：确认读取 2027-01/02 → 成稿 2027-03/05 → 投稿 2027-08。论文 A 先在 2026 workshop 落地，论文 B 在主会落地，是一条干净的两步走。

**第 5 位｜JFDS，作为论文 B 的期刊平行/备选（rolling，12–16 周）。**
若作者更想要期刊而非会议，JFDS 是「现实档」里最合适的：实务读者、欢迎实测成本、对诚实评估有胃口。可与 ICAIF 2027 二选一（不可同时）。

**不建议**：ICLR 2027（追高，且封存结果未出的金融应用论文 fit 差）；NeurIPS FMTS（**NeurIPS 2026 无金融 workshop**，FMTS 自述范围不含金融，且 Sydney + 4 页非存档，投入产出比低）；JFEC（作者 §10 自评「证据不够」，我同意）；FRL（2,500 词装不下）；Pacific-Basin（唯一可核的金融 RR 期刊，但亚太范围与美股宇宙冲突）。

---

## Questions for Authors

**Q1｜若折 05–35 读出的结果是 H1 不拒绝，这还是同一篇论文吗？**
§9 把「阶段 3 读取」排在「阶段 4 论文初稿」之前，字面读来成稿依赖结果。请写出**两种结果下各自的论文骨架与题目**并公开——这是预注册的核心承诺，也是把「不确认」与「信号死了」区分开的唯一保障（`confirmation_protocol_v4_revisions.md` 的四态输出已有此意识，但计划书没有把它写成投稿承诺）。

**Q2｜题目的 zero-shot 与 §4 的 k=2（含微调臂）如何调和？头条数字用哪一臂？**
请明确：摘要的 0.021 / t 3.6 / BE 21.9bp / 毛 10.48% 是否统一改为零样本口径（0.0192 / BE 13.57bp / 毛 4.78%）？若不改，题目是否去掉 zero-shot？（见 W2 的三个选项。）

**Q3｜若成本小试因 AUM / 学校实盘基金未落地而不能启动，第四支柱怎么办？**
是否接受 W4 提出的降级交付（BE 的 CI + 券商费率表的确定项 + 一个显式标注为未测的缺口）作为**论文 B 的备用形态**？如果接受，请现在就把它写进 §9 作为分支，而不是等到发现小试跑不起来。

**Q4｜是否愿意在同管线上跑一个具名的通用 TSFM 零样本对照（E8）？**
这是把「金融原生预训练才有效」从引用变成本文证据的最便宜一步，不消耗任何确认折，且是 ICAIF 读者最想看到的那张表（见 W7）。若不做，请在 §8 里显式承认这条论断是借来的。

---

## Minor Issues

- **摘要开销分配**：`pretrained on 12B K-lines across 45 exchanges` 用了 8 个词描述**别人的**模型卡；而流动性迁移（本文最好的发现）只分到 `migrates along the liquidity spectrum across eras` 7 个词。建议对调权重。
- **摘要缺 ICAIF 读者要的钩子**：没有 "we release"（代码 / 预注册文本 / 登记簿哈希 / 评估协议）。ICAIF 近年对 artifact 友好，而本文的 artifact 恰是它的贡献本身。加一句 `We release the pre-registration, the append-only ledger, and the sealing harness.`
- **§1 的四问 (a)(b)(c)(d) 结构很好，但与 §4 的 H1–H5+C 不是一一对应**。建议给 §1 的四问各标上对应的假设编号，读者能一眼看到「哪一问由哪个检验回答、哪一问没有检验回答」。
- **§5 表格的「消耗折？」列是很好的设计**，建议保留到正文——ICAIF 审稿人很少见到有人明确标注每个实验消耗多少统计资本。
- **§2 缺 Kronos 之外的金融原生预训练对照**（若有同期工作）。若确无，请在正文写明「据我们检索，Kronos 是唯一权重公开的金融原生 K 线预训练模型」并给检索日期。
- **§8 的「WRDS 学术口径须书面确认可用于学校实盘项目」**：这一条在论文里必须写成已解决或已规避，不能以未决状态出现在正式投稿的 limitation 节——审稿人会视为伦理/合规风险而非局限。（`HANDOFF.md` §12.5 第 1 条把它标为「比任何因子都重要」，我同意。）
- **术语**：`套袖`（sleeve / staggered tranche）在英文稿里需要一个固定译名并在首次出现时定义；`折`统一译 fold 并明确 fold 与 walk-forward window 的关系。
- **中文题目与英文题目不等价**：中文「有没有可部署的横截面 alpha？」是疑问句，英文 *Does … Carry Deployable …* 也是疑问句，但英文副标题的三个形容词（Pre-Registered, Sealed-Confirmation, Cost-Measured）在中文里是「预注册、封存确认、实测成本」——中文少了 "on US Large Caps" 的限定。统一。

---

*本报告仅代表 Journal-Fit 席位的意见，不涵盖统计推断正确性（Reviewer 1 席位）。所有引用的仓库内容截至 2026-09-03，均未触碰任何 sealed 路径或 `scores.parquet`。*
