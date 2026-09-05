# 任务书草稿：exp13 Compustat 开发折诊断（SUE / 财报临近 / 财报日方差）

> **状态：草稿，待用户裁定是否启动。** 写于 2026-09-05，写在任何结果之前（`CLAUDE.md` §二）。
> 派工：opus（`CLAUDE.md` §九）。agent 开工第一步是把 §3–§5 全文抄进脚本 docstring，**然后才读第一行数据**。
> 本任务**不改 v4、不读折 05–35 / fold44–45 / 封存窗**，与解封链无关。

---

## 0. 背景（自足）

- 主信号：Kronos 时序基础模型在 CRSP 日线上的 6 日截面打分，t 日收盘算分、t+1 开盘成交（规范 §4）。
- H1b 张成检验（`ledger.md:483` `[gpt-exp 11]`；v4 §2.4）用同频价量控制集（K6b 八因子 + turnover 三项 + hi52 + 市场，SIC2 行业内去均值）在开发折 36–42 上得 S-TH-ind alpha **+6.76%/年、保留率 109%、6/7 折**。**两个缺口原样写在冻结协议里**：财报日虚拟缺外部日历、SUE 缺 Compustat，「均未做」。
- 2026-09-05 Compustat 年度 / 季度宽表到位（`F:\quant\external\compustat\received_20260905\`；`ledger.md` 当日 `data-receipt` 条）。**CCM 链接表 `crsp.ccmxpf_lnkhist` 未随包提供。**
- `docs/调研_信号源盘点_2026-09-03.md` §6 与 §10 第 5 项：用户曾提议「未来 6 日内有财报就不买」；【文献】Frazzini–Lamont 2007（公告月多空 >60bp/月）、Savor–Wilson 2016（预定公告者年化异常收益 9.9%）指向相反方向。「6 日窗口内一个财报日的方差贡献」被记为**唯一能推翻该结论的量**，当时无数据未算。
- 协议状态：v4 已冻结（sha256 `c3c823e2…fcbe`），`:202`「本线在读折 05–35 前不得再改」。

## 1. 目的与层级

三个交付，全部是**开发折诊断、估计交付、无门槛、不判定**（折 36–42 已被消耗，其读数为方向性证据，`CLAUDE.md` §四）：

| 编号 | 内容 | 回答的问题 |
|---|---|---|
| **D1** | 链接与覆盖审计 | header-CUSIP 链接在 top500 宇宙里逐折覆盖多少名字；SUE / EA 非缺占比 |
| **D2** | H1b 扩展规格开发折读数 | 开发折上 S-TH-ind 的 alpha 有多少能被盈利意外 / 财报临近因子解释 |
| **D3** | 财报日方差贡献 | 含实际公告日的 6 日持有窗，其方差与均值收益相对不含窗的比值 |

用途：(i) 论文里「两个缺口已在开发折上补做」一句；(ii) 为解封后是否另行登记探索性分析提供先验；(iii) 回答 §6 的方差问题。
**不得据此改 v4、不得据此挑规格、不得进入任何判定、不得作为部署依据。**

## 2. 硬禁令（违反即全部作废）

1. 只读折 36–42 的 `scores.parquet`（FT 臂，同 `scripts/exp11_spanning_extended.py:339` 的 `scores_path(fold, "ft")`）。任何对折 05–35、fold44–45、封存窗、`outputs/*sealed*` 的读取即全部作废；若 `src/crsp_pipeline/sealed.py` 的 `assert_readable` 守卫可用，必须走它。
2. 不改 `experiments/confirmation_protocol_v4.md`、`scripts/k6b_spanning.py`、`scripts/exp11_spanning_extended.py`。管道**只 import**；不可 import 则逐字抄并注明来源行号。
3. 全脚本无 `label`；交付附 `grep -n label` 结果。
4. 禁止自动 `append_ledger`；登记由主会话手写。
5. Compustat 派生物**不进 git**：放 `F:\quant\external\compustat\derived\`，附 `MANIFEST.json`（源 ZIP sha256 + 生成脚本 sha256 + 规则版本 + 生成时间）。
6. 内存：季度 CSV 解压后 4.57 GB。**必须 `usecols` + `chunksize`**，只转一次 parquet（列裁剪到 §4 所需），此后不再碰 CSV；转换峰值提交内存 < 8 GB（其它会话在跑，`CLAUDE.md` §七）。
7. 不得编数字；没核到的一律写「未核」。

## 3. 数据处理规则（先写死）

### 3.1 季度表筛选
`indfmt=INDL, datafmt=STD, consol=C`（验收显示全表已如此，脚本仍显式过滤并断言）。`fic` **不筛**（CRSP 宇宙决定样本）；`curcdq` 不筛，但报告 top500 内非 USD 占比。

### 3.2 重复键（1999–2025 有 599 组 `gvkey+datadate`，每组 2 行，`fyr` 均不同）
成因【推断，未核】为财年变更时新旧财年口径并存。规则：**同组保留 `fyearq` 最大者；再平则保留 `rdq` 非缺者；再平则 `fqtr` 最大者。** 零参数、确定性。交付时报告该规则在 top500 宇宙内实际触及的股票-季度数。

### 3.3 GVKEY → PERMNO（CCM 缺失下的替代链接）
- 键：`fundq.cusip`（9 位，Compustat 当前 header CUSIP【推断，须查 PDF 核】）↔ `security_info_history.hdrcusip9`（CRSP header CUSIP；【本项目实测 09-05】25,331 个 permno 全部非空、一一对应、无一对多）。
- 一个 gvkey ↔ 多个 permno（多股份类别）：全部保留，季度数据广播到每个 permno；报告数量。
- 一个 permno ↔ 多个 gvkey：视为链接冲突，**该 permno 整段置缺**并报告数量，不做挑选。
- 已知劣势（原样写进报告）：header 对 header 不是历史时点匹配；CUSIP 变更后两库 header 可能不同步；退市多年的公司两库 header 可能指向不同证券。**若日后拿到 `ccmxpf_lnkhist`，本节整体替换，D1–D3 全部重跑并登记两版差异。**

### 3.4 时点规则（前视禁令的核心，`CLAUDE.md` §一.1）
- 季度 q 的任何数值，**可用起点 = `rdq_q` 之后的首个交易日**（rdq 当日不用：公告可能在收盘后【推断】）。
- `rdq` 缺失的季度**不可用**。不用 `datadate + 固定滞后` 回填——两套时点混用等于给不同名字不同的信息集。
- 陈旧上限：信号日 t 距最近可用季度的 `datadate` **≤ 180 日历日**，超过置缺（B 层先验：Fama–French 年度 6 个月滞后惯例的季度类比；唯一常数）。
- 不做任何重述处理。fundq 数值是否按首次披露保留：**未核**。agent 须在 `documentation/Comp_Quarterly6126.pdf` 查 `epspxq` / `ibq` / `ajexq` 条目并原文摘录，标「有证据 / 未核」。

## 4. 构造定义（先写死；常数全部取文献默认，B 层优先序 ①②）

### 4.1 SUE（Livnat–Mendenhall 2006 口径【文献】）
- `E_q = epspxq / ajexq`（基本每股收益扣非常项目，按累计调整因子复权）。
- `SUE_q = (E_q − E_{q−4}) / σ_q`，`σ_q` = 前 8 个季度（q−7..q）季节差 `(E_j − E_{j−4})` 的样本标准差；**非缺 < 6 个置缺；σ = 0 置缺**。
- `q−4` 按 `(fyearq, fqtr)` 对齐，**不是按行位移**。
- 信号日 t 取 §3.4 规则下最近可用季度的 SUE；截面上照 exp11 的 `rank(pct=True)` 在当日候选池内秩化。

### 4.2 EA（财报临近；Frazzini–Lamont 2007 公告溢价因子的零前视版【文献 + 推断】）
- 预期公告日 `rdq_hat_q` = 同 `fqtr` **上一财年**的实际 `rdq` + 1 年（无则置缺）。**只用过去的 rdq**。
- `ea_prox_t = −min(交易日数(t → 最近的未来 rdq_hat), 63)`；无可用 `rdq_hat` 置缺。63 = 一个季度的交易日数，唯一常数。分数越高 = 越临近公告。
- 敏感性 `ea_real_t`：用**实际** `rdq` 计 `−min(交易日数(t → 最近的未来 rdq), 63)`。**这是事后日历**（现实中提前 2–4 周知晓），只作敏感性并标注。

### 4.3 张成规格（在 S-TH-ind 冻结定义上追加，不改 S-TH-ind 本身）

| 规格 | 控制集 |
|---|---|
| S-TH-ind-SUE | S-TH-ind + `sue` |
| S-TH-ind-EA | S-TH-ind + `ea_prox` |
| S-TH-ind-SUE-EA | S-TH-ind + `sue` + `ea_prox` |
| S-TH-ind-EAreal（敏感性） | S-TH-ind + `ea_real` |

- 因子走 K6b 同管道（top500 / 六档错位 **NT=6** / 进前 10% / 跌出前 30% 才卖 / t+1 开盘 / 多空价差腿 / 毛收益）。**与 exp11 相同的 NT=6 口径，读数不得与 NT=5 并列**（`CLAUDE.md` §八）。
- `sue` / `ea_prox` 与 S-TH-ind 其它因子一样做 point-in-time SIC2 行业内去均值（逻辑照 `scripts/xsec_context_probe.py:126-150`，逐日区间匹配 + `secinfoenddt` 失效判定）。
- 候选池 = 当天有分的名字；因子缺失的名字**不进该因子的候选池**。agent 须先核对 exp11 `:237-320` 对 `hi52` 缺失的实际处理并照抄，交付时写明。
- 三个主规格**全部报告，不得择优**；S-TH-ind 原读数并列作基线行。

### 4.4 D3 财报日方差（描述统计，允许用事后 rdq）
- 样本：折 36–42 各自验证窗，top500 宇宙（同 K6b 候选池），每个股票-日 d 的日收益 r（`panel_raw` 口径，含退市收益）。
- `ea_day_d = 1` 若 `rdq ∈ {d−1, d, d+1}`。
- 报告：(a) `mean(r² | ea=1) / mean(r² | ea=0)`；(b) `mean(r | ea=1) − mean(r | ea=0)` 与 NW(5) 95% CI；(c) 以 6 日持有窗（t+1..t+6）为单位：含 ≥1 个 `ea_day` 的窗占全部窗的比例，及其收益方差占全部窗方差的份额；(d) 以上逐折 + 合并。

## 5. 判据（先写死）

- **无门槛、无 PASS/FAIL、无「有价值 / 无价值」措辞。**
- 每个量报点估计 + NW(5) 95% CI + 7 折正折数；D2 另报各追加因子的载荷 β 及其 CI。
- **MDE 未算，因为本任务不是检验。** 若日后要把任何一项升为检验，须先按 `CLAUDE.md` §二独立写 SESOI / MDE / 功效，并按比较分层归 A 层或 B 层。
- 措辞模板：「在开发折 36–42（已消耗，方向性证据）上，加入 X 后 S-TH-ind 的 alpha 由 a 变为 b（保留率 c → d，N/7 正）；β_X = e（CI [·,·]）。不作判定。」
- exp11 的限定原句照抄：「仅限本控制集、本开发样本与冻结构造；保留率 >100% 只表示正向暴露于本样本内亏钱的因子，**不得写成无限定的 survives spanning**。」

## 6. 交付

- 代码：`scripts/build_compustat_link.py`（CSV → 裁剪 parquet + 去重 + 链接表 + `MANIFEST.json`）；`scripts/exp13_compustat_dev_diag.py`（docstring 含 §3–§5 全文；`--smoke` 用合成数据跑通）。
- 测试 `tests/test_exp13_*.py`：链接一对一断言；SUE 时点（合成数据：rdq 之后才可见、rdq 当日不可见、180 日陈旧置缺、q−4 按财季对齐、<6 个非缺置缺）；`rdq_hat` 只用上一财年；去重规则确定性；全脚本无 `label`。
- 产物：`outputs/exp13_compustat_dev_diag/{summary.json, link_coverage.json, report.md}`；`F:\quant\external\compustat\derived\{fundq_slim.parquet, gvkey_permno_link.parquet, MANIFEST.json}`。
- `report.md` 内容顺序：D1 覆盖（逐折 top500 内 SUE / `ea_prox` 非缺占比、链接冲突数、非 USD 占比、去重触及数）→ D2 四规格表 + S-TH-ind 基线行 → D3 四个量 → 前视自查 grep → §7 未核项逐条回答 → ledger 草稿行（主会话定稿）。读数标【实测】，解释标【推断】。
- 预算：转换 + 链接约 1 小时 CPU；D2 每规格与 exp11 同量级；D3 分钟级。全程不用 GPU。

## 7. 未核项（交付时必须逐条回答或写「未核」）

1. fundq 数值是否按首次披露保留（重述问题），PDF 原文摘录。
2. Compustat `cusip` 列是否为当前 header CUSIP，PDF 原文摘录。
3. 599 组重复键的成因是否为财年变更（PDF `fyr` / `datafqtr` 条目）。
4. 原 WRDS 查询条件与下载日期——**问合作者**，不得用 ZIP 时间戳回填。
5. `crsp.ccmxpf_lnkhist` 能否补拿——**问合作者，优先于一切替代链接**。
