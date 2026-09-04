# 美股 DL 量化管线规范 v1.2(冻结)

**状态**：Claude × GPT 四轮对抗审查后冻结。
**日期**：2026-08-13

---

## 0. 目标与执行约定

- 任务：Kronos 微调 → 日频横截面 score → 低频执行的美股选股策略。
- 时间线：t 日收盘数据 → 盘后推理 → t+1 开盘成交。允许使用 t 日收盘后可得的全部信息。
- 研究频率与执行频率解耦：信号日频；实盘调仓周频/月频、阈值触发（§7）。

## 1. 数据源（CRSP CIZ，全链统一）

- legacy SIZ（`crsp.dsf`）冻结于 2024-12 数据，样本至 2026 必须用 CIZ。
- 表：`crsp.dsf_v2`（WRDS 视图，原生 `crsp.StkDlySecurityData`）；`crsp.stksecurityinfohist`；CIZ distribution / corporate-action 表（复权与股息处理必需，§4/§8）；基准 `crsp.wrds_dailyindexret_query` 取 `VWRETD/EWRETD`；Fama-French 库。表名以 WRDS CIZ 文档为准，代码第一步 list tables 核对。
- 日线字段：`PERMNO, DlyCalDt, DlyOpen, DlyHigh, DlyLow, DlyClose, DlyPrcFlg, DlyVol, DlyPrcVol, DlyRet, DlyRetMissFlg, DlyFacPrc, DlyDelFlg, DlyCap` + 退市字段（`DelActionType/DelStatusType/DelReasonType/DelPaymentType`）。
  - close 用 `DlyClose`；市值用 `DlyCap`（单位按 $千，代码时用 AAPL 总市值做量级校验）；不使用 legacy `SHROUT` 口径。
  - amount 用 `DlyPrcVol`，注明 = DlyPrc × DlyVol，非逐笔 dollar volume。
- 范围：2000-01-01 至 `data_cut_date`（§11）。原始层按年分区 Parquet，零变换。

## 2. Universe 筛选（选股面板）

CIZ 普通股筛选写死：

```sql
ShareType='NS' AND SecurityType='EQTY' AND SecuritySubType='COM'
AND USIncFlg='Y' AND IssuerType IN ('ACOR','CORP')
AND PrimaryExch IN ('N','A','Q') AND ConditionalType='RW'
AND TradingStatusFlg='A'
```

流动性条件（全部按 t 日信息滚动，禁止回填）：

- 有效 `DlyClose` ≥ $5；
- `ADV20_t = mean(DlyPrcVol[t-19 : t])`，要求窗口内 ≥15 个有效观测，ADV20_t ≥ $5m；
- 上市 ≥ 120 个交易日（非自然日）；
- `DlyCap` 排名前 1500（按 t 日值）。

## 3. 双面板分离

- 选股面板：上述过滤只回答「t 日能否入选」。
- 收益面板：持有期收益一律从未过滤全量面板提取（active 条件会滤掉 `DlyDelFlg='Y'` 记录）。

## 4. 标签：execution-return engine（总财富收益）

标签为 t+1 开盘建仓、t+6 开盘退出的 total return，分段复合：

1. 建仓：t+1 实际 `DlyOpen`（未复权原始价）。若 t+1 无有效 open（停牌/已不可交易）→ 记为 unfillable，单独归类并报告原因（停牌 vs 退市），不删除、不假设成交；
2. 首日段：t+1 open → t+1 close 价格收益 = `DlyClose(t+1)/DlyOpen(t+1) − 1`（建仓在 ex 之后，不含 t+1 股息）；
3. 中段：t+2 … t+5 逐日复合 CIZ `DlyRet`（close-to-close 含息收益）；
4. 退出段：t+5 close → t+6 open 隔夜价格差；若 t+6 为 ex-date，持仓过夜者有权获得该股息，此段 = `(DlyOpen(t+6) + Div(t+6)) / DlyClose(t+5) − 1`，Div 取自 distribution 表（v1.1 未覆盖此情形，v1.2 新增）；
5. 退市接管：持有期内出现 `DlyDelFlg='Y'` → 从该点起复合至退市终值记录的 `DlyRet`，期末即为终值，不再要求 t+6 open 存在。

规则：

- 禁止对存续股票用「复权 open 比值」当标签（那是价格收益，与退市股的总收益口径混用会系统性偏置横截面）；
- `DlyRet` 缺失的业绩类退市按 Shumway 插补（−30%/−55%）作敏感性假设，报告 ±档位；
- score = `predOpen(t+6)/predOpen(t+1) − 1`（模型只见 OHLCV，score 是价格收益预测；标签是总收益——此口径差记入文档，股息在 5 日尺度横截面影响小，v2 若加基本面再收敛）。

## 5. 复权（独立模块，不与标签引擎混用）

- 假设 `DlyFacPrc` 为当期（事件）因子而非累计因子——该语义待代码时验证（GPT 所引出处已失效）；无论哪种语义，统一路径：从 CIZ distribution/corporate-action 事件构造以固定基准日为锚的累计复权因子，禁止逐日裸乘 `DlyFacPrc`。
- Kronos 蜡烛复权只处理拆股与股票股利；现金股息、分拆、配股不写入 OHLC。
- 验证：AAPL 2020-08-31、NVDA 2024-06-10 拆股，双路构造（事件累计 vs 直接用 DlyFacPrc）对比，不一致即锁定语义结论。

## 6. Kronos 微调

- 官方生成式目标不动；官方默认（finetune/config.py）lookback=90, predict=10, max_context=512。
- 本项目 predict_window = 6；lookback 消融 {60, 90, 200, 400}（400+6+1=407 ≤ 512 ✓）。
- 特征 open/high/low/close/volume（+amount）；推理按官方采样参数多路径取均值（参数名代码时核对）。

## 7. 切分与测试纪律

- Walk-forward：训 3 年 → 验 6 个月 → 滚动；
- 封存 OOS：最后 ≥2 年，设计冻结前禁止查看（lookback/阈值/调仓规则的所有选择只允许消耗验证期）；
- Purge：标签视野 6 个交易日；代码断言：

  ```
  train_label_end = train_signal_date + 6 个交易日
  max(train_label_end) < min(val_signal_date)
  ```

  （显式按交易日历加 6，防日期索引实现差异的 off-by-one）。

## 8. 成本模型（分段函数，非统一 bp）

- `Cost_t = fee_schedule(order_value, order_type) + 监管结算费 + 价差/滑点（5/15/30bp 三档）`；
- fee_schedule 按账户后台实际费率表写成分段函数：整股每单固定费（参考 moomoo AU 美股 US$0.99/单）与碎股按金额计费（可能设上限）分开处理；行动项：导出费率表原文；
- 换手假设写死：缓冲区替换制（换一只 = 卖一单 + 买一单），不做全组合重新等权配平（配平会使单数倍增，若将来改配平制，费用表重算）；
- 固定费用对 $3000 的约束结论不变：≤5 仓、周频以下、score 改善超阈值才换仓；回测与实盘同约束。

## 9. 清洗规则（CIZ 语义）

- 无负价格逻辑（那是 legacy SIZ）：`DlyPrc` 已取绝对值，报价中点由 `DlyPrcFlg` 标识；`DlyPrcFlg='BA'` 的日期单独统计与处理；
- 退市收益：CIZ `DlyRet` 在 `DlyDelFlg` 记录上已并入退市收益，禁止再套 (1+RET)(1+DLRET)−1；
- 停牌/缺失不删行、不插值、不压缩时间：市场休市日不生成行；个股在开市日缺失有效 OHLC 时，该日留空；lookback 或预测区间含缺口的训练样本整体排除（防止相隔数日的蜡烛被当作相邻），按年份 × 交易所报告排除率；收益面板不受此规则影响；缺口 mask 方案归 v2；
- 标签与评估不截尾：原始标签、组合收益、主 RankIC 用真实值（截尾会削弱退市尾部评估）；winsorized RankIC 仅作附加稳健性指标；ranking label 截尾随 ranking head 归 v2。

## 10. 审计与验证

CIZ coverage audit（2000+、过滤后 universe，逐年 × 交易所）：

- DlyOpen 缺失率；`DlyPrcFlg='BA'` 占比；low ≤ open/close ≤ high 违反率；拆股日复权连续性（§5 双路测试）；DlyCap 覆盖与量级校验。
- 合格 → 全链 CRSP；不合格 → Norgate Platinum 作训练蜡烛备选（注意跨供应商错配）。

Golden fixtures + 不变量（替代 v1.1 的硬编码断言）：

- Lehman 2008：断言其历史行、退市终值记录、收益处理路径存在且被引擎走到；具体数值从首次下载数据固化为 golden fixture，此后回归比对；
- 每年退市终值记录数 > 0；历史退市证券存在于原始层；active universe 数量合理波动（不断言单调递减）；累计 distinct PERMNO 非递减；
- PERMNO+date 唯一；插补只发生在业绩类退市码上；unfillable 样本占比按年报告。

## 11. 可复现性冻结

每次数据快照记录：`data_cut_date`、实际最大交易日、查询时间戳、SQL 全文 hash、WRDS 表版本标识。实验引用快照 ID，不引用「至今」。

## 12. v2 backlog

ranking head / 二阶段 ranker（含 label 截尾）、缺口 mask、基本面（SEC EDGAR XBRL, PIT）、regime 过滤、容量分析、配平制费用重估。

---

## 修订记录

- **v1.1 → v1.2**：复权改为事件累计构造（DlyFacPrc 语义待验证，双路测试定夺）；标签改 execution-return engine（五段复合、退市接管、unfillable 路径、退出日 ex-date 股息段）；删除 abs/负价逻辑（改 DlyPrcFlg）；SHROUT→DlyCap；ADV20 窗口写死；停牌不压缩时间 + 样本排除率报告；删除标签截尾；增加封存 OOS 与 purge 断言；费用改分段函数 + 缓冲区替换假设；锚点测试改 golden fixture + 不变量；新增可复现性冻结节。
