# 美股 DL 量化管线规范 v1.3(草案)

**状态**:在 v1.2 冻结版基础上修订,待确认后冻结。v1.2 原文保留不动。
**日期**:2026-08-25

---

## 0. 目标与执行约定

- 任务:Kronos 微调 → 日频横截面 score → 低频执行的美股选股策略。
- 时间线:t 日收盘数据 → 盘后推理 → t+1 开盘成交。允许使用 t 日收盘后可得的全部信息。
- 研究频率与执行频率解耦:信号日频;实盘调仓周频/月频、阈值触发(§7)。

## 1. 数据源(CRSP CIZ,全链统一)

- legacy SIZ(`crsp.dsf`)冻结于 2024-12 数据,样本至 2026 必须用 CIZ。
- 表:`crsp.dsf_v2`(WRDS 视图,原生 `crsp.StkDlySecurityData`);`crsp.stksecurityinfohist`;CIZ distribution / corporate-action 表(复权与股息处理必需,§4/§8);基准 `crsp.wrds_dailyindexret_query` 取 `VWRETD/EWRETD`。表名以 WRDS CIZ 文档为准,代码第一步 list tables 核对。
- **归因因子(v1.3 修改:弃用 Fama-French 外部表,全链保持单一 CRSP 数据源)**:归因诊断所需因子一律从已下载的 CRSP 面板自建——市场因子用 `VWRETD`(评估口径统一为对 VWRETD 的超额,不引入无风险利率,故不需要 FF 的 RF 列);规模因子代理用 `DlyCap` 十分位构造小减大组合;动量因子用 12-1 月收益构造赢减输组合。构造规则冻结前写死,universe 过滤与 §2 同一套,禁止另起口径。价值因子需账面数据,CRSP 无法自建,随 v2 基本面(SEC EDGAR XBRL)再加。自建因子仅服务验证期归因诊断,非核心评估指标(§7.5 的通过标准不依赖它);代价是与学术 FF 口径不可直接对比,记入文档,若将来需要标准口径,Ken French 官网免费文件可随时补挂,不影响主链。
- 日线字段:`PERMNO, DlyCalDt, DlyOpen, DlyHigh, DlyLow, DlyClose, DlyPrcFlg, DlyVol, DlyPrcVol, DlyRet, DlyRetMissFlg, DlyFacPrc, DlyDelFlg, DlyCap` + 退市字段(`DelActionType/DelStatusType/DelReasonType/DelPaymentType`)。
  - close 用 `DlyClose`;市值用 `DlyCap`(单位按 $千,代码时用 AAPL 总市值做量级校验);不使用 legacy `SHROUT` 口径。
  - amount 用 `DlyPrcVol`,注明 = DlyPrc × DlyVol,非逐笔 dollar volume。
- 范围:2000-01-01 至 `data_cut_date`(§11)。原始层按年分区 Parquet,零变换。

## 2. Universe 筛选(选股面板)

CIZ 普通股筛选写死:

```sql
ShareType='NS' AND SecurityType='EQTY' AND SecuritySubType='COM'
AND USIncFlg='Y' AND IssuerType IN ('ACOR','CORP')
AND PrimaryExch IN ('N','A','Q') AND ConditionalType='RW'
AND TradingStatusFlg='A'
```

流动性条件(全部按 t 日信息滚动,禁止回填):

- 有效 `DlyClose` ≥ $5;
- `ADV20_t = mean(DlyPrcVol[t-19 : t])`,要求窗口内 ≥15 个有效观测,ADV20_t ≥ $5m;
- 上市 ≥ 120 个交易日(非自然日);
- `DlyCap` 排名前 1500(按 t 日值)。

## 3. 双面板分离

- 选股面板:上述过滤只回答「t 日能否入选」。
- 收益面板:持有期收益一律从未过滤全量面板提取(active 条件会滤掉 `DlyDelFlg='Y'` 记录)。

## 4. 标签:execution-return engine(总财富收益)

标签为 t+1 开盘建仓、t+6 开盘退出的 total return,分段复合:

1. 建仓:t+1 实际 `DlyOpen`(未复权原始价)。若 t+1 无有效 open(停牌/已不可交易)→ 记为 unfillable,单独归类并报告原因(停牌 vs 退市),不删除、不假设成交;
2. 首日段:t+1 open → t+1 close 价格收益 = `DlyClose(t+1)/DlyOpen(t+1) − 1`(建仓在 ex 之后,不含 t+1 股息);
3. 中段:t+2 … t+5 逐日复合 CIZ `DlyRet`(close-to-close 含息收益);
4. 退出段:t+5 close → t+6 open 隔夜价格差;若 t+6 为 ex-date,持仓过夜者有权获得该股息,此段 = `(DlyOpen(t+6) + Div(t+6)) / DlyClose(t+5) − 1`,Div 取自 distribution 表(v1.1 未覆盖此情形,v1.2 新增);
5. 退市接管:持有期内出现 `DlyDelFlg='Y'` → 从该点起复合至退市终值记录的 `DlyRet`,期末即为终值,不再要求 t+6 open 存在。

规则:

- 禁止对存续股票用「复权 open 比值」当标签(那是价格收益,与退市股的总收益口径混用会系统性偏置横截面);
- `DlyRet` 缺失的业绩类退市按 Shumway 插补(−30%/−55%)作敏感性假设,报告 ±档位;
- score = `predOpen(t+6)/predOpen(t+1) − 1`(模型只见 OHLCV,score 是价格收益预测;标签是总收益——此口径差记入文档,股息在 5 日尺度横截面影响小,v2 若加基本面再收敛)。

## 5. 复权(独立模块,不与标签引擎混用)

- 假设 `DlyFacPrc` 为当期(事件)因子而非累计因子——该语义待代码时验证(GPT 所引出处已失效);无论哪种语义,统一路径:从 CIZ distribution/corporate-action 事件构造以固定基准日为锚的累计复权因子,禁止逐日裸乘 `DlyFacPrc`。
- Kronos 蜡烛复权只处理拆股与股票股利;现金股息、分拆、配股不写入 OHLC。
- 验证:AAPL 2020-08-31、NVDA 2024-06-10 拆股,双路构造(事件累计 vs 直接用 DlyFacPrc)对比,不一致即锁定语义结论。

## 6. Kronos 微调

- 官方生成式目标不动;官方默认(finetune/config.py)lookback=90, predict=10, max_context=512。
- 本项目 predict_window = 6;lookback 消融 {60, 90, 200}(2026-08-27 按先验删 400,理由见预注册 §2;512 上限对余档无约束)。
- 特征 open/high/low/close/volume(+amount);推理按官方采样参数多路径取均值(参数名代码时核对)。

## 7. 切分与测试纪律

- Walk-forward:训 3 年 → 验 6 个月 → 滚动;
- 封存 OOS:最后 ≥2 年,设计冻结前禁止查看(lookback/阈值/调仓规则的所有选择只允许消耗验证期);
- Purge:标签视野 6 个交易日;代码断言:

  ```
  train_label_end = train_signal_date + 6 个交易日
  max(train_label_end) < min(val_signal_date)
  ```

  (显式按交易日历加 6,防日期索引实现差异的 off-by-one)。

## 7.5 评估协议(信号层 / 执行层分离,v1.3 新增)

评估分两层,回答两个不同的问题。统计检验只发生在信号层;执行层只做可行性确认。

**信号层——横截面信号是否存在(全广度):**

- 对验证期内全部通过 §2 过滤的股票计算,不受仓位数约束。
- 指标:RankIC(原始为主,winsorized 附加,§9);十分位组合 top−bottom 价差(等权,用 §4 execution-return 标签,扣 30bp 档成本);按验证窗口逐期报告,对 IC 序列均值做 Newey-West t 检验。
- **中性化诊断**:score 对 (i) 市场 beta、(ii) 个股 60 日已实现波动率做横截面回归,报告残差 RankIC。残差 IC 相对原始 IC 大幅衰减 → 信号主要是波动率/beta 排序,按失败处理(逐股 OHLCV 生成式预测的已知风险:预测里的市场共同分量与波动率会主导横截面排名)。
- 通过标准(冻结前写死,只许消耗验证期):原始与残差 RankIC 验证期均值 > 0 且 t > 2;十分位价差扣成本后 > 0。

**执行层——$3000 约束下是否可实施(全约束模拟):**

- §8 全部成本(悲观 30bp 档 + 实际费率表)、仓位数上限、缓冲区替换、阈值触发、t+1 开盘成交、unfillable 处理,全按实盘规则模拟。
- 指标:净值对 VWRETD 的超额、最大回撤、年换手、成本分解(固定费/点差/滑点占毛收益比例)。
- **选择噪声带**:每个调仓日在 score 邻近名次内随机重抽持仓,≥200 次 Monte Carlo,报告净值分布带。执行层结论只看实际路径是否落在噪声带内且带中位为正;**不做显著性声明**——小广度、周频、2 年 OOS 的统计功效不足以区分运气与边际,此局限预先写入。

**结论规则(预承诺,防事后合理化):**

1. 信号层不通过 → 停止,无论执行层净值多好看;
2. 信号层通过、执行层成本吃掉全部边际 → 信号成立但不可实施,转 v2(扩资金或改执行结构);
3. 双层通过 → 可上实盘。实盘记录仅作执行校验(实际成交价 vs 模拟假设)与监控,**不作为策略有效性的证据**——此条预先写死,防止日后用小广度实盘盈亏反推策略成败。

## 8. 成本模型(分段函数,非统一 bp)

- `Cost_t = fee_schedule(order_value, order_type) + 监管结算费 + 价差/滑点(5/15/30bp 三档)`;
- fee_schedule 按账户后台实际费率表写成分段函数:整股每单固定费(参考 moomoo AU 美股 US$0.99/单)与碎股按金额计费(可能设上限)分开处理;行动项:导出费率表原文;
- **仓位结构(v1.3 修改):广度优先。通道费率已初查(2026-08-25 网页核实,开户前以账户后台费率表为准)**:

  | 通道 | 美股费用 | $185/单的单边固定费成本 |
  | --- | --- | --- |
  | moomoo AU 整股 | US$0.99/单 | ($600/单时 16.5bp) |
  | moomoo AU 碎股 | min(0.99%, $0.99)/单,豁免 SEC/TAF 等过手费 | 50–99bp |
  | IBKR AU tiered(含碎股) | $0.0035/股,min $0.35/单,max 1% | ≈19bp |
  | Alpaca(API 券商) | $0 佣金,碎股 $1 起,称支持澳洲税务居民 | ≈0(仅点差 + 卖出侧监管过手费) |

  - 结论:moomoo 碎股费用封顶 $0.99 ≈ 整股固定费,对广度**无**改善,弃用;**主方案:Alpaca(首选,零佣金 + API 直连管线)或 IBKR tiered,等权碎股**。广度提升 IR 约 √(N/5) 倍,且执行层组合更接近信号层评估对象,缩小 §7.5 两层错配;
  - **仓位数 N 不写死,作预注册消融:N ∈ {10, 15, 20, 25}**,只消耗验证期,冻结前定案。判据取**平台而非尖峰**:10–25 档表现应大体相当;若收益对 N 敏感(仅个别档赚钱),按过拟合警报处理,回到 §7.5 信号层排查。理由:1500 只 universe 下 top-25 仍在 score 分布前 1.7%,排名 5→25 的信号稀释远小于 1/√N 的方差摊薄,N 的合理域宽,让数据定;
  - 若两条低费通道实测均不可行 → 退回 moomoo ≤5 仓整股方案,广度错配作为已知局限记入文档,§7.5 执行层的「不做显著性声明」相应更严格;
  - **已确认**:碎股不参与开盘竞价(moomoo 碎股为 9:30–16:00 ET 盘中单、当日有效、不支持附加单;Alpaca 碎股仅限盘中市价单)→ 执行层成交假设统一改为「t+1 open + 滑点档」的盘初市价单,与 §4 标签的 open 口径差单独量化报告(用开盘价 vs 开盘后 5 分钟 VWAP 的历史差作滑点档校准);
  - 行动项余留:Alpaca 澳洲个人户实测(开户、入金通道与 FX 成本、碎股成交质量);IBKR AU 的 FX 转换(约 0.002%,min $2)与入金流程;确认所选通道无月费/闲置费;
- 换手假设写死:缓冲区替换制(换一只 = 卖一单 + 买一单),不做全组合重新等权配平(配平会使单数倍增,若将来改配平制,费用表重算);
- 固定费用约束(整股回退方案下):≤5 仓、周频以下、score 改善超阈值才换仓;回测与实盘同约束。

## 9. 清洗规则(CIZ 语义)

- 无负价格逻辑(那是 legacy SIZ):`DlyPrc` 已取绝对值,报价中点由 `DlyPrcFlg` 标识;`DlyPrcFlg='BA'` 的日期单独统计与处理;
- 退市收益:CIZ `DlyRet` 在 `DlyDelFlg` 记录上已并入退市收益,禁止再套 (1+RET)(1+DLRET)−1;
- 停牌/缺失不删行、不插值、不压缩时间:市场休市日不生成行;个股在开市日缺失有效 OHLC 时,该日留空;lookback 或预测区间含缺口的训练样本整体排除(防止相隔数日的蜡烛被当作相邻),按年份 × 交易所报告排除率;收益面板不受此规则影响;缺口 mask 方案归 v2;
- 标签与评估不截尾:原始标签、组合收益、主 RankIC 用真实值(截尾会削弱退市尾部评估);winsorized RankIC 仅作附加稳健性指标;ranking label 截尾随 ranking head 归 v2。

## 10. 审计与验证

CIZ coverage audit(2000+、过滤后 universe,逐年 × 交易所):

- DlyOpen 缺失率;`DlyPrcFlg='BA'` 占比;low ≤ open/close ≤ high 违反率;拆股日复权连续性(§5 双路测试);DlyCap 覆盖与量级校验。
- 合格 → 全链 CRSP;不合格 → Norgate Platinum 作训练蜡烛备选(注意跨供应商错配)。

Golden fixtures + 不变量(替代 v1.1 的硬编码断言):

- Lehman 2008:断言其历史行、退市终值记录、收益处理路径存在且被引擎走到;具体数值从首次下载数据固化为 golden fixture,此后回归比对;
- 每年退市终值记录数 > 0;历史退市证券存在于原始层;active universe 数量合理波动(不断言单调递减);累计 distinct PERMNO 非递减;
- PERMNO+date 唯一;插补只发生在业绩类退市码上;unfillable 样本占比按年报告。

## 11. 可复现性冻结

每次数据快照记录:`data_cut_date`、实际最大交易日、查询时间戳、SQL 全文 hash、WRDS 表版本标识。实验引用快照 ID,不引用「至今」。

## 12. v2 backlog

ranking head / 二阶段 ranker(含 label 截尾)、缺口 mask、基本面(SEC EDGAR XBRL, PIT)、regime 过滤、容量分析、配平制费用重估、扩资金后的执行结构重估、**小市值第二 universe**(市值排名 1500 以下、ADV ≥ $1m:横截面异象在小盘更强且机构受容量约束进不来,是小资金唯一的结构性优势;点差/滑点档需单独实测,CIZ 数据本身已覆盖)。

---

## 修订记录

- **v1.1 → v1.2**:复权改为事件累计构造(DlyFacPrc 语义待验证,双路测试定夺);标签改 execution-return engine(五段复合、退市接管、unfillable 路径、退出日 ex-date 股息段);删除 abs/负价逻辑(改 DlyPrcFlg);SHROUT→DlyCap;ADV20 窗口写死;停牌不压缩时间 + 样本排除率报告;删除标签截尾;增加封存 OOS 与 purge 断言;费用改分段函数 + 缓冲区替换假设;锚点测试改 golden fixture + 不变量;新增可复现性冻结节。
- **v1.2 → v1.3(本版,草案)**:弃用 Fama-French 外部表,归因因子改 CRSP 面板自建(市场 VWRETD、规模 DlyCap 十分位、动量 12-1;价值随 v2 基本面),全链保持单一数据源;新增 §7.5 两层评估协议——信号层全广度统计检验(RankIC + 十分位价差 + beta/波动率中性化诊断 + 写死的通过标准)与执行层全约束模拟(成本分解 + Monte Carlo 选择噪声带,不做显著性声明)分离,附预承诺结论规则,实盘定位改为执行校验而非有效性证据;仓位结构改广度优先——低费通道 + 等权碎股为主方案,仓位数 N ∈ {10,15,20,25} 预注册消融、验证期定案,整股 ≤5 仓降为回退方案,新增碎股开盘竞价可行性行动项(后确认碎股不参与开盘竞价,执行假设改盘初市价 + 滑点档)。
