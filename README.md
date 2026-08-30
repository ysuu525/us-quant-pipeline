# us-quant-pipeline

美股 DL 量化管线：Kronos 微调 → 日频横截面 score → 低频执行。
规范见 [docs/美股DL量化管线规范_v1.3.md](docs/美股DL量化管线规范_v1.3.md)（冻结依据，代码一律对应其章节号）。

**数据不进 git。** CRSP 是授权数据，快照（parquet）只在本地/移动硬盘/网盘之间传；
git 里只有代码、测试、配置和快照 manifest 的引用（§11 可复现性）。

## 仓库结构

```
docs/                  规范 v1.2 / v1.3、持仓管理规则（只读参考）
downloader/            CRSP CIZ 下载器（独立可用，见其 README）
src/crsp_pipeline/     管线代码
  calendar.py            交易日历（§7/§9 日期算术基础）
  universe.py            选股面板：静态 CIZ 筛选 + 流动性条件（§2/§3）
  labels.py              execution-return 标签引擎（§4）
  adjust.py              事件累计复权 + DlyFacPrc 双路验证（§5）
  cleaning.py            BA 统计、lookback 缺口排除（§9）
  splits.py              walk-forward + purge 断言 + 封存 OOS（§7）
  factors.py             自建归因因子 + beta/波动率暴露（§1/§7.5）
  signal_eval.py         信号层：RankIC / Newey-West / 十分位价差 / 中性化 + 冻结通过标准（§7.5）
  costs.py               分段成本模型、通道预设、监管过手费、滑点档（§8）
  execution_sim.py       执行层：缓冲区替换模拟 + Monte Carlo 选择噪声带（§7.5/§8）
src/kronos_ft/         Kronos 微调层（§6 + docs/预注册_v1.md）
  windows.py             训练/推理窗口索引 + 内层验证切分（purge 同 §7）
  dataset.py             官方归一化契约的 torch Dataset
  models.py              预训练加载 / 冒烟小模型 / SWA 权重平均 / device 降级
  train.py               单卡两阶段微调：内层 loss 早停 + SWA-3 + 登记簿 + --smoke
  infer.py               多路径采样打分：score = predOpen(t+6)/predOpen(t+1) − 1
third_party/kronos/    官方仓库 submodule（钉死 commit，不改动）
configs/               default.yaml / experiment.yaml 模板；本机路径写 local.yaml（不进 git）
tests/                 合成数据单元测试（标签 golden case、purge off-by-one 等）
```

## 环境（Mac / Windows 通用）

Python ≥ 3.10。克隆要带 submodule：

```bash
git clone --recurse-submodules https://github.com/ysuu525/us-quant-pipeline.git
```

（已克隆的补一句 `git submodule update --init`。）然后：

```bash
pip install -e ".[dev,train]"
pytest
python -m kronos_ft.train --smoke
```

torch 按平台装：Mac 直接 `pip install torch`（MPS/CPU）；Windows 用 CUDA
wheel（Phase 4 的 setup 脚本负责）。克隆后第一件事是 `pytest` 全绿 +
冒烟 PASS 才继续。（2026-08-26 起开发与训练都在 Windows 机上进行。）

## Windows（4080 Super）使用流程

全部步骤见 [RUNBOOK_WINDOWS.md](RUNBOOK_WINDOWS.md)。一句话版：

```powershell
git clone --recurse-submodules https://github.com/ysuu525/us-quant-pipeline.git
cd us-quant-pipeline
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

脚本自动完成 venv、CUDA torch、依赖、CUDA 自检、pytest、训练冒烟；之后
按 RUNBOOK 配置数据路径开训。

## 尚未落地 / 待真实数据核对的口径

这些点在代码里以**参数/谓词注入**留了口子，注释里均有标记，接入真实数据后冻结。
**2026-08-26 用真实快照核对后已冻结的**（探查与审计见 `scripts/prepare_data.py`
产出的 `audit/report.md`）：

- ~~CIZ distribution 复权事件码~~：**已冻结** = `distype='FRS'`
  （disdetailtype ∈ {STKSPL, STKDIV}），factor = 1 + disfacshr
  （`adjust.split_events_from_distributions`）；现金股息 = `distype='CD'`
  （标签退出段用，待 labels 接线时复核）；
- ~~`DlyFacPrc` 语义~~：**已验证 = 当期事件因子**（AAPL 2020-08-31、
  NVDA 2024-06-10 双路均判 event）；管线仍走事件累计路径，DlyFacPrc 只作交叉审计；
- ~~上市日~~：`prepare_data.py` 的 universe 阶段已改传
  `stksecurityinfohist.securitybegdt` 为 `first_trade_dates`；
- 另经核对：`DlyCap` 单位 = $千（AAPL 量级校验过）；`DlyPrcVol` =
  DlyClose × DlyVol 精确成立。

**仍待核对的**：

- CIZ 业绩类退市码集合（`labels.compute_label` 的 `is_performance_delist`，
  Shumway 插补只允许作用于业绩类退市）；
- 退市终值记录的 `DlyClose` 是否即终值（退市恰发生在 t+1 时首日段的口径），
  用 §10 Lehman 2008 golden fixture 核对；
- 监管过手费（SEC fee / FINRA TAF）默认值为公开档位，开户后按账户后台
  费率表原文重配 `costs.RegulatoryFees` 与通道 `FeeSchedule`（§8 行动项）；
- 执行层时序近似：模拟按「t+1 交易日边界成交、买入腿享当日 close-to-close
  收益」处理，open 与前收的隔夜差并入滑点档；Phase 5 接真实 `DlyOpen` 精化
  并量化与 §4 标签的口径差。

## 阶段进度

- [x] Phase 0 仓库骨架 / 配置
- [x] Phase 1 数据层与标签层（§2–§5、§7、§9）+ 合成数据测试
- [x] Phase 2 评估与成本模块（§7.5、§8、§1 自建归因因子）
- [x] Phase 3 Kronos 微调接入（§6 + 预注册 v1）+ Mac 冒烟
- [x] Phase 4 Windows 部署材料（setup_windows.ps1、RUNBOOK_WINDOWS.md）
- [ ] Phase 5 真实数据核对——进行中（2026-08-26）：
  - [x] 快照加载层 `crsp_pipeline.snapshot` + 准备脚本 `scripts/prepare_data.py`
  - [x] coverage audit §10（逐年缺失率/BA 占比/OHLC 一致性/DlyCap 量级/退市不变量）
  - [x] 双路复权验证 §5（AAPL/NVDA → DlyFacPrc = 当期事件因子；事件码 FRS 冻结）
  - [ ] golden fixtures（Lehman 2008 退市路径、标签引擎实数据 golden case）
  - [ ] 业绩类退市码集合冻结 + labels 接线
