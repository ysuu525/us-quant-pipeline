# 交接文档（2026-08-27 深夜）

> 给下一个会话/下一个人。读完这份就能接着干。
> 权威细节在 `docs/` 与 `experiments/ledger.md`，本文只做导航。

---

## 1. 一句话现状

数据、环境、口径、工具链全部就绪并经实测校验；**训练池已定案 = universe
池（B 臂）**；**lookback 粗筛第一轮无淘汰，三档全存活，第二轮（补 seeds
{1,2}，24 格）正在跑**，约 2 天。

### 1.1 ⚠ 需要人拍板的一件事（在第二轮结果出来前想清楚）

第一轮读数：lb90 +0.01784（最优）> lb200 +0.01637 > lb60 +0.00587，但按
冻结判据**一档都没淘汰**（lb60 的自助分布 96.9% 偏向 lb90，差一点没够
99% 线）。若第二轮补了 seed 仍无法淘汰，**预注册的决胜规则是"取最小
lookback" → 会选中 lb60，即点估计最差的那一档**。

这条规则写在看到任何结果之前，理由是"参数暴露最少、训练推理最快"——用于
在**统计上不可分**的档之间取简。现在的问题是 lb60 与 lb90 的差距看起来
不像"不可分"。**但此刻改规则就是事后调参，会毁掉整套预注册的可信度。**
正确做法只有两条：(a) 照规则执行并把这一局限写进最终报告；(b) 若认为
规则本身有缺陷，作为一次**公开记录的设计修订**在登记簿写明理由与代价，
并接受"此决定受已见数据影响"的标注。**不要默默改。**

## 2. 正在跑什么 / 下一步

**正在跑**：`scripts\run_screen_round2.ps1`（后台，2026-08-28 起）——粗筛
第二轮，3 档 × 4 折 × seeds{1,2} = 24 格训练+评估，约 2 天。脚本幂等：
中断后原样重跑即从断点续。第一轮（`run_screen_lookback.ps1`）已完成，
结果见 `outputs/screen_lookback_round1.md`。

**已挂自动接力**（2026-08-27 19:31）：`scripts\after_screen_compare.ps1`
在后台守着上面那个跑批，12 个评估目录（3 档 × 4 折）全齐后自动执行下面
这条 compare_arms 并写 `outputs\screen_lookback_round1.md` + 追加登记簿。
日志 `outputs\after_screen_compare.log`。若训练中途死了而结果不全，它会
**拒跑并在日志里列出缺哪几折**——那时重跑 run_screen_lookback.ps1 续训，
再重挂 watcher 即可。所以下面这条命令通常**不需要你手动敲**：

**第二轮跑完后**——把每档 3 个 seed 的评估目录**全部**传给 compare_arms
（同一 (臂,折) 多个目录会自动按 seed 逐日平均），阈值收到 **95%**，判据
同前；仍 >1 档存活 → 见 §1.1 的决胜规则问题。命令形如下（第一轮版本，
第二轮需为每档补上 `_s1_` / `_s2_` 的 12 个目录）：

```bash
cd F:\quant\us-quant-pipeline && .venv\Scripts\python.exe scripts\compare_arms.py --arm lb90=outputs\fold01_lb90_s0_poolB_universe\eval_poolB_universe --arm lb90=outputs\fold02_lb90_s0_poolB_universe\eval_poolB_universe_fold02 --arm lb90=outputs\fold03_lb90_s0_poolB_universe\eval_poolB_universe_fold03 --arm lb90=outputs\fold04_lb90_s0_poolB_universe\eval_poolB_universe_fold04 --arm lb60=outputs\fold01_lb60_s0_poolB_universe\eval_screen_lb60_fold01 --arm lb60=outputs\fold02_lb60_s0_poolB_universe\eval_screen_lb60_fold02 --arm lb60=outputs\fold03_lb60_s0_poolB_universe\eval_screen_lb60_fold03 --arm lb60=outputs\fold04_lb60_s0_poolB_universe\eval_screen_lb60_fold04 --arm lb200=outputs\fold01_lb200_s0_poolB_universe\eval_screen_lb200_fold01 --arm lb200=outputs\fold02_lb200_s0_poolB_universe\eval_screen_lb200_fold02 --arm lb200=outputs\fold03_lb200_s0_poolB_universe\eval_screen_lb200_fold03 --arm lb200=outputs\fold04_lb200_s0_poolB_universe\eval_screen_lb200_fold04 --baseline lb90 --out outputs\screen_lookback_round1.md --ledger
```

**判据（预注册 §2 第一轮，已冻结）**：淘汰某档当且仅当"最优档 − 该档"的
99% 块自助 CI 全体 > 0 **且** ≥3/4 折同向。存活档进第二轮（补 seeds
{1,2}，阈值收到 95%）；仍平局 → **取最小 lookback**。
**不要凭均值眼看下结论**——池子消融已经演示过：三折时自助 CI 说 B 赢但
折同向只有 2/3，判据正确判了"不可分"，第四折补上才成立。

**再下一步**：胜出档 → 42 折全跑（每折随机 seed）→ §7.5 信号层评估。

### 2.1 已定案：训练池 = B（universe 内 anchor）

四折配对终审（`outputs/ablation_pool_lb90.md`）：平均配对差 **+0.00855**，
99% CI [+0.00108, +0.01959] 全体 > 0，折内同向 **3/4** → 两轮判据均判胜出。
四折 RankIC 均值 A=+0.00928 / B=+0.01784。C 臂（质量过滤）随之弃用。
**此后所有训练一律加 `--universe-parquet`。**

## 3. 环境与路径

| 东西 | 位置 |
|---|---|
| 代码 | `F:\quant\us-quant-pipeline`（git repo，分支 main） |
| 训练 venv | `.venv`（Python 3.12 + CUDA torch；**不要往里装 wrds**） |
| 下载器 venv | `downloader\.venv-dl`（wrds 会把 pandas 降级，故隔离） |
| CRSP 快照 | 日线 `F:\quant\crsp_ciz_snapshots\crsp_ciz_2026-08-24_...`；小表 `F:\quant\crsp_ciz_2026-08-25_...`（两处是预期布局，数据齐全） |
| 整理后面板 | `F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z\` |
| JKP 外部因子 | `F:\quant\external\jkp\`（305MB，CC BY-NC **禁商用**） |
| 训练产物 | `outputs\fold<NN>_lb90_s0[_poolB_universe\|_poolC_quality]\` |
| 登记簿 | `experiments\ledger.md`（append-only，所有决定与验证窗查看） |

常用命令：

```bash
.venv\Scripts\python.exe -m pytest -q
```

```bash
.venv\Scripts\python.exe -m kronos_ft.train --panel F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z\panel_kronos_adj.parquet --index-parquet F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z\market_index.parquet --train-start 2000-01-03 --train-end 2002-12-20 --lookback 90 --seed 0 --stage both --out outputs\foldNN_lbXX_sY --index-cache F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z\index_cache\lbXX_full.parquet
```

```bash
.venv\Scripts\python.exe scripts\evaluate_fold.py --model-dir outputs\foldNN_lbXX_sY --processed F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z --val-start YYYY-MM-DD --val-end YYYY-MM-DD --lookback XX --tag <标识>
```

折边界（42 折，oos_start=2024-01-01）：fold01 训 2000-01-03..2002-12-20 /
验 2003-01-02..2003-06-30，此后每折整体前滚 6 个月。完整列表用
`crsp_pipeline.splits.walk_forward_folds` 生成。

## 4. 今天定下的决定（全部在登记簿 + 预注册修订记录里）

**数据口径（真实数据核对后冻结）**
- 复权事件 = `distype='FRS'`，factor = 1 + disfacshr；
- `DlyFacPrc` = 当期事件因子（AAPL/NVDA 双路验证）；
- `DlyCap` 单位 $千；`DlyPrcVol` = DlyClose × DlyVol 精确；
- §10 审计合格 → 全链 CRSP，不需要 Norgate 备选。

**实验设计**
- 训练池：A/B/C 三臂消融（进行中，见 §2）；
- lookback 消融集 = **{60, 90, 200}**（按先验删 400，五条理由在预注册 §2）；
- 筛选 = racing 两轮制 + 配对块自助判据 + 最小 lookback 决胜 + 全折随机
  seed + 第 21 折漂移复检（预算 ~92 次训练）；
- **§7.5 头条统计只用折 05–42**（折 01–04 被选档消耗，单独报告）；
- OOS 封存起点 2024-01-01 不变，但**主判读窗 = 2024-07..2025-12**
  （Kronos 预训练数据截至 2024-06 且含美股，2024H1 被基座见过）。

**与官方 Kronos 的关系**
- 三处已对齐：抽样改回官方 n_iter 契约、tokenizer 早停指标改 `mse(z,x)`、
  batch 50；
- 刻意偏离（有理由）：predict=6、窗口少取 1 行、早停+SWA、内层验证切法、
  seed 随 cfg.seed 变化、num_workers=0、无 DDP；
- **完整清单：`docs/与官方微调差异清单_2026-08-27.md`**——不在清单上的
  行为即与官方一致；新偏离必须先记清单再进代码。

**上线相关（尚未执行）**
- **PDT 规则已于 2026-06-04 废除**（FINRA Notice 26-10，官网核实）→
  预注册 §4 已改写为保证金约束；
- 新增预注册 §4.1 衰减监控与停机规则（**占位待填数**）；
- 券商排除 IBKR Pro（$120 仓位往返 1.67%）；
- FF 平替用 JKP（数据已下载；注意其 FF3 复制因子未公开，用替代映射）。

## 5. 代码变更清单（相对上游 main）

**新增**
- `src/crsp_pipeline/snapshot.py` — 快照加载层
- `scripts/prepare_data.py` — 快照 → 面板/universe/审计
- `scripts/evaluate_fold.py` — 单折打分 + RankIC/分层 IC/十分位
- `scripts/compare_arms.py` — 配对块自助判据（预注册 §2 实现）
- `docs/` 六份调研与核查文档、`experiments/ledger.md`

**修改**
- `adjust.py` +`split_events_from_distributions`/`adjust_panel`（冻结规则）
- `cleaning.py` +`quality_ok_mask`（C 臂质量过滤）
- `windows.py` +`extra_valid` 参数、`filter_index_by_universe`（B/C 臂）
- `dataset.py` 官方抽样契约 + 预载裁剪
- `train.py` 抽样/验证口径对齐、**断点续训**、索引缓存、池子消融开关
- `infer.py` **fast 打分路径**（CPU 侧 98×，整体 22%；与官方逐位一致）
- `setup_windows.ps1` 存成 UTF-8 **带 BOM**（否则 PS 5.1 解析中文注释报错）

测试 **106 项全绿**（`pytest -q`）。

## 6. 坑（都踩过）

| 坑 | 处理 |
|---|---|
| 装 `wrds` 会把训练 venv 的 pandas 降级 | 下载器用独立 venv，已隔离并在其 README 写了警告 |
| 训练中 CUDA OOM（桌面程序抢显存） | 训练本身只用约 5GB/16GB；**原命令重跑即自动续训**（已完成阶段跳过、未完成阶段逐位续上） |
| 训练进程占 12–22GB 内存，此时跑 pytest 会撞页面文件上限 | 等训练间隙跑测试，或先停训练 |
| PowerShell 传 `-c "..."` 时引号被吃 | 写成脚本文件再跑 |
| `.ps1` 存成无 BOM 的 UTF-8 | PS 5.1 会把中文注释解析成语法错误，必须带 BOM |
| **不要在消融/粗筛期间改 batch size 或精度** | 会改变 RNG 消耗 → 采样路径变化 → 臂间差异混入采样噪声。留到全部定案后再动并重跑对照 |
| 训练无声退出、无 traceback | 查 `Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'}`。2026-08-28 出过一次 Event 153（驱动错误/GPU 复位）打断跑批。跑批脚本已加**单步重试 ×3**（靠续训零成本）；偶发即可，若变频繁需查驱动/供电 |

## 7. 待办（按优先级）

1. **[卡住流程]** fold04-B 评估完成 → 跑 §2 那条终审命令 → 池子定案；
2. lookback 粗筛第一轮（12 训 + 12 评）；
3. `git commit`（**目前全部改动未提交**，见 §8）；
4. Phase 5 剩余：Lehman 2008 golden fixture、业绩类退市码冻结 + labels 接线；
5. 预注册 §4.1 填数（材料已备齐在 `docs/上线决策调研_2026-08-27.md`）；
6. 消融定案后：评估的 GPU 侧提速（batch/bf16）+ 对照重跑。

## 8. git 状态（重要）

**所有工作都未提交**，分支 `main`，上游最新提交是 `6338cb3 Phase 4`。
18 个文件被修改、10 个未跟踪。交接前建议：

- 新建分支再提交（不要直接堆在 main 上）；
- `.gitignore` 需排除：`downloader/.venv-dl/`、`.venv/`、`outputs/`、
  `configs/local.yaml`（本机路径）；
- `experiments/ledger.md` **要提交**（预注册 §3 要求 append-only 随 git 走）；
- 数据一律不进 git（§11：CRSP 是授权数据）。
