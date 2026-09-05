# opus 执行者草稿：树基线封存计算（2026-09-05）

> 主会话审后并入 `experiments/ledger.md`。下列条目逐字可用。
> **本文件与其产物均不含任何封存窗口（2005-01-03 起）的绩效读数。**

---

## 草稿 1：口径延伸（配置 sha256 变化的正式登记）

- 2026-09-05 | caliber-extension | **[opus-gbdt] 树基线封存计算的口径延伸：只扩窗口与 JKP 派生快照，超参网格 / seeds / 内层选参规则 / 轮数档位一字未改。** 授权依据为同日 `authorisation-and-sealed-mode` 行（H3 处置取第一条）。**冻结配置分叉为新文件**：`configs/gbdt_strong_v2.json`（sha256 `18d10bc6bd41347bf3deabf03a9c015e9a154d30b3bb19c1e3bb56fec444ee24`，`ledger.md:185` 登记值，**本次未改动、逐字节保持**）→ 新增 `configs/gbdt_strong_v2_sealed.json`（sha256 `c65ba72992f439219f2f07516715d95630383964de3bbd6708c04b1e6d66ca57`）。两份配置的**唯一实质差异是两个字段**（`json.dumps(sort_keys=True)` 逐行 diff 实测，除文档块外只有这两行）：`features.jkp_state_snapshot` 与 `features.jkp_state_snapshot_sha256`。新增的 `sealed_extension_2026_09_05` 块只记录本次延伸的理由与范围，不参与任何计算。**未改**：`grids`（LGBM/XGB/CatBoost 三张 2×2×2 网格）、`seeds`=[11,29,47]、`tuning_seed`=20260831、`round_checkpoints`=[300,700]、`inner_validation`（尾部 126 交易日、每日 384 只、选参指标与并列打破规则）、`final_fit`、`training_label`、`num_threads`=6、`max_rss_gb`=8.0、`compute`、`sealed_raw_data_cutoff`=2023-12-29、26 个个股特征的定义。**新 JKP 派生快照**：`F:\quant\external\jkp\derived\usa_all_factors_daily_vw_cap_state_lag1_5_20_60_2001-10-01_through_2023-12-29.parquet`，sha256 `669a1e2381cf3889b263c4c70374fefed22d8e451a56204b594b8696f709d944`，5601 日 × 459 列，物理窗 2001-10-01..2023-12-29（旧快照 sha256 `bbcfc2a2d9f1b02d67c658ccb3cbbe78ebef9af1fab7452e271702302ab5f906`，1760 日 × 459 列，物理窗 2017-01-03..2023-12-29）；源 CSV `usa_all_factors_daily_vw_cap.csv` sha256 `b4a248b40e071b544dd5fbb0bce525718f58fb4d5f55cb639702292eec90f7a8`；公式 / 窗 / lag 与旧快照逐字相同（`expm1(rolling_sum(log1p(ret.shift(1))))`，窗 5/20/60，lag=1）。**重建脚本 `scripts/build_jkp_state_snapshot.py`（新增）**。**保真度断言（两项，均实测）**：① **同窗重建逐位相同** —— 用同一脚本在旧快照自己的物理窗 2017-01-03..2023-12-29 上重建，与旧快照 459 列 × 1760 行**最大绝对差 0.0**、NaN 图样完全一致、列名与列序完全一致；② **扩窗快照在共享区间上的最大绝对差 1.11e-16**（794835 个双方有定义的格子），**低于预注册容差 1e-12**。**唯一的结构性差异是旧快照自己的滚动预热**：旧快照按 2017-01-03 物理截断后再滚动，故前 5/20/60 行（cum5/cum20/cum60，止于 2017-03-29）为 NaN，共 13005 个格子；新快照因有更长历史而把这些格子填满。**这些日期从未进入任何折**（旧口径下 `_prepare_jkp_cache` 的有限性断言只覆盖 `date >= min(train_start) = 2017-07-03`），故对已发布的七折开发读数零影响。**代码 sha256 的连带变化（必须披露）**：`scripts/gbdt_baseline.py` 由 `3c4e59f4349b27333c5277341f7a74e69890dcd2e25b1b6be1967bd5b89254f8`（= 折 36–42 各 `fold_summary.json` 里 `provenance.pipeline_script_sha256` 的登记值，已核对与 `git show HEAD:scripts/gbdt_baseline.py` 逐位一致）改为 `641b8d4ef089d60cb2261b36a4c4398269b768369946bc297a23cf028089a43e`；`PIPELINE_BUILD_ID` 保持 `gbdt-strong-v2-20260831` 不变。**后果**：以当前工作区重跑 `--summarize-only` 会因 provenance 不符而 fail-closed 报错，这是设计内的 fail-closed 行为，**不是结果失效**；复核折 36–42 需先 `git show <本次提交的父提交>:scripts/gbdt_baseline.py` 取回原脚本。**改动清单（只加不改）**：`FOLDS` 七折默认值一字未动，新增可注入的 `ACTIVE_FOLDS` 与 `--folds-json`；新增 `--sealed` / `--append-ledger` / `--cache-dir`；年度缓存的年份区间由折表推出（对七折默认口径逐字节等价，实测 `target_hi` 取值不变）；`src/crsp_pipeline/sealed.py` 的 `audit_dir` 增加可选 `extra_allowed`、`write_seal` 的 `authorisation` 改为 `setdefault`（不传时行为与旧版逐字相同）。

## 草稿 2：封存计算的执行与守卫（不含任何读数）

- 2026-09-05 | sealed-compute-run | **[opus-gbdt] 树基线（XGBoost 主口径）在折 05–35 上完成计算专用的训练与打分；只产出分数，未计算任何指标。** 范围严格按同日授权：**只跑 XGBoost**，LightGBM / CatBoost 不跑（披露：v4 §2.6 的另两口径在确认集上无产物）。折表由 `.venv\Scripts\python.exe scripts\emit_folds.py --processed <P> --first 5 --last 35` 机械产出，落盘 `outputs/gbdt_strong_jkp_v2/folds_05_35.json`（31 折，fold05 train=[2002-01-03..2004-12-22] val=[2005-01-03..2005-07-01]，fold35 val 止于 2020-07-02），**未手写任何窗口**。年度特征缓存重建 2002–2020（`--prepare-only`），落在 `outputs/gbdt_strong_jkp_v2/cache_sealed/`；**该缓存带训练目标列 `y`（截面去均值的未来 6 个市场交易日复合回报），故整个目录自带 `SEALED` 哨兵与清单**，`crsp_pipeline.sealed.assert_readable` 覆盖其下全部文件 —— 这是授权里「标签不得以可读形式落盘」的落实方式（授权原文允许「把含标签的缓存写到带 SEALED 哨兵的目录并在报告里说明」）。开发折的 `outputs/gbdt_strong_jkp_v2/cache/` 与 `preregister.json` **一字未动**，封存侧另写 `preregister_sealed.json`。每折产物在 `outputs/gbdt_strong_jkp_v2/xgboost/sealed/foldNN/`：`scores.parquet`（列 `PERMNO, signal_date, score`，三 seed 算术均值，即开发折 `scores_ensemble` 的同一口径）+ 三个 `scores_seed{11,29,47}.parquet` 及其 sidecar + `tuning.json`（内层选参记录）+ `SEALED` + `SEALED_MANIFEST.json`，全部过 `sealed.write_seal` 与 `audit_dir`（clean=true）。**未写**：`labels.parquet`、`daily_ic_*.parquet`、`fold_summary.json`、`scores_ensemble.parquet`、`metrics.json`、`summary.json` —— `--sealed` 路径在 `run_fold` 里于加载标签之前硬性 return，冒烟里把 `_load_labels` / `_daily_ic` / `_fold_result` / `_label_path` 四个入口全部替换成会抛异常的地雷后仍能跑通，即物理上进不了标签分支。**内层选参的一处必须披露的实情**：冻结规则是「训练窗尾部 126 交易日」，对靠后的折（如 fold35，内层验证段落在 2019H2）该段本身落在封存区间内；该量是冻结训练流程的内部量、授权明确覆盖「训练（含冻结的内层选参）」，故 `tuning.json` **留在带哨兵的封存目录内、全程不打印、不导出**（`--sealed` 下把原来的 best-inner 打印替换为不含数值的一行）。自动测试新增 `tests/test_gbdt_sealed.py`（端到端实现在 `scripts/gbdt_sealed_smoke.py`，因 `.venv-gbdt` 无 pytest、`.venv` 无 xgboost），断言封存目录无禁止产物、清单字段齐全、读取守卫生效、折表不可手写、封存模式拒绝 `--summarize-only`、含标签缓存自带哨兵。队列脚本 `scripts/run_gbdt_sealed_queue.ps1`（互斥锁 + 四次原命令重试 + 按 `SEALED_MANIFEST.json` 断点续跑），日志在 `outputs/gbdt_strong_jkp_v2/xgboost/sealed/_logs/`。**逐折登记行由 `--append-ledger` 自动追加，格式仿 2026-09-01 Kronos 队列，只含 tag / 模型 / 验证窗 / 行数 / 日数 / seeds，不含任何指标；按 tag 幂等，续跑不会重复。** 计算授权 != 读取授权：这些分数在用户对冻结版 v4 给出书面「读」之前不得打开。

---

## 附：本次涉及的全部 sha256

| 对象 | 旧 | 新 |
|---|---|---|
| `configs/gbdt_strong_v2.json` | `18d10bc6bd41347bf3deabf03a9c015e9a154d30b3bb19c1e3bb56fec444ee24` | 未改动 |
| `configs/gbdt_strong_v2_sealed.json` | —（新增） | `c65ba72992f439219f2f07516715d95630383964de3bbd6708c04b1e6d66ca57` |
| JKP 派生快照 | `bbcfc2a2d9f1b02d67c658ccb3cbbe78ebef9af1fab7452e271702302ab5f906` | `669a1e2381cf3889b263c4c70374fefed22d8e451a56204b594b8696f709d944` |
| JKP 源 CSV | `b4a248b40e071b544dd5fbb0bce525718f58fb4d5f55cb639702292eec90f7a8` | 未改动 |
| `scripts/gbdt_baseline.py` | `3c4e59f4349b27333c5277341f7a74e69890dcd2e25b1b6be1967bd5b89254f8` | `641b8d4ef089d60cb2261b36a4c4398269b768369946bc297a23cf028089a43e` |
| `src/crsp_pipeline/sealed.py` | （见本次提交的父提交） | `e2c8dc027e1c4be7ba7e1ad1b1b6308d1a7eec27c986408573fb30bb26f3d754` |

队列完成折数与总耗时：见执行者最终消息与 `outputs/gbdt_strong_jkp_v2/xgboost/sealed/_logs/`。
快照重建的完整比对记录：`outputs/gbdt_strong_jkp_v2/jkp_snapshot_rebuild_report.json`。

---

## 续跑 / 排障

- **续跑命令**（幂等，已完成的折按 `SEALED_MANIFEST.json` 跳过）：
  `powershell -NoProfile -ExecutionPolicy Bypass -File "F:\quant\us-quant-pipeline\scripts\run_gbdt_sealed_queue.ps1"`
- **锁文件**：`outputs/gbdt_strong_jkp_v2/xgboost/sealed/.queue.lock`（内容为队列进程 PID；
  进程已死时脚本会自动清锁，进程活着时直接退出，不会并发）。
- **完成哨兵**：`outputs/gbdt_strong_jkp_v2/xgboost/sealed/QUEUE.DONE`。
- **日志**：`outputs/gbdt_strong_jkp_v2/xgboost/sealed/_logs/fold_foldNN.log(.err)`。
- **单折重跑**：
  `.venv-gbdt\Scripts\python.exe scripts\gbdt_baseline.py --config configs\gbdt_strong_v2_sealed.json --out-dir outputs\gbdt_strong_jkp_v2 --cache-dir outputs\gbdt_strong_jkp_v2\cache_sealed --folds-json outputs\gbdt_strong_jkp_v2\folds_05_35.json --models xgboost --folds foldNN --sealed --append-ledger`
- **踩过的坑（已修）**：`.ps1` 必须写成 **UTF-8 with BOM**。PS 5.1 在本机（ANSI 代码页 936）
  把无 BOM 的 UTF-8 脚本按 GBK 解码，中文注释会吞掉后续行，导致 `emit_folds` 那一行根本没执行、
  `$LASTEXITCODE` 保持 `$null` 而误判失败。既有的 `scripts/run_sealed_confirm_queue.ps1` 带 BOM，
  新脚本首版没带，首次启动即失败一次（已加 BOM，第二次启动正常）。

---

## 队列中断与续跑的实测记录（2026-09-05）

- 首轮队列（受管后台）在本地 09-05 05:09 前后随会话一并被杀，**fold09 停在第一个 seed**。
- 断点续跑实测**按设计工作**：重启后 fold05–08 按 `SEALED_MANIFEST.json` 跳过；fold09 复用了
  已通过 provenance 校验的 `tuning.json` 与 `scores_seed11.parquet`（不重跑内层选参、不重跑该 seed），
  直接从 seed29 继续。全部 `.err` 日志为 0 字节。
- **单折峰值 RSS 随年代上行**：fold05 5.51 / fold06 6.53 / fold07 6.74 / fold08 6.95 GB，
  冻结硬限 8.0 GB。开发折（2017–2023 训练窗）的历史峰值是 6.88 GB，故预期在 6.9–7.0 GB 附近平台化，
  但余量只有约 1 GB。**若后续折触到 8.0 GB 上限，`_check_memory` 会 fail-closed 抛 MemoryError，
  队列的四次原命令重试救不了（该失败是确定性的）**；届时须由用户裁定是否调整 `max_rss_gb`
  ——那是冻结配置里的字段，执行者不得自行改。

### 受管后台任务的 60 分钟硬上限（实测，需用户裁定）

**现象**：以受管后台方式启动的队列进程**每约 60 分钟被杀一次**，两次实测一致 ——
第一轮 04:11 起、约 05:09 死（58 min）；第二轮 13:37 起、14:37 死（60 min）。
被杀时整棵进程树一起死（无孤儿 python，锁文件的 PID 随之失效，脚本下次启动会自动清锁），
**不存在并发双跑的风险**。

**代价可控**：seed 级断点续跑实测有效，每次被杀最多损失一个正在拟合的 seed
（约 2–5 分钟），已完成的 `tuning.json` 与 `scores_seed*.parquet` 都按 provenance 校验后复用。
fold09、fold12 两次都是这样接上的。

**后果**：队列无法在一次会话内自己跑完 31 折（剩余约 6 h，需 6–7 次重新拉起），
且**会话结束后队列不会自己继续**。

**待裁定**：是否授权改用**脱离会话的启动方式**（如 `schtasks /create /sc once` 或
计划任务托管）让队列自行跑完。CLAUDE.md §七 记的是「分离启动的后台进程会夭折」，
与这里的失效模式相反，故不宜由执行者自行改变启动方式 —— 保持受管后台并逐次重启，
或由用户裁定改为计划任务。

### 会话外启动（`schtasks`）实测失败，已退回受管模式（2026-09-05，主会话裁定 1 的执行结果）

**任务名**：`GbdtSealedQueue20260905`
**创建**：2026-09-05 14:41（本地），`schtasks /create /tn GbdtSealedQueue20260905
/tr "F:\quant\us-quant-pipeline\scripts\run_gbdt_sealed_queue_task.cmd" /sc once /st 14:42 /f`
—— 当前用户 `admin`、非提权、Logon Mode = `Interactive only`；包装脚本
`scripts/run_gbdt_sealed_queue_task.cmd`（纯 ASCII，只加一层控制台重定向到
`_logs/queue_task.log`，解释器 / 配置 / 日志 / 锁与受管方式完全一致）。
**删除**：2026-09-05 14:43（本地），`schtasks /delete /f`，已确认查询报 "cannot find the file"。

**结果：夭折，存活约 12 秒。** 14:42:01 启动、正确清掉上一份的失效锁、跳过 fold05–11、
14:42:03 开始 fold12 的 seed29 拟合，14:42:13 整棵树消失；任务状态回到 `Ready`，
**`Last Result: -1073741510` = `0xC000013A` = `STATUS_CONTROL_C_EXIT`** ——
进程收到 Ctrl+C / 控制台关闭事件而非自身报错（`fold_fold12.log.err` 为空，
python 侧没有任何异常）。这正是 CLAUDE.md §七「分离启动的后台进程会夭折」记录的现象，
本次给出了它的具体退出码。

**按裁定 1 条件 (b) 退回受管重启模式**：14:43:09 以受管后台重新拉起（PS PID 36024，
python 子进程 6.54 GB 工作集），锁与日志均在写，fold12 从 seed29 续上。

**留给用户的一个未试选项（执行者不自行改动，因超出裁定给的参数）**：失败原因是
`Interactive only` 会把任务挂到交互式控制台会话上。若改成「不管用户是否登录都运行」
（`/ru <user> /rp <password>` 或 S4U `/np`），任务将脱离交互控制台，可能不再收到
`CTRL_CLOSE_EVENT`。这需要改变裁定里写死的「当前用户、非提权」参数（S4U 还涉及
「作为批处理作业登录」权限），故须用户另行裁定。

### 内存上限的预授权（主会话裁定 2，2026-09-05）—— 截至目前**未触发、未使用**

用户预授权：若任何折触发 `_check_memory` 的 8.0 GB fail-closed，允许把
`configs/gbdt_strong_v2_sealed.json` 的 `max_rss_gb` 由 8.0 提到 12.0（保护参数，
不是训练参数；其余字段逐字不变）。改后 `config_sha256` 会变，届时须在 `SEALED_MANIFEST.json`
与本草稿里写明**哪些折在旧上限下跑、哪些在新上限下跑**。

**当前状态：未触发。** 已发布七折的峰值 RSS 为 5.51 / 6.53 / 6.74 / 6.95 /（fold09 续跑段
2.97）/ fold10、fold11 见日志，全部 < 8.0 GB，故 `max_rss_gb` 保持 8.0、
`config_sha256` 仍为 `c65ba72992f439219f2f07516715d95630383964de3bbd6708c04b1e6d66ca57`。
若后续触发，将在此处补记切换点与两段的 sha256。

**用户裁定（2026-09-05，紧接上条）**：上面那个「未试选项」**不试** —— `/ru`+`/rp` 与 S4U `/np`
都需要账户凭据或「作为批处理作业登录」权限，凭据类操作主会话不做、也不让 agent 做。
**维持受管重启模式直到队列完成**，每次被杀后按现有方式重新拉起。
退出码 `0xC000013A` 与 `Logon Mode = Interactive only` 这条根因保留在案。
