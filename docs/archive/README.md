# 归档：已过期 / 已被取代的文档

放这里的文件**内容一字未改**，只是移出 `docs/` 主视野，避免新会话通读时
被过期内容污染上下文。历史引用若指向旧路径，按下表一跳即达。

| 旧路径 | 现路径 | 归档理由（2026-09-04 核） |
|---|---|---|
| `docs/HANDOFF_2026-08-27_已过期.md` | `docs/archive/` | 文件名自陈过期；已被 `HANDOFF.md` 取代，全仓零引用 |
| `docs/美股DL量化管线规范_v1.2.md` | `docs/archive/` | 被 `docs/美股DL量化管线规范_v1.3.md` 取代（README 指向 v1.3），零引用 |
| `docs/moomoo持仓管理规则_v0.1.md` | `docs/archive/` | 券商已定 Alpaca（HANDOFF §11.4a），moomoo 路线作废，零引用 |
| `docs/FF平替调研_2026-08-26.md` | `docs/archive/` | 08-26 因子源调研；结论已并入 JKP 方案，仅被同批归档的《开源因子评估》引用 |
| `docs/开源因子评估_2026-08-26.md` | `docs/archive/` | 同上，零外部引用 |
| `docs/变现结构与同类基准调研_2026-08-27.md` | `docs/archive/` | 08-27 变现调研；**仍被 `docs/审稿_席位报告_2026-09-03/R1_methodology.md:143` 按行号引为证据**（`:402` 处的 NW lag 论点），行号未变，跟到本目录即可 |
| `HANDOFF_STRONG_BASELINE.md`（仓库根） | `docs/archive/` | GBDT+JKP 强基线任务已完成（`outputs/gbdt_strong_jkp_v2/`），三份权威文件零引用 |

## 明确**没有**归档、仍在 `docs/` / `experiments/` 原位的近似候选

- `docs/审稿_研究计划书_2026-09-03.md` —— 被**当前权威**的 `docs/研究计划书_v0.2_2026-09-04.md`
  开篇引为所逐条回应的审稿意见；投稿渠道决定也记在其中。
- `experiments/signal2_prereg_v1.md` —— 被 `experiments/signal2_prereg_v2.md`（在预注册清单内）正文引用。
- `experiments/confirmation_protocol_v3.md`、`docs/研究计划书_2026-09-03.md` ——
  **在 `scripts/preregistration_manifest.py` 的固定清单里**，被 OTS 时间戳按仓库根相对路径钉了 sha256。
  内容虽已被 v4 修订清单 / v0.2 取代，但**移动或删除会破坏 `--verify`**，一律不动。
