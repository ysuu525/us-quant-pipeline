"""报告渲染。**释放清单的最后一道闸门。**

三条硬约束（v4 §3 / 任务书 §0）：

1. **只出现释放清单允许的量**：合并点估计、CI、正折数、逐折**符号**计数、
   收缩比与相交关系、换手、BE ± CI、成本网格、H6 的均值/CI/t/事件数/触发日数/
   同向折数。清单以外的一切量不写。
2. **不得出现逐折 IC 表**——报告里连折号都不出现，只写「折 05–35（31 折）」这样的
   区间与计数；逐折数值留在 `<out>/<arm>/fold<NN>/metrics.json` 作审计，不进报告。
3. **不得出现任何 FT−ZS 差值**。两臂各自的量并列**呈现**，差值与其显著性不算不看。

措辞：读数标 **【实测】**，解释性文字标 **【推断】**；B 层与功效不足一律写
「在本样本量下该问题不可回答」，不得写「不可分」「两者相等」。
"""
from __future__ import annotations

from typing import Any

from . import config as C

__all__ = ["render_report"]

_ARM_LABEL = {"ft": "FT（微调）", "zs": "ZS（零样本）"}


def _f(x, nd: int = 5) -> str:
    if x is None:
        return "未核"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "未核"
    return f"{v:+.{nd}f}"


def _u(x, nd: int = 2) -> str:
    """无符号格式（比率、次数这类正负号没有含义的量）。"""
    if x is None:
        return "未核"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "未核" if v != v else f"{v:.{nd}f}"


def _ci(d: dict, nd: int = 5) -> str:
    return f"[{_f(d.get('ci_low'), nd)}, {_f(d.get('ci_high'), nd)}]"


def _yn(x) -> str:
    return "未核" if x is None else ("是" if x else "否")


def _h1(lines: list, res: dict, n_folds: int) -> None:
    lines += ["## H1 —— 全池逐日 RankIC（区间估计，不做二元 PASS）", ""]
    lines += ["| 臂 | 点估计 | NW(5) 95% CI | 正折数 | 实现收缩比 | 收缩比 95% CI |",
              "|---|---:|---|---:|---:|---|"]
    for arm in C.ARMS:
        r = res.get(arm)
        if not r:
            continue
        s = r["rank_ic"]
        rc = r["realized_shrinkage_ratio_ci95"]
        lines.append(
            f"| {_ARM_LABEL[arm]} | {_f(s['mean'])} | {_ci(s)} | "
            f"{r['folds_positive']}/{r['folds_total']} | "
            f"{_f(r['realized_shrinkage_ratio'], 3)} | "
            f"[{_f(rc[0], 3)}, {_f(rc[1], 3)}] |")
    lines += ["", "**【实测】预注册预测的相交关系**（v4 §2.3）：", ""]
    lines += ["| 臂 | 主预测区间 | 与 95% CI 相交 | 敏感性预测 | 与 95% CI 相交 | H4 读取闸门（CI 下界 > 0） |",
              "|---|---|---|---|---|---|"]
    for arm in C.ARMS:
        r = res.get(arm)
        if not r:
            continue
        m, s = r["main_prediction_interval"], r["sensitivity_prediction_interval"]
        lines.append(
            f"| {_ARM_LABEL[arm]} | [{m[0]:.7f}, {m[1]:.7f}] | "
            f"{_yn(r['main_prediction_intersects_ci95'])} | "
            f"[{s[0]:.4f}, {s[1]:.4f}] | "
            f"{_yn(r['sensitivity_prediction_intersects_ci95'])} | "
            f"{_yn(r['h4_read_gate_ci_low_gt_0'])} |")
    lines += [
        "",
        "**【推断】**不相交只否定「选择偏差估计正确」这一元主张，"
        "**不构成对信号的否定**（v4 §2.3.1）。敏感性预测一并披露，"
        "**兑现与否以主预测区间为准**。",
        "",
        "**【实测】口径限定**：收缩比分母按 v4 §3 取 FT 0.0207 / ZS 0.01919"
        f"（FT 的实验 6 实算基数为 {C.H1_DEV_BASE_EXP6_FT:.7f}，差 0.0000320）；"
        "FT 基数是逐日合并均值、ZS 基数是七折均值，两者口径不同（v4 §2.3.2 限定 1）；"
        "ZS 的三档 haircut 是由 FT 的 p 值机械等比例外推，不是对 ZS 自身重跑 "
        "Harvey–Liu（限定 3）。",
        "",
        "**【实测】H4 状态**：信号 #2 已按预注册准入规则判「不准入」"
        "（`ledger:477`），**H4 空置**；闸门只记录，不触发任何动作。",
        "",
        "**【实测】必须披露的技术缺陷**：FT 臂相邻折共享约 83% 训练窗，而 "
        "`confirm_power_v3_joint.py` 按折 i.i.d. 抽样，折数门的 size 被低估；"
        "ZS 臂无此问题（R1 指出，未在本项目复算）。",
        "",
    ]


def _h1b(lines: list, res: dict | None) -> None:
    lines += ["## H1b —— 扩充控制集张成后的 alpha（估计交付；v4 §2.4 裁定 C-1）", ""]
    if not res:
        lines += [
            "**本次运行未跑**（`--no-h1b`）。v4 §2.4 的裁定是 **C-1 采纳**（估计交付），"
            "默认应当交付；此处缺失属运行选项，不是协议变更。", ""]
        return
    meta = res["meta"]["unseal"]
    lines += [
        f"**口径警告**：{meta['caliber_warning']}",
        f"**臂别（披露的设计假设，不是性能发现）**：H1b **只在 "
        f"{'、'.join(a.upper() for a in C.H1B_ARMS)} 臂交付**"
        "——ZS 臂没有可外推的开发折 SE，未算 MDE 的终点不得进入 v4；"
        "**不得据确认结果回溯改臂或补做 ZS 臂**（v4 §2.4 第 4 条）。",
        f"主规格先验固定为 **{meta['primary_spec']}**，"
        f"{'、'.join(meta['secondary_specs'])} 作次规格并列报告，"
        "**不得按结果更换主规格，也不得从三规格中挑一个说「H1b 通过了」**。",
        "**无 SESOI，故不作功效关判定**；本协议生效期内不得为 H1b 补写 SESOI 并据以判定。",
        "", "| 规格 | alpha（%/年） | 保留率（%） | NW5 t | 逐折 alpha>0 | 冻结期披露的 MDE80（NT=6 外推） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for spec in C.H1B_SPECS:
        r = res["results"].get(spec)
        if not r:
            continue
        star = "（主）" if spec == C.H1B_PRIMARY_SPEC else ""
        lines.append(
            f"| {spec}{star} | {_f(r['alpha_ann_pct'], 4)} | "
            f"{_u(r['retention_pct'], 2)} | {_f(r['nw5_t_alpha'], 3)} | "
            f"{r['folds_alpha_positive']}/{len(res['meta']['folds'])} | "
            f"{_u(C.H1B_MDE80_NT6.get(spec), 4)} |")
    caveats = {res["results"][s]["required_caveat"] for s in C.H1B_SPECS
               if s in res["results"]}
    lines += [
        "", "**【实测】限定**：" + " ".join(sorted(caveats)),
        "**缺口**：财报日虚拟需外部日历、SUE 需 Compustat，**均未做**。",
        "**【推断】**MDE80 一列是 v4 §2.4 冻结期按 **NT=6** 口径外推的披露值，"
        "**不作解封后的引用值**；不设门槛、不产生 PASS/FAIL。", ""]


def _h2(lines: list, res: dict) -> None:
    lines += [
        "## H2 —— ΔADV（现代型流动性放大；符号 + CI + 折数，**无量级门**）", "",
        f"CI 用 z = {C.Z_BONFERRONI_K2:.6f}（k=2 配置族 Bonferroni）。", "",
        "| 臂 | ΔADV | CI | 正折数 | mean RankIC(top500) | 三门全过 |",
        "|---|---:|---|---:|---:|---|",
    ]
    for arm in C.ARMS:
        r = res.get(arm)
        if not r:
            continue
        lines.append(
            f"| {_ARM_LABEL[arm]} | {_f(r['delta_adv']['mean'])} | {_ci(r['delta_adv'])} | "
            f"{r['folds_positive']}/{r['folds_total']} | "
            f"{_f(r['ic_top500']['mean'])} | {_yn(r['all_gates_pass'])} |")
    lines += ["", "**三门**：① CI 下界 > 0；② 正折数 ≥ "
              f"{C.H2_POSITIVE_FOLDS_GATE}/{C.H2_TOTAL_FOLDS}；③ mean RankIC(top500) > 0。", ""]
    # 门名用中文渲染：报告里连折号形状的标识都不出现（释放清单纪律）。
    gate_label = {
        "ci_low_gt_0": "① CI 下界 > 0",
        f"folds_positive_ge_{C.H2_POSITIVE_FOLDS_GATE}_of_{C.H2_TOTAL_FOLDS}":
            f"② 正折数 ≥ {C.H2_POSITIVE_FOLDS_GATE}/{C.H2_TOTAL_FOLDS}",
        "mean_ic_top500_gt_0": "③ mean RankIC(top500) > 0",
    }
    for arm in C.ARMS:
        r = res.get(arm)
        if not r:
            continue
        got = "、".join(f"{gate_label.get(k, '门')}={_yn(v)}"
                        for k, v in r["gates"].items())
        lines.append(f"- **【实测】{_ARM_LABEL[arm]}**：{got}。{r['reading']}")
    lines += [
        "",
        "**【推断】**三门未全过**不得**写成「机制不存在」——真值 0.0026 处功效仅 "
        "45–52%（`ledger:412`）。E 与 H2 不合成总 PASS。", "",
    ]


def _h2_era(lines: list, res: dict) -> None:
    lines += [
        "## H2-era —— 年代二分（估计交付、无门槛、只切一次）", "",
        f"切点先验固定为 **{C.ERA_CUT}**（按 val_start 机械二分，"
        "不扫切点、不试其它分法）。", "",
        "| 臂 | 段 | 折数 | ΔADV | ΔADV 95% CI | RankIC(top500) | RankIC(full) | ΔADV 正折数 |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for arm in C.ARMS:
        r = res.get(arm)
        if not r:
            continue
        for key in ("early", "late"):
            s = r["segments"][key]
            lines.append(
                f"| {_ARM_LABEL[arm]} | {s['label']} | {s['n_folds']} | "
                f"{_f(s['delta_adv']['mean'])} | {_ci(s['delta_adv'])} | "
                f"{_f(s['ic_top500']['mean'])} | {_f(s['ic_full_pool']['mean'])} | "
                f"{s['folds_delta_adv_positive']}/{s['n_folds']} |")
    lines += ["", "**年代交互项（晚期 − 早期）**：", "",
              "| 臂 | 量 | 点估计 | 95% CI | MDE80（v4 §2.5.1 披露） | 区间覆盖 0 |",
              "|---|---|---:|---|---:|---|"]
    for arm in C.ARMS:
        r = res.get(arm)
        if not r:
            continue
        for key, label in (("interaction_late_minus_early_delta_adv", "ΔADV"),
                           ("interaction_late_minus_early_ic_top500", "RankIC(top500)")):
            i = r.get(key)
            if not i:
                continue
            lines.append(
                f"| {_ARM_LABEL[arm]} | {label} | {_f(i['mean'])} | "
                f"[{_f(i['ci_low'])}, {_f(i['ci_high'])}] | "
                f"{_u(i.get('mde80_disclosed'), 5)} | {_yn(i['covers_zero'])} |")
    note = next((r["boundary_note"] for r in res.values() if r), C.ERA_BOUNDARY_NOTE)
    lines += [
        "",
        f"**【实测】归段**：早期段 {C.ERA_EARLY}（{len(C.ERA_EARLY_FOLDS)} 折）、"
        f"晚期段 {C.ERA_LATE}（{len(C.ERA_LATE_FOLDS)} 折）。",
        f"**【实测】必须披露的边界事实**：{note}",
        "",
        f"**【推断】**交互区间覆盖 0 时按措辞强制写「{C.UNANSWERABLE}」，"
        "**不得**写「两个年代相同」。MDE80 一列原样引用 v4 §2.5.1，只作披露；"
        "该表按折间独立抽样，而 FT 臂相邻折训练窗重叠约 83%，故真实 SE 被低估、"
        "MDE 偏乐观。", "",
    ]


def _h3(lines: list, res: dict | None) -> None:
    lines += ["## H3 —— Kronos vs 冻结树基线的配对 ΔIC（B 层 / 估计交付）", ""]
    if not res:
        lines += [
            "**本次运行未跑**（`--no-h3`，或树基线封存队列尚未覆盖本次折范围）。", ""]
        return
    lines += [
        f"主口径 **XGBoost × {res['arm'].upper()}**（v4.1 附录连带变更；"
        "v4 §2.6 原定 XGBoost × ZS，ZS 臂已抛弃）。配对观测集 = 标签 ok ∧ 两侧分数非缺。",
        "", "| 量 | 点估计 | NW(5) 95% CI | 正折数 | 配对日数 |",
        "|---|---:|---|---:|---:|",
    ]
    for key, label in (("delta_ic", "ΔIC（Kronos − 树）"),
                       ("ic_kronos", "Kronos 全池 RankIC"),
                       ("ic_tree", "树基线全池 RankIC")):
        s = res[key]
        folds = (f"{res['folds_positive']}/{res['folds_total']}"
                 if key == "delta_ic" else "—")
        lines.append(
            f"| {label} | {_f(s['mean'])} | {_ci(s)} | {folds} | "
            f"{res['n_paired_days']} |")
    lines += [
        "",
        f"**【实测】功效披露**：MDE80 开发折 7 折 {res['mde80_dev7_disclosed']:.4f}、"
        f"31 折外推约 {res['mde80_31fold_extrapolated']:.4f}；候选 SESOI "
        f"{res['sesoi_candidates'][0]:.4f} / {res['sesoi_candidates'][1]:.4f}"
        "——**两者都低于 MDE80**，故 H3 定为 B 层并已退出 family（链缩为 H1 → H4）。",
        f"**【实测】{res['reading']}**",
        f"**【推断】必须一并披露**：{res['caveat']}",
        "",
    ]


def _h6(lines: list, res: dict | None) -> None:
    lines += ["## H6 —— ICT P1「扫流动性后收回」（探索性估计交付）", ""]
    if not res:
        lines += ["**未运行。**", ""]
        return
    lines += [
        f"只跑 {'、'.join(C.H6_MARKERS)}，只算 {'、'.join(C.H6_QUESTIONS)}；"
        f"NW lag = {C.H6_NW_LAG}；SESOI = {C.H6_SESOI_BP:.0f} bp / 6 日持有。"
        "Q2(b) 的五分位用 Kronos FT 封存分数（sample_count=5，跑前已核对清单）。",
        "",
        "| 问题 | 标记 | 预期方向 | 均值(bp) | NW(6) 95% CI(bp) | t | 事件数 | 触发日数 | 同向折数/有效折数 |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for s in res.get("summaries", []):
        if s.get("y") != "y" or s.get("question") not in C.H6_QUESTIONS:
            continue
        se = s.get("se_bp")
        lo = hi = None
        if se is not None and se == se:
            lo, hi = s["mean_bp"] - C.Z95 * se, s["mean_bp"] + C.Z95 * se
        lines.append(
            f"| {s['question']} | {s['marker']} | "
            f"{'+' if s.get('expected_sign', 0) > 0 else '−'} | "
            f"{_f(s.get('mean_bp'), 2)} | [{_f(lo, 2)}, {_f(hi, 2)}] | "
            f"{_f(s.get('t'), 2)} | {s.get('n_events', 0)} | {s.get('n_days', 0)} | "
            f"{s.get('n_same_dir_folds', 0)}/{s.get('n_valid_folds', 0)} |")
    u = res.get("unseal", {})
    lines += [
        "",
        "**【推断】层级**：Q1 的 31 折 MDE80 外推 ≈ SESOI（**A 层，边界**），"
        "Q2(b) ≈ 2.2× SESOI（**B 层，只报数不判定**）。两者都不产生 PASS/FAIL。",
        f"**用途限制**：{u.get('use_restriction', '')}",
        f"**【推断】必须一并披露的替代解释**：{u.get('alternative_explanation', '')}",
        "",
    ]


def _e(lines: list, res: dict) -> None:
    lines += [
        "## E —— 经济端（估计交付；**无 5% 阈值、无二元 PASS**）", "",
        f"冻结构造：NT={C.NT} 袖套 / top{C.TOPN} / 进前 10% / 跌出前 "
        f"{int(C.EXIT_PCT * 100)}% 才卖 / t 收盘算分 → t+1 开盘成交。", "",
        "| 臂 | 毛年化(%) | 95% CI(%) | 正折数 | 年单边交易(×) | 每 bp 拖累(%/年) | BE 单边(bp) | BE 95% CI(bp) |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for arm in C.ARMS:
        r = res.get(arm)
        if not r:
            continue
        ci = r["gross_annual_ci95_pct"]
        bci = r["breakeven_oneway_bp_ci95"]
        lines.append(
            f"| {_ARM_LABEL[arm]} | {_f(r['gross_annual_pct'], 2)} | "
            f"[{_f(ci[0], 2)}, {_f(ci[1], 2)}] | "
            f"{r['folds_positive']}/{r['folds_total']} | "
            f"{_u(r['oneway_trades_per_year'], 1)} | "
            f"{_u(r['drag_per_bp_annual_pct'], 4)} | "
            f"{_f(r['breakeven_oneway_bp'], 2)} | "
            f"[{_f(bci[0], 2)}, {_f(bci[1], 2)}] |")
    lines += ["", "**成本网格下的净年化（%）**（敏感性展示，**不承担判据功能**）：", "",
              "| 臂 | " + " | ".join(f"{bp}bp" for bp in C.COST_GRID_BP) + " |",
              "|---|" + "---:|" * len(C.COST_GRID_BP)]
    for arm in C.ARMS:
        r = res.get(arm)
        if not r:
            continue
        cells = " | ".join(_f(r["net_annual_pct_by_cost_bp"][str(bp)], 2)
                           for bp in C.COST_GRID_BP)
        lines.append(f"| {_ARM_LABEL[arm]} | {cells} |")
    lines += [
        "", "**部署线（v4 §2.2；是规则，不是检验；C 未实测前只报线、不判 go）**：",
        "", f"`go ⟺ C ≤ BE_dev × (1 − h) − {C.DEPLOY_RESERVE_BP:.0f}bp`，"
        f"**h = {C.DEPLOY_HAIRCUT_MAIN} 主判**"
        f"（实验 6 三档 Harvey–Liu haircut 区间 "
        f"[{C.DEPLOY_HAIRCUT_RANGE[0]}, {C.DEPLOY_HAIRCUT_RANGE[1]}] 的中点）。", "",
        "| 臂 | 开发折 BE(bp) | h | 1 − h | BE_disc(bp) | **C 上限(bp)** | 口径 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for arm in C.ARMS:
        r = res.get(arm)
        if not r:
            continue
        d = r["deployment_line"]
        be = d["be_dev_bp"]
        rows = [
            (C.DEPLOY_HAIRCUT_RANGE[0], "敏感性（BHY，最松）"),
            (C.DEPLOY_HAIRCUT_MAIN, "**主判（中点）**"),
            (C.DEPLOY_HAIRCUT_RANGE[1], "敏感性（Bonferroni，最严）"),
            (C.DEPLOY_HAIRCUT_LEGACY, "旧线对照（×0.75，v4 生效后作废）"),
        ]
        for h, tag in rows:
            disc = be * (1 - h)
            lines.append(
                f"| {_ARM_LABEL[arm]} | {be:.1f} | {h:.3f} | {1 - h:.3f} | "
                f"{_u(disc, 2)} | {_u(disc - C.DEPLOY_RESERVE_BP, 2)} | {tag} |")
    ft = res.get("ft", {}).get("deployment_line", {})
    zs = res.get("zs", {}).get("deployment_line", {})
    tight = ft.get("legacy_line", {}).get("tightening_vs_legacy_pct")
    lines += [
        "",
        f"**【实测】新线较旧线收紧**：FT {_u(tight, 1)}%、"
        f"ZS {_u(zs.get('legacy_line', {}).get('tightening_vs_legacy_pct'), 1)}%。"
        "旧线 `×0.75` 是「审稿估选择偏差 20–28%」的代理，与 §2.3 已改用的登记簿实测 "
        "haircut 不是同一口径；**v4 生效后旧线作废，只作本行对照**。",
        "",
        f"**【实测】必须随新线一起披露的连带事实**：ZS 臂的新部署线 "
        f"{_u(zs.get('c_max_bp_main'), 2)}bp **已低于 C_stop 的 "
        f"{C.C_STOP_BP:.0f}bp**，即在 h = {C.DEPLOY_HAIRCUT_MAIN} 口径下，"
        "ZS 可部署的成本区间整体落在「15.5 年历史无法在统计上证明净超额 > 0」的区间之内。"
        "这是算术后果，**不改变 C_stop「只作披露、不门控部署」的定位**，"
        "也不改变两臂均满足时部署 FT 的先验顺序。",
        "",
        "**【推断】**不得说「达到 / 未达到经济目标」——5% 阈值已作废（修订 1）；"
        "不得把「BE 的 CI 覆盖实测 C」说成「策略无效」。", "",
    ]


def render_report(payload: dict[str, Any], *, title: str, scope: str,
                  n_folds: int) -> str:
    """把汇总字典渲染成 Markdown。``scope`` 形如「折 05–35（31 折，2005–2020）」。"""
    meta = payload.get("run_meta", {})
    lines: list[str] = [
        f"# {title}", "",
        "> **一次性、不可重开的读取。** 本报告只含确认协议 v4 §3 释放清单允许的量；",
        "> 逐折数值留在各折 `metrics.json` 作审计，**报告内不出现逐折 IC 表**；",
        "> **不含任何 FT−ZS 差值**（两臂各自的量并列呈现，差值不算不看）。",
        f"> 读数标 **【实测】**，解释标 **【推断】**；B 层与功效不足一律写「{C.UNANSWERABLE}」。",
        "",
        f"- **所依据的协议**：`{meta.get('protocol_path', '未核')}`，"
        f"sha256 `{meta.get('protocol_sha256', '未核')}`（运行时现算），"
        f"tag `{meta.get('protocol_tag', C.PROTOCOL_TAG)}`",
        f"- 读取范围：{scope}",
        f"- 运行标识：`{meta.get('run_id', '未核')}`；UTC `{meta.get('run_utc', '未核')}`",
        f"- 用户读取授权（登记簿凭据）：`{meta.get('authorisation_ref', '未核')}`",
        f"- 读取的臂：**{'、'.join(a.upper() for a in meta.get('arms_read', [])) or '未核'}**"
        + ("　—— **ZS 臂按 v4.1 附录已抛弃，未读取**（附录 "
           f"`{meta.get('addendum_path', '未核')}`，sha256 "
           f"`{meta.get('addendum_sha256', '未核')}`）"
           if meta.get("zs_dropped") else "（v4 §1 的 k=2 双臂）"),
        f"- 代码提交：`{meta.get('git_commit', '未核')}`；"
        f"工作树 porcelain 行数 {meta.get('git_porcelain_lines', '未核')}",
        f"- 配置哈希：`{meta.get('config_sha256', '未核')}`；"
        f"代码快照哈希：`{meta.get('code_sha256_of_snapshot', '未核')}`",
        f"- 口径核对：{meta.get('n_verified', 0)} 个（折 × 臂）全部通过 "
        "`scores_sha256` 与 config 五项（lookback 90 / predict 6 / sample_count 5 / "
        "amp bf16 / batch_size 128）",
        "",
    ]
    if meta.get("zs_dropped"):
        lines += ["", "> **【实测】ZS 臂的处置（随确认结果一并披露）**：" + C.ZS_DROPPED_NOTE, ""]
    _h1(lines, payload.get("h1", {}), n_folds)
    _h1b(lines, payload.get("h1b"))
    _h2(lines, payload.get("h2", {}))
    _h2_era(lines, payload.get("h2_era", {}))
    _h3(lines, payload.get("h3"))
    _h6(lines, payload.get("h6"))
    _e(lines, payload.get("e", {}))
    lines += [
        "## 未交付项", "",
        "- **H4（Kronos + 信号 #2 合成）**：**空置**——信号 #2 未过预注册准入"
        "（`ledger:477`）。条文冻结保留，日后新候选过同一套准入规则即可直接执行。",
        "",
        "## 措辞强制（`CLAUDE.md` §二）", "",
        f"任何 B 层或功效不足的读数一律写「**{C.UNANSWERABLE}，按预先规则取 X**」，"
        "**不得**写「不可分」「两者相等」「已检验为无差异」。",
        "",
    ]
    return "\n".join(lines) + "\n"
