"""实验 10 / P4：adaptation dynamics（TSFMAudit 式），**不改 train.py**。

判据 / 用途限制（先写死；来源 ``docs/实验指令_2026-09-04.md`` 实验 10）
--------------------------------------------------------------------
- **诊断，无阈值，不作判定。** 只报 ``‖θ_epoch − θ_0‖₂`` 与逐轮 loss 的**曲线形状**。
- **限定（必须随读数一起出现）**：TSFMAudit 的读法是「见过的数据适应得更快 / 位移
  更小」，而本项目**只有开发折**可做微调（干净窗不许微调），**没有对照臂**，
  因此**只能报曲线形状，不能下污染结论**。
- **不碰封存队列的任何日志与产物**：只在开发折 36 / 39 / 42 的训练输出上做。
- ZS 臂已按用户 2026-09-05 裁定整体抛弃；P4 本就只涉及微调，与该裁定无冲突。

实现口径（两选一，**先对拍再选**；本文件是被选中的方案）
-------------------------------------------------------
选 **(ii) 新脚本包一层，不改 ``src/kronos_ft/train.py`` 一个字节**。理由：

1. ``train_stage``（``src/kronos_ft/train.py:220-224``）**每个 epoch 都已经把
   完整 ``state_dict`` 存成 ``{stage}_epoch{NNN}.pt``**，``{stage}_summary.json``
   里也已经有逐轮 ``train_loss`` / ``inner_loss``。P4 要的两条曲线**已经在磁盘上**，
   是纯事后计算，一个 GPU 小时都不用花。
2. 方案 (i)（给 train.py 加 ``--log-adaptation``）即使默认关闭也要改动被
   ``SEALED_MANIFEST.json`` 的 ``code_sha256`` 钉住的文件
   （``scripts/evaluate_fold.py:245-250`` 记录 ``kronos_ft/train.py`` 的 sha256），
   会让新旧封存清单的代码指纹对不上；而 (ii) 的路径逐位不变是**构造性的**，
   不需要靠冒烟对拍去论证——train.py 的 sha256 前后一致即证毕。

θ_0（参照点）
------------
- tokenizer 阶段：``NeoQuasar/Kronos-Tokenizer-base``
- predictor 阶段：``NeoQuasar/Kronos-small``
（``scripts/run_supp_folds.ps1`` / ``run_probe_recent.ps1`` 的训练命令都用
``kronos_ft.train`` 的默认预训练权重，即上面两个。）

分层
----
按 ``state_dict`` 的键前缀分组：tokenizer 阶段 = embed / encoder / quant_embed /
post_quant_embed / decoder / head / quantizer；predictor 阶段 = embedding /
time_emb / transformer / norm / head / dep_layer。同时报总位移与相对位移
``‖Δθ‖₂ / ‖θ_0‖₂``。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp10_contamination_probes import (  # noqa: E402
    ALLOWED_MODEL_FOLDS,
    BoundaryError,
    _esc,
    _rel,
    assert_read_ok,
    assert_write_ok,
    log,
)
from kronos_ft.models import build_tiny, load_pretrained  # noqa: E402

BASE_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
BASE_PREDICTOR = "NeoQuasar/Kronos-small"

#: 键前缀 → 分层名。顺序敏感：先匹配到的胜出。
GROUPS = {
    "tokenizer": (("embed.", "embed"), ("encoder.", "encoder"),
                  ("quant_embed.", "quant_embed"),
                  ("post_quant_embed_pre.", "post_quant_embed_pre"),
                  ("post_quant_embed.", "post_quant_embed"),
                  ("decoder.", "decoder"), ("head.", "head"),
                  ("tokenizer.", "quantizer")),
    "predictor": (("embedding.", "embedding"), ("time_emb.", "time_emb"),
                  ("transformer.", "transformer"), ("norm.", "norm"),
                  ("head.", "head"), ("dep_layer.", "dep_layer")),
}
EPOCH_RE = re.compile(r"^(tokenizer|predictor)_epoch(\d{3})\.pt$")


def _group_of(stage: str, key: str) -> str:
    for prefix, name in GROUPS[stage]:
        if key.startswith(prefix):
            return name
    return "other"


def _norms(stage: str, sd: dict, base: dict) -> dict:
    """返回 {group: {delta_l2, base_l2, rel}} 与 total。只算浮点张量。"""
    acc: dict[str, list[float]] = {}
    for k, v in base.items():
        if k not in sd or not torch.is_floating_point(v):
            continue
        g = _group_of(stage, k)
        d = (sd[k].float() - v.float())
        a = acc.setdefault(g, [0.0, 0.0, 0])
        a[0] += float(torch.sum(d * d))
        a[1] += float(torch.sum(v.float() * v.float()))
        a[2] += int(v.numel())
    out, tot_d, tot_b, tot_n = {}, 0.0, 0.0, 0
    for g, (dd, bb, n) in sorted(acc.items()):
        out[g] = {"delta_l2": float(np.sqrt(dd)), "base_l2": float(np.sqrt(bb)),
                  "rel": float(np.sqrt(dd) / np.sqrt(bb)) if bb > 0 else None,
                  "n_params": n}
        tot_d += dd
        tot_b += bb
        tot_n += n
    out["__total__"] = {"delta_l2": float(np.sqrt(tot_d)),
                        "base_l2": float(np.sqrt(tot_b)),
                        "rel": float(np.sqrt(tot_d) / np.sqrt(tot_b)) if tot_b else None,
                        "n_params": tot_n}
    return out


def _load_sd(path: Path) -> dict:
    return torch.load(assert_read_ok(path), map_location="cpu", weights_only=True)


def analyse_dir(run_dir: Path, base_sds: dict[str, dict],
                guard: bool = True) -> dict:
    """一个训练输出目录 → 逐轮 ‖Δθ‖ 与 loss 曲线。"""
    run_dir = Path(run_dir)
    entry: dict = {"dir": _rel(run_dir), "stages": {}}
    for stage in ("tokenizer", "predictor"):
        ckpts = sorted(p for p in run_dir.iterdir()
                       if EPOCH_RE.match(p.name) and p.name.startswith(stage + "_"))
        summ_path = run_dir / f"{stage}_summary.json"
        if not ckpts or not summ_path.exists():
            continue
        summ = json.loads((assert_read_ok(summ_path) if guard else summ_path)
                          .read_text(encoding="utf-8"))
        base = base_sds[stage]
        rows = []
        for p in ckpts:
            ep = int(EPOCH_RE.match(p.name).group(2))
            sd = _load_sd(p) if guard else torch.load(p, map_location="cpu",
                                                      weights_only=True)
            rows.append({"epoch": ep, "norms": _norms(stage, sd, base)})
            del sd
        hist = {int(h["epoch"]): h for h in summ.get("history", [])}
        for r in rows:
            h = hist.get(r["epoch"], {})
            r["train_loss"] = h.get("train_loss")
            r["inner_loss"] = h.get("inner_loss")
        entry["stages"][stage] = {
            "best_epoch": summ.get("best_epoch"),
            "stopped_epoch": summ.get("stopped_epoch"),
            "swa_epochs": summ.get("swa_epochs"),
            "best_inner_loss": summ.get("best_inner_loss"),
            "swa_inner_loss": summ.get("swa_inner_loss"),
            "n_epochs_on_disk": len(rows),
            "epochs": rows,
        }
    return entry


# --------------------------------------------------------------------------
# SVG
# --------------------------------------------------------------------------

def write_svg(path: Path, payload: dict) -> None:
    runs = payload["runs"]
    series = []
    for name, e in sorted(runs.items()):
        for stage, s in e.get("stages", {}).items():
            xs = [r["epoch"] for r in s["epochs"]]
            dn = [r["norms"]["__total__"]["rel"] for r in s["epochs"]]
            tl = [r.get("train_loss") for r in s["epochs"]]
            il = [r.get("inner_loss") for r in s["epochs"]]
            series.append((f"{name}/{stage}", xs, dn, tl, il))
    if not series:
        assert_write_ok(path).write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'/>",
            encoding="utf-8")
        return
    W, H = 1180, 520
    T, B = 58, 60
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' "
             f"viewBox='0 0 {W} {H}' font-family='DejaVu Sans,Arial,sans-serif'>",
             f"<rect width='{W}' height='{H}' fill='white'/>",
             f"<text x='{W/2}' y='24' font-size='16' text-anchor='middle'>"
             f"P4 adaptation dynamics：相对参数位移与 loss（诊断，无阈值）</text>",
             f"<text x='{W/2}' y='42' font-size='11' fill='#666' text-anchor='middle'>"
             f"只有开发折可做微调、无对照臂 —— 只报曲线形状，不下污染结论</text>"]
    palette = ["#c0392b", "#2980b9", "#16a085", "#8e44ad", "#d35400", "#2c3e50"]

    def panel(x0, w, title, pick, fmt="{:.4f}"):
        vals = [v for _, _, dn, tl, il in series
                for v in pick(dn, tl, il) if v is not None]
        if not vals:
            return
        lo, hi = min(vals), max(vals)
        pad = 0.06 * max(hi - lo, 1e-9)
        lo, hi = lo - pad, hi + pad
        xmax = max(max(xs) for _, xs, _, _, _ in series)

        def yy(v):
            return H - B - (v - lo) / (hi - lo) * (H - T - B)

        def xx(e):
            return x0 + 40 + (e - 1) / max(xmax - 1, 1) * (w - 60)
        parts.append(f"<line x1='{x0+40}' y1='{T}' x2='{x0+40}' y2='{H-B}' stroke='#333'/>")
        parts.append(f"<line x1='{x0+40}' y1='{H-B}' x2='{x0+w-20}' y2='{H-B}' stroke='#333'/>")
        for i in range(5):
            v = lo + (hi - lo) * i / 4
            parts.append(f"<text x='{x0+36}' y='{yy(v)+4:.1f}' font-size='9' "
                         f"text-anchor='end'>{fmt.format(v)}</text>")
        for e in range(1, xmax + 1, max(1, xmax // 6)):
            parts.append(f"<text x='{xx(e):.1f}' y='{H-B+14}' font-size='9' "
                         f"text-anchor='middle'>{e}</text>")
        parts.append(f"<text x='{x0+40}' y='{T-8}' font-size='11'>{_esc(title)}</text>")
        parts.append(f"<text x='{x0+w-20}' y='{H-B+30}' font-size='9' fill='#666' "
                     f"text-anchor='end'>epoch</text>")
        for k, (nm, xs, dn, tl, il) in enumerate(series):
            ys = pick(dn, tl, il)
            pts = [(xx(e), yy(v)) for e, v in zip(xs, ys) if v is not None]
            if not pts:
                continue
            c = palette[k % len(palette)]
            d = " ".join(("M" if i == 0 else "L") + f"{a:.1f},{b:.1f}"
                         for i, (a, b) in enumerate(pts))
            parts.append(f"<path d='{d}' fill='none' stroke='{c}' stroke-width='1.6'/>")
            parts.append(f"<text x='{x0+48}' y='{T+13*(k+1)}' font-size='9' fill='{c}'>"
                         f"{_esc(nm)}</text>")

    panel(0, W // 2, "相对参数位移 ‖θ_e − θ_0‖₂ / ‖θ_0‖₂",
          lambda dn, tl, il: dn)
    panel(W // 2, W // 2, "逐轮 loss（实线=train，虚线在 JSON 里）",
          lambda dn, tl, il: tl, "{:.3f}")
    parts.append("</svg>")
    assert_write_ok(path).write_text("\n".join(parts), encoding="utf-8")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="实验 10 / P4：adaptation dynamics")
    ap.add_argument("--folds", default="36,39,42",
                    help="开发折号（写死 36/39/42；白名单外报错）")
    ap.add_argument("--dir-template",
                    default="outputs/fold{f:02d}_lb90_s0_poolB_universe")
    ap.add_argument("--extra-dir", action="append", default=[],
                    help="额外的训练输出目录（P4 重跑队列的产物）")
    ap.add_argument("--base-tokenizer", default=BASE_TOKENIZER)
    ap.add_argument("--base-predictor", default=BASE_PREDICTOR)
    ap.add_argument("--out", default="outputs/exp10_p4_adaptation.json")
    ap.add_argument("--svg", default="outputs/exp10_p4_adaptation.svg")
    ap.add_argument("--smoke-dir", default=None,
                    help="tiny 冒烟：分析该目录，θ_0 由 build_tiny()+seed 重建")
    ap.add_argument("--smoke-seed", type=int, default=0)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    torch.set_num_threads(2)

    if args.smoke_dir:
        from kronos_ft.train import set_seed
        set_seed(args.smoke_seed)
        tk, md = build_tiny()
        base_sds = {"tokenizer": {k: v.cpu() for k, v in tk.state_dict().items()},
                    "predictor": {k: v.cpu() for k, v in md.state_dict().items()}}
        entry = analyse_dir(Path(args.smoke_dir), base_sds, guard=False)
        payload = {"experiment": "exp10-P4 冒烟（tiny，合成数据）",
                   "runs": {"smoke": entry},
                   "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=float))
        return

    folds = [int(s) for s in args.folds.split(",") if s.strip()]
    bad = [f for f in folds if f not in ALLOWED_MODEL_FOLDS]
    if bad:
        raise BoundaryError(f"折 {bad} 不在开发折白名单 {ALLOWED_MODEL_FOLDS} 内。")

    log("加载 θ_0（预训练基座）…")
    tk, md = load_pretrained(args.base_tokenizer, args.base_predictor)
    base_sds = {"tokenizer": {k: v.cpu() for k, v in tk.state_dict().items()},
                "predictor": {k: v.cpu() for k, v in md.state_dict().items()}}
    del tk, md

    runs = {}
    dirs = [(f"fold{f}", REPO_ROOT / args.dir_template.format(f=f)) for f in folds]
    dirs += [(Path(d).name, Path(d) if Path(d).is_absolute() else REPO_ROOT / d)
             for d in args.extra_dir]
    for name, d in dirs:
        if not d.exists():
            log(f"跳过（不存在）：{_rel(d)}")
            continue
        log(f"分析 {_rel(d)} …")
        runs[name] = analyse_dir(d, base_sds)

    payload = {
        "experiment": "exp10-P4 adaptation dynamics（TSFMAudit 式）",
        "criterion": ("诊断，无阈值，不作判定。只报 ‖θ_epoch − θ_0‖₂ 与逐轮 loss 的曲线形状。"),
        "qualification": (
            "TSFMAudit 的读法是『见过的数据适应得更快 / 位移更小』；本项目只有开发折"
            "可做微调（干净窗不许微调），没有对照臂，因此只能报曲线形状，不能下污染结论。"),
        "implementation_choice": (
            "方案 (ii)：新脚本包一层，src/kronos_ft/train.py 一个字节未改（sha256 前后一致），"
            "逐轮 checkpoint 与 loss 由 train_stage 既有行为写在磁盘上，本脚本纯事后计算。"),
        "zero_shot_arm": "ZS 臂已按用户 2026-09-05 裁定整体抛弃；P4 只涉及微调，无冲突。",
        "base": {"tokenizer": args.base_tokenizer, "predictor": args.base_predictor},
        "runs": runs,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = assert_write_ok(REPO_ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    write_svg(REPO_ROOT / args.svg, payload)
    log(f"完成 → {_rel(out)}")


if __name__ == "__main__":
    main()
