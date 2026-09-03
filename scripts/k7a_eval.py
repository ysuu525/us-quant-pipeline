"""K7a 判读：折01-04 零样本 vs 微调，按预注册的三条阈值。

预注册（ledger, 2026-08-31，落笔于见到任何结果之前）：
  ① 零样本收回早今 top500 IC 缺口（0.0142）的至少一半，即改善 >= +0.007
  ② 至少 3/4 个早期折同向
  ③ top-decile 毛超额收回早今差距的至少一半，约 >= +4.8%/年
"""
import numpy as np, pandas as pd, os

P = 'F:/quant/processed/crsp_ciz_2026-08-24_20260825T130601Z/'
OUT = 'F:/quant/us-quant-pipeline/outputs/'
TOPN = 500

FT = {'fold01': OUT + 'fold01_lb90_s0_poolB_universe/eval_poolB_universe',
      'fold02': OUT + 'fold02_lb90_s0_poolB_universe/eval_poolB_universe_fold02',
      'fold03': OUT + 'fold03_lb90_s0_poolB_universe/eval_poolB_universe_fold03',
      'fold04': OUT + 'fold04_lb90_s0_poolB_universe/eval_poolB_universe_fold04'}
ZS = {f'fold0{i}': OUT + f'zeroshot_base/eval_zs_fold0{i}' for i in range(1, 5)}

p = pd.read_parquet(P + 'panel_raw.parquet', columns=['PERMNO', 'DlyCalDt', 'DlyPrcVol'],
                    filters=[('DlyCalDt', '>=', pd.Timestamp('2002-10-01')),
                             ('DlyCalDt', '<=', pd.Timestamp('2005-01-05'))])
p['DlyCalDt'] = pd.to_datetime(p['DlyCalDt'])
p = p.sort_values(['PERMNO', 'DlyCalDt'])
p['adv20'] = (p.groupby('PERMNO')['DlyPrcVol'].rolling(20, min_periods=10).mean()
               .reset_index(level=0, drop=True)).groupby(p['PERMNO']).shift(1)
adv = {d: dict(zip(g.PERMNO, g.adv20)) for d, g in p.groupby('DlyCalDt')}
del p


def measure(sdir, ldir):
    l = pd.read_parquet(os.path.join(ldir, 'labels.parquet'))
    s = pd.read_parquet(os.path.join(sdir, 'scores.parquet'))
    m = s.merge(l[['PERMNO', 'signal_date', 'label', 'status']],
                on=['PERMNO', 'signal_date'])
    m = m[(m.status == 'ok') & m.label.notna() & m.score.notna()]
    m['signal_date'] = pd.to_datetime(m['signal_date'])
    icf, ict, top = [], [], []
    for dt, g in m.groupby('signal_date'):
        if len(g) < 50:
            continue
        a = adv.get(dt, {})
        g2 = g[[np.isfinite(a.get(x, np.nan)) for x in g.PERMNO]]
        if len(g2) > TOPN:
            g2 = g2.assign(_a=[a[x] for x in g2.PERMNO]).nlargest(TOPN, '_a')
        if len(g2) < 50:
            continue
        icf.append(g.score.rank().corr(g.label.rank()))
        ict.append(g2.score.rank().corr(g2.label.rank()))
        q = g2.score.quantile(0.9)
        top.append(g2.label[g2.score >= q].mean() - g2.label.mean())
    return np.mean(icf), np.mean(ict), np.mean(top) / 6 * 252 * 100, len(ict)


print(f"{'折':<8}{'全池IC(ZS)':>12}{'全池IC(FT)':>12}{'t500IC(ZS)':>12}"
      f"{'t500IC(FT)':>12}{'ΔIC':>10}{'top10%年毛(ZS)':>15}{'(FT)':>10}{'Δ':>9}")
rows = []
for f in FT:
    zf, zt, ztop, n = measure(ZS[f], FT[f])
    ff, ft, ftop, _ = measure(FT[f], FT[f])
    rows.append((f, zf, ff, zt, ft, zt - ft, ztop, ftop, ztop - ftop))
    print(f"{f:<8}{zf:>+12.5f}{ff:>+12.5f}{zt:>+12.5f}{ft:>+12.5f}{zt-ft:>+10.5f}"
          f"{ztop:>+14.2f}%{ftop:>+9.2f}%{ztop-ftop:>+8.2f}%")

R = pd.DataFrame(rows, columns=['fold', 'zs_full', 'ft_full', 'zs_t500', 'ft_t500',
                                'd_ic', 'zs_top', 'ft_top', 'd_top'])
print(f"\n{'均值':<8}{R.zs_full.mean():>+12.5f}{R.ft_full.mean():>+12.5f}"
      f"{R.zs_t500.mean():>+12.5f}{R.ft_t500.mean():>+12.5f}{R.d_ic.mean():>+10.5f}"
      f"{R.zs_top.mean():>+14.2f}%{R.ft_top.mean():>+9.2f}%{R.d_top.mean():>+8.2f}%")

print("\n" + "=" * 78)
print("按预注册三条阈值判读")
print("=" * 78)
c1 = R.d_ic.mean() >= 0.007
c2 = int((R.d_ic > 0).sum()) >= 3
c3 = R.d_top.mean() >= 4.8
print(f"  ① 零样本改善 top500 IC >= +0.007 ?   实测 {R.d_ic.mean():+.5f}   -> {'通过' if c1 else '不通过'}")
print(f"  ② 至少 3/4 折同向 ?                  实测 {int((R.d_ic>0).sum())}/4      -> {'通过' if c2 else '不通过'}")
print(f"  ③ top-decile 毛超额改善 >= +4.8%/年 ? 实测 {R.d_top.mean():+.2f}%  -> {'通过' if c3 else '不通过'}")
print(f"\n  **{'零样本明显救回' if (c1 and c2 and c3) else '未达『明显救回』标准'}**")
print(f"\n  参照：近代 top500 IC = +0.0258，早期微调 top500 IC = {R.ft_t500.mean():+.5f}，"
      f"缺口 {0.0258-R.ft_t500.mean():+.5f}")
print(f"        零样本 top500 IC = {R.zs_t500.mean():+.5f}，仍差近代 "
      f"{0.0258-R.zs_t500.mean():+.5f}")
