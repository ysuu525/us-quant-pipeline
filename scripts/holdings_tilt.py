"""持仓在四个特征上的百分位倾斜（相对池均值 0.50）——K9 只约束了其中一个。"""
import numpy as np, pandas as pd, os
P='F:/quant/processed/crsp_ciz_2026-08-24_20260825T130601Z/'
OUT='F:/quant/us-quant-pipeline/outputs/'
EXIT,NT,TOPN=0.30,6,500
p=pd.read_parquet(P+'panel_raw.parquet',
   columns=['PERMNO','DlyCalDt','DlyClose','DlyPrcVol','DlyRet'],
   filters=[('DlyCalDt','>=',pd.Timestamp('2020-04-01')),('DlyCalDt','<=',pd.Timestamp('2024-01-05'))])
p['DlyCalDt']=pd.to_datetime(p['DlyCalDt']); p=p.sort_values(['PERMNO','DlyCalDt'])
g=p.groupby('PERMNO')
p['adv20']=g['DlyPrcVol'].rolling(20,min_periods=10).mean().reset_index(level=0,drop=True).groupby(p.PERMNO).shift(1)
p['ivol']=g['DlyRet'].rolling(60,min_periods=40).std().reset_index(level=0,drop=True).groupby(p.PERMNO).shift(1)
p['invpx']=1.0/p.DlyClose.abs().where(p.DlyClose.abs()>0)
E=pd.read_parquet(OUT+'edge_monthly.parquet'); E=E[E.edge.notna()&(E.edge>0)]
E['use_ym']=(pd.PeriodIndex(E.ym,freq='M')+1).astype(str)
edge={(int(a),b):float(c) for a,b,c in zip(E.PERMNO,E.use_ym,E.edge)}
FEATS=['adv20','invpx','ivol','edge']
day={d:gg for d,gg in p.groupby('DlyCalDt')}; del p
res={f:[] for f in FEATS}
for fold in range(36,43):
    sc=pd.read_parquet(OUT+f'fold{fold}_lb90_s0_poolB_universe/eval_amp_lb90_fold{fold}/scores.parquet',
                       columns=['PERMNO','signal_date','score']).dropna()
    sc['signal_date']=pd.to_datetime(sc['signal_date'])
    by={d:dict(zip(gg.PERMNO,gg.score)) for d,gg in sc.groupby('signal_date') if len(gg)>=50}
    days=sorted(by); book=[None]*NT
    for i,d in enumerate(days):
        f=day.get(d)
        if f is None: continue
        s=by[d]; a=dict(zip(f.PERMNO,f.adv20))
        elig=[x for x in s if np.isfinite(a.get(x,np.nan))]
        if len(elig)>TOPN: elig=sorted(elig,key=lambda x:-a[x])[:TOPN]
        s={x:v for x,v in s.items() if x in set(elig)}
        if len(s)<50: continue
        n=len(s); pct=(pd.Series(s).rank()/n).to_dict(); k=max(1,n//10)
        order=sorted(pct,key=lambda x:-pct[x]); j=i%NT
        prev=book[j]
        book[j]=list(order[:k]) if prev is None else (
            [x for x in prev if x in pct and pct[x]>=1-EXIT][:k] +
            [x for x in order if x not in set([y for y in prev if y in pct and pct[y]>=1-EXIT][:k])][:k-len([y for y in prev if y in pct and pct[y]>=1-EXIT][:k])])
        if i<NT: continue
        held=[x for t in range(NT) if book[t] for x in book[t]]
        ym=pd.Period(d,freq='M').strftime('%Y-%m')
        sub=f[f.PERMNO.isin(s)]
        sub=sub.assign(edge=[edge.get((x,ym),np.nan) for x in sub.PERMNO])
        for feat in FEATS:
            v=sub[[ 'PERMNO',feat]].dropna()
            if len(v)<50: continue
            r=dict(zip(v.PERMNO,v[feat].rank(pct=True)))
            hp=[r[x] for x in held if x in r]
            if hp: res[feat].append(np.mean(hp))
print(f"{'特征':<22}{'持仓平均百分位':>16}{'相对池均值0.50':>18}{'K6b 收益相关':>14}{'K9 约束?':>10}")
NM={'adv20':'ADV20（流动性）','invpx':'1/价格','ivol':'特质波动(60日)','edge':'EDGE 有效价差'}
K6={'adv20':'+0.326(Amihud)','invpx':'+0.382','ivol':'+0.317','edge':'+0.291(代理)'}
for f in FEATS:
    m=np.mean(res[f])
    print(f"{NM[f]:<22}{m:>16.4f}{m-0.5:>+18.4f}{K6[f]:>14}{'是' if f=='adv20' else '否':>10}")
