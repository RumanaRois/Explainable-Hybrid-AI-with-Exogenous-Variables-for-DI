#!/usr/bin/env python3
"""
Explainable Hybrid AI with Exogenous Variables for Three-hourly Discomfort Index Forecasting
in a Dense Tropical City (Dhaka, Bangladesh) -- complete analysis code in one file.

Authors: R. N. Kowbi, M. M. H. Khan, R. Rois (Jahangirnagar University / BMD)

Framework
  Stage 1 (primary learner) receives the exogenous variables (rainfall, cloud cover, pressure, wind),
          DI lags (3, 9, 18 h) and calendar terms; such learners carry the suffix X in results
          (DTX, RFX, XGBX, SVRX, ANNX, LSTMX, GRUX, FPX, SARIMAX).
          Raw prediction files use the plain algorithm names; the 'analyse' step attaches the labels.
  Stage 2 (residual learner) receives ONLY the estimated error term:
          ETS, SARIMA or TBATS after an ML/DL learner (ML-TS hybrids, e.g. GRUX-SARIMA), or an ML/DL learner
          on lagged SARIMAX residuals (TS-ML hybrids, e.g. SARIMAX-GRU). Residual models run one step (3 h) ahead.
  41 configurations x 2 tracks (raw DI, STL-adjusted DI) = 82 models, plus persistence and seasonal naive.

Requirements (Python >= 3.10):
  pip install numpy pandas openpyxl scipy scikit-learn statsmodels xgboost tbats prophet torch shap matplotlib

Usage (run from any folder; edit DATA/OUT/FIG below or pass --data/--out/--fig):
  python DI_XAI_analysis.py eda                     # trend tests, Fig. 3 (observed DI), Supp. Fig. S1
  python DI_XAI_analysis.py framework               # Fig. 2 (workflow diagram)
  python DI_XAI_analysis.py model raw A SARIMAX     # one model group: track raw|stl, scenario A|B, group
        groups: naive SARIMAX DT RF XGB SVR ANN LSTM GRU FP SX-DT SX-RF SX-XGB SX-SVR SX-ANN SX-LSTM SX-GRU SX-FP
        (run SARIMAX before the SX-* groups of the same track; add --notbats for scenario B)
  python DI_XAI_analysis.py all-models              # every group used in the paper (several hours on 2 CPUs)
  python DI_XAI_analysis.py analyse                 # metrics, Diebold-Mariano, category skill, conformal PIs, Fig. 4
  python DI_XAI_analysis.py xai                     # TreeSHAP for XGBoost (Figs 5-6, Supp. Fig. S2)
  python DI_XAI_analysis.py xai-gru                 # expected-gradients SHAP for the GRU network
  python DI_XAI_analysis.py values                  # every number quoted in the manuscript -> values.json
Scenario A: exogenous variables valid at the target time (perfect prognosis).
Scenario B: exogenous variables lagged by 3 h (operational check).
"""
import sys, os, re, time, glob, json, pickle, argparse, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from scipy.stats import kendalltau, theilslopes, norm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------- paths (edit or use CLI options)
DATA = 'DI_data.xlsx'          # BMD 3-hourly data: DateTime, Season, Temperature, Humidity, DI, Rainfall, Atmosphere, Cloud, Wind
OUT = 'out'                    # predictions, metrics, JSON summaries
FIG = 'fig'                    # figures (300 dpi PNG)
SEED = 42
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})

# ============================================================================= 1. Data preparation

EXOG=['Rainfall','Cloud','Atmosphere','Wind']
SEAS=['Season_Spring','Season_Summer','Season_Winter']
CAL=['Hour','Day','Month']
LAGS=[1,3,6]
def load():
    df=pd.read_excel(DATA); df['DateTime']=pd.to_datetime(df['DateTime'])
    df=df.set_index('DateTime').asfreq('3h')
    d=pd.get_dummies(df['Season'],prefix='Season').astype(float)
    for c in SEAS: df[c]=d[c]
    df['Hour']=df.index.hour; df['Day']=df.index.day; df['Month']=df.index.month
    return df
N_TOTAL=34576; TEST0=int(N_TOTAL*0.8); CAL0=int(TEST0*0.8)
def design(df, target='DI', scenario='A'):
    X=pd.DataFrame(index=df.index)
    ex=df[EXOG] if scenario=='A' else df[EXOG].shift(1)
    for c in EXOG: X[c]=ex[c]
    for c in CAL: X[c]=df[c]
    for l in LAGS: X[f'DI_lag_{l}']=df[target].shift(l)
    for c in SEAS: X[c]=df[c]
    return X, df[target]
def mase_den(ytr): return np.mean(np.abs(np.diff(np.asarray(ytr))))
def metrics(y,p,den):
    y=np.asarray(y); p=np.asarray(p); e=y-p
    return dict(MSE=np.mean(e**2),RMSE=np.sqrt(np.mean(e**2)),MAE=np.mean(np.abs(e)),
                MAPE=100*np.mean(np.abs(e/y)),MASE=np.mean(np.abs(e))/den,
                R2=1-np.sum(e**2)/np.sum((y-y.mean())**2))


# ============================================================================= 2. Forecasting models (stage 1 and stage 2)


SEED = 42
np.random.seed(SEED)

# ------------------------------------------------------------------ data
def build(track, scenario):
    df = load()
    z = df['DI'].copy()
    s = pd.Series(0.0, index=df.index)
    if track == 'stl':
        st = STL(z.iloc[:TEST0], period=8, robust=True).fit()
        s.iloc[:TEST0] = st.seasonal.values
        last = st.seasonal.iloc[-8 * 30:]
        prof = last.groupby(last.index.hour).mean()
        s.iloc[TEST0:] = prof.loc[df.index[TEST0:].hour].values
        df['DI_sa'] = z - s
        X, zt = design(df, 'DI_sa', scenario)
    else:
        X, zt = design(df, 'DI', scenario)
    return df, X, zt, s

# ------------------------------------------------------------------ primary learners
def nn_fit_predict(kind, Xtr, ytr, Xte, Xprev=None, win=6, seed=SEED):
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(1)
    sx = StandardScaler().fit(Xtr); sy = StandardScaler().fit(ytr.reshape(-1, 1))
    A = sx.transform(Xtr).astype(np.float32); b = sy.transform(ytr.reshape(-1, 1)).astype(np.float32).ravel()
    B = sx.transform(Xte).astype(np.float32)
    if kind == 'ann':
        Atr, btr, Ate = A, b, B
        net = nn.Sequential(nn.Linear(A.shape[1], 256), nn.ReLU(), nn.Dropout(0.1),
                            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
                            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        fwd = lambda x: net(x).squeeze(-1)
    else:
        def seq(M, head=None):
            M2 = M if head is None else np.vstack([head, M])
            idx = np.arange(win - 1, len(M2))
            return np.stack([M2[i - win + 1:i + 1] for i in idx])
        Atr = seq(A); btr = b[win - 1:]
        Ate = seq(B, A[-(win - 1):])            # test windows use the last training rows as history
        Rnn = nn.LSTM if kind == 'lstm' else nn.GRU
        class Net(nn.Module):
            def __init__(s):
                super().__init__(); s.r = Rnn(A.shape[1], 50, num_layers=2, batch_first=True, dropout=0.1); s.o = nn.Linear(50, 1)
            def forward(s, x):
                h, _ = s.r(x); return s.o(h[:, -1]).squeeze(-1)
        net = Net(); fwd = net
    # chronological validation block = last 10 % of the fitting data
    nv = int(0.1 * len(Atr)); Xt_, yt_ = torch.tensor(Atr[:-nv]), torch.tensor(btr[:-nv])
    Xv_, yv_ = torch.tensor(Atr[-nv:]), torch.tensor(btr[-nv:])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); lossf = nn.MSELoss()
    best, bad, state = 1e9, 0, None
    for ep in range(60):
        net.train(); perm = torch.randperm(len(Xt_))
        for i in range(0, len(perm), 256):
            j = perm[i:i + 256]; opt.zero_grad(); l = lossf(fwd(Xt_[j]), yt_[j]); l.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl = lossf(fwd(Xv_), yv_).item()
        if vl < best - 1e-5: best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 8: break
    net.load_state_dict(state); net.eval()
    with torch.no_grad(): p = fwd(torch.tensor(Ate)).numpy()
    return sy.inverse_transform(p.reshape(-1, 1)).ravel()

def prophet_fit_predict(Xtr, ytr, Xte, idx_tr, idx_te):
    from prophet import Prophet
    import logging; logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    tr = Xtr.copy(); tr['ds'] = idx_tr; tr['y'] = ytr
    te = Xte.copy(); te['ds'] = idx_te
    m = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=True)
    for c in Xtr.columns: m.add_regressor(c)
    m.fit(tr)
    return m.predict(te)['yhat'].values

TUNED = {}
def tuned_params(name, Xa, ya, Xb, yb):
    """Light grid search on the calibration block (fit on first 80 % of training, score on last 20 %)."""
    if name in TUNED: return TUNED[name]
    if name == 'DT':
        grid = [dict(max_depth=d, min_samples_leaf=l) for d in (6, 8, 10, 12) for l in (20, 50)]
        mk = lambda g: DecisionTreeRegressor(random_state=SEED, **g)
    elif name == 'RF':
        grid = [dict(max_depth=d, min_samples_leaf=l) for d in (12, 20) for l in (5, 20)]
        mk = lambda g: RandomForestRegressor(n_estimators=200, max_features=0.5, n_jobs=2, random_state=SEED, **g)
    elif name == 'XGB':
        grid = [dict(n_estimators=n, max_depth=d) for n in (300, 600) for d in (4, 6)]
        mk = lambda g: XGBRegressor(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=2, **g)
    sc = []
    for g in grid:
        p = mk(g).fit(Xa, ya).predict(Xb); sc.append(np.mean((yb - p) ** 2))
    TUNED[name] = grid[int(np.argmin(sc))]
    print(name, 'tuned', TUNED[name], flush=True)
    return TUNED[name]

def primary(name, Xtr, ytr, Xte, idx_tr=None, idx_te=None, tune_data=None):
    Xtr_, Xte_ = Xtr.values, Xte.values
    if name in ('DT', 'RF', 'XGB'):
        g = tuned_params(name, *tune_data)
        m = {'DT': lambda: DecisionTreeRegressor(random_state=SEED, **g),
             'RF': lambda: RandomForestRegressor(n_estimators=200, max_features=0.5, n_jobs=2, random_state=SEED, **g),
             'XGB': lambda: XGBRegressor(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=2, **g)}[name]()
        return m.fit(Xtr_, ytr).predict(Xte_), m
    if name == 'SVR':
        m = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale'))
        return m.fit(Xtr_, ytr).predict(Xte_), m
    if name == 'ANN': return nn_fit_predict('ann', Xtr_, ytr, Xte_), None
    if name == 'LSTM': return nn_fit_predict('lstm', Xtr_, ytr, Xte_), None
    if name == 'GRU': return nn_fit_predict('gru', Xtr_, ytr, Xte_), None
    if name == 'FP': return prophet_fit_predict(Xtr, ytr, Xte, idx_tr, idx_te), None
    raise ValueError(name)

# ------------------------------------------------------------------ residual (secondary) models, one-step-ahead
def hw_filter(params, y, m=8):
    """Damped additive-trend, additive-season Holt-Winters one-step filter with fixed parameters."""
    a, b_, g, phi = params['smoothing_level'], params['smoothing_trend'], params['smoothing_seasonal'], params['damping_trend']
    l, b = params['initial_level'], params['initial_trend']
    s = list(np.asarray(params['initial_seasons'], float))
    out = np.empty(len(y))
    for t, yt in enumerate(y):
        st = s[-m]; out[t] = l + phi * b + st
        ln = a * (yt - st) + (1 - a) * (l + phi * b)
        bn = b_ * (ln - l) + (1 - b_) * phi * b
        s.append(g * (yt - l - phi * b) + (1 - g) * st)
        l, b = ln, bn
    return out

def resid_onestep(kind, r_cal, r_test):
    full = np.concatenate([r_cal, r_test]); n = len(r_cal)
    if kind == 'ETS':
        f = ExponentialSmoothing(r_cal, trend='add', damped_trend=True, seasonal='add', seasonal_periods=8,
                                 initialization_method='estimated').fit()
        return hw_filter(f.params, full)[n:]
    if kind == 'SARIMAX':
        f = SARIMAX(r_cal, order=(1, 0, 1), seasonal_order=(1, 0, 1, 8)).fit(disp=False)
        return f.apply(full).predict()[n:]
    if kind == 'TBATS':
        from tbats import TBATS
        est = TBATS(seasonal_periods=[8], use_box_cox=False, use_trend=False, use_damped_trend=False, n_jobs=1, show_warnings=False)
        mdl = est.fit(r_cal)
        mdl.fit(full)                      # same parameters, one-step in-series predictions over cal+test
        return np.asarray(mdl.y_hat)[n:]
    raise ValueError(kind)

# ------------------------------------------------------------------ SARIMAX primary (one-step)
SX_EXOG = EXOG + SEAS
def sarimax_onestep(df, z, scenario, n_end):
    ex = df[SX_EXOG].astype(float) if scenario == 'A' else df[SX_EXOG].astype(float).assign(**{c: df[c].shift(1) for c in EXOG})
    ex = ex.bfill()
    cache = f'{OUT}/sarimax_{TRACK}_{scenario}.pkl'
    if os.path.exists(cache): return pickle.load(open(cache, 'rb'))
    best = None
    for o, so in [((0, 0, 1), (1, 0, 0, 8)), ((1, 0, 1), (1, 0, 1, 8)), ((2, 0, 1), (1, 0, 1, 8))]:
        t = time.time()
        w0 = n_end - 8760                    # parameters estimated on the last 3 years of the training period
        f = SARIMAX(z.iloc[w0:n_end].values, exog=ex.iloc[w0:n_end].values, order=o, seasonal_order=so).fit(disp=False, maxiter=100)
        print('SARIMAX', o, so, 'AIC', round(f.aic, 1), round(time.time() - t), 's', flush=True)
        if best is None or f.aic < best[0]: best = (f.aic, o, so, f)
    f = best[3].apply(z.values, exog=ex.values)
    fitted = f.predict()                     # one-step-ahead predictions over the whole series
    res = dict(order=best[1], sorder=best[2], aic=best[0], pred=fitted, ex=ex)
    pickle.dump(res, open(cache, 'wb'))
    return res

# ------------------------------------------------------------------ main


def run_model(track, scen, group, NOTBATS=False):
    """Fit one model group for one track/scenario and save its test-period predictions."""
    global TRACK
    TRACK, SCEN, GROUP = track, scen, group
    df, X, z, s = build(TRACK, SCEN)
    first = 8                                       # drop rows without full lag history
    I_fit_a = slice(first, CAL0); I_cal = slice(CAL0, TEST0); I_fit = slice(first, TEST0); I_te = slice(TEST0, None)
    y_te = df['DI'].values[TEST0:]; s_te = s.values[TEST0:]; s_cal = s.values[CAL0:TEST0]
    fn = f'{OUT}/pred_{TRACK}_{SCEN}_{GROUP}.pkl'
    P = {}
    tune_data = (X.values[I_fit_a], z.values[I_fit_a], X.values[I_cal], z.values[I_cal])
    pre = 'STL-' if TRACK == 'stl' else ''

    if GROUP == 'naive':
        P['Persistence'] = df['DI'].shift(1).values[TEST0:]
        P['Seasonal naive'] = df['DI'].shift(8).values[TEST0:]

    elif GROUP in ('DT', 'RF', 'XGB', 'SVR', 'ANN', 'LSTM', 'GRU', 'FP'):
        nm = GROUP; t0 = time.time()
        p_cal, _ = primary(nm, X.iloc[I_fit_a], z.values[I_fit_a], X.iloc[I_cal], X.index[I_fit_a], X.index[I_cal], tune_data)
        p_te, model = primary(nm, X.iloc[I_fit], z.values[I_fit], X.iloc[I_te], X.index[I_fit], X.index[I_te], tune_data)
        r_cal = z.values[I_cal] - p_cal; r_te = z.values[I_te] - p_te
        P[pre + nm] = p_te + s_te
        np.save(f'{OUT}/calres_{TRACK}_{SCEN}_{nm}.npy', r_cal)
        if nm == 'XGB' and model is not None: pickle.dump(model, open(f'{OUT}/xgb_{TRACK}_{SCEN}.pkl', 'wb'))
        for k in (('ETS', 'SARIMAX') if NOTBATS else ('ETS', 'SARIMAX', 'TBATS')):
            t = time.time()
            e = resid_onestep(k, r_cal, r_te)
            P[f'{pre}{nm}-' + ('SARIMA' if k == 'SARIMAX' else k)] = p_te + e + s_te   # stage 2 has no exogenous inputs
            print(nm, 'SARIMA' if k == 'SARIMAX' else k, round(time.time() - t), 's', flush=True)
        print(nm, 'done', round(time.time() - t0), 's', flush=True)

    elif GROUP == 'SARIMAX':
        res = sarimax_onestep(df, z, SCEN, TEST0)
        P[pre + 'SARIMAX'] = res['pred'][TEST0:] + s_te
        print('SARIMAX selected', res['order'], res['sorder'])

    elif GROUP.startswith('SX-'):
        # SARIMAX primary + ML residual learner (TS-ML hybrids)
        nm = GROUP[3:]
        res = sarimax_onestep(df, z, SCEN, TEST0)
        r = pd.Series(z.values - res['pred'], index=df.index)
        R = pd.DataFrame({f'res_lag_{l}': r.shift(l) for l in (1, 2, 3, 6, 8)}, index=df.index)
        XR = R.copy()      # stage 2 sees only the error term (lagged SARIMAX residuals); exogenous inputs enter stage 1 only
        tdR = (XR.values[I_fit_a], r.values[I_fit_a], XR.values[I_cal], r.values[I_cal])
        TUNED.clear()
        I_r = slice(TEST0 - 8760, TEST0) if nm == 'SVR' else I_fit   # kernel SVR: last 3 years only (computational cost)
        e_te, _ = primary(nm, XR.iloc[I_r], r.values[I_r], XR.iloc[I_te], XR.index[I_r], XR.index[I_te], tdR)
        lab = ('SARIMAX-' if TRACK == 'raw' else 'STL-SARIMAX-') + nm
        P[lab] = res['pred'][TEST0:] + e_te + s_te
        print(lab, 'done', flush=True)

    P['_y'] = y_te; P['_idx'] = df.index[TEST0:]
    pickle.dump(P, open(fn, 'wb'))



# ============================================================================= 3. Exploratory analysis and trends

def run_eda():
    """Trend tests (Mann-Kendall, Sen slope), observed-DI figure and correlation matrix."""
    EDGES = [-np.inf, 21, 24, 27, 29, 32, np.inf]
    CATS = ['No discomfort', '<50% discomfort', '>50% discomfort', 'Most discomfort', 'Severe discomfort', 'Medical emergency']
    CCOL = ['#7fbf7b', '#d9f0a3', '#fee391', '#fdae6b', '#e6550d', '#7f2704']
    df = load(); di = df['DI']
    cat = pd.cut(di, EDGES, labels=False, right=False)
    res = {}
    # ---------------- trend statistics
    ann = di.groupby(df.index.year).mean()
    full = ann.loc[2014:2024]
    tau, p = kendalltau(full.index, full.values); sl = theilslopes(full.values, full.index)
    res['annual_mean'] = full.round(3).to_dict(); res['MK_tau'] = tau; res['MK_p'] = p; res['sen_slope_per_decade'] = sl[0] * 10
    share = pd.crosstab(df.index.year, cat, normalize='index') * 100
    hi = share[[4, 5]].sum(1).loc[2014:2024]                     # severe + emergency share
    tau2, p2 = kendalltau(hi.index, hi.values); sl2 = theilslopes(hi.values, hi.index)
    res['severe_share'] = hi.round(2).to_dict(); res['severe_MK_tau'] = tau2; res['severe_MK_p'] = p2; res['severe_sen_per_decade'] = sl2[0] * 10
    hi27 = share[[3, 4, 5]].sum(1).loc[2014:2024]; t3, p3 = kendalltau(hi27.index, hi27.values); s3 = theilslopes(hi27.values, hi27.index)
    res['ge27_share'] = hi27.round(2).to_dict(); res['ge27_MK_p'] = p3; res['ge27_sen_per_decade'] = s3[0] * 10
    # seasonal (JJAS and MAM) means
    for nm, mons in {'MAM': [3, 4, 5], 'JJAS': [6, 7, 8, 9], 'DJF': [12, 1, 2]}.items():
        sub = di[df.index.month.isin(mons)].groupby(di[df.index.month.isin(mons)].index.year).mean().loc[2014:2024]
        t_, p_ = kendalltau(sub.index, sub.values); s_ = theilslopes(sub.values, sub.index)
        res[f'{nm}_sen_per_decade'] = s_[0] * 10; res[f'{nm}_MK_p'] = p_
    res['desc'] = df[['Temperature', 'Humidity', 'DI', 'Rainfall', 'Atmosphere', 'Cloud', 'Wind']].describe().T.round(2).to_dict()
    res['cat_share_all'] = (cat.value_counts(normalize=True).sort_index() * 100).round(2).to_dict()
    json.dump(res, open(f'{OUT}/eda.json', 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != 'desc'}, indent=1, default=float))

    # ---------------- Figure 2: observed DI
    fig = plt.figure(figsize=(7.4, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=.42, wspace=.5)
    ax = fig.add_subplot(gs[0, :])
    for i in range(6):
        lo, hi_ = max(EDGES[i], 8), min(EDGES[i + 1], 35)
        ax.axhspan(lo, hi_, color=CCOL[i], alpha=.28, lw=0, label=CATS[i])
    ax.plot(di.index, di.values, lw=.25, color='#3b4a6b', alpha=.55)
    mm = di.resample('MS').mean(); ax.plot(mm.index, mm.values, color='#08306b', lw=1.1, label='Monthly mean')
    yy = pd.to_datetime([f'{y}-07-01' for y in full.index])
    ax.plot(yy, full.values, 'o', color='k', ms=3, label='Annual mean')
    ax.plot(yy, sl[1] + sl[0] * full.index.values, '--', color='k', lw=1, label=f'Sen slope {sl[0]*10:+.2f} °C/decade')
    ax.set_ylim(9, 34.5); ax.set_xlim(di.index[0], di.index[-1]); ax.set_ylabel('DI (°C)')
    ax.legend(ncol=5, fontsize=6.3, frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.25))
    ax.set_title('a', loc='left', fontweight='bold', x=-.06)
    ax2 = fig.add_subplot(gs[1, 0])
    hm = di.groupby([df.index.month, df.index.hour]).mean().unstack()
    im = ax2.imshow(hm.T.values, aspect='auto', cmap='YlOrRd', origin='lower', extent=[.5, 12.5, -1.5, 22.5])
    ax2.set_xticks(range(1, 13)); ax2.set_xticklabels('JFMAMJJASOND')
    ax2.set_yticks(range(0, 24, 3)); ax2.set_yticklabels([f'{h:02d}' for h in range(0, 24, 3)])
    ax2.set_ylabel('Observation hour (UTC)'); ax2.set_xlabel('Month')
    cs = ax2.contour(np.arange(1, 13), np.arange(0, 24, 3), hm.T.values, levels=[21, 24, 27, 29], colors='k', linewidths=.5)
    ax2.clabel(cs, fontsize=5.5, fmt='%d')
    cb = fig.colorbar(im, ax=ax2, fraction=.05, pad=.02); cb.set_label('Mean DI (°C)')
    ax2.set_title('b', loc='left', fontweight='bold', x=-.18)
    ax3 = fig.add_subplot(gs[1, 1])
    sh = share.loc[2014:2025]
    bottom = np.zeros(len(sh))
    for i in range(6):
        v = sh[i].values if i in sh.columns else np.zeros(len(sh))
        ax3.bar(sh.index, v, bottom=bottom, color=CCOL[i], width=.8, edgecolor='white', lw=.3); bottom += v
    ax3.set_ylabel('Share of 3-hourly observations (%)'); ax3.set_ylim(0, 100)
    ax3.set_xticks(sh.index); ax3.set_xticklabels([f"'{str(y)[2:]}" for y in sh.index], fontsize=7)
    ax3.text(2025, 101, 'Jan–Oct', ha='center', fontsize=5.5)
    ax3.set_title('c', loc='left', fontweight='bold', x=-.16)
    fig.savefig(f'{FIG}/fig_observed.png', dpi=300, bbox_inches='tight'); plt.close(fig)

    # ---------------- Supplementary: correlation heatmap (3-hourly)
    cols = ['Temperature', 'Humidity', 'Rainfall', 'Atmosphere', 'Cloud', 'Wind', 'DI']
    C = df[cols].corr()
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(C.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(7)); ax.set_xticklabels(['T', 'RH', 'Rain', 'Pressure', 'Cloud', 'Wind', 'DI'], rotation=45)
    ax.set_yticks(range(7)); ax.set_yticklabels(['T', 'RH', 'Rain', 'Pressure', 'Cloud', 'Wind', 'DI'])
    for i in range(7):
        for j in range(7): ax.text(j, i, f'{C.values[i,j]:.2f}', ha='center', va='center', fontsize=6, color='w' if abs(C.values[i, j]) > .6 else 'k')
    fig.colorbar(im, fraction=.046); fig.tight_layout(); fig.savefig(f'{FIG}/figS_corr.png', dpi=300); plt.close(fig)
    json.dump(C.round(3).to_dict(), open(f'{OUT}/corr.json', 'w'), indent=1)


# ============================================================================= 4. Workflow figure

def run_framework():
    """Workflow diagram of the explainable hybrid framework."""
    fig, ax = plt.subplots(figsize=(7.4, 4.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis('off')
    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec='#33415c', lw=.8))
        ax.text(x + w / 2, y + h - 1.6, title, ha='center', va='top', fontweight='bold', fontsize=7.8)
        ax.text(x + w / 2, y + h - 5.0, body, ha='center', va='top', fontsize=6.6, linespacing=1.35)
    def arr(x1, y1, x2, y2): ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='-|>', color='#33415c', lw=.9))
    C1, C2, C3, C4 = '#e8eef7', '#fdf0e1', '#e6f4ea', '#f3e8f7'
    box(1, 44, 30, 16, '1  Data (BMD, Dhaka)', '3-hourly T, RH, rainfall, pressure,\ncloud cover and wind speed\nJan 2014 – Oct 2025 (n = 34,576)', C1)
    box(35, 44, 30, 16, '2  Target and predictors', 'Thom DI = T − 0.55(1 − 0.01RH)\n× (T − 14.5); predictors: DI lags\n(3, 9, 18 h), exogenous meteorology,\ncalendar and season', C1)
    box(69, 44, 30, 16, '3  Chronological split', 'Fitting: Jan 2014 – Jul 2021\nCalibration: Jul 2021 – Jun 2023\nTest: Jun 2023 – Oct 2025\n(n = 6,916 forecasts)', C1)
    box(1, 22, 30, 17, 'Stage 1: primary learner', 'with exogenous inputs (X):\nDTX, RFX, XGBX, SVRX, ANNX,\nLSTMX, GRUX, FPX or SARIMAX\n(raw DI or STL-adjusted DI)', C2)
    box(35, 22, 30, 17, 'Stage 2: residual learner', 'residuals only (no exogenous\ninputs), updated one step ahead:\nETS · SARIMA · TBATS (ML-TS)\nor ML/DL learner (TS-ML)', C2)
    box(69, 22, 30, 17, 'Stage 3: combination', 'DI(t) = primary(t) + residual(t)\n41 configurations × 2 tracks\n(raw, STL) = 82 models\n+ persistence & seasonal naive', C2)
    box(1, 1, 47, 16, 'Evaluation', 'MSE, RMSE, MAE, MAPE, MASE, R²\nDiebold–Mariano tests (HAC)\nDI-category skill (accuracy, F1, hit rate ≥ 29 °C)\nrolling conformal 90%/95% prediction intervals\noperational check: exogenous inputs lagged by 3 h', C3)
    box(52, 1, 47, 16, 'Explainable AI (SHAP)', 'TreeSHAP attributions of XGBoost (exact, additive)\nglobal importance & feature-group shares\ndependence of DI on pressure, cloud, rain, wind\nseasonal attribution & local event explanation', C4)
    arr(31.8, 52, 34.2, 52); arr(65.8, 52, 68.2, 52)
    arr(31.8, 30.5, 34.2, 30.5); arr(65.8, 30.5, 68.2, 30.5)
    ax.annotate('', (16, 39.8), (84, 43.5), arrowprops=dict(arrowstyle='-|>', color='#33415c', lw=.9, connectionstyle='angle,angleA=180,angleB=90,rad=0'))
    arr(84, 21.2, 70, 17.8); arr(84, 21.2, 30, 17.8)
    fig.savefig(f'{FIG}/fig_framework.png', dpi=300, bbox_inches='tight'); plt.close(fig)


# ============================================================================= 5. Evaluation

def run_analyse():
    """Metrics, Diebold-Mariano tests, DI-category skill, rolling conformal intervals, forecast figure."""
    df = load(); den = mase_den(df['DI'].iloc[:TEST0]); y = df['DI'].values[TEST0:]; idx = df.index[TEST0:]
    EDGES = [-np.inf, 21, 24, 27, 29, 32, np.inf]

    ML_ = {'DT', 'RF', 'FP', 'SVR', 'XGB', 'ANN', 'LSTM', 'GRU'}
    def rn(k):
        """Stage-2 residual models use no exogenous inputs: <learner>-SARIMAX -> <learner>-SARIMA; STL-ARIMAX -> STL-SARIMAX."""
        k = k.replace('STL-ARIMAX', 'STL-SARIMAX')
        if k.endswith('-SARIMAX') and k[:-8].split('-')[-1] in ML_: k = k[:-8] + '-SARIMA'
        t = k.split('-'); i0 = 1 if t[0] == 'STL' else 0          # first-stage learner receives exogenous inputs -> suffix X
        if len(t) > i0 and t[i0] in ML_: t[i0] += 'X'
        return '-'.join(t)
    def collect(track, scen):
        P = {}
        for f in sorted(glob.glob(f'{OUT}/pred_{track}_{scen}_*.pkl')):
            d = pickle.load(open(f, 'rb')); P.update({rn(k): v for k, v in d.items() if not k.startswith('_')})
        return P

    def dm(e1, e2, lag=None):
        """Diebold-Mariano with Newey-West HAC variance and HLN small-sample correction (squared loss).
        Negative statistic -> model 1 more accurate."""
        d = e1 ** 2 - e2 ** 2; n = len(d); dbar = d.mean(); dc = d - dbar
        L = lag if lag is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
        v = np.sum(dc ** 2) / n
        for k in range(1, L + 1):
            v += 2 * (1 - k / (L + 1)) * np.sum(dc[k:] * dc[:-k]) / n
        stat = dbar / np.sqrt(v / n)
        return stat, 2 * norm.sf(abs(stat))

    tabs = {}
    for track, scen in (('raw', 'A'), ('stl', 'A'), ('raw', 'B')):
        P = collect(track, scen)
        if not P: continue
        rows = {k: metrics(y, v, den) for k, v in P.items() if np.all(np.isfinite(v))}
        tabs[f'{track}{scen}'] = pd.DataFrame(rows).T
        pickle.dump(P, open(f'{OUT}/all_{track}_{scen}.pkl', 'wb'))
    for k, t in tabs.items():
        t.round(4).to_csv(f'{OUT}/metrics_{k}.csv'); print(k, len(t)); print(t.sort_values('MSE').round(4).head(12))

    A = collect('raw', 'A'); S = collect('stl', 'A'); B = collect('raw', 'B')
    allA = {**A, **S}
    TA = pd.concat([tabs['rawA'], tabs.get('stlA', pd.DataFrame())]).drop_duplicates()
    best = TA.drop(['Persistence', 'Seasonal naive'], errors='ignore')['MSE'].idxmin()
    print('BEST', best)
    summary = dict(best=best, n_models=int(len(TA) - 2), mase_den=den)
    # ---------------- DM tests: best vs every other model (scenario A)
    eb = y - allA[best]
    dmres = {}
    for k, v in allA.items():
        if k == best: continue
        s, p = dm(eb, y - v); dmres[k] = dict(DM=s, p=p)
    DMT = pd.DataFrame(dmres).T.sort_values('DM'); DMT.round(4).to_csv(f'{OUT}/dm_best_vs_all.csv')
    summary['dm_n_sig'] = int((DMT['p'] < 0.05).sum()); summary['dm_n'] = int(len(DMT))
    summary['dm_not_sig'] = DMT[DMT['p'] >= 0.05].index.tolist()
    for k in ('XGBX', 'XGBX-ETS', 'XGBX-SARIMA', 'XGBX-TBATS', 'Persistence', 'SVRX', 'ANNX'):
        if k in dmres: summary[f'dm_{k}'] = dmres[k]
    # DM: hybrid vs its own primary
    pairs = {}
    for k in A:
        for r in ('-ETS', '-SARIMA', '-TBATS'):
            if k.endswith(r) and k[:-len(r)] in A:
                s, p = dm(y - A[k], y - A[k[:-len(r)]]); pairs[k] = dict(DM=s, p=p)
    pd.DataFrame(pairs).T.round(4).to_csv(f'{OUT}/dm_hybrid_vs_primary.csv')
    # STL vs raw counterpart
    stlp = {}
    for k in [k for k in S if k.startswith('STL-')]:
        base = k.replace('STL-ARIMAX', 'SARIMAX').replace('STL-', '')
        if base in A:
            s, p = dm(y - S[k], y - A[base]); stlp[k] = dict(DM=s, p=p, dMSE=np.mean((y - S[k]) ** 2) - np.mean((y - A[base]) ** 2))
    STLT = pd.DataFrame(stlp).T; STLT.round(4).to_csv(f'{OUT}/dm_stl_vs_raw.csv')
    summary['stl_better_sig'] = int(((STLT.DM < 0) & (STLT.p < .05)).sum()); summary['stl_worse_sig'] = int(((STLT.DM > 0) & (STLT.p < .05)).sum())
    summary['stl_n'] = int(len(STLT)); summary['stl_median_dMSE'] = float(STLT.dMSE.median())

    # ---------------- category skill
    def cats(v): return pd.cut(v, EDGES, labels=False, right=False).astype(int)
    cy = cats(y); cs = {}
    for k in [best, 'XGBX', 'SVRX', 'ANNX', 'LSTMX', 'GRUX', 'Persistence', 'Seasonal naive', 'SARIMAX']:
        if k not in allA: continue
        cp = cats(allA[k]); hy, hp = y >= 29, allA[k] >= 29
        cs[k] = dict(acc=accuracy_score(cy, cp), within1=np.mean(np.abs(cy - cp) <= 1), f1_macro=f1_score(cy, cp, average='macro'),
                     kappa_w=cohen_kappa_score(cy, cp, weights='quadratic'),
                     POD29=(hy & hp).sum() / hy.sum(), FAR29=(~hy & hp).sum() / max(hp.sum(), 1),
                     CSI29=(hy & hp).sum() / (hy | hp).sum())
    CS = pd.DataFrame(cs).T; CS.round(4).to_csv(f'{OUT}/category_skill.csv'); print(CS.round(3))
    summary['n_ge29_test'] = int((y >= 29).sum())

    # ---------------- rolling conformal intervals for the best model
    e = y - allA[best]
    cal_res = np.load(f'{OUT}/calres_raw_A_XGB.npy') if os.path.exists(f'{OUT}/calres_raw_A_XGB.npy') else e[:0]
    hist = list(np.abs(cal_res[-2920:]))
    W = 2920; lo = {}; cover = {}
    for a in (0.10, 0.05):
        h = []; buf = list(hist)
        for t in range(len(e)):
            q = np.quantile(buf[-W:], 1 - a) if buf else np.nan; h.append(q); buf.append(abs(e[t]))
        h = np.array(h); lo[a] = h
        cover[a] = dict(coverage=float(np.mean(np.abs(e) <= h)), mean_width=float(np.mean(2 * h)))
    summary['conformal'] = {f'{int((1-a)*100)}%': v for a, v in cover.items()}
    seas = df['Season'].values[TEST0:]
    summary['conformal_by_season'] = {s_: float(np.mean(np.abs(e[seas == s_]) <= lo[0.05][seas == s_])) for s_ in np.unique(seas)}
    # error by season and hour
    summary['rmse_by_season'] = {s_: float(np.sqrt(np.mean(e[seas == s_] ** 2))) for s_ in np.unique(seas)}
    hr = idx.hour
    summary['rmse_by_hour'] = {int(h_): float(np.sqrt(np.mean(e[hr == h_] ** 2))) for h_ in np.unique(hr)}

    # ---------------- scenario B comparison (operational, lagged exogenous)
    if 'rawB' in tabs:
        comp = pd.DataFrame({'A_MSE': tabs['rawA']['MSE'], 'B_MSE': tabs['rawB']['MSE']}).dropna()
        comp['change_%'] = 100 * (comp.B_MSE / comp.A_MSE - 1); comp.round(4).to_csv(f'{OUT}/scenario_AB.csv'); print(comp.round(3))

    json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))

    # ---------------- Figure: forecast skill
    fig = plt.figure(figsize=(7.4, 5.4)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=.45, wspace=.75)
    ax = fig.add_subplot(gs[0, :])
    w0, w1 = pd.Timestamp('2024-04-10'), pd.Timestamp('2024-05-10')
    m = (idx >= w0) & (idx < w1)
    ax.fill_between(idx[m], allA[best][m] - lo[0.05][m], allA[best][m] + lo[0.05][m], color='#9ecae1', alpha=.6, lw=0, label='95% conformal interval')
    ax.plot(idx[m], y[m], color='k', lw=1, label='Observed')
    ax.plot(idx[m], allA[best][m], color='#d94801', lw=1, ls='--', label=f'{best}')
    ax.plot(idx[m], allA['Persistence'][m], color='#6a51a3', lw=.7, alpha=.7, label='Persistence')
    ax.axhline(29, color='#a50f15', lw=.6, ls=':'); ax.text(idx[m][2], 29.15, 'Severe discomfort (29 °C)', fontsize=6, color='#a50f15')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y')); ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.set_ylabel('DI (°C)'); ax.legend(ncol=4, fontsize=6.5, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.0))
    ax.set_title('a', loc='left', fontweight='bold', x=-.07)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hexbin(y, allA[best], gridsize=45, cmap='Blues', mincnt=1, bins='log', linewidths=0)
    ax2.plot([10, 34], [10, 34], 'k-', lw=.6); ax2.set_xlabel('Observed DI (°C)'); ax2.set_ylabel('Forecast DI (°C)')
    ax2.text(11, 31.5, f"R² = {TA.loc[best,'R2']:.3f}\nRMSE = {TA.loc[best,'RMSE']:.3f}", fontsize=6.5)
    ax2.set_title('b', loc='left', fontweight='bold', x=-.3)
    ax3 = fig.add_subplot(gs[1, 1])
    top = TA.drop(['Persistence', 'Seasonal naive'], errors='ignore').sort_values('MSE')
    fam = {}
    for k in top.index:
        f = k.replace('STL-', '').split('-')[0]; fam.setdefault(f, k)
    show = [fam[f] for f in fam][:9] + ['Persistence']
    vals = TA.loc[show, 'RMSE']
    colors = ['#d94801' if k == best else ('#9e9ac8' if k == 'Persistence' else '#6baed6') for k in show]
    ax3.barh(range(len(show))[::-1], vals.values, color=colors)
    ax3.set_yticks(range(len(show))[::-1]); ax3.set_yticklabels(show, fontsize=6.3); ax3.set_xlabel('RMSE (°C)')
    ax3.set_title('c', loc='left', fontweight='bold', x=-.55)
    ax4 = fig.add_subplot(gs[1, 2])
    hrs = sorted(summary['rmse_by_hour']); ax4.plot(hrs, [summary['rmse_by_hour'][h] for h in hrs], 'o-', color='#d94801', ms=3, label=best)
    ep = y - allA['Persistence']; ax4.plot(hrs, [np.sqrt(np.mean(ep[hr == h] ** 2)) for h in hrs], 's-', color='#9e9ac8', ms=3, label='Persistence')
    ax4.set_xticks(hrs); ax4.set_xticklabels([f'{h:02d}' for h in hrs], fontsize=6.5); ax4.set_xlabel('Hour (UTC)'); ax4.set_ylabel('RMSE (°C)')
    ax4.legend(fontsize=6, frameon=False); ax4.set_title('d', loc='left', fontweight='bold', x=-.3)
    fig.savefig(f'{FIG}/fig_forecast.png', dpi=300, bbox_inches='tight'); plt.close(fig)


# ============================================================================= 6. Explainable AI

def run_xai():
    """Exact TreeSHAP attributions of the XGBoost primary learner (scenarios A and B)."""
    import shap
    os.makedirs(FIG, exist_ok=True)
    warnings.filterwarnings('ignore')
    NICE = {'DI_lag_1': 'DI (t−1, 3 h)', 'DI_lag_3': 'DI (t−3, 9 h)', 'DI_lag_6': 'DI (t−6, 18 h)', 'Atmosphere': 'Pressure',
            'Cloud': 'Cloud cover', 'Rainfall': 'Rainfall', 'Wind': 'Wind speed', 'Hour': 'Hour of day', 'Day': 'Day of month',
            'Month': 'Month', 'Season_Spring': 'Season: spring', 'Season_Summer': 'Season: summer', 'Season_Winter': 'Season: winter'}
    GROUP = {'DI_lag_1': 'Autoregressive DI', 'DI_lag_3': 'Autoregressive DI', 'DI_lag_6': 'Autoregressive DI',
             'Atmosphere': 'Exogenous meteorology', 'Cloud': 'Exogenous meteorology', 'Rainfall': 'Exogenous meteorology',
             'Wind': 'Exogenous meteorology', 'Hour': 'Calendar', 'Day': 'Calendar', 'Month': 'Calendar',
             'Season_Spring': 'Calendar', 'Season_Summer': 'Calendar', 'Season_Winter': 'Calendar'}
    res = {}
    for scen in ('A', 'B'):
        fn = f'{OUT}/xgb_raw_{scen}.pkl'
        if not os.path.exists(fn): continue
        m = pickle.load(open(fn, 'rb'))
        df = load(); X, y = design(df, 'DI', scen)
        Xte = X.iloc[TEST0:]
        ex = shap.TreeExplainer(m)
        sv = ex(Xte.values); sv.feature_names = [NICE[c] for c in X.columns]
        S = sv.values
        imp = pd.Series(np.abs(S).mean(0), index=X.columns).sort_values(ascending=False)
        grp = pd.Series(np.abs(S).mean(0), index=X.columns).groupby(GROUP).sum()
        res[scen] = dict(base=float(ex.expected_value), imp=imp.round(4).to_dict(), group=(grp / grp.sum()).round(4).to_dict(),
                         additivity=float(np.max(np.abs(S.sum(1) + ex.expected_value - m.predict(Xte.values)))))
        # seasonal decomposition of exogenous SHAP
        sea = df['Season'].iloc[TEST0:].values
        tab = pd.DataFrame(np.abs(S), columns=X.columns); tab['Season'] = sea
        res[scen]['by_season'] = tab.groupby('Season')[['Atmosphere', 'Cloud', 'Rainfall', 'Wind', 'DI_lag_1']].mean().round(4).to_dict()
        np.save(f'{OUT}/shap_{scen}.npy', S)
        if scen == 'A':
            # ---- Figure: global importance (beeswarm) + grouped bar
            fig = plt.figure(figsize=(7.2, 3.6))
            ax1 = fig.add_axes([0.22, 0.13, 0.42, 0.8])
            plt.sca(ax1); shap.plots.beeswarm(sv, max_display=13, show=False, plot_size=None, color_bar=True)
            ax1.set_xlabel('SHAP value (contribution to DI, °C)', fontsize=8); ax1.tick_params(labelsize=7.5)
            ax1.set_title('a  Local contributions (test set)', loc='left', fontsize=9, fontweight='bold')
            ax2 = fig.add_axes([0.80, 0.13, 0.18, 0.8])
            g = (grp / grp.sum() * 100).sort_values()
            ax2.barh(g.index, g.values, color=['#9aa5b1', '#2b6cb0', '#c05621'][:len(g)])
            for i, v in enumerate(g.values): ax2.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=7.5)
            ax2.set_xlim(0, 110); ax2.set_xlabel('Share of mean |SHAP| (%)', fontsize=8); ax2.tick_params(labelsize=7.5)
            ax2.set_title('b  Feature groups', loc='left', fontsize=9, fontweight='bold')
            fig.savefig(f'{FIG}/fig_shap_global.png', dpi=300); plt.close(fig)
            # ---- Figure: dependence plots for exogenous drivers
            fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.5), gridspec_kw=dict(wspace=0.55))
            cols = ['Atmosphere', 'Cloud', 'Rainfall', 'Wind']
            colvals = df['Month'].iloc[TEST0:].values
            for a, c in zip(axs, cols):
                j = list(X.columns).index(c); x = Xte[c].values
                sc = a.scatter(x, S[:, j], c=colvals, cmap='twilight_shifted', s=2, alpha=.6, rasterized=True)
                a.axhline(0, color='k', lw=.5)
                if c in ('Rainfall', 'Wind'): a.set_xscale('symlog', linthresh=1)
                a.set_xlabel({'Atmosphere': 'Pressure (hPa)', 'Cloud': 'Cloud cover (okta)', 'Rainfall': 'Rainfall (mm)', 'Wind': 'Wind (kt)'}[c], fontsize=8)
                a.tick_params(labelsize=7)
                if c == 'Cloud': a.set_xlim(-0.5, 8.5)
                if c in ('Rainfall', 'Wind'): a.set_xlim(left=0)
            axs[0].set_ylabel('SHAP value (°C)', fontsize=8)
            cb = fig.colorbar(sc, ax=axs, fraction=.02, pad=.01); cb.set_label('Month', fontsize=7); cb.ax.tick_params(labelsize=6)
            for i, a in enumerate(axs): a.set_title('abcd'[i], loc='left', fontweight='bold', fontsize=9)
            fig.savefig(f'{FIG}/fig_shap_dependence.png', dpi=300, bbox_inches='tight'); plt.close(fig)
            # ---- local explanation for the most extreme test observation
            k = int(np.argmax(y.iloc[TEST0:].values))
            res['A']['extreme'] = dict(time=str(Xte.index[k]), obs=float(y.iloc[TEST0 + k]), pred=float(m.predict(Xte.values[k:k + 1])[0]),
                                       contrib={c: round(float(S[k, i]), 3) for i, c in enumerate(X.columns)})
            fig = plt.figure(figsize=(6.0, 3.4))
            shap.plots.waterfall(sv[k], max_display=9, show=False)
            plt.gcf().set_size_inches(6.4, 3.6); plt.tight_layout()
            plt.savefig(f'{FIG}/fig_shap_waterfall.png', dpi=300, bbox_inches='tight'); plt.close('all')
    json.dump(res, open(f'{OUT}/xai.json', 'w'), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str)[:4000])

def run_xai_gru():
    """Expected-gradients SHAP of the GRU primary learner (scenario A, raw track)."""
    import shap, torch, torch.nn as nn
    SEED = 42; win = 6
    torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(2)
    df = load(); X, y = design(df, 'DI', 'A'); first = 8
    Xtr, ytr = X.values[first:TEST0], y.values[first:TEST0]; Xte = X.values[TEST0:]
    sx = StandardScaler().fit(Xtr); sy = StandardScaler().fit(ytr.reshape(-1, 1))
    A = sx.transform(Xtr).astype(np.float32); b = sy.transform(ytr.reshape(-1, 1)).astype(np.float32).ravel(); B = sx.transform(Xte).astype(np.float32)
    def seq(M, head=None):
        M2 = M if head is None else np.vstack([head, M]); idx = np.arange(win - 1, len(M2))
        return np.stack([M2[i - win + 1:i + 1] for i in idx])
    Atr = seq(A); btr = b[win - 1:]; Ate = seq(B, A[-(win - 1):])
    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.r = nn.GRU(A.shape[1], 50, num_layers=2, batch_first=True, dropout=0.1); s.o = nn.Linear(50, 1)
        def forward(s, x):
            h, _ = s.r(x); return s.o(h[:, -1])
    net = Net(); nv = int(.1 * len(Atr))
    Xt_, yt_, Xv_, yv_ = map(torch.tensor, (Atr[:-nv], btr[:-nv], Atr[-nv:], btr[-nv:]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); L = nn.MSELoss(); best, bad = 1e9, 0
    for ep in range(60):
        net.train(); perm = torch.randperm(len(Xt_))
        for i in range(0, len(perm), 256):
            j = perm[i:i + 256]; opt.zero_grad(); l = L(net(Xt_[j]).squeeze(-1), yt_[j]); l.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl = L(net(Xv_).squeeze(-1), yv_).item()
        if vl < best - 1e-5: best, bad, st = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 8: break
    net.load_state_dict(st); net.eval()
    with torch.no_grad(): p = sy.inverse_transform(net(torch.tensor(Ate)).numpy()).ravel()
    mse = float(np.mean((y.values[TEST0:] - p) ** 2)); print('GRU test MSE', mse)
    rng = np.random.default_rng(SEED)
    bg = torch.tensor(Atr[rng.choice(len(Atr), 300, replace=False)])
    sel = rng.choice(len(Ate), 1500, replace=False)
    torch.backends.cudnn.enabled = False
    net.train(False)
    ex = shap.GradientExplainer(net, bg)
    sv = ex.shap_values(torch.tensor(Ate[sel]), nsamples=200)
    sv = np.asarray(sv).reshape(len(sel), win, A.shape[1]) * sy.scale_[0]      # in DI units
    imp_feat = np.abs(sv.sum(1)).mean(0)            # aggregate over the 6-step window per feature
    imp_step = np.abs(sv).sum(2).mean(0)            # aggregate over features per time step (t-5 ... t)
    cols = list(X.columns)
    G = {'Autoregressive DI': [c for c in cols if c.startswith('DI_lag')], 'Exogenous meteorology': ['Rainfall', 'Cloud', 'Atmosphere', 'Wind'],
         'Calendar': ['Hour', 'Day', 'Month', 'Season_Spring', 'Season_Summer', 'Season_Winter']}
    tot = imp_feat.sum()
    res = dict(test_mse=mse, imp={c: float(v) for c, v in zip(cols, imp_feat)},
               group={g: float(sum(imp_feat[cols.index(c)] for c in cs) / tot) for g, cs in G.items()},
               step_share={f't-{win-1-i}': float(v / imp_step.sum()) for i, v in enumerate(imp_step)})
    json.dump(res, open(f'{OUT}/xai_gru.json', 'w'), indent=1); print(json.dumps(res, indent=1))


# ============================================================================= 7. Numbers for the manuscript

def run_values():
    """Derive every number quoted in the manuscript and Table 2 (values.json in OUT)."""
    E = json.load(open(f'{OUT}/eda.json')); S = json.load(open(f'{OUT}/summary.json')); X = json.load(open(f'{OUT}/xai.json'))
    G = json.load(open(f'{OUT}/xai_gru.json'))
    tA = pd.read_csv(f'{OUT}/metrics_rawA.csv', index_col=0); tS = pd.read_csv(f'{OUT}/metrics_stlA.csv', index_col=0); tB = pd.read_csv(f'{OUT}/metrics_rawB.csv', index_col=0)
    TA = pd.concat([tA, tS.drop(['Persistence', 'Seasonal naive'], errors='ignore')])
    CS = pd.read_csv(f'{OUT}/category_skill.csv', index_col=0)
    DM = pd.read_csv(f'{OUT}/dm_best_vs_all.csv', index_col=0)
    HP = pd.read_csv(f'{OUT}/dm_hybrid_vs_primary.csv', index_col=0)
    ST = pd.read_csv(f'{OUT}/dm_stl_vs_raw.csv', index_col=0)
    df = load(); di = df['DI']
    f2 = lambda v: f'{v:.2f}'; f3 = lambda v: f'{v:.3f}'; f1 = lambda v: f'{v:.1f}'
    def fp(p): return '< 0.001' if p < 0.001 else f'{p:.3f}'
    v = {}
    d = E['desc']
    v['di_mean'] = f2(d['mean']['DI']); v['di_sd'] = f2(d['std']['DI']); v['di_min'] = f2(d['min']['DI']); v['di_max'] = f2(d['max']['DI'])
    cs = E['cat_share_all']; v['sh2'] = f1(cs['2']); v['sh3'] = f1(cs['3']); v['sh4'] = f1(cs['4']); v['n_emerg'] = int((di >= 32).sum())
    hm = di.groupby([df.index.month, df.index.hour]).mean().unstack()
    mx = hm.stack().idxmax(); v['peak_val'] = f1(hm.stack().max()); v['peak_hour'] = f'{mx[1]:02d}:00'
    v['peak_month'] = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][mx[0] - 1]
    rng = (hm.max(1) - hm.min(1)); v['_diurnal_range'] = rng.round(2).to_dict()
    v['ann_slope'] = f2(E['sen_slope_per_decade']); v['ann_p'] = fp(E['MK_p'])
    for s_ in ('JJAS', 'MAM', 'DJF'): v[f'{s_.lower()}_slope'] = f2(E[f'{s_}_sen_per_decade']); v[f'{s_.lower()}_p'] = fp(E[f'{s_}_MK_p'])
    v['sev_2014'] = f1(E['severe_share']['2014']); v['sev_2024'] = f1(E['severe_share']['2024'])
    v['sev_slope'] = f1(E['severe_sen_per_decade']); v['sev_p'] = fp(E['severe_MK_p']); v['ge27_slope'] = f1(E['ge27_sen_per_decade']); v['ge27_p'] = fp(E['ge27_MK_p'])
    pers = TA.loc['Persistence', 'MSE']
    mods = TA.drop(['Persistence', 'Seasonal naive'])
    assert len(mods) == 82, len(mods)
    red = 100 * (1 - mods['MSE'] / pers)
    v['pers_rmse'] = f3(TA.loc['Persistence', 'RMSE']); v['snaive_rmse'] = f3(TA.loc['Seasonal naive', 'RMSE'])
    v['min_red'] = f'{red.min():.0f}'; v['max_red'] = f'{red.max():.0f}'
    singles = ['SARIMAX', 'DTX', 'RFX', 'FPX', 'SVRX', 'XGBX', 'ANNX', 'LSTMX', 'GRUX']
    sg = tA.loc[singles].sort_values('MSE')
    v['best_single'] = sg.index[0]; v['best_single_rmse'] = f3(sg.iloc[0]['RMSE']); v['best_single_mae'] = f3(sg.iloc[0]['MAE'])
    for k in ('RF', 'XGB', 'ANN', 'FP'): v[f'{k.lower()}_rmse'] = f3(tA.loc[k + 'X', 'RMSE'])
    mid = tA.loc[['SVRX', 'SARIMAX', 'DTX'], 'RMSE']; v['mid_lo'] = f3(mid.min()); v['mid_hi'] = f3(mid.max())
    v['n_hyb_better'] = int((HP['DM'] < 0).sum()); v['n_hyb_sig'] = int(((HP['DM'] < 0) & (HP['p'] < .05)).sum()); v['_n_hyb'] = int(len(HP))
    prim = ['DT', 'RF', 'FP', 'SVR', 'XGB', 'ANN', 'LSTM', 'GRU']
    bestres = {p: tA.loc[[f'{p}X-ETS', f'{p}X-SARIMA', f'{p}X-TBATS'], 'MSE'].idxmin().split('-')[-1] for p in prim}
    v['n_sx_best'] = sum(1 for x in bestres.values() if x == 'SARIMA'); v['_bestres'] = bestres
    b = S['best']; v['best'] = b
    for m_ in ('RMSE', 'MAE', 'MAPE', 'MASE', 'R2'): v[f'best_{m_.lower()}'] = f3(TA.loc[b, m_]) if m_ != 'MAPE' else f2(TA.loc[b, m_])
    v['best_red_pers'] = f1(100 * (1 - TA.loc[b, 'MSE'] / pers)); v['best_red_single'] = f1(100 * (1 - TA.loc[b, 'MSE'] / sg.iloc[0]['MSE']))
    v['dm_n_sig'] = S['dm_n_sig']; v['dm_n'] = S['dm_n']
    ns = [k for k in S['dm_not_sig']]
    v['dm_not_sig'] = ', '.join(ns[:-1]) + (' and ' if len(ns) > 1 else '') + ns[-1] if ns else 'none'
    tsml = tA.loc[[i for i in tA.index if i.startswith('SARIMAX-')]].sort_values('MSE')
    v['n_tsml_better'] = int((tsml['MSE'] < tA.loc['SARIMAX', 'MSE']).sum()); v['tsml_worst'] = tsml.index[-1]; v['best_tsml'] = tsml.index[0]; v['best_tsml_rmse'] = f3(tsml.iloc[0]['RMSE'])
    v['stl_dmse'] = f3(ST['dMSE'].median()); v['stl_worse_sig'] = int(((ST.DM > 0) & (ST.p < .05)).sum()); v['stl_n'] = int(len(ST))
    v['_stl_better'] = int((ST.dMSE < 0).sum())
    bs = tS.drop(['Persistence', 'Seasonal naive'], errors='ignore').sort_values('MSE'); v['best_stl'] = bs.index[0]; v['best_stl_rmse'] = f3(bs.iloc[0]['RMSE'])
    tb = tB.drop(['Persistence', 'Seasonal naive']).sort_values('MSE'); v['bestB'] = tb.index[0]; v['bestB_rmse'] = f3(tb.iloc[0]['RMSE'])
    v['bestB_change'] = f'{100 * (tb.iloc[0]["MSE"] / TA.loc[b, "MSE"] - 1):.0f}'; v['bestB_red_pers'] = f'{100 * (1 - tb.iloc[0]["MSE"] / pers):.0f}'
    c = CS.loc[b]; pc = CS.loc['Persistence']
    v['cat_acc'] = f1(100 * c.acc); v['cat_w1'] = f1(100 * c.within1); v['cat_kappa'] = f3(c.kappa_w); v['pcat_acc'] = f1(100 * pc.acc); v['pcat_kappa'] = f3(pc.kappa_w)
    v['pod'] = f2(c.POD29); v['far'] = f2(c.FAR29); v['csi'] = f2(c.CSI29); v['ppod'] = f2(pc.POD29); v['pfar'] = f2(pc.FAR29); v['pcsi'] = f2(pc.CSI29)
    v['pod_pct'] = f'{100 * c.POD29:.0f}'; v['n29'] = f"{S['n_ge29_test']:,}"
    rs = S['rmse_by_season']; v['rmse_spring'] = f3(rs['Spring']); v['rmse_summer'] = f3(rs['Summer']); v['_rmse_season'] = rs
    rh = {int(k): x for k, x in S['rmse_by_hour'].items()}; v['err_peak_hour'] = f'{max(rh, key=rh.get):02d}:00'; v['_rmse_hour'] = rh
    cf = S['conformal']; v['cov90'] = f1(100 * cf['90%']['coverage']); v['cov95'] = f1(100 * cf['95%']['coverage'])
    v['w90'] = f2(cf['90%']['mean_width']); v['w95'] = f2(cf['95%']['mean_width'])
    cb = S['conformal_by_season']; v['cov_lo'] = f1(100 * min(cb.values())); v['cov_hi'] = f1(100 * max(cb.values()))
    A = X['A']; imp = A['imp']; gA = A['group']; gB = X['B']['group']
    v['shap_lag1'] = f2(imp['DI_lag_1']); v['shap_hour'] = f2(imp['Hour']); v['shap_lag6'] = f2(imp['DI_lag_6'])
    v['shap_p'] = f2(imp['Atmosphere']); v['shap_rain'] = f2(imp['Rainfall']); v['shap_cloud'] = f2(imp['Cloud']); v['shap_wind'] = f3(imp['Wind'])
    v['g_ar'] = f1(100 * gA['Autoregressive DI']); v['g_cal'] = f1(100 * gA['Calendar']); v['g_ex'] = f1(100 * gA['Exogenous meteorology']); v['gB_ex'] = f1(100 * gB['Exogenous meteorology'])
    v['shapB_rain'] = f3(X['B']['imp']['Rainfall'])
    v['gru_ar'] = f1(100 * G['group']['Autoregressive DI']); v['gru_cal'] = f1(100 * G['group']['Calendar']); v['gru_ex'] = f1(100 * G['group']['Exogenous meteorology'])
    v['gru_last'] = f'{100 * G["step_share"]["t-0"]:.0f}'
    Sv = np.load(f'{OUT}/shap_A.npy'); cols = ['Rainfall', 'Cloud', 'Atmosphere', 'Wind', 'Hour', 'Day', 'Month', 'DI_lag_1', 'DI_lag_3', 'DI_lag_6', 'Season_Spring', 'Season_Summer', 'Season_Winter']
    v['p_max'] = f1(Sv[:, cols.index('Atmosphere')].max()); v['rain_min'] = f1(Sv[:, cols.index('Rainfall')].min())
    bsn = A['by_season']; v['rain_win'] = f3(bsn['Rainfall']['Winter']); v['rain_spr'] = f3(bsn['Rainfall']['Spring']); v['p_win'] = f3(bsn['Atmosphere']['Winter'])
    ex = A['extreme']; v['ext_obs'] = f2(ex['obs']); v['ext_pred'] = f2(ex['pred']); v['ext_time'] = pd.Timestamp(ex['time']).strftime('%H:%M on %d %B %Y')
    v['ext_lag1'] = f2(ex['contrib']['DI_lag_1']); v['ext_hour'] = f2(ex['contrib']['Hour']); v['ext_p'] = f2(ex['contrib']['Atmosphere'])
    v['mase_den'] = f'{S["mase_den"]:.3f}'
    # ---------- Table 2
    def row(name, t, src):
        r = t.loc[name]; dmv = '—' if name == b else (f"{DM.loc[name,'DM']:.2f} ({fp(DM.loc[name,'p'])})" if name in DM.index else '')
        return f"| {src} | {name} | {r.RMSE:.3f} | {r.MAE:.3f} | {r.MAPE:.2f} | {r.MASE:.3f} | {r.R2:.4f} | {dmv} |"
    lines = ["Table: ⟦B⟧**Table 2.** Test-period accuracy of the reference forecasts, the nine single models and the six most accurate hybrid configurations (scenario A, 6,916 three-hourly forecasts, 20 June 2023 – 31 October 2025). DM, Diebold–Mariano statistic of the best model against each model (negative values favour the best model; *p* in parentheses). Complete results for all 82 configurations are given in Supplementary Tables S2–S3.", "",
             "| Group | Model | RMSE (°C) | MAE (°C) | MAPE (%) | MASE | *R*^2^ | DM (*p*) |", "|:--|:--|--:|--:|--:|--:|--:|--:|"]
    lines[2] = lines[2].replace('| Group', '| ⟦T⟧Group')
    for n in ('Persistence', 'Seasonal naive'): lines.append(row(n, TA, 'Reference'))
    for n in tA.loc[singles].sort_values('MSE').index: lines.append(row(n, TA, 'Single'))
    hyb = mods.drop(singles, errors='ignore').sort_values('MSE').index[:6]
    for n in hyb: lines.append(row(n, TA, 'Hybrid'))
    v['TABLE2'] = '\n'.join(lines)
    json.dump(v, open(f'{OUT}/values.json', 'w'), indent=1, default=str)
    print(json.dumps({k: x for k, x in v.items() if k != 'TABLE2'}, indent=1, default=str)); print(v['TABLE2'])


# ============================================================================= Command-line interface
ALL_JOBS = ([('raw', 'A', g) for g in ['naive', 'SARIMAX', 'DT', 'RF', 'XGB', 'SVR', 'ANN', 'LSTM', 'GRU', 'FP']] +
            [('stl', 'A', g) for g in ['naive', 'SARIMAX', 'DT', 'RF', 'XGB', 'SVR', 'ANN', 'LSTM', 'GRU', 'FP']] +
            [(t, 'A', 'SX-' + g) for t in ('raw', 'stl') for g in ['DT', 'RF', 'XGB', 'SVR', 'ANN', 'LSTM', 'GRU', 'FP']] +
            [('raw', 'B', g) for g in ['naive', 'SARIMAX', 'DT', 'RF', 'XGB', 'SVR', 'ANN', 'LSTM', 'GRU', 'FP']])

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Explainable hybrid AI for three-hourly DI forecasting')
    ap.add_argument('step', choices=['eda', 'framework', 'model', 'all-models', 'analyse', 'xai', 'xai-gru', 'values'])
    ap.add_argument('args', nargs='*', help='for "model": track (raw|stl) scenario (A|B) group')
    ap.add_argument('--data', default=DATA); ap.add_argument('--out', default=OUT); ap.add_argument('--fig', default=FIG)
    ap.add_argument('--notbats', action='store_true', help='skip TBATS residual models (used for scenario B)')
    a = ap.parse_args()
    DATA, OUT, FIG = a.data, a.out, a.fig
    os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    if a.step == 'eda': run_eda()
    elif a.step == 'framework': run_framework()
    elif a.step == 'model':
        tr, sc, gr = a.args; run_model(tr, sc, gr, NOTBATS=a.notbats or sc == 'B')
    elif a.step == 'all-models':
        for tr, sc, gr in ALL_JOBS:
            t0 = time.time(); run_model(tr, sc, gr, NOTBATS=(sc == 'B')); print(f'{tr} {sc} {gr} done in {time.time()-t0:.0f} s', flush=True)
    elif a.step == 'analyse': run_analyse()
    elif a.step == 'xai': run_xai()
    elif a.step == 'xai-gru': run_xai_gru()
    elif a.step == 'values': run_values()
