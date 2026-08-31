#!/usr/bin/env python3
"""Build aligned WTI spot-futures dataset and run cointegration diagnostics.
Only numpy + pandas are available (no scipy/statsmodels), so ADF, Johansen
trace, and ARCH-LM tests are implemented directly."""
import numpy as np, pandas as pd, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(HERE, "raw")

MONTHS = ["01","02","03","04","05","06","07","08","09","10","11","12"]

def read_spot(fn):
    d, v = [], []
    for line in open(fn):
        line=line.strip()
        if not line: continue
        dt, val = line.split()
        d.append(dt); v.append(float(val))
    return pd.Series(v, index=pd.to_datetime(d), name="spot")

def read_futures(fn, name):
    idx, val = [], []
    for line in open(fn):
        parts=line.split()
        if not parts: continue
        yr=parts[0]; cells=parts[1:]
        for m,c in zip(MONTHS,cells):
            idx.append(pd.Timestamp(f"{yr}-{m}-01"))
            val.append(np.nan if c=="NA" else float(c))
    return pd.Series(val, index=pd.DatetimeIndex(idx), name=name)

spot = read_spot(os.path.join(RAW,"spot_MCOILWTICO.txt"))
f1 = read_futures(os.path.join(RAW,"RCLC1m.txt"),"F1")
f2 = read_futures(os.path.join(RAW,"RCLC2m.txt"),"F2")
f3 = read_futures(os.path.join(RAW,"RCLC3m.txt"),"F3")
f4 = read_futures(os.path.join(RAW,"RCLC4m.txt"),"F4")

df = pd.concat([spot,f1,f2,f3,f4],axis=1).sort_index()
# aligned sample = rows where ALL five series present
levels = df.dropna()
print("Full spot range:", spot.index.min().date(), "to", spot.index.max().date(), "n=",spot.notna().sum())
for s in [f1,f2,f3,f4]:
    print(f"  {s.name} range:", s.dropna().index.min().date(),"to",s.dropna().index.max().date(),"n=",s.notna().sum())
print("Aligned (all 5) range:", levels.index.min().date(),"to",levels.index.max().date(),"n=",len(levels))
print("Any negative/zero prices in aligned sample?", (levels<=0).any().any())

# log prices
log = np.log(levels)
log.columns = ["l_"+c for c in levels.columns]

out = pd.concat([levels, log], axis=1)
out.index.name="date"
csv_path = os.path.join(HERE,"wti_spot_futures.csv")
os.makedirs(os.path.dirname(csv_path),exist_ok=True)
out.round(6).to_csv(csv_path)
print("Wrote", csv_path, "shape", out.shape)

# ---------- helpers ----------
def gammq(a,x):
    """Regularized upper incomplete gamma Q(a,x) (Numerical Recipes)."""
    if x<0 or a<=0: return float('nan')
    if x==0: return 1.0
    gln=math.lgamma(a)
    if x < a+1.0:  # series
        ap=a; s=1.0/a; d=s
        for _ in range(1000):
            ap+=1; d*=x/ap; s+=d
            if abs(d)<abs(s)*1e-12: break
        return 1.0 - s*math.exp(-x+a*math.log(x)-gln)
    else:  # continued fraction
        b=x+1.0-a; c=1e30; dd=1.0/b; h=dd
        for i in range(1,1000):
            an=-i*(i-a); b+=2.0
            dd=an*dd+b
            if abs(dd)<1e-30: dd=1e-30
            c=b+an/c
            if abs(c)<1e-30: c=1e-30
            dd=1.0/dd; delta=dd*c; h*=delta
            if abs(delta-1.0)<1e-12: break
        return math.exp(-x+a*math.log(x)-gln)*h

def chi2_sf(x,k):   return gammq(k/2.0, x/2.0)
def norm_cdf(x):    return 0.5*math.erfc(-x/math.sqrt(2))

def ols(y,X):
    b,_,_,_=np.linalg.lstsq(X,y,rcond=None)
    e=y-X@b
    n,k=X.shape
    s2=e@e/(n-k)
    XtXinv=np.linalg.inv(X.T@X)
    se=np.sqrt(np.diag(s2*XtXinv))
    return b,se,e

# ---------- ADF ----------
# MacKinnon (2010) asymptotic 5% critical values
ADF_CV = {("c",0.05):-2.86, ("ct",0.05):-3.41, ("c",0.01):-3.43, ("ct",0.01):-3.96,
          ("c",0.10):-2.57, ("ct",0.10):-3.13}
def adf(y, lags, trend="c"):
    y=np.asarray(y,float)
    dy=np.diff(y)
    n=len(dy)
    Z=[]  # regressors
    yl=y[lags:-1] if lags>0 else y[:-1]
    # build with lags of dy
    rows=n-lags
    Yd=dy[lags:]
    cols=[y[lags:-1]]  # level lag (aligned)
    for j in range(1,lags+1):
        cols.append(dy[lags-j:-j])
    X=np.column_stack(cols)
    det=np.ones((rows,1))
    if trend=="ct":
        det=np.column_stack([det, np.arange(rows)])
    X=np.column_stack([X,det])
    b,se,e=ols(Yd,X)
    tstat=b[0]/se[0]
    return tstat

# ---------- Johansen trace (restricted constant, "ci") ----------
# Osterwald-Lenum (1992) / MacKinnon-Haug-Michelis 5% trace CVs, restricted const
# keyed by (n-r) = number of common-trend hypotheses tested
TRACE_CV_05 = {1:9.24,2:19.96,3:34.91,4:53.12,5:76.07}
TRACE_CV_01 = {1:12.97,2:24.60,3:41.07,4:60.16,5:84.45}

def johansen_trace(Y, k):
    """Y: T x p levels. k = number of lags in the VAR (>=1). Restricted constant.
    Returns eigenvalues (desc) and trace stats for r=0..p-1."""
    Y=np.asarray(Y,float)
    T,p=Y.shape
    dY=np.diff(Y,axis=0)                    # (T-1) x p
    # regressors: lagged diffs 1..k-1
    Z=[]
    start=k-1
    dZ=[]
    for i in range(1,k):
        dZ.append(dY[start-i:-i])
    dY0=dY[start:]                          # (T-k) x p   dependent
    Ylag=Y[start:-1]                        # (T-k) x p   levels lag (t-1)
    n=dY0.shape[0]
    # restricted constant -> append 1 to Ylag block (enters cointegration space)
    Ylag_r=np.column_stack([Ylag, np.ones((n,1))])
    # deterministic in short run: none extra (const restricted). Stack lagged diffs.
    if dZ:
        W=np.column_stack(dZ)
    else:
        W=np.zeros((n,0))
    # partial out W from dY0 and Ylag_r
    if W.shape[1]>0:
        Wp=np.column_stack([W])  # no unrestricted const (it's restricted)
        P=Wp@np.linalg.pinv(Wp)
        R0=dY0 - P@dY0
        R1=Ylag_r - P@Ylag_r
    else:
        R0=dY0; R1=Ylag_r
    S00=R0.T@R0/n; S11=R1.T@R1/n; S01=R0.T@R1/n; S10=S01.T
    M=np.linalg.inv(S11)@S10@np.linalg.inv(S00)@S01
    eig=np.linalg.eigvals(M).real
    eig=np.sort(eig)[::-1]
    eig=eig[:p]                              # p eigenvalues of interest
    eig=np.clip(eig,1e-12,1-1e-12)
    trace=[]
    for r in range(p):
        stat=-n*np.sum(np.log(1-eig[r:]))
        trace.append(stat)
    return eig, np.array(trace), n

# ---------- VAR + ARCH-LM ----------
def var_resid(Y,k):
    Y=np.asarray(Y,float); T,p=Y.shape
    rows=T-k
    Ydep=Y[k:]
    cols=[np.ones((rows,1))]
    for j in range(1,k+1):
        cols.append(Y[k-j:T-j])
    X=np.column_stack(cols)
    B,_,_,_=np.linalg.lstsq(X,Ydep,rcond=None)
    E=Ydep-X@B
    return E

def arch_lm(e, q):
    e=np.asarray(e,float); e2=e**2
    n=len(e2)-q
    y=e2[q:]
    cols=[np.ones(n)]
    for j in range(1,q+1):
        cols.append(e2[q-j:len(e2)-j])
    X=np.column_stack(cols)
    b,_,_=ols(y,X)
    yhat=X@b; ybar=y.mean()
    ss_res=np.sum((y-yhat)**2); ss_tot=np.sum((y-ybar)**2)
    R2=1-ss_res/ss_tot
    LM=n*R2
    return LM, chi2_sf(LM,q)

print("\n================ DIAGNOSTICS ================")
LL = log.values
names=list(log.columns)
print("\n-- ADF tests (5% CV in brackets; H0: unit root) --")
print("Levels (trend=ct):")
for i,nm in enumerate(names):
    t=adf(LL[:,i],lags=2,trend="ct")
    print(f"  {nm:8s} ADF={t:7.3f}  [5% CV {ADF_CV[('ct',0.05)]:.2f}]  {'reject' if t<ADF_CV[('ct',0.05)] else 'fail-to-reject'}")
print("First differences (trend=c):")
for i,nm in enumerate(names):
    t=adf(np.diff(LL[:,i]),lags=1,trend="c")
    print(f"  d{nm:7s} ADF={t:7.3f}  [5% CV {ADF_CV[('c',0.05)]:.2f}]  {'reject' if t<ADF_CV[('c',0.05)] else 'fail-to-reject'}")

print("\n-- Johansen trace test on (l_spot,l_F1..l_F4), VAR lag k=2, restricted const --")
eig,trace,neff=johansen_trace(LL,k=2)
p=len(names)
print(f"  effective obs = {neff}")
print(f"  {'H0:r<=':7s}{'trace':>10s}{'5% CV':>10s}{'1% CV':>10s}  decision")
for r in range(p):
    nmr=p-r
    cv5=TRACE_CV_05[nmr]; cv1=TRACE_CV_01[nmr]
    dec="reject" if trace[r]>cv5 else "fail-to-reject"
    print(f"  r<={r:<4d}{trace[r]:10.2f}{cv5:10.2f}{cv1:10.2f}  {dec}")
print("  eigenvalues:", np.round(eig,4))

print("\n-- ARCH-LM on VAR(2) residuals (per equation), q=4 lags --")
E=var_resid(LL,k=2)
for i,nm in enumerate(names):
    LM,pv=arch_lm(E[:,i],q=4)
    print(f"  {nm:8s} LM={LM:8.2f}  p={pv:.3g}  {'ARCH' if pv<0.05 else 'no ARCH'}")
# multivariate: ARCH-LM on sum of squared standardized resids proxy
print("\n-- Basis (l_Fi - l_spot) ADF (should be stationary if cointegrated) --")
for i,nm in enumerate(names[1:],start=1):
    basis=LL[:,i]-LL[:,0]
    t=adf(basis,lags=2,trend="c")
    print(f"  {nm}-l_spot ADF={t:7.3f} [5% {ADF_CV[('c',0.05)]:.2f}] {'reject(stationary)' if t<ADF_CV[('c',0.05)] else 'fail-to-reject'}")

print("\n-- Volatility regime markers (|monthly log-return of spot|) --")
r=np.diff(LL[:,0])
dts=log.index[1:]
sr=pd.Series(r,index=dts)
big=sr[abs(sr)>0.15]
print("  months with |log-return|>15%:")
for d,val in big.items():
    print(f"    {d.date()}  {val:+.3f}")
print(f"  full-sample monthly return sd={sr.std():.4f}")
for lab,a,b in [("2007-2009",'2007-01','2009-12'),("2014-2016",'2014-01','2016-12'),("2020",'2020-01','2020-12')]:
    sub=sr[(sr.index>=a)&(sr.index<=b)]
    print(f"  sd {lab}: {sub.std():.4f}")
