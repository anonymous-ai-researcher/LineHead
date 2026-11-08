import numpy as np
from scipy import stats, integrate, optimize
eps,delta=0.25,0.05
a=lambda e:0.5*(e-np.log1p(e)); b=lambda e:0.5*(-e-np.log1p(-e))
def c(e):
    s=np.sqrt(1+4*e*e); return 0.5*(s-1-np.log((1+s)/2))
g=np.linspace(1e-4,1-1e-4,10000)
print("min c-a on grid",(c(g)-a(g)).min(), "min a-e^2/12",(a(g)-g**2/12).min(), "1/a(1/4)",1/a(eps))
def least(f):
    d=1
    while f(d)>delta: d+=1
    return d
def ip_tail(d,e):
    # P(<ea,eb> >= e), ea,eb ~ N(0,I/d): condition on ||eb||^2 = R/d, R~chi2_d; <ea,eb> ~ N(0, R/d^2)
    f=lambda R: stats.chi2.pdf(R,d)*stats.norm.sf(e*d/np.sqrt(R))
    return integrate.quad(f,0,np.inf,limit=200)[0]
for m in (64,150,600):
    lm=least(lambda d:2*m*m*np.exp(-d*eps**2/16))
    f1=least(lambda d:m*np.exp(-d*a(eps))+m*np.exp(-d*b(eps))+m*(m-1)*np.exp(-d*c(eps)))
    f2=least(lambda d:2*m*m*np.exp(-d*a(eps)))
    # exact tails union bound: search near
    def ex(d): return m*(stats.chi2.sf(d*(1+eps),d)+stats.chi2.cdf(d*(1-eps),d))+m*(m-1)*ip_tail(d,eps)
    lo=200
    while ex(lo)>delta: lo+=1
    print(m,"LM",lm,"chernoff-first",f1,"2m^2e^{-da}",f2,"12/eps^2 form",int(np.ceil(12/eps**2*np.log(2*m*m/delta))),"exact",lo)
# 32 dims
d=32
p=2*ip_tail(d,eps)
print("32-dim P(|ip|>=1/4)",p,"expected pairs of 2016",p*2016)
rng=np.random.default_rng(0); held=0; viol=[]
for _ in range(200):
    E=rng.normal(0,1/np.sqrt(d),(64,d)); G=E@E.T
    off=np.abs(G[np.triu_indices(64,1)]); nrm=np.abs(np.diag(G)-1)
    viol.append((off>=eps).sum()); held+= (off<eps).all() and (nrm<eps).all()
print("mean violating pairs",np.mean(viol),"E held",held,"/200")
tau2=2*np.log(4*600/delta); print("4tau^2/eps",4*tau2/eps)
