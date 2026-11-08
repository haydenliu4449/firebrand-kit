import sys, json, numpy as np, cv2
sys.path.insert(0,'work/calib')
ru=np.load('work/calib/ridge_undist.npy').astype(np.float32)
poly=np.array([[1750,1500],[5900,1350],[5900,3400],[2600,3400],[1900,2400]],np.int32)
roi=np.zeros(ru.shape,np.uint8); cv2.fillPoly(roi,[poly],1)
# exclude the burned-in label (undistorted) and keep away from the tyre-track band later via support
H=json.load(open('work/calib/hough.json'))
ys,xs=np.nonzero((ru>4)&(roi>0)); wv=ru[ys,xs]
def refine(r,t,band=7):
    n=np.array([np.cos(t),np.sin(t)])
    for _ in range(3):
        d=xs*n[0]+ys*n[1]-r; s=np.abs(d)<band
        if s.sum()<50: return None
        P=np.c_[xs[s],ys[s]].astype(float); w=wv[s]
        c=(P*w[:,None]).sum(0)/w.sum(); C=((P-c).T*w)@(P-c)
        ev,evec=np.linalg.eigh(C); n=evec[:,0]; r=float(n@c)
        if n[1]<0: n=-n; r=-r
    # support: fraction of samples along the line (inside roi) with ridge>4 within 3 px
    t_=np.arctan2(n[1],n[0]); dvec=np.array([-n[1],n[0]]); p0=n*r
    ss=np.arange(-6000,6000,4); pts=p0+np.outer(ss,dvec)
    ok=(pts[:,0]>=0)&(pts[:,0]<ru.shape[1]-1)&(pts[:,1]>=0)&(pts[:,1]<ru.shape[0]-1)
    pts=pts[ok]; inr=roi[pts[:,1].astype(int),pts[:,0].astype(int)]>0; pts=pts[inr]
    if len(pts)<100: return None
    best=np.zeros(len(pts))
    for o in (-3,-2,-1,0,1,2,3):
        q=pts+o*n; best=np.maximum(best,ru[q[:,1].astype(int),q[:,0].astype(int)])
    return dict(n=n.tolist(),r=r,support=float((best>4).mean()),len=len(pts)*4)
def clean(lines,min_support):
    R=[x for x in (refine(r,t) for r,t in lines) if x and x['support']>=min_support]
    R.sort(key=lambda x:-x['support']); out=[]
    cpt=np.array([3800,2400])
    for x in R:
        n=np.array(x['n']); ang=np.arctan2(n[1],n[0])
        dup=False
        for y in out:
            m=np.array(y['n']); 
            if abs(np.angle(np.exp(1j*(ang-np.arctan2(m[1],m[0])))))<np.deg2rad(1.5) and abs((n@cpt-x['r'])-(m@cpt-y['r']))<30: dup=True; break
        if not dup: out.append(x)
    return out
A=clean([tuple(a) for a in H['A']],0.45); B=clean([tuple(b) for b in H['B']],0.18)
# order by signed offset at the ROI centre along a common normal
def order(L):
    n0=np.array(L[0]['n']); cpt=np.array([3800,2400])
    for x in L: x['off']=float(np.array(x['n'])@cpt-x['r']) if np.array(x['n'])@n0>0 else float(-(np.array(x['n'])@cpt-x['r']))
    L.sort(key=lambda x:x['off']); return L
A=order(A); B=order(B)
for name,L in (('ROW',A),('CROSS',B)):
    print(name,len(L))
    prev=None
    for i,x in enumerate(L):
        g='' if prev is None else f"gap {x['off']-prev:7.1f}"
        print(f"  {i:2d} off {x['off']:8.1f} angle {np.rad2deg(np.arctan2(x['n'][1],x['n'][0])):6.1f} support {x['support']:.2f}  {g}")
        prev=x['off']
json.dump(dict(A=A,B=B),open('work/calib/lines_clean.json','w'))
v=cv2.cvtColor(np.clip(ru*12,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
for col,L in (((0,255,0),A),((255,0,255),B)):
    for i,x in enumerate(L):
        n=np.array(x['n']); d=np.array([-n[1],n[0]]); p0=n*x['r']
        p1=tuple((p0-7000*d).astype(int)); p2=tuple((p0+7000*d).astype(int)); cv2.line(v,p1,p2,col,3)
        c=p0+d*((np.array([3800,2400])-p0)@d); cv2.putText(v,str(i),tuple(c.astype(int)),0,2.2,col,5)
cv2.polylines(v,[poly],True,(0,200,255),4)
cv2.imwrite('work/calib/grid_clean.jpg',cv2.resize(v[1100:3456,1500:6144],None,fx=0.34,fy=0.34,interpolation=cv2.INTER_AREA))
