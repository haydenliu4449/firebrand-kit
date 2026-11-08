import json, glob, numpy as np, cv2, sys
sys.path.insert(0,'work/calib')
from scipy.optimize import minimize_scalar
W,H=3840,2160; cx,cy=W/2,H/2; R=np.hypot(cx,cy)
Rr=np.load('work/calib/ridge.npy').astype(np.float32); Rs=cv2.GaussianBlur(Rr,(0,0),2.0)
def undist(p,k):
    p=np.asarray(p,float); d=(p-[cx,cy])/R; r2=(d**2).sum(1,keepdims=True)
    return np.array([cx,cy])+d/(1+k*r2)*R
def trace(x,y,a,sign,step=6,maxn=600,search=6,ymin=0):
    pts=[]; miss=0
    for _ in range(maxn):
        x+=sign*step*np.cos(a); y+=sign*step*np.sin(a)
        if not(5<x<W-5 and max(ymin,5)<y<H-5): break
        nx,ny=-np.sin(a),np.cos(a); offs=np.arange(-search,search+1)
        xs=np.clip((x+offs*nx).round().astype(int),0,W-1); ys=np.clip((y+offs*ny).round().astype(int),0,H-1)
        v=Rs[ys,xs]; k=int(np.argmax(v))
        if v[k]<2.5:
            miss+=1
            if miss>5: break
            continue
        miss=0; x,y=float(xs[k]),float(ys[k]); pts.append((x,y))
        if len(pts)>=10:
            d=np.array(pts[-1])-np.array(pts[-10]); na=np.arctan2(d[1],d[0])+(np.pi if sign<0 else 0)
            if abs(np.angle(np.exp(1j*(na-a))))<0.12: a=na
    return pts
# drain edges + wall-frame edges: seeds on a horizontal scan, steep down-right orientation
extra=[]
for ys in (1700,1900):
    row=Rs[ys,400:1300]; pk=[i for i in range(3,len(row)-3) if row[i]==row[i-3:i+4].max() and row[i]>6]
    for i in pk:
        x=400+i
        for a in np.deg2rad([55,58,61,64,67]):
            pts=trace(float(x),float(ys),a,-1,ymin=1300)[::-1]+[(float(x),float(ys))]+trace(float(x),float(ys),a,+1)
            if len(pts)>80: extra.append(np.array(pts)); break
print('steep edges traced:',len(extra),[ (int(p[0,0]),int(p[0,1]),int(p[-1,0]),int(p[-1,1])) for p in extra])
rows=[]
for f in sorted(glob.glob('work/calib/rows_*.json')):
    if 'merged' in f: continue
    for l in json.load(open(f)):
        p=np.array(l['pts']); n=len(p)
        if n<60: continue
        p=p[int(n*.1):int(n*.9)]
        if np.ptp(p[:,0])+np.ptp(p[:,1])>=900: rows.append(p)
def robust_rms(p):
    m=np.ones(len(p),bool)
    for _ in range(3):
        q=p[m]; c=q.mean(0); u,s,vt=np.linalg.svd(q-c,full_matrices=False); nrm=vt[-1]
        r=np.abs((p-c)@nrm); thr=max(3*np.median(r[m]),1.0); m=r<thr
    return float(np.sqrt((r[m]**2).mean())), m.mean()
def total(k,ls):
    vals=[robust_rms(undist(p,k))[0] for p in ls]; return float(np.mean(np.square(vals)))
for name,ls in (('row joints',rows),('steep edges',extra),('both',rows+extra)):
    if not ls: continue
    res=minimize_scalar(lambda k:total(k,ls),bounds=(-0.6,0.6),method='bounded')
    print(f'{name:12s} lines {len(ls):2d}  best k {res.x:+.3f}  rms {np.sqrt(res.fun):.2f} px   at k=0: {np.sqrt(total(0,ls)):.2f}   curve:',' '.join(f'{k:+.1f}:{np.sqrt(total(k,ls)):.2f}' for k in (-0.4,-0.3,-0.2,-0.1,0.0)))
json.dump([p.tolist() for p in extra],open('work/calib/steep_edges.json','w'))
