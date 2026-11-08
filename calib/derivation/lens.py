"""Plumb-line lens fit: one-parameter division model, centre = image centre.
Distorted d -> undistorted u = c + (d - c) / (1 + k * |d - c|^2 / R^2). Each traced
straight line must be straight after undistortion; k minimises the summed residual."""
import json, glob, numpy as np, cv2
from scipy.optimize import minimize_scalar
W,H=3840,2160; cx,cy=W/2,H/2; R=np.hypot(cx,cy)
def undist(p,k):
    p=np.asarray(p,float); d=(p-[cx,cy])/R; r2=(d**2).sum(1,keepdims=True)
    return np.array([cx,cy])+d/(1+k*r2)*R
def line_rms(p):
    p=p-p.mean(0); s=np.linalg.svd(p,full_matrices=False)[1]; return s[-1]/np.sqrt(len(p))
lines=[]
for f in sorted(glob.glob('work/calib/rows_*.json')):
    if 'merged' in f: continue
    for l in json.load(open(f)):
        p=np.array(l['pts'])
        if len(p)<60: continue
        n=len(p); p=p[int(n*.1):int(n*.9)]
        if np.ptp(p[:,0])+np.ptp(p[:,1])<900: continue
        lines.append(p)
# trench drain edges: trace the strong ridges in the lower-left (seeds by scan along y=1800)
Rr=np.load('work/calib/ridge.npy').astype(np.float32)
import importlib.util
print('row-joint lines used:',len(lines))
def total(k,ls):
    return sum(line_rms(undist(p,k))**2*len(p) for p in ls)/sum(len(p) for p in ls)
for name,ls in (('row joints',lines),):
    res=minimize_scalar(lambda k:total(k,ls),bounds=(-0.6,0.6),method='bounded')
    print(name,'best k',round(res.x,4),'rms px',round(np.sqrt(res.fun),2),' at k=0:',round(np.sqrt(total(0,ls)),2))
    for k in (-0.3,-0.2,-0.1,0,0.1):
        print('   k',k,'rms',round(np.sqrt(total(k,ls)),2))
json.dump(dict(k_rows=float(res.x)),open('work/calib/lens_rows.json','w'))
