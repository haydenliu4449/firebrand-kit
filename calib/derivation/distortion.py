"""Lens distortion (one-parameter division model, centre = image centre) by plumb lines:
the right k makes the scene's straight edges longest. Writes work/calib/k_scan.csv."""
import cv2, numpy as np, json
img=cv2.imread('work/calib/scene_clip9_median.png',0); H,W=img.shape
cx,cy=W/2,H/2; R=np.hypot(cx,cy)
def remap_for(k):
    # output (undistorted) pixel u -> source (distorted) pixel d. Division model:
    # u = c + (d-c)/(1+k*rd^2)  (rd normalised by R). Invert per pixel by fixed point.
    yu,xu=np.mgrid[0:H,0:W].astype(np.float32)
    ux,uy=(xu-cx)/R,(yu-cy)/R; ru=np.hypot(ux,uy)
    rd=ru.copy()
    for _ in range(20):
        rd=ru*(1+k*rd**2)
    s=np.where(ru>0,rd/np.maximum(ru,1e-9),1)
    return (cx+ux*s*R).astype(np.float32),(cy+uy*s*R).astype(np.float32)
lsd=cv2.createLineSegmentDetector(0)
small=lambda a:cv2.resize(a,(W//2,H//2),interpolation=cv2.INTER_AREA)
out=[]
for k in np.round(np.arange(-0.40,0.05,0.025),3):
    mx,my=remap_for(k); u=cv2.remap(img,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    seg=lsd.detect(small(cv2.GaussianBlur(u,(0,0),1.2)))[0]
    if seg is None: out.append((k,0,0)); continue
    seg=seg.reshape(-1,4); L=np.hypot(seg[:,2]-seg[:,0],seg[:,3]-seg[:,1])
    out.append((float(k),float(L[L>80].sum()),int((L>150).sum())))
    print(f'k={k:+.3f}  length in segments >80px {L[L>80].sum():8.0f}  segments >150px {(L>150).sum()}',flush=True)
best=max(out,key=lambda r:r[1]); print('best',best)
json.dump(dict(scan=out,best_k=best[0]),open('work/calib/k_scan.json','w'))
