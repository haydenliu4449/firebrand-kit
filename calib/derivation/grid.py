import sys, json, numpy as np, cv2
sys.path.insert(0,'work/calib'); import undist_lib as U
ru=np.load('work/calib/ridge_undist.npy').astype(np.float32)
# driveway region in the UNDISTORTED canvas (generous polygon, below the kerb, right of the drain)
poly=np.array([[1750,1500],[5900,1350],[5900,3400],[2600,3400],[1900,2400]],np.int32)
m=np.zeros(ru.shape,np.uint8); cv2.fillPoly(m,[poly],1)
bw=((ru>5)&(m>0)).astype(np.uint8)*255
bw=cv2.ximgproc.thinning(bw) if hasattr(cv2,'ximgproc') else bw
def hough(theta_lo,theta_hi,thr):
    L=cv2.HoughLines(bw,1,np.pi/720,thr,min_theta=np.deg2rad(theta_lo),max_theta=np.deg2rad(theta_hi))
    return [] if L is None else [tuple(l[0]) for l in L]
def merge(lines,drho=25,dth=np.deg2rad(2.0)):
    out=[]
    for r,t in lines:                       # HoughLines returns strongest first
        if all(abs(r-r2)>drho or abs(t-t2)>dth for r2,t2 in out): out.append((r,t))
    return out
# normal angle theta: row joints run down-right at ~25-35 deg -> normal ~115-125 deg; cross joints run down-left ~ -45..-60 -> normal ~30-50
A=merge(hough(100,135,700)); B=merge(hough(25,65,450))
print('row lines',len(A),'cross lines',len(B))
print('A theta',[round(np.rad2deg(t),1) for r,t in A]); print('B theta',[round(np.rad2deg(t),1) for r,t in B])
json.dump(dict(A=[list(map(float,x)) for x in A],B=[list(map(float,x)) for x in B]),open('work/calib/hough.json','w'))
v=cv2.cvtColor(np.clip(ru*12,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
def draw(r,t,col):
    a,b=np.cos(t),np.sin(t); x0,y0=a*r,b*r; p1=(int(x0+8000*(-b)),int(y0+8000*a)); p2=(int(x0-8000*(-b)),int(y0-8000*a)); cv2.line(v,p1,p2,col,3)
for r,t in A: draw(r,t,(0,255,0))
for r,t in B: draw(r,t,(255,0,255))
cv2.polylines(v,[poly],True,(0,200,255),4)
cv2.imwrite('work/calib/grid_lines.jpg',cv2.resize(v[1100:3456,1500:6144],None,fx=0.34,fy=0.34,interpolation=cv2.INTER_AREA))
