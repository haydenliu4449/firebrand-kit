import cv2,numpy as np,json
R=np.load('work/calib/ridge.npy').astype(np.float32); H,W=R.shape
Rs=cv2.GaussianBlur(R,(0,0),2.0)
# local orientation from the structure tensor of the ridge map
gx=cv2.Sobel(Rs,cv2.CV_32F,1,0,3); gy=cv2.Sobel(Rs,cv2.CV_32F,0,1,3)
Jxx=cv2.GaussianBlur(gx*gx,(0,0),6); Jyy=cv2.GaussianBlur(gy*gy,(0,0),6); Jxy=cv2.GaussianBlur(gx*gy,(0,0),6)
ori=0.5*np.arctan2(2*Jxy,Jxx-Jyy)+np.pi/2      # direction ALONG the ridge
def trace(x,y,ang,sign,step=6,maxn=700,search=7):
    pts=[]; a=ang
    for _ in range(maxn):
        x+=sign*step*np.cos(a); y+=sign*step*np.sin(a)
        if not(5<x<W-5 and 1050<y<H-5): break
        nx,ny=-np.sin(a),np.cos(a)
        offs=np.arange(-search,search+1)
        xs=np.clip((x+offs*nx).round().astype(int),0,W-1); ys=np.clip((y+offs*ny).round().astype(int),0,H-1)
        v=Rs[ys,xs]; k=int(np.argmax(v))
        if v[k]<2.5: 
            # allow short gaps
            pts.append(None); 
            if len(pts)>=6 and all(p is None for p in pts[-6:]): break
            continue
        x,y=float(xs[k]),float(ys[k]); pts.append((x,y))
        if len(pts)>=3:
            q=[p for p in pts[-8:] if p is not None]
            if len(q)>=3:
                d=np.array(q[-1])-np.array(q[0]); na=np.arctan2(d[1],d[0])
                if sign<0: na+=np.pi
                if abs(np.angle(np.exp(1j*(na-a))))<0.15: a=na
    return [p for p in pts if p is not None]
X0=int(__import__('sys').argv[1]) if len(__import__('sys').argv)>1 else 2000
col=Rs[1100:2150,X0]; 
peaks=[i for i in range(2,len(col)-2) if col[i]==col[i-2:i+3].max() and col[i]>4]
lines=[]
for p in peaks:
    y=1100+p; a=ori[y,X0]; a=(a+np.pi/2)%np.pi-np.pi/2
    deg=np.rad2deg(a)
    if not(10<deg<45): continue          # row joints run down-right
    left=trace(float(X0),float(y),a,-1); right=trace(float(X0),float(y),a,+1)
    pts=left[::-1]+[(float(X0),float(y))]+right
    if len(pts)>40: lines.append(dict(seed_y=y,deg=float(deg),pts=pts))
print(len(peaks),'peaks;',len(lines),'row-joint lines traced; seeds y:',[l['seed_y'] for l in lines])
json.dump(lines,open(f'work/calib/rows_{X0}.json','w'))
v=cv2.cvtColor(np.clip(R*10,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
cols=[(0,255,0),(0,200,255),(255,0,255),(255,255,0),(0,0,255),(255,128,0)]
for i,l in enumerate(lines):
    p=np.array(l['pts'],np.int32); cv2.polylines(v,[p],False,cols[i%6],3); cv2.putText(v,str(i),tuple(p[len(p)//2]),0,1.5,cols[i%6],3)
cv2.imwrite(f'work/calib/rows_{X0}.jpg',cv2.resize(v[1000:2160,:],None,fx=0.42,fy=0.42,interpolation=cv2.INTER_AREA))
