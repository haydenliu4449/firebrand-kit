import json, numpy as np, cv2
ru=np.load('work/calib/ridge_undist.npy').astype(np.float32); ru=cv2.GaussianBlur(ru,(0,0),1.2)
L=json.load(open('work/calib/lines_clean.json'))['A'][:7]          # row lines 0..6 (7 and 8 dropped)
def pt_on(line,s):           # point on line at parameter s (along its direction)
    n=np.array(line['n']); d=np.array([-n[1],n[0]]); return n*line['r']+np.outer(s,d), d
def inter(l1,p,u):           # intersect line l1 with ray p + t*u
    n=np.array(l1['n']); t=(l1['r']-p@n)/(u@n); return p+t*u
segs=[]
for j in range(6):
    a,b=L[j],L[j+1]
    s=np.arange(-5000,5000,3.0); P,d=pt_on(a,s)
    ok=(P[:,0]>1900)&(P[:,0]<5800)&(P[:,1]>1500)&(P[:,1]<3350); s,P=s[ok],P[ok]
    best=np.zeros(len(P)); bphi=np.zeros(len(P)); bq=np.zeros((len(P),2))
    for phi in np.deg2rad(np.arange(-80,-20,1.0)):        # cross joints run down-left (image) => up-right
        u=np.array([np.cos(phi),np.sin(phi)])
        nb=np.array(b['n']); t=(b['r']-P@nb)/(u@nb)       # reach line b along u
        good=(t>40)&(t<700)
        ts=np.linspace(0.1,0.9,25)
        Q=P[:,None,:]+t[:,None,None]*ts[None,:,None]*u[None,None,:]
        xi=np.clip(Q[...,0].astype(int),0,ru.shape[1]-1); yi=np.clip(Q[...,1].astype(int),0,ru.shape[0]-1)
        v=ru[yi,xi]; score=np.where(good,np.minimum(v,15).mean(1)-0.5*v.std(1),0)
        m=score>best; best[m]=score[m]; bphi[m]=phi; bq[m]=P[m]+t[m,None]*u
    # peaks along s
    pk=[i for i in range(5,len(best)-5) if best[i]==best[i-5:i+6].max() and best[i]>4.0]
    for i in pk: segs.append(dict(j=j,s=float(s[i]),score=float(best[i]),phi=float(np.rad2deg(bphi[i])),p0=P[i].tolist(),p1=bq[i].tolist()))
    print(f'band {j}: {len(pk)} cross joints, phi',[round(np.rad2deg(bphi[i])) for i in pk])
json.dump(segs,open('work/calib/cross_segs.json','w'))
v=cv2.cvtColor(np.clip(ru*12,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
for x in L:
    n=np.array(x['n']); d=np.array([-n[1],n[0]]); p0=n*x['r']; cv2.line(v,tuple((p0-7000*d).astype(int)),tuple((p0+7000*d).astype(int)),(0,255,0),2)
for sg in segs: cv2.line(v,tuple(map(int,sg['p0'])),tuple(map(int,sg['p1'])),(255,0,255),5)
cv2.imwrite('work/calib/cross.jpg',cv2.resize(v[1300:3456,1700:6144],None,fx=0.36,fy=0.36,interpolation=cv2.INTER_AREA))
