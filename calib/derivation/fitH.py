import json, numpy as np, cv2
from scipy.optimize import least_squares
L=json.load(open('work/calib/lines_clean.json'))['A'][:7]; S=json.load(open('work/calib/cross_segs.json'))
S=[s for s in S if s['phi']<-30]                      # cross joints steeper than 30 deg; the -21/-25 pair lay on a tyre track
def samples(line,n=40):
    nn=np.array(line['n']); d=np.array([-nn[1],nn[0]]); p0=nn*line['r']
    s=np.linspace(-5000,5000,4000); P=p0+np.outer(s,d)
    ok=(P[:,0]>2000)&(P[:,0]<5800)&(P[:,1]>1500)&(P[:,1]<3350); P=P[ok]
    return P[np.linspace(0,len(P)-1,n).astype(int)]
rowpts=[(j,samples(L[j])) for j in range(7)]
def proj(Hm,P):                     # image (undistorted canvas) -> ground, via Hm = ground->image inverse
    Q=np.c_[P,np.ones(len(P))]@Hm.T; return Q[:,:2]/Q[:,2:3]
a,b=S[0],S[-1]
src=np.array([a['p0'],a['p1'],b['p0'],b['p1']],float)
dst=np.array([[0,0.6*a['j']],[0,0.6*(a['j']+1)],[1,0.6*b['j']],[1,0.6*(b['j']+1)]],float)
H0=cv2.getPerspectiveTransform(src.astype(np.float32),dst.astype(np.float32)).astype(float)   # image -> ground
def unpack(h): return np.append(h,1.0).reshape(3,3)
def resid(h):
    Hm=unpack(h); r=[]
    for j,P in rowpts: r.append(proj(Hm,P)[:,1]-0.6*j)
    for s in S:
        g=proj(Hm,np.array([s['p0'],s['p1']])); r.append([(g[0,0]-g[1,0])*3])
    ga=proj(Hm,np.array([a['p0']]))[0]; gb=proj(Hm,np.array([b['p0']]))[0]
    r.append([(ga[0]-0)*10,(gb[0]-1)*10])
    return np.concatenate(r)
h0=(H0/H0[2,2]).ravel()[:8]
res=least_squares(resid,h0,loss='soft_l1',f_scale=0.02)
Hm=unpack(res.x)
print('row-line residual (m): rms',[round(float(np.sqrt(np.mean((proj(Hm,P)[:,1]-0.6*j)**2))),4) for j,P in rowpts])
xs=[]
for s in S:
    g=proj(Hm,np.array([s['p0'],s['p1']])); xs.append((s['j'],float(g.mean(0)[0]),float(g[0,0]-g[1,0])))
xs.sort()
print('cross joints (band, x in arbitrary units, end mismatch):'); [print('  ',j,round(x,3),round(dx,3)) for j,x,dx in xs]
# along-row scale: the x values should sit on a lattice of spacing u (0.6 m), with parity by band
best=None
for u in np.linspace(0.05,2.0,4000):
    ph=np.array([x/u-0.5*(j%2) for j,x,_ in xs]); c=np.abs(np.mean(np.exp(2j*np.pi*ph)))
    if best is None or c>best[0]: best=(c,u)
c,u=best; print('lattice fit: coherence',round(c,3),'unit',round(u,4))
ph=np.array([x/u-0.5*(j%2) for j,x,_ in xs]); off=np.angle(np.mean(np.exp(2j*np.pi*ph)))/(2*np.pi)
dev=((ph-off+0.5)%1-0.5)*0.6
print('deviation of each cross joint from the 0.6 m lattice (m):',np.round(dev,3).tolist())
# also without parity (pattern check): coherence if rows are NOT offset by half
best2=max(((np.abs(np.mean(np.exp(2j*np.pi*np.array([x/u2 for j,x,_ in xs])))),u2) for u2 in np.linspace(0.05,2.0,4000)))
print('without the half-offset: coherence',round(best2[0],3),'unit',round(best2[1],4))
json.dump(dict(H_img_to_ground=Hm.tolist(),x_unit=u,cross=xs),open('work/calib/H_fit.json','w'))
