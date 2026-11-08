import numpy as np, cv2
W,H=3840,2160; cx,cy=W/2,H/2; R=np.hypot(cx,cy); K=-0.446
OW,OH=int(W*1.6),int(H*1.6); ocx,ocy=OW/2,OH/2
def to_undist(p,k=K):
    """distorted 4K pixel -> undistorted canvas pixel (OWxOH, same scale, centred)"""
    p=np.atleast_2d(np.asarray(p,float)); d=(p-[cx,cy])/R; r2=(d**2).sum(1,keepdims=True)
    u=d/(1+k*r2); return np.c_[ocx+u[:,0]*R, ocy+u[:,1]*R]
def maps(k=K):
    yu,xu=np.mgrid[0:OH,0:OW].astype(np.float32); ux,uy=(xu-ocx)/R,(yu-ocy)/R; ru=np.hypot(ux,uy)
    disc=1-4*k*ru**2; d=np.where(ru>1e-9,(1-np.sqrt(np.maximum(disc,0)))/(2*k*np.maximum(ru,1e-9)),0)
    s=np.where(ru>1e-9,d/np.maximum(ru,1e-9),1)
    return (cx+ux*s*R).astype(np.float32),(cy+uy*s*R).astype(np.float32)
