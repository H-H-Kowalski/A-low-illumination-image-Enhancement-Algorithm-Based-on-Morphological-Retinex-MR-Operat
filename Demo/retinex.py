# encoding:utf-8
'''

References:
[1] D. J. Jobson, Z. Rahman and G. A. Woodell, "A multiscale retinex for bridging the 
    gap between color images and the human observation of scenes," in IEEE 
    Transactions on Image Processing, vol. 6, no. 7, pp. 965-976, July 1997.
[2] Frankle,Jonathan and McCann, John, “Method and Apparatus for Lightness Imaging”, 
    USPatent #4,384,336, May 17, 1983.
[3] Funt, Brian & Ciurea, Florian & McCann, John. (2004). Retinex in Matlab. Journal 
    of Electronic Imaging - J ELECTRON IMAGING. 13. 48-57. 10.1117/1.1636761.
[4] Ana Belén Petro, Catalina Sbert, and Jean-Michel Morel, Multiscale Retinex, Image 
    Processing On Line, (2014), pp. 71–88. https://doi.org/10.5201/ipol.2014.107

Provided by muggledy on 2020-3-18
'''

import numpy as np
from .tools import eps,gauss_blur,simplest_color_balance

### Frankle-McCann Retinex[2,3]
def retinex_FM(img,iter=4):
    '''log(OP(x,y))=1/2{log(OP(x,y))+[log(OP(xs,ys))+log(R(x,y))-log(R(xs,ys))]*}, see
       matlab code in https://www.cs.sfu.ca/~colour/publications/IST-2000/'''
    if len(img.shape)==2:
        img=img[...,None]
    ret=np.zeros(img.shape,dtype='uint8')
    def update_OP(x,y):
        nonlocal OP
        IP=OP.copy()
        if x>0 and y==0:
            IP[:-x,:]=OP[x:,:]+R[:-x,:]-R[x:,:]
        if x==0 and y>0:
            IP[:,y:]=OP[:,:-y]+R[:,y:]-R[:,:-y]
        if x<0 and y==0:
            IP[-x:,:]=OP[:x,:]+R[-x:,:]-R[:x,:]
        if x==0 and y<0:
            IP[:,:y]=OP[:,-y:]+R[:,:y]-R[:,-y:]
        IP[IP>maximum]=maximum
        OP=(OP+IP)/2
    for i in range(img.shape[-1]):
        R=np.log(img[...,i].astype('double')+1)
        maximum=np.max(R)
        OP=maximum*np.ones(R.shape)
        S=2**(int(np.log2(np.min(R.shape))-1))
        while abs(S)>=1: #iterations is slow
            for k in range(iter):
                update_OP(S,0)
                update_OP(0,S)
            S=int(-S/2)
        OP=np.exp(OP)
        mmin=np.min(OP)
        mmax=np.max(OP)
        ret[...,i]=(OP-mmin)/(mmax-mmin)*255
    return ret.squeeze()

### Multi-Scale Retinex[4]
def MultiScaleRetinex(img,sigmas=[15,80,250],weights=None,flag=True):
    if weights==None:
        weights=np.ones(len(sigmas))/len(sigmas)
    elif not abs(sum(weights)-1)<0.00001:
        raise ValueError('sum of weights must be 1!')
    r=np.zeros(img.shape,dtype='double')
    img=img.astype('double')
    #print(sigmas)
    for i,sigma in enumerate(sigmas): #遍历
        r+=(np.log(img+1)-np.log(gauss_blur(img,sigma)+1))*weights[i]
    if flag:
        mmin=np.min(r,axis=(0,1),keepdims=True)
        mmax=np.max(r,axis=(0,1),keepdims=True)
        r=(r-mmin)/(mmax-mmin)*255  # maybe indispensable when used in MSRCR or Gimp, make pic vibrant
        r=r.astype('uint8')
    return r

def retinex_MSRCR(img,sigmas=[12,80,250],s1=0.01,s2=0.01):
    alpha=125
    img=img.astype('double')+1 #
    csum_log=np.log(np.sum(img,axis=2))
    msr=MultiScaleRetinex(img-1,sigmas) #-1
    r=(np.log(alpha*img)-csum_log[...,None])*msr
    for i in range(r.shape[-1]):
        r[...,i]=simplest_color_balance(r[...,i],0.01,0.01)
    return r.astype('uint8')


def retinex_gimp(img,sigmas=[12,80,250],dynamic=2):
    '''指的是GIMP中的实现，它改进了基于
       在MSRCR上，引入均值和标准差以及动态参数
       消除色差，实验表明它效果很好。看到
       源代码在 https://github.com/piksels-and-lines-orchestra/gimp/blob/master/plug-ins/common/contrast-retinex.c'''
    alpha=128
    gain=1
    offset=0
    img=img.astype('double')+1 #
    csum_log=np.log(np.sum(img,axis=2))
    msr=MultiScaleRetinex(img-1,sigmas) #-1
    r=gain*(np.log(alpha*img)-csum_log[...,None])*msr+offset
    mean=np.mean(r,axis=(0,1),keepdims=True)
    var=np.sqrt(np.sum((r-mean)**2,axis=(0,1),keepdims=True)/r[...,0].size)
    mmin=mean-dynamic*var
    mmax=mean+dynamic*var
    stretch=(r-mmin)/(mmax-mmin)*255
    stretch[stretch>255]=255
    stretch[stretch<0]=0
    return stretch.astype('uint8')

### Multi-Scale Retinex with Chromaticity Preservation, see[4] Algorithm 2 in section 4
def retinex_MSRCP(img,sigmas=[12,80,250],s1=0.01,s2=0.01):
    '''与他人相比，简单又快速'''
    Int=np.sum(img,axis=2)/3
    Diffs=[]
    for sigma in sigmas:
        Diffs.append(np.log(Int+1)-np.log(gauss_blur(Int,sigma)+1))
    MSR=sum(Diffs)/3
    Int1=simplest_color_balance(MSR,s1,s2)
    B=np.max(img,axis=2)
    A=np.min(np.stack((255/(B+eps),Int1/(Int+eps)),axis=2),axis=-1)
    return (A[...,None]*img).astype('uint8')

def retinex_AMSR(img,sigmas=[12,80,250]):
    img=img.astype('double')+1 #
    msr=MultiScaleRetinex(img-1,sigmas,flag=False) #
    y=0.05  
    for i in range(msr.shape[-1]):
        v,c=np.unique((msr[...,i]*100).astype('int'),return_counts=True)
        sort_v_index=np.argsort(v)
        sort_v,sort_c=v[sort_v_index],c[sort_v_index] #plot hist
        zero_ind=np.where(sort_v==0)[0][0]
        zero_c=sort_c[zero_ind]
        #
        _=np.where(sort_c[:zero_ind]<=zero_c*y)[0]
        if len(_)==0:
            low_ind=0
        else:
            low_ind=_[-1]
        _=np.where(sort_c[zero_ind+1:]<=zero_c*y)[0]
        if len(_)==0:
            up_ind=len(sort_c)-1
        else:
            up_ind=_[0]+zero_ind+1
        #
        low_v,up_v=sort_v[[low_ind,up_ind]]/100 #low clip value and up clip value
        msr[...,i]=np.maximum(np.minimum(msr[:,:,i],up_v),low_v)
        mmin=np.min(msr[...,i])
        mmax=np.max(msr[...,i])
        msr[...,i]=(msr[...,i]-mmin)/(mmax-mmin)*255
    r=(np.log(125*img)-np.log(np.sum(img,axis=2))[...,None])*msr
    mmin,mmax=np.min(r),np.max(r)
    return ((r-mmin)/(mmax-mmin)*255).astype('uint8')
    # return msr.astype(np.uint8)



'''颜色恢复的步骤，也许没事

'''
