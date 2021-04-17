import time
import numpy as np
from functools import wraps
import cv2
import os.path

eps=np.finfo(np.double).eps


def my_guidedFilter_oneChannel(srcImg, guidedImg, rad=9, eps=0.01):
    srcImg = srcImg / 255.0
    guidedImg = guidedImg / 255.0
    img_shape = np.shape(srcImg)
    P_mean = cv2.boxFilter(srcImg, -1, (rad, rad), normalize=True)
    I_mean = cv2.boxFilter(guidedImg, -1, (rad, rad), normalize=True)
    I_square_mean = cv2.boxFilter(np.multiply(guidedImg, guidedImg), -1, (rad, rad), normalize=True)
    I_mul_P_mean = cv2.boxFilter(np.multiply(srcImg, guidedImg), -1, (rad, rad), normalize=True)
    var_I = I_square_mean - np.multiply(I_mean, I_mean)
    cov_I_P = I_mul_P_mean - np.multiply(I_mean, P_mean)
    a = cov_I_P / (var_I + eps)
    b = P_mean - np.multiply(a, I_mean)
    a_mean = cv2.boxFilter(a, -1, (rad, rad), normalize=True)
    b_mean = cv2.boxFilter(b, -1, (rad, rad), normalize=True)
    dstImg = np.multiply(a_mean, guidedImg) + b_mean
    return dstImg * 255.0

def my_guidedFilter_threeChannel(srcImg, guidedImg, rad=9, eps=0.01):
    img_shape = np.shape(srcImg)

    dstImg = np.zeros(img_shape, dtype=float)

    for ind in range(0, img_shape[2]):
        dstImg[:, :, ind] = my_guidedFilter_oneChannel(srcImg[:, :, ind],
                                                       guidedImg[:, :, ind], rad, eps)
    dstImg = dstImg.astype(np.uint8)

    return dstImg

def filter_1(img,rad=5,eps=0.01):
    dstimg = my_guidedFilter_threeChannel(img, img, rad, eps)
    return dstimg



def my_heq(img):
    '''process heq on 3-channel image just like single-channel, most of the time
       it's worse than func cv2_heq'''
    hist,_=np.histogram(img,256,[0,256])
    cdf=hist.cumsum()
    gaimi=cdf/cdf.max()
    return np.floor((gaimi*255)[img.ravel()]).astype('uint8').reshape(img.shape)

def cv2_heq(img,yuv=False):
    '''cv2's histogram equalization can only be implemented on single channel image,
       sometimes it doesn't work very well for 3-channel pic, you can contrast with
       func my_heq:
          >>> img=cv2.imread('./img/lena0.8.png')
          >>> contrast_plot(img,[cv2_heq(img),my_heq(img)],['cv2 heq','my heq'])
    '''
    if len(img.shape)==2:
        img=img[...,None]
    if yuv:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    ret=img.copy()
    for i in range(img.shape[-1]):
        ret[...,i]=cv2.equalizeHist(img[...,i])
        if yuv:
            break
    if yuv:
        return cv2.cvtColor(ret,cv2.COLOR_YCrCb2BGR)
    return ret.squeeze()

def get_gauss_kernel(sigma,dim=2):
    '''1D gaussian function: G(x)=1/(sqrt{2π}σ)exp{-(x-μ)²/2σ²}. Herein, μ:=0, after
       normalizing the 1D kernel, we can get 2D kernel version by
       matmul(1D_kernel',1D_kernel), having same sigma in both directions. Note that
       if you want to blur one image with a 2-D gaussian filter, you should separate
       it into two steps(i.e. separate the 2-D filter into two 1-D filter, one column
       filter, one row filter): 1) blur image with first column filter, 2) blur the
       result image of 1) with the second row filter. Analyse the time complexity: if
       m&n is the shape of image, p&q is the size of 2-D filter, bluring image with
       2-D filter takes O(mnpq), but two-step method takes O(pmn+qmn)'''
    ksize=int(np.floor(sigma*6)/2)*2+1 #kernel size("3-σ"法则) refer to
    #https://github.com/upcAutoLang/MSRCR-Restoration/blob/master/src/MSRCR.cpp
    k_1D=np.arange(ksize)-ksize//2
    k_1D=np.exp(-k_1D**2/(2*sigma**2))
    k_1D=k_1D/np.sum(k_1D)
    if dim==1:
        return k_1D
    elif dim==2:
        return k_1D[:,None].dot(k_1D.reshape(1,-1))

def gauss_blur_original(img,sigma):
    '''suitable for 1 or 3 channel image'''
    row_filter=get_gauss_kernel(sigma,1)
    t=cv2.filter2D(img,-1,row_filter[...,None])
    return cv2.filter2D(t,-1,row_filter.reshape(1,-1))

def gauss_blur_recursive(img,sigma):
    '''参考“高斯滤波器的递归实现”
    (doi: 10.1016 / 0165 - 1684 (95) 00020 - e)。纸认为它比
    FFT(快速傅里叶变换)实现的一个高斯滤波器。
    适用于1或3通道图像'''
    pass

def gauss_blur(img,sigma,method='original'):
    if method=='original':
        return gauss_blur_original(img,sigma)
    elif method=='recursive':
        return gauss_blur_recursive(img,sigma)

def simplest_color_balance(img_msrcr,s1,s2):
    '''see section 3.1 in “Simplest Color Balance”(doi: 10.5201/ipol.2011.llmps-scb).
    Only suitable for 1-channel image'''
    sort_img=np.sort(img_msrcr,None)
    N=img_msrcr.size
    Vmin=sort_img[int(N*s1)]
    Vmax=sort_img[int(N*(1-s2))-1]
    img_msrcr[img_msrcr<Vmin]=Vmin
    img_msrcr[img_msrcr>Vmax]=Vmax
    return (img_msrcr-Vmin)*255/(Vmax-Vmin)

def get_dir(flag=True):
    '''trace back this function's caller(flag=True) or this script's(flag=False) directory'''
    if flag:
        import traceback
        t=traceback.extract_stack()
        return os.path.dirname(os.path.realpath(t[0].filename))
    else:
        return os.path.dirname(os.path.realpath(__file__))
