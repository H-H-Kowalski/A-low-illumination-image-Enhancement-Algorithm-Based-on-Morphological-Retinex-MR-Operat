import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from sympy import symbols, sin, cos, exp, N

def causal_parameters(sigma):
    a1, a2, b1, b2, y1, y2, w1, w2, s = symbols('a1 a2 b1 b2 y1 y2 w1 w2 s')
    ap0, ap1, ap2, ap3 = symbols('ap0 ap1 ap2 ap3')
    bp1, bp2, bp3, bp4 = symbols('bp1 bp2 bp3 bp4')
    ap0 = a1 + a2
    ap1 = exp(-y2/s)*(b2*sin(w2/s) - (a2 + 2*a1)*cos(w2/s)) + exp(-y1/s)*(b1*sin(w1/s)- (2*a2+a1)*cos(w1/s))
    ap2 = 2*exp(-(y1+y2)/s)*((a1+a2)*cos(w2/s)*cos(w1/s) - cos(w2/s)*b1*sin(w1/s) - cos(w1/s)*b2*sin(w2/s)) + a2*exp(-2*y1/s) + a1*exp(-2*y2/s)
    ap3 = exp(-(y2+2*y1)/s)*(b2*sin(w2/s) - a2*cos(w2/s)) + exp(-(y1 + 2*y2)/s)*(b1*sin(w1/s) - a1*cos(w1/s))
    bp4 = exp(-(2*y1 + 2*y2)/s)
    bp3 = - 2*cos(w1/s)*exp(-(y1+2*y2)/s) - 2*cos(w2/s)*exp(-(y2+2*y1)/s)
    bp2 = 4*cos(w2/s)*cos(w1/s)*exp(-(y1+y2)/s) + exp(-2*y1/s) + exp(-2*y2/s)
    bp1 = - 2*exp(-y2/s)*cos(w2/s) - 2*exp(-y1/s)*cos(w1/s)
    params = {a1:1.6800, a2:-0.6803, b1:3.7350, b2:-0.2598, y1:1.7830, y2:1.7230, w1:0.6318, w2:1.9970, s:sigma}
    A = [N(ap0.subs(params)), N(ap1.subs(params)), N(ap2.subs(params)), N(ap3.subs(params))]
    B = [N(bp1.subs(params)), N(bp2.subs(params)), N(bp3.subs(params)), N(bp4.subs(params))]
    return A, B


def anti_causal_parameters(a, b):
    ap0, ap1, ap2, ap3 = symbols('ap0 ap1 ap2 ap3')
    bp1, bp2, bp3, bp4 = symbols('bp1 bp2 bp3 bp4')
    an1, an2, an3, an4 = symbols('an1 an2 an3 an4')
    an1 = ap1 - bp1*ap0
    an2 = ap2 - bp2*ap0
    an3 = ap3 - bp3*ap0
    an4 = - bp4*ap0
    params = {ap0:a[0], ap1:a[1], ap2:a[2], ap3:a[3], bp1:b[0], bp2:b[1], bp3:b[2], bp4:b[3]}
    A = [N(an1.subs(params)), N(an2.subs(params)), N(an3.subs(params)), N(an4.subs(params))]
    return A, b

def causal_sum(x, y, row, n, ap, bp):
    xSum = 0.0
    ySum = 0.0
    for i in range(4):
        xSum += ap[i]*x[row, n-i]
        ySum += bp[i]*y[row, n-(i+1)]
    return xSum - ySum

def anti_causal_sum(x, y, row, n, an, bn):
    xSum = 0.0
    ySum = 0.0
    for i in range(4):
        xSum += an[i]*x[row, n+i+1]
        ySum += bn[i]*y[row, n+i+1]
    return xSum - ySum


def filter1D(X, ap, bp, an, bn):
    M,N = X.shape[:2]
    yp = np.zeros_like(X)
    ym = np.zeros_like(X)
    final = X.copy()
    multiplier = 1./(sigma * np.sqrt(2*np.pi))
    for m in range(M):
        for n in range(4,N-4,+1):
            yp[m,n] = causal_sum(X, yp, m, n, ap, bp)
        for n in range(N-5,5,-1):
            ym[m,n] = anti_causal_sum(X, ym, m, n, an, bn)
    return multiplier*(yp+ym)

def gaussianFilter(I, sigma):
    ap, bp = causal_parameters(sigma)
    an, bn = anti_causal_parameters(ap, bp)
    X = np.pad(I, [4,4], 'constant', constant_values=[0,0])
    filtered = filter1D(X, ap, bp, an, bn)
    final = filter1D(filtered.T, ap, bp, an, bn)
    return final.T

sigma = 2       #change sigma
I1 = cv2.imread("D:\pyt\\test6.jpg").astype('float')
filtered_image = gaussianFilter(I1, sigma)
blur = gaussian_filter(I1,sigma)
cv2.imshow("1",I1)
cv2.imshow("2",blur)
cv2.imshow("3",filtered_image)

cv2.waitKey()