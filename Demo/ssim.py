import numpy
import numpy as np
import math
import cv2
from skimage import metrics
# 计算图片评分
def SMD(img):
    ''' :param img:narray 二维灰度图像 :return: float 图像约清晰越大 '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(1, shape[1]):
            out+=math.fabs(int(img[x,y])-int(img[x,y-1]))
            out+=math.fabs(int(img[x,y]-int(img[x+1,y])))
    return out

def ssim(imageA, imageB):
    imageA = np.array(imageA, dtype=np.uint8)
    imageB = np.array(imageB, dtype=np.uint8)
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    grayScore = metrics.structural_similarity(grayA, grayB)
    print("{}".format(grayScore))

def psnr(img1, img2):  # 这里输入的是（0,255）的灰度或彩色图像，如果是彩色图像，则numpy.mean相当于对三个通道计算的结果再求均值
    mse = numpy.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:  # 如果两图片差距过小代表完美重合
        return 100
    PIXEL_MAX = 1.0
    print("", 20 * math.log10(PIXEL_MAX / math.sqrt(mse)))  # 将对数中pixel_max的平方放了下来

def entropy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    out = 0
    count = np.shape(img)[0]*np.shape(img)[1]
    p = np.bincount(np.array(img).flatten())
    for i in range(0, len(p)):
        if p[i]!=0:
            out-=p[i]*math.log(p[i]/count)/count
    return out


for i in range(1,4):
    name1=r'./image/ori -'+str(i)+'.png'
    name2=r'./image/our -'+str(i)+'.png'
    original = cv2.imread(name1)
    contrast = cv2.imread(name2,cv2.IMREAD_GRAYSCALE)
    # contrast = cv2.imread(name2)
    # print( skimage.measure.shannon_entropy(contrast,base=2))
    # print(SMD(contrast))
    # print(cv2.Laplacian(contrast, cv2.CV_64F).var())  #laplacian
    print(entropy(contrast))  #laplacian
    # psnrValue = psnr(original, contrast) #psnr
    # ssimValue = ssim(original, contrast) #ssim