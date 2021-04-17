import matplotlib.pyplot as plt
from Demo.retinex import retinex_FM, retinex_MSRCR, retinex_gimp, retinex_MSRCP, retinex_AMSR
from Demo.tools import cv2_heq, simplest_color_balance, gauss_blur, eps, filter_1
import cv2
import numpy as np
import mahotas
import time

def gamma(img, gamma1=2.0):
    imgIn = img.astype(float) / 255
    msr = imgIn ** gamma1
    msr *= 255
    msr[msr < 0] = 0
    msr[msr > 255] = 255
    msr = msr.astype(np.uint8)
    return msr
def image_hist_demo(image):
    color = {"blue", "green", "red"}
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据下标和数据，一般用在 for 循环当中。
    for i, color in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        frame = plt.gca()
        frame.axes.get_yaxis().set_visible(False)
        plt.xlim([0, 256])

for i in range(1,4):
    start = time.time()
    name="./image/ori -"+str(i)+".png" #路径可修改也可改成非循环模式
    ori = cv2.imread(name)
    img = cv2.imread(name)
    ori = retinex_AMSR(ori)
    ori_tc = mahotas.morph.tophat_open(mahotas.morph.tophat_close(ori))
    ori_to = mahotas.morph.tophat_close(mahotas.morph.tophat_open(ori))

    im = ((ori_tc - ori))
    im2 = (ori - ori_to)
    im2 = im2.astype(np.uint8)
    im = im.astype(np.uint8)
    fin = gamma(im * 0.1 + im2 * 0.1 + ori).astype("uint8")

    # cv2.imshow("fin",fin)
    # cv2.imshow("ori",img)

    # msr = retinex_MSRCR(img)
    end = time.time()
    # cv2.imwrite("msr-"+str(i)+'.png',msr)
    cv2.imwrite("./image/ours-"+str(i)+'.png',fin)
    print(str(end-start)+"s") #计算算法时间
cv2.waitKey()
