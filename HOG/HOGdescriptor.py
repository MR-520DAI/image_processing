import cv2

if __name__ == '__main__':

    #读入图像
    img = cv2.imread(".\person.PNG")

    #参数设置
    winSize = (128,128) #窗口大小
    blockSize = (64,64) #block块大小
    blockStride = (8,8) #block块步长
    cellSize = (16,16)  #cell块大小
    nbins = 9   #梯度方向数

    #定义HOG特征检测器
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    winStride = (8,8)
    padding = (8,8)

    #计算特征
    feature = hog.compute(img, winStride, padding)

    print(feature.shape)
