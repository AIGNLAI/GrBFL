import cv2
import numpy as np
import time
import networkx as nx
import scipy.sparse as sp
import random
random.seed(0)

# 返回所有像素梯度信息
def get_grad(img, dir = None):
    """
    :param img:
    :return: map{(x,y):grad} and grad matric
    """
    # get grad list
    # 分别计算x、y方向：右减左，下减上
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy2 = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    return sobelxy2

def cal_bound(img, center, Rx, Ry):
    """
    :param img: 原图，需要获取原图的大小
    :param center: 中心点坐标
    :param Rx, Ry: 不同方向的半径
    :return: 上下左右边界
    """
    h, w = img.shape
    left = center[1] - Rx
    right = center[1] + Rx
    up = center[0] - Ry
    down = center[0] + Ry

    if left < 0 :
        left = 0
    if right >= w :
        right = w - 1
    if up < 0 :
        up = 0
    if down >= h :
        down = h - 1

    return int(left), int(right), int(up), int(down)

def center_select(img_grad, img_label):
    """
    :param img_grad:
    :param img_label:
    :return:
    """
    minPix_xy = np.where(img_grad == img_grad.min()) # 返回一个二维元组，前一个为所有梯度最小的点的横坐标，后一个为所有梯度最小的点的纵坐标

    while True:
        pos = random.randint(0, len(minPix_xy[0]) - 1)  # 随机挑选一个中心点
        #print(len(minPix_xy[0]))
        if (img_label[minPix_xy[0][pos],minPix_xy[1][pos]] != 1):
            return [minPix_xy[0][pos],minPix_xy[1][pos]]

def cal_Radius(img, center, purity, threshold, var_threshold):
    """
    :param img: 输入图片
    :param center:  中心点坐标 [x, y]
    :param purity:  1 - （异类点个数 / 总个数)
    :param threshold:  判断是否是异类点， 与中心点灰度值的差值的绝对值 / 中心点的灰度值
    :return: Rx, Ry, 输入中心点对应的半径
    该方法待优化
    """
    Rx = 0 # 初始化半径
    Ry = 0

    flag = True
    flag_x = True
    flag_y = True
    item_count = 0
    temp_pixNum = 0
    center_value = int(img[center[0], center[1]])

    while True:
        if flag_x == True and flag_y == True:
            item_count += 1
        else:
            if flag_x:
                item_count = 1
            if flag_y:
                item_count = 2
        if flag_x and item_count % 2 != 0:
            Rx += 1

        if flag_y and item_count % 2 == 0:
            Ry += 1

        #计算切片边界
        left, right, up, down = cal_bound(img, center, Rx, Ry)
        pixNum = (down - up + 1) * (right - left + 1)  # 当前总像素点
        if pixNum == temp_pixNum:
            return Rx, Ry
        # 计算异类点个数
        count = len(np.where(abs(np.int_(img[up:down + 1, left:right + 1]) - center_value) > threshold)[0])
        temp_purity = 1 - count / pixNum
        var = np.var(img[up:down + 1, left:right + 1])
        temp_pixNum = pixNum

        if temp_purity > purity and var < var_threshold:
            if purity < 0.99:
                purity = purity * 1.005
            else:
                purity = 0.99
            flag = True
        else:
            flag = False

        if flag == False and item_count % 2 != 0:
            flag_x = False
            Rx -= 1

        if flag == False and item_count % 2 == 0:
            flag_y = False
            Ry -= 1

        if flag_x == False and flag_y == False:
            return Rx, Ry

def img2graph(img, purity=0.7, threshold=10, var_threshold=30):
    """ 本函数用于对图片进行基础粒矩聚类，得到基础粒矩列表以及基础图进行后续操作，例如：可视化，上下采样，旋转，反转等。
    :param img: 输入图片
    :param purity:
    :param threshold:
    :param var_threshold:
    :return: 粒矩列表：center，图：g
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 彩色图转为灰度图
    img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
    img_label = np.zeros(img.shape)  # 创建label矩阵
    img_grad = get_grad(img)  # 计算梯度图
    max_Grad = img_grad.max()  # 计算梯度图最大值
    center = []  # 创建中心列表
    center_count = 0  # 中心点个数计数

    start = time.time()
    while 0 in img_label: # 存在没有被划分的点
        # 选择一个梯度最小且没有被划分过的点为中心点
        temp_center = center_select(img_grad, img_label)
        # 计算半径 Rx, Ry -> x 方向的半径与 y 方向的半径
        Rx, Ry = cal_Radius(img, temp_center, purity, threshold, var_threshold)
        # 计算实际的矩形在图片中的位置(存在粒矩大小超出图像范围，所以不能直接用半径进行切片)，方便后续使用切片进行特征提取等操作
        left, right, up, down = cal_bound(img, temp_center, Rx, Ry)
        # 添加粒矩 （[x, y], Rx, Ry） 一个粒矩一个元组.最基础特征:中心点坐标， Rx， Ry
        center.append((temp_center, Rx, Ry)) # 粒矩存储方式待优化
        # 将本次迭代生成的粒矩包含的像素点标记 （下次迭代就不会选取这些点作为中心点）
        img_label[up:down + 1, left:right + 1] = 1
        # 将本次迭代生成的粒矩包含的像素点对应位置的梯度设为梯度最大值
        img_grad[up:down + 1, left:right + 1] = max_Grad
        # 粒矩计数
        center_count += 1
    end = time.time()
    print("粒矩聚类时间:%.2f秒" % (end - start))
    print("共生成" + str(center_count) + "个粒矩")

    # 聚类完成，开始构建 Graph (本代码使用 networkx 进行 Graph 的构建，后续如果要考虑构图速度也可以使用其他方法)
    # 图的基本组成 -> 节点集和边集：节点集就是所有粒矩的中心点，边集就是判断两个粒矩是否有重叠的像素点，有就将两个粒矩的中心点相连，即这两个节点之间存在边
    # 初始化 Graph
    g = nx.Graph()
    # 1. 添加节点
    for i in range(len(center)):
        g.add_node(str(i))
    # 2. 生成边 (使用粒矩的位置关系进行边的生成)
    for i in range(len(center)):
        for j in range(i + 1, len(center)):
            center_1 = center[i]
            center_2 = center[j]
            if (abs(center_1[0][0] - center_2[0][0]) - 1) < center_1[2] + center_2[2] and (
                    abs(center_1[0][1] - center_2[0][1]) - 1) < center_1[1] + center_2[1]:  # 相接无边
                g.add_edge(str(i), str(j))
    # 3. 生成 GNN 需要的数据 -> 存有节点的矩阵，大小为 N * F (N 为节点的个数，F 为节点特征的维度)
    #                        边集，大小为 2 * N (N 为边的个数，每一列为该边连接的两个节点在节点矩阵中的索引)
    #                        边特征矩阵 (可选，部分图神经网络可用)
    a = nx.adjacency_matrix(g)
    a = sp.csr_matrix(a)
    # 邻接矩阵 adj （边集）
    adj = a.A
    adj = sp.coo_matrix(adj)
    adj = np.vstack((adj.row, adj.col))

    center_ = np.zeros((len(center), 9))  # 粒矩基础属性 -> 中心坐标, Rx, Ry （不同的数据可手动提取不同的特征）
    # 生成节点属性和粒矩基础属性数组,这里还可以给出剩余5个特征的构造方法？？
    for id in range(len(center)):
        # 粒矩基础属性 center_
        center_[id] = [center[id][0][0], center[id][0][1], center[id][1], center[id][2],0,0,0,0,0]

    return adj,center, g,center_
