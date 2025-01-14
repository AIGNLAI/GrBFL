import cv2
import numpy as np
import time
import networkx as nx
import scipy.sparse as sp
import random
random.seed(0)
N = 5000
sum_of_square = np.zeros((N,N))
sum_of_num = np.zeros((N,N))
sum_of_pur = np.zeros((N,N))

def get_grad(img, dir = None):

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy2 = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    return sobelxy2

def cal_bound(img, center, Rx, Ry):
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
    minPix_xy = np.where(img_grad == img_grad.min()) 
    while True:
        pos = random.randint(0, len(minPix_xy[0]) - 1)
        if (img_label[minPix_xy[0][pos],minPix_xy[1][pos]] != 1):
            return [minPix_xy[0][pos],minPix_xy[1][pos]]

def cal_Radius(img, center, purity, threshold, var_threshold):
    a = 0
    Rx = 0
    Ry = 0
    flag = True
    flag_x = True
    flag_y = True
    item_count = 0
    temp_pixNum = 0
    center_value = int(img[center[0], center[1]])
    while True:
        a += 1
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
        left, right, up, down = cal_bound(img, center, Rx, Ry)
        pixNum = (down - up + 1) * (right - left + 1)
        if pixNum == temp_pixNum:
            return Rx, Ry
        count = len(np.where(abs(np.int_(img[up:down + 1, left:right + 1]) - center_value) > threshold)[0])
        temp_purity = 1 - count / pixNum
        var = (sum_of_square[down + 1][right + 1] - sum_of_square[down + 1][left] - sum_of_square[up][right + 1] + sum_of_square[up][left])/pixNum - \
            ((sum_of_num[down + 1][right + 1] - sum_of_num[down + 1][left] - sum_of_num[up][right + 1] + sum_of_num[up][left])/pixNum)**2
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


def cal_Radius_bin(img, center, purity, threshold, var_threshold):
    rx,ry = 0,0
    flag = True
    center_value = int(img[center[0], center[1]])
    ttmp = 4
    for i in range(1,ttmp):
        left, right, up, down = cal_bound(img, center, i, i)
        pixNum = (down - up + 1) * (right - left + 1)
        count = len(np.where(abs(np.int_(img[up:down + 1, left:right + 1]) - center_value) > threshold)[0])
        temp_purity = 1 - count / pixNum
        var = (sum_of_square[down + 1][right + 1] - sum_of_square[down + 1][left] - sum_of_square[up][right + 1] + sum_of_square[up][left])/pixNum - \
            ((sum_of_num[down + 1][right + 1] - sum_of_num[down + 1][left] - sum_of_num[up][right + 1] + sum_of_num[up][left])/pixNum)**2
        if temp_purity > purity and var < var_threshold:
            rx,ry = i,i
        else:
            break
    if rx == ttmp-1:
        bin_r = len(img) - 1
        bin_l = 0
        while bin_l < bin_r:
            mid = (bin_l + bin_r) >> 1
            left, right, up, down = cal_bound(img, center, mid, mid)
            pixNum = (down - up + 1) * (right - left + 1)
            count = len(np.where(abs(np.int_(img[up:down + 1, left:right + 1]) - center_value) > threshold)[0])
            temp_purity = 1 - count / pixNum
            var = (sum_of_square[down + 1][right + 1] - sum_of_square[down + 1][left] - sum_of_square[up][right + 1] + sum_of_square[up][left])/pixNum - \
            ((sum_of_num[down + 1][right + 1] - sum_of_num[down + 1][left] - sum_of_num[up][right + 1] + sum_of_num[up][left])/pixNum)**2
            if temp_purity > purity and var < var_threshold:
                rx,ry = mid,mid
                break
            else:
                bin_r = mid
    flag = True
    left, right, up, down = cal_bound(img, center, rx + 1, ry)
    pixNum = (down - up + 1) * (right - left + 1)
    count = len(np.where(abs(np.int_(img[up:down + 1, left:right + 1]) - center_value) > threshold)[0])
    temp_purity = 1 - count / pixNum
    var = np.var(img[up:down + 1, left:right + 1])
    if temp_purity > purity and var < var_threshold:
        flag = True
    else:
        flag = False
    if flag:
        for i in range(1,ttmp):
            left, right, up, down = cal_bound(img, center, rx + 1, ry)
            pixNum = (down - up + 1) * (right - left + 1)
            count = len(np.where(abs(np.int_(img[up:down + 1, left:right + 1]) - center_value) > threshold)[0])
            temp_purity = 1 - count / pixNum
            var = (sum_of_square[down + 1][right + 1] - sum_of_square[down + 1][left] - sum_of_square[up][right + 1] + sum_of_square[up][left])/pixNum - \
            ((sum_of_num[down + 1][right + 1] - sum_of_num[down + 1][left] - sum_of_num[up][right + 1] + sum_of_num[up][left])/pixNum)**2
            if temp_purity > purity and var < var_threshold:
                rx += 1
            else:
                break
        if rx - ry == ttmp-1:
            bin_r = len(img) - 1
            bin_l = 2
            while bin_l < bin_r:
                mid = (bin_l + bin_r) >> 1
                left, right, up, down = cal_bound(img, center, rx + mid, ry)
                pixNum = (down - up + 1) * (right - left + 1)
                count = len(np.where(abs(np.int_(img[up:down + 1, left:right + 1]) - center_value) > threshold)[0])
                temp_purity = 1 - count / pixNum
                var = (sum_of_square[down + 1][right + 1] - sum_of_square[down + 1][left] - sum_of_square[up][right + 1] + sum_of_square[up][left])/pixNum - \
            ((sum_of_num[down + 1][right + 1] - sum_of_num[down + 1][left] - sum_of_num[up][right + 1] + sum_of_num[up][left])/pixNum)**2
                if temp_purity > purity and var < var_threshold:
                    rx += mid
                    break
                else:
                    bin_r = mid
        
    else:
        for i in range(1,ttmp):
            left, right, up, down = cal_bound(img, center, rx , ry + 1)
            pixNum = (down - up + 1) * (right - left + 1)
            count = len(np.where(abs(np.int_(img[up:down + 1, left:right + 1]) - center_value) > threshold)[0])
            temp_purity = 1 - count / pixNum
            var = (sum_of_square[down + 1][right + 1] - sum_of_square[down + 1][left] - sum_of_square[up][right + 1] + sum_of_square[up][left])/pixNum - \
            ((sum_of_num[down + 1][right + 1] - sum_of_num[down + 1][left] - sum_of_num[up][right + 1] + sum_of_num[up][left])/pixNum)**2
            if temp_purity > purity and var < var_threshold:
                ry += 1
            else:
                break
        if rx - ry == -ttmp+1:
            bin_r = len(img) - 1
            bin_l = 2
            sep = []
            while bin_l < bin_r:
                mid = (bin_l + bin_r) >> 1
                left, right, up, down = cal_bound(img, center, rx , ry + mid)
                pixNum = (down - up + 1) * (right - left + 1)
                count = len(np.where(abs(np.int_(img[up:down + 1, left:right + 1]) - center_value) > threshold)[0])
                temp_purity = 1 - count / pixNum
                var = (sum_of_square[down + 1][right + 1] - sum_of_square[down + 1][left] - sum_of_square[up][right + 1] + sum_of_square[up][left])/pixNum - \
            ((sum_of_num[down + 1][right + 1] - sum_of_num[down + 1][left] - sum_of_num[up][right + 1] + sum_of_num[up][left])/pixNum)**2
                if temp_purity > purity and var < var_threshold:
                    ry += mid
                    break
                else:
                    bin_r = mid
        
    return rx,ry

def img2graph(img, purity=0.9, threshold=10, var_threshold=20):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    for i in range(1,len(img) + 1):
        for j in range(1,len(img[0]) + 1):
            sum_of_num[i][j] = sum_of_num[i - 1][j] + sum_of_num[i][j - 1] - sum_of_num[i - 1][j - 1] + img[i - 1][j - 1]
            sum_of_square[i][j] = sum_of_square[i - 1][j] + sum_of_square[i][j - 1] - sum_of_square[i - 1][j - 1] + img[i - 1][j - 1]**2

    img_label = np.zeros(img.shape) 
    img_grad = get_grad(img)
    max_Grad = img_grad.max()
    center = []
    center_count = 0 

    start = time.time()
    while 0 in img_label: 
        temp_center = center_select(img_grad, img_label)
        Rx, Ry = cal_Radius_bin(img, temp_center, purity, threshold, var_threshold)
        left, right, up, down = cal_bound(img, temp_center, Rx, Ry)
        center.append((temp_center, np.mean(img[up:down + 1, left:right + 1]),np.var(img[up:down + 1, left:right + 1]),Rx, Ry,np.max(img[up:down + 1, left:right + 1]),np.min(img[up:down + 1, left:right + 1]))) # 粒矩存储方式待优化
        img_label[up:down + 1, left:right + 1] = 1
        img_grad[up:down + 1, left:right + 1] = max_Grad
        center_count += 1
    end = time.time()

    g = nx.Graph()
    for i in range(len(center)):
        g.add_node(str(i))
    for i in range(len(center)):
        for j in range(i + 1, len(center)):
            center_1 = center[i]
            center_2 = center[j]
            if (abs(center_1[0][0] - center_2[0][0]) - 1) < center_1[2] + center_2[2] and (
                    abs(center_1[0][1] - center_2[0][1]) - 1) < center_1[1] + center_2[1]:
                g.add_edge(str(i), str(j))

    a = nx.adjacency_matrix(g)
    a = sp.csr_matrix(a)
    adj = a.A
    adj = sp.coo_matrix(adj)
    adj = np.vstack((adj.row, adj.col))

    center_ = np.zeros((len(center), 9))
    theta = np.random.randint(0,360)
    mode = np.random.randint(0,4)
    if mode == 0:
        for id in range(len(center)):
            center_[id] = [center[id][0][0], center[id][0][1], center[id][1], center[id][2],center[id][3],center[id][4],center[id][5],center[id][6],0]
    elif mode == 1:
        for id in range(len(center)):
            x_new = (center[id][3] - center[id][0][0])*np.cos(theta) + (center[id][4] - center[id][0][1])*np.sin(theta) + center[id][0][0]
            y_new = (center[id][4] - center[id][0][1])*np.cos(theta) - (center[id][3] - center[id][0][0])*np.sin(theta) + center[id][0][1]
            center_[id] = [center[id][0][0], center[id][0][1], center[id][1], center[id][2],x_new,y_new,center[id][5],center[id][6],0]
    elif mode == 2:
        for id in range(len(center)):
            x_new = center[id][0][0]
            y_new = len(img)-center[id][0][1]
            center_[id] = [center[id][0][0], center[id][0][1], center[id][1], center[id][2],x_new,y_new,center[id][5],center[id][6],0]
    else:
        for id in range(len(center)):
            x_new = len(img[0])-center[id][0][0]
            y_new = center[id][0][1]
            center_[id] = [center[id][0][0], center[id][0][1], center[id][1], center[id][2],x_new,y_new,center[id][5],center[id][6],0]
    return adj,center, g,center_
from PIL import Image, ImageDraw, ImageOps
import random
import numpy as np

def generate_color(vm, vv, vmax, vmin):
    mean = vm
    std_dev = np.sqrt(vv)

    r = int(np.clip(np.random.normal(mean, std_dev), vmin, vmax))
    g = int(np.clip(np.random.normal(mean, std_dev), vmin, vmax))
    b = int(np.clip(np.random.normal(mean, std_dev), vmin, vmax))

    return (r, g, b)

def generate_image(center, g, img_size):
    img = Image.new('RGB', img_size, (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for c in center:
        (x, y), vm, vv, rx, ry, vmax, vmin = c
        x += random.randint(-2, 2)
        y += random.randint(-2, 2)
        rx += random.randint(0, 2)
        ry += random.randint(0, 2)
        fill_color = generate_color(vm, vv, vmax, vmin)
        draw.rectangle((x-rx, y-ry, x+rx, y+ry), fill=fill_color)

    img = ImageOps.mirror(img)
    img = img.rotate(-90) 
    return img
