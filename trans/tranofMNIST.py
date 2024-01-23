import torch
import torchvision
from torchvision import datasets, transforms
import os
from img2graph import *
random.seed(0)


# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='../data/MNIST', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='../data/MNIST', train=False, download=True, transform=transform)

# 创建保存图像的文件夹
train_dir = '../data/MNIST/train'
test_dir = '../data/MNIST/test'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
import h5py
os.environ['HDF5_FREESPACE_THRESHOLD'] = '214748364800'  # 200 MB

# 保存训练集图像
for i, (image, label) in enumerate(train_dataset):
    filename = f'{train_dir}/{i}.png'  # 图像文件名格式为索引.png
    image = image.squeeze().numpy() * 255  # 转换为numpy数组并恢复像素值范围
    image = image.astype('uint8')
    torchvision.transforms.functional.to_pil_image(image).save(filename)
    img_name = filename # 图片路径
    RGB_img = cv2.imread(img_name)
    h, w = RGB_img.shape[0:2]
    adj,center, g,center_ = img2graph(RGB_img)
    edge_attr = []
    for k in range(len(adj)):
        s,e = adj[k][0],adj[k][1]
        this_attr = []
        for j in range(9):
            this_attr.append(center_[s][j]+center_[e][j])
        edge_attr.append(this_attr)
    for v in center:
        y, x, Rx, Ry = v[0][0], v[0][1], v[1], v[2]
        cv2.rectangle(RGB_img, (x - Rx, y - Ry), (x + Rx, y + Ry), (0, 255, 0), 1)
    #cv2.imwrite("./data/SLIC/SLIC_" + str(i) + "_" + str(label) + ".jpg", RGB_img)
    with h5py.File('../data/MNIST/h5_attr_all/train/' + str(i) + '.hdf5', 'w') as f:
        #dset = f.create_dataset('image', data="./data/SLIC/SLIC_" + str(i) + "_" + str(label) + ".jpg")
        f['y'] = label
        f['edge_index'] = adj
        f['x'] = center_
        f['edge_attr'] = edge_attr
        f.close()


# 保存测试集图像
for i, (image, label) in enumerate(test_dataset):
    filename = f'{test_dir}/{i}.png'  # 图像文件名格式为索引.png
    image = image.squeeze().numpy() * 255  # 转换为numpy数组并恢复像素值范围
    image = image.astype('uint8')
    torchvision.transforms.functional.to_pil_image(image).save(filename)
    img_name = filename # 图片路径
    RGB_img = cv2.imread(img_name)
    h, w = RGB_img.shape[0:2]
    adj,center, g,center_ = img2graph(RGB_img)
    edge_attr = []
    for k in range(len(adj)):
        s,e = adj[k][0],adj[k][1]
        this_attr = []
        for j in range(9):
            this_attr.append(center_[s][j]+center_[e][j])
        edge_attr.append(this_attr)
    for v in center:
        y, x, Rx, Ry = v[0][0], v[0][1], v[1], v[2]
        cv2.rectangle(RGB_img, (x - Rx, y - Ry), (x + Rx, y + Ry), (0, 255, 0), 1)
    #cv2.imwrite("./data/SLIC/SLIC_" + str(i) + "_" + str(label) + ".jpg", RGB_img)
    with h5py.File('../data/MNIST/h5_attr_all/test/' + str(i) + '.hdf5', 'w') as f:
        #dset = f.create_dataset('image', data="./data/SLIC/SLIC_" + str(i) + "_" + str(label) + ".jpg")
        f['y'] = label
        f['edge_index'] = adj
        f['x'] = center_
        f['edge_attr'] = edge_attr
        f.close()


print("图像保存、转换完成！")

