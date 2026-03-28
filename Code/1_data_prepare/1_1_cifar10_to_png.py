```python
# coding:utf-8  # 指定源码文件编码为 UTF-8
"""
    将cifar10的data_batch_12345 转换成 png格式的图片
    每个类别单独存放在一个文件夹，文件夹名称为0-9
"""
from imageio import imwrite  # 使用 imageio 的 imwrite 将数组保存为图片文件
import numpy as np  # 导入 numpy，用于数据 reshape、transpose 等操作
import os  # 导入 os，用于路径拼接与文件夹创建
import pickle  # 导入 pickle，用于读取 cifar-10 的二进制批次文件

base_dir = "D:/python   11/新建文件夹/practise/pytorch" # 基础目录（需要修改为你的实际Data目录所在绝对路径）
data_dir = os.path.join(base_dir, "Data", "cifar-10-batches-py") # CIFAR-10 原始 batch 文件所在目录
train_o_dir = os.path.join(base_dir, "Data", "cifar-10-png", "raw_train") # 训练集图片输出根目录
test_o_dir = os.path.join(base_dir, "Data", "cifar-10-png", "raw_test") # 测试集图片输出根目录

Train = False   # 控制是否解压训练集：True=解压训练集；False=不解压训练集，仅解压测试集

# 解压缩，返回解压后的字典
def unpickle(file):
    with open(file, 'rb') as fo:  # 以二进制方式打开 pickle 文件
        dict_ = pickle.load(fo, encoding='bytes')  # 反序列化并以 bytes 编码读取数据
    return dict_  # 返回解压后的字典

def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):  # 若目录不存在
        os.makedirs(my_dir)  # 创建目录（包含多级目录）

# 生成训练集图片，
if __name__ == '__main__':
    if Train:  # 若需要处理训练集
        for j in range(1, 6):  # 遍历 data_batch_1 到 data_batch_5
            data_path = os.path.join(data_dir, "data_batch_" + str(j))  # 拼接每个 batch 的路径
            train_data = unpickle(data_path)  # 读取并解包当前 batch
            print(data_path + " is loading...")  # 打印加载提示

            for i in range(0, 10000):  # 每个 data_batch 包含 10000 张图
                img = np.reshape(train_data[b'data'][i], (3, 32, 32))  # 将一张图从 (3*32*32,) reshape 为 (3,32,32)
                img = img.transpose(1, 2, 0)  # 将通道维从 CHW 转为 HWC 以便保存图片

                label_num = str(train_data[b'labels'][i])  # 取出该图片对应的类别标签，并转为字符串
                o_dir = os.path.join(train_o_dir, label_num)  # 该类别对应的输出文件夹路径
                my_mkdir(o_dir)  # 确保该文件夹存在

                img_name = label_num + '_' + str(i + (j - 1)*10000) + '.png'  # 生成图片文件名（带类别+全局索引）
                img_path = os.path.join(o_dir, img_name)  # 拼接图片保存路径
                imwrite(img_path, img)  # 将图像数组保存为 png

            print(data_path + " loaded.")  # 当前 batch 处理完成提示

    print("test_batch is loading...")  # 开始加载测试集 batch

    # 生成测试集图片
    test_data_path = os.path.join(data_dir, "test_batch")  # 测试集 batch 路径
    test_data = unpickle(test_data_path)  # 读取并解包测试集
    for i in range(0, 10000):  # 测试集同样包含 10000 张图
        img = np.reshape(test_data[b'data'][i], (3, 32, 32))  # 将单张图 reshape 为 (3,32,32)
        img = img.transpose(1, 2, 0)  # 转换为 (32,32,3) 方便保存

        label_num = str(test_data[b'labels'][i])  # 取出类别标签
        o_dir = os.path.join(test_o_dir, label_num)  # 对应类别的输出目录
        my_mkdir(o_dir)  # 创建目录（如果不存在）

        img_name = label_num + '_' + str(i) + '.png'  # 生成图片名（带类别+索引）
        img_path = os.path.join(o_dir, img_name)  # 拼接保存路径
        imwrite(img_path, img)  # 保存 png 文件

    print("test_batch loaded.")  # 测试集处理完成提示

