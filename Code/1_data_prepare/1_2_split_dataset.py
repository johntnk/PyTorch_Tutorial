# coding: utf-8
"""
将原始数据集进行划分成训练集、验证集和测试集
"""

import os              # 用于路径拼接、遍历目录等
import glob            # 用于匹配文件（比如 *.png）
import random          # 用于随机打乱图片列表
import shutil          # 用于文件复制

# 你的原始数据集目录（里面按类别分文件夹，比如 raw_test/class1/*.png）
dataset_dir = os.path.join("..", "..", "Data", "cifar-10-png", "raw_test")

# 输出的训练/验证/测试目录
train_dir = os.path.join("..", "..", "Data", "train")
valid_dir = os.path.join("..", "..", "Data", "valid")
test_dir  = os.path.join("..", "..", "Data", "test")

# 划分比例
train_per = 0.8        # 训练集占比
valid_per = 0.1        # 验证集占比
test_per  = 0.1        # 测试集占比（实际上代码中用 imgs_num - valid_point 来得到）

def makedir(new_dir):
    """
    如果目录不存在，就创建它
    """
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

if __name__ == '__main__':

    # 遍历原始数据集目录
    # root：当前遍历到的目录
    # dirs：当前目录下的子目录（通常对应类别）
    # files：当前目录下的文件（这里没用到，因为我们用 glob 去找 png）
    for root, dirs, files in os.walk(dataset_dir):

        # 这里主要遍历每个子目录（也就是每个类别）
        for sDir in dirs:

            # 找出该类别文件夹下所有 png 图片路径
            imgs_list = glob.glob(os.path.join(root, sDir, '*.png'))

            # 固定随机种子，让划分结果可复现
            # 注意：你在这里对每个类别都 seed(666)，每次都会从同一个随机序列开始
            random.seed(666)

            # 打乱该类别的图片列表，保证划分更均匀
            random.shuffle(imgs_list)

            # 该类别图片总数
            imgs_num = len(imgs_list)

            # 训练集分割点：前 train_point 张放训练集
            train_point = int(imgs_num * train_per)

            # 验证集分割点：前 valid_point 张放到 train+valid
            # 即 train(0~train_point) + valid(train_point~valid_point)
            valid_point = int(imgs_num * (train_per + valid_per))

            # 遍历该类别所有图片，并拷贝到不同输出目录
            for i in range(imgs_num):

                # i < train_point：训练集
                if i < train_point:
                    out_dir = os.path.join(train_dir, sDir)

                # train_point <= i < valid_point：验证集
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sDir)

                # i >= valid_point：测试集
                else:
                    out_dir = os.path.join(test_dir, sDir)

                # 确保输出类别目录存在
                makedir(out_dir)

                # 输出文件路径：保持原文件名不变
                out_path = os.path.join(out_dir, os.path.split(imgs_list[i])[-1])

                # 复制文件
                shutil.copy(imgs_list[i], out_path)

            # 打印每个类别的划分统计信息
            # train：train_point
            # valid：valid_point - train_point
            # test：imgs_num - valid_point
            print('Class:{}, train:{}, valid:{}, test:{}'.format(
                sDir, train_point, valid_point - train_point, imgs_num - valid_point
            ))
