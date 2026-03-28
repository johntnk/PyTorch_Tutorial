# coding:utf-8
import os
'''
    为数据集生成对应的 txt 文件
    作用：遍历 Data/train 或 Data/valid 目录下的子文件夹，
         找到其中的 png 图片，并为每张图片生成一行：
         图片路径 + 标签
    常用于：目标检测/分类/分割任务的数据集格式整理
'''

# 训练集 txt 文件保存路径（会生成/覆盖该文件）
train_txt_path = os.path.join("..", "..", "Data", "train.txt")
# 训练集图片目录（内部一般是：类别文件夹/图片）
train_dir = os.path.join("..", "..", "Data", "train")

# 验证集 txt 文件保存路径
valid_txt_path = os.path.join("..", "..", "Data", "valid.txt")
# 验证集图片目录
valid_dir = os.path.join("..", "..", "Data", "valid")


def gen_txt(txt_path, img_dir):
    # 以写入模式打开 txt 文件（如果不存在就创建，存在就清空再写）
    f = open(txt_path, 'w')

    # 遍历 img_dir 目录树
    # root：当前遍历到的目录
    # s_dirs：当前 root 下的所有子目录名（topdown=True 表示从上往下遍历）
    # _：第三个返回值通常是文件名列表，这里不关心所以用 _
    for root, s_dirs, _ in os.walk(img_dir, topdown=True):
        # 遍历每个子目录（通常每个子目录代表一个类别）
        for sub_dir in s_dirs:
            # 拼出该类别目录的绝对路径
            i_dir = os.path.join(root, sub_dir)
            # 列出类别目录下的所有文件（这里假设是 png 图片）
            img_list = os.listdir(i_dir)

            # 遍历类别目录下的每个文件名
            for i in range(len(img_list)):
                # 如果不是 png 文件，跳过
                # 注意：这里用 endswith('png')，更规范写法是 endswith('.png')
                if not img_list[i].endswith('png'):
                    continue

                # 从文件名中提取标签：
                # 假设图片命名格式类似：label_xxx.png
                # 例如：cat_001.png -> label='cat'
                label = img_list[i].split('_')[0]

                # 拼出图片的完整路径
                img_path = os.path.join(i_dir, img_list[i])

                # 写入 txt 一行：图片路径 + 空格 + 标签 + 换行
                line = img_path + ' ' + label + '\n'
                f.write(line)

    # 关闭文件
    f.close()


if __name__ == '__main__':
    # 生成训练集 txt：Data/train.txt
    gen_txt(train_txt_path, train_dir)
    # 生成验证集 txt：Data/valid.txt
    gen_txt(valid_txt_path, valid_dir)
