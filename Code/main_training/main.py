# coding: utf-8

# 导入 PyTorch 主库
import torch
# 导入 DataLoader 用于批量加载数据
from torch.utils.data import DataLoader
# 导入 torchvision 的图像变换工具（数据预处理）
import torchvision.transforms as transforms
# 导入 NumPy，用于数值计算/混淆矩阵等
import numpy as np
# 导入 os，用于路径拼接与文件夹创建
import os
# 从 autograd 导入 Variable（老写法，现代PyTorch通常不需要）
from torch.autograd import Variable
# 导入神经网络模块基类 nn
import torch.nn as nn
# 导入功能性函数 F（如 relu）
import torch.nn.functional as F
# 导入优化器 optim（SGD等）
import torch.optim as optim
# 导入 sys，用于修改 Python 路径
import sys
# 将上一级目录加入路径，以便导入自定义 utils
sys.path.append("..")
# 从自定义 utils 模块导入：MyDataset、validate、show_confMat
from utils.utils import MyDataset, validate, show_confMat
# 导入 tensorboardX 的 SummaryWriter，用于写入TensorBoard日志
from tensorboardX import SummaryWriter
# 导入 datetime，用于生成日志目录名
from datetime import datetime

# 训练集文本文件路径（相对当前文件）
train_txt_path = os.path.join("..", "..", "Data", "train.txt")
# 验证集文本文件路径（相对当前文件）
valid_txt_path = os.path.join("..", "..", "Data", "valid.txt")

# CIFAR-10类别名称（10个类别）
classes_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 训练批大小
train_bs = 16
# 验证批大小
valid_bs = 16
# 初始学习率
lr_init = 0.001
# 最大epoch数
max_epoch = 1

# 结果保存目录路径
result_dir = os.path.join("..", "..", "Result")

# 获取当前时间
now_time = datetime.now()
# 格式化时间字符串，用于区分不同训练实验
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

# 当前实验的日志目录路径
log_dir = os.path.join(result_dir, time_str)
# 如果日志目录不存在，就创建
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 创建 TensorBoard 的写入器，并指定日志目录
writer = SummaryWriter(log_dir=log_dir)

# ------------------------------------ step 1/5 : 加载数据------------------------------------

# 用于归一化的均值（每个通道的mean）
normMean = [0.4948052, 0.48568845, 0.44682974]
# 用于归一化的标准差（每个通道的std）
normStd = [0.24580306, 0.24236229, 0.2603115]
# 创建归一化变换
normTransform = transforms.Normalize(normMean, normStd)

# 训练集数据增强/预处理流程
trainTransform = transforms.Compose([
    # 将图像缩放为 32x32
    transforms.Resize(32),
    # 随机裁剪到32x32，padding=4相当于先四周padding再裁剪
    transforms.RandomCrop(32, padding=4),
    # 转为Tensor（shape从HWC变为CHW，数值范围[0,1]）
    transforms.ToTensor(),
    # 进行归一化
    normTransform
])

# 验证集预处理流程（不做随机增强）
validTransform = transforms.Compose([
    # 转为Tensor
    transforms.ToTensor(),
    # 进行归一化
    normTransform
])

# 构建训练集Dataset实例
train_data = MyDataset(txt_path=train_txt_path, transform=trainTransform)
# 构建验证集Dataset实例
valid_data = MyDataset(txt_path=valid_txt_path, transform=validTransform)

# 构建训练 DataLoader：shuffle打乱训练样本顺序
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)
# 构建验证 DataLoader：通常不shuffle
valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs)

# ------------------------------------ step 2/5 : 定义网络------------------------------------

# 定义卷积神经网络类 Net
class Net(nn.Module):
    # 初始化网络结构
    def __init__(self):
        # 调用父类构造函数
        super(Net, self).__init__()
        # 第一个卷积层：输入3通道输出6通道，卷积核大小5x5（默认stride=1 padding=0）
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 第一个最大池化：窗口2x2，stride=2
        self.pool1 = nn.MaxPool2d(2, 2)
        # 第二个卷积层：输入6通道输出16通道，卷积核大小5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 第二个最大池化层
        self.pool2 = nn.MaxPool2d(2, 2)
        # 全连接层：输入维度 16*5*5（根据前面卷积+池化计算得出），输出120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 全连接层：120 -> 84
        self.fc2 = nn.Linear(120, 84)
        # 输出层：84 -> 10（对应10分类）
        self.fc3 = nn.Linear(84, 10)

    # 前向传播
    def forward(self, x):
        # conv1 + relu + pool1
        x = self.pool1(F.relu(self.conv1(x)))
        # conv2 + relu + pool2
        x = self.pool2(F.relu(self.conv2(x)))
        # 将特征图展平为一维向量（batch维保留）
        x = x.view(-1, 16 * 5 * 5)
        # fc1 + relu
        x = F.relu(self.fc1(x))
        # fc2 + relu
        x = F.relu(self.fc2(x))
        # fc3 输出 logits（不做softmax，因为CrossEntropyLoss内部会处理）
        x = self.fc3(x)
        # 返回分类输出
        return x

    # 定义权值初始化方法
    def initialize_weights(self):
        # 遍历网络中所有子模块
        for m in self.modules():
            # 若是卷积层
            if isinstance(m, nn.Conv2d):
                # 使用 Xavier 正态分布初始化卷积核权重
                torch.nn.init.xavier_normal_(m.weight.data)
                # 如果有bias则置0
                if m.bias is not None:
                    m.bias.data.zero_()
            # 若是BatchNorm层
            elif isinstance(m, nn.BatchNorm2d):
                # gamma置1
                m.weight.data.fill_(1)
                # beta置0
                m.bias.data.zero_()
            # 若是全连接层
            elif isinstance(m, nn.Linear):
                # 使用正态分布初始化线性层权重
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                # bias置0
                m.bias.data.zero_()

# 创建一个网络实例
net = Net()
# 调用权值初始化方法
net.initialize_weights()

# ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------

# 使用交叉熵作为损失函数（多分类）
criterion = nn.CrossEntropyLoss()
# 使用SGD优化器：学习率lr_init，动量momentum=0.9，阻尼dampening=0.1（注意：dampening用于momentum的变体）
optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)
# 学习率调度器：每step_size=50个epoch后学习率乘gamma=0.1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# ------------------------------------ step 4/5 : 训练 --------------------------------------------------

# 遍历epoch
for epoch in range(max_epoch):

    # 记录本epoch训练loss累计值
    loss_sigma = 0.0
    # 记录累计正确预测数量
    correct = 0.0
    # 记录累计样本数量
    total = 0.0
    # 更新学习率（在当前epoch开始时调用）
    scheduler.step()

    # 训练集迭代
    for i, data in enumerate(train_loader):
        # 获取图片与标签
        inputs, labels = data
        # 将inputs、labels封装为Variable（旧写法）
        inputs, labels = Variable(inputs), Variable(labels)

        # 梯度清零（为下一次反向传播做准备）
        optimizer.zero_grad()
        # 前向传播得到模型输出
        outputs = net(inputs)
        # 计算损失（CrossEntropyLoss接收logits和类别index）
        loss = criterion(outputs, labels)
        # 反向传播计算梯度
        loss.backward()
        # 根据梯度更新参数
        optimizer.step()

        # 取预测类别：outputs最大值对应的类别index
        _, predicted = torch.max(outputs.data, 1)
        # 累加总样本数
        total += labels.size(0)
        # 统计当前batch预测正确数量并累加
        correct += (predicted == labels).squeeze().sum().numpy()
        # 累加当前loss
        loss_sigma += loss.item()

        # 每10个iteration打印一次训练信息
        if i % 10 == 9:
            # 计算这10次的平均loss
            loss_avg = loss_sigma / 10
            # 重置loss累计
            loss_sigma = 0.0
            # 打印训练进度与loss/acc
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg, correct / total))

            # 写入tensorboard：训练loss
            writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
            # 写入tensorboard：当前学习率（取optimizer的第一个param group）
            writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)
            # 写入tensorboard：训练准确率
            writer.add_scalars('Accuracy_group', {'train_acc': correct / total}, epoch)

    # 每个epoch后，将每层参数的梯度和权值以直方图形式写入tensorboard
    for name, layer in net.named_parameters():
        # 写入梯度直方图（如果grad为None会报错，这里没有做判断）
        writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        # 写入参数权值直方图
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

    # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
    # 每2个epoch进行一次验证
    if epoch % 2 == 0:
        # 验证loss累计
        loss_sigma = 0.0
        # 类别数
        cls_num = len(classes_name)
        # 初始化混淆矩阵：行是真实类别，列是预测类别
        conf_mat = np.zeros([cls_num, cls_num])

        # 切换为评估模式（例如会影响Dropout/BatchNorm等）
        net.eval()
        # 遍历验证集
        for i, data in enumerate(valid_loader):

            # 取出图像和标签
            images, labels = data
            # Variable封装
            images, labels = Variable(images), Variable(labels)

            # 前向传播
            outputs = net(images)
            # detach_与验证无梯度无关（且没有no_grad），这里只是断开计算图（但没有阻止图构建）
            outputs.detach_()

            # 计算验证loss
            loss = criterion(outputs, labels)
            # 累加loss
            loss_sigma += loss.item()

            # 取预测类别
            _, predicted = torch.max(outputs.data, 1)

            # 构建混淆矩阵：统计每个样本真实类别 vs 预测类别
            for j in range(len(labels)):
                # 真实类别index
                cate_i = labels[j].numpy()
                # 预测类别index
                pre_i = predicted[j].numpy()
                # 在混淆矩阵对应位置加1
                conf_mat[cate_i, pre_i] += 1.0

        # 混淆矩阵对角线和 / 总和 = accuracy
        print('{} set Accuracy:{:.2%}'.format('Valid', conf_mat.trace() / conf_mat.sum()))
        # 写入tensorboard：验证loss
        writer.add_scalars('Loss_group', {'valid_loss': loss_sigma / len(valid_loader)}, epoch)
        # 写入tensorboard：验证accuracy
        writer.add_scalars('Accuracy_group', {'valid_acc': conf_mat.trace() / conf_mat.sum()}, epoch)

# 训练结束提示
print('Finished Training')

# ------------------------------------ step5: 保存模型 并且绘制混淆矩阵图 ------------------------------------

# 保存模型参数文件路径
net_save_path = os.path.join(log_dir, 'net_params.pkl')
# 将模型state_dict保存到文件
torch.save(net.state_dict(), net_save_path)

# 使用validate函数在训练集上验证并返回混淆矩阵与准确率
conf_mat_train, train_acc = validate(net, train_loader, 'train', classes_name)
# 在验证集上验证
conf_mat_valid, valid_acc = validate(net, valid_loader, 'valid', classes_name)

# 绘制训练集混淆矩阵图并保存到log_dir
show_confMat(conf_mat_train, classes_name, 'train', log_dir)
# 绘制验证集混淆矩阵图并保存到log_dir
show_confMat(conf_mat_valid, classes_name, 'valid', log_dir)

