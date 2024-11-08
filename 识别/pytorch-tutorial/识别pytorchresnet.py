import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import sys


# 1. 定义使用 ResNet-18 的模型
class ResNet18ForMNIST(nn.Module):
    def __init__(self):
        super(ResNet18ForMNIST, self).__init__()
        # 加载预训练的 ResNet-18
        self.resnet18 = models.resnet18(pretrained=True)

        # 修改输入通道数，因为 ResNet 默认接受3通道输入图像
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # 修改最后的全连接层，将输出类别数修改为 10（MNIST有10个类别）
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 10)

    def forward(self, x):
        return self.resnet18(x)


# 2. 数据预处理与加载
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 将单通道图像转换为3通道
    transforms.Resize((224, 224)),  # ResNet-18 默认输入大小为 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用ImageNet的均值和标准差
])

# 加载 MNIST 数据集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


# 3. 训练模型
def train_model(model, trainloader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()  # 设定模型为训练模式
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()  # 清除梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}')


# 4. 测试模型
def test_model(model, testloader):
    model.eval()  # 设定模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on the test set: {100 * correct / total:.2f}%')


# 5. 加载并预测图像
def predict_image(model, image_path):
    image = Image.open(image_path).convert('L')  # 将图像转换为灰度图
    image = transform(image).unsqueeze(0).cuda()  # 转换为Tensor并增加batch维度
    output = model(image)
    _, predicted = torch.max(output, 1)
    print(f'Predicted digit: {predicted.item()}')


# 6. 保存与加载模型
def save_model(model, path='resnet18_mnist.pth'):
    torch.save(model.state_dict(), path)


def load_model(path='resnet18_mnist.pth'):
    model = ResNet18ForMNIST().cuda()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# 7. 入口函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型，损失函数和优化器
    model = ResNet18ForMNIST().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, trainloader, criterion, optimizer, num_epochs=5)

    # 测试模型
    test_model(model, testloader)

    # 保存模型
    save_model(model)

    # 图像预测
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f'Predicting digit for image: {image_path}')
        model = load_model()  # 加载已训练的模型
        predict_image(model, image_path)
    else:
        print('No image path provided for prediction.')


if __name__ == '__main__':
    main()
