import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import sys


# 1. 定义卷积神经网络（CNN）
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)

        x = x.view(-1, 128 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# 2. 数据预处理与加载
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图
                                transforms.Resize((28, 28)),  # 调整大小到28x28
                                transforms.ToTensor(),  # 转换为Tensor
                                transforms.Normalize((0.5,), (0.5,))])  # 标准化

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


# 3. 训练模型
def train_model(model, trainloader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}')


# 4. 测试模型
def test_model(model, testloader):
    model.eval()
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


# 5. 加载保存的模型并进行预测
def predict_image(model, image_path):
    image = Image.open(image_path).convert('L')  # 转换为灰度图
    image = transform(image).unsqueeze(0).cuda()  # 转换为Tensor并增加batch维度
    output = model(image)
    _, predicted = torch.max(output, 1)
    print(f'Predicted digit: {predicted.item()}')


# 6. 保存和加载模型
def save_model(model, path='mnist_cnn.pth'):
    torch.save(model.state_dict(), path)


def load_model(path='mnist_cnn.pth'):
    model = SimpleCNN().cuda()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# 7. 入口函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型，定义损失函数和优化器
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #
    # # 训练模型
    # train_model(model, trainloader, criterion, optimizer, num_epochs=5)
    #
    # # 测试模型
    # test_model(model, testloader)
    #
    # # 保存模型
    # save_model(model)

    # 处理图片预测

    # image_path = sys.argv[1]
    image_path = r'C:\work\team\blcok\1111_PIL.png'
    print(f'Predicting digit for image: {image_path}')
    model = load_model()  # 加载已训练好的模型
    predict_image(model, image_path)



if __name__ == '__main__':
    main()
