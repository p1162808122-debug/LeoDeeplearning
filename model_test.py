import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import torch.nn.functional as F
from model import AlexNet   # ✅ 改成你的 AlexNet 模型
import matplotlib.pyplot as plt
import numpy as np

# FashionMNIST 的 10 个类别
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def test_data_process():
    """加载 FashionMNIST 测试集"""
    transform = transforms.Compose([
        transforms.Resize(size=227),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    test_data = FashionMNIST(
        root='./data',
        train=False,
        transform=transform,
        download=True
    )
    test_dataloader = Data.DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    return test_dataloader


def test_model_process(model, test_dataloader):
    """评估模型在测试集上的整体准确率"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # ✅ 模型设为评估模式

    test_corrects = 0
    test_num = 0

    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x, test_data_y = test_data_x.to(device), test_data_y.to(device)
            output = model(test_data_x)
            pre_lab = torch.argmax(output, 1)
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            test_num += test_data_x.size(0)

    test_acc = test_corrects.double().item() / test_num
    print(f"\n📊 测试集总体准确率: {test_acc:.4f}")


def test_single_samples(model, test_dataloader, num_samples=10):
    """随机展示若干样本的预测结果"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (b_x, b_y) in enumerate(test_dataloader):
            if i >= num_samples:
                break
            b_x, b_y = b_x.to(device), b_y.to(device)
            output = model(b_x)
            pre_lab = torch.argmax(output, 1)
            result = pre_lab.item()
            label = b_y.item()
            print(f"样本 {i+1}: 预测 → {classes[result]}，真实 → {classes[label]}")

            # 可视化图像
            img = b_x.cpu().squeeze(0)  # 去掉batch维度，只保留[C,H,W]或[H,W]

            if img.dim() == 3:  # 彩色图像（如AlexNet输入）
                img = img.permute(1, 2, 0).numpy()
                plt.imshow(img)
            elif img.dim() == 2:  # 灰度图像
                img = img.numpy()
                plt.imshow(img, cmap='gray')
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")

            plt.title(f"Pred: {classes[result]} | True: {classes[label]}")
            plt.axis('off')
            plt.show()


if __name__ == "__main__":
    model = AlexNet()
    model.load_state_dict(torch.load('./best_alexnet_model.pth', map_location='cpu'))
    test_dataloader = test_data_process()
    test_model_process(model, test_dataloader)
    test_single_samples(model, test_dataloader, num_samples=5)
