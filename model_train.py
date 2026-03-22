import copy
import time
import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
import pandas as pd
from model import AlexNet


def train_val_data_process(batch_size=128):
    """加载 FashionMNIST 并划分训练集与验证集"""
    transform = transforms.Compose([
        transforms.Resize((227, 227)),  # AlexNet 输入大小通常为 227x227
        transforms.ToTensor()
    ])
    full_train_data = FashionMNIST(root='./data',
                                   train=True,
                                   transform=transform,
                                   download=True)
    train_data, val_data = Data.random_split(full_train_data,
                                              [int(0.8*len(full_train_data)),
                                              int(0.2*len(full_train_data))])
    train_loader = Data.DataLoader(train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=0)
    val_loader = Data.DataLoader(val_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=0)

    return train_loader, val_loader

def train_model_process(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    """训练并验证 AlexNet 模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_all, val_loss_all = [], []
    train_acc_all, val_acc_all = [], []

    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-"*30)

        # --- 训练阶段 ---
        model.train()
        train_loss, train_corrects, train_num = 0.0, 0, 0

        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)

            optimizer.zero_grad()
            outputs = model(b_x)
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(preds == b_y).item()
            train_num += b_x.size(0)

        # --- 验证阶段 ---
        model.eval()
        val_loss, val_corrects, val_num = 0.0, 0, 0

        with torch.no_grad():
            for b_x, b_y in val_loader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                outputs = model(b_x)
                loss = criterion(outputs, b_y)
                preds = torch.argmax(outputs, dim=1)

                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(preds == b_y).item()
                val_num += b_x.size(0)

        # 记录每轮数据
        train_loss_epoch = train_loss / train_num
        val_loss_epoch = val_loss / val_num
        train_acc_epoch = train_corrects / train_num
        val_acc_epoch = val_corrects / val_num

        train_loss_all.append(train_loss_epoch)
        val_loss_all.append(val_loss_epoch)
        train_acc_all.append(train_acc_epoch)
        val_acc_all.append(val_acc_epoch)

        print(f"Train Loss: {train_loss_epoch:.4f}  Train Acc: {train_acc_epoch:.4f}")
        print(f"Val Loss: {val_loss_epoch:.4f}  Val Acc: {val_acc_epoch:.4f}")

        # 更新最优模型
        if val_acc_epoch > best_acc:
            best_acc = val_acc_epoch
            best_model_wts = copy.deepcopy(model.state_dict())

    time_use = time.time() - since
    print(f"训练和验证耗时: {int(time_use//60)}m {int(time_use%60)}s")

    # 保存最优模型
    torch.save(best_model_wts, './best_alexnet_model.pth')

    # 保存训练记录
    train_process = pd.DataFrame({
        'epoch': range(num_epochs),
        'train_loss_all': train_loss_all,
        'train_acc_all': train_acc_all,
        'val_loss_all': val_loss_all,
        'val_acc_all': val_acc_all
    })

    return train_process


def matplot_acc_loss(train_process, save_path=None):
    """可视化训练过程"""
    plt.figure(figsize=(12, 4))

    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process['train_loss_all'], 'r-', label='Train Loss')
    plt.plot(train_process['epoch'], train_process['val_loss_all'], 'b-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process['train_acc_all'], 'r-', label='Train Acc')
    plt.plot(train_process['epoch'], train_process['val_acc_all'], 'b-', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    train_loader, val_loader = train_val_data_process(batch_size=128)
    model = AlexNet()  # FashionMNIST 10 类
    train_process = train_model_process(model, train_loader, val_loader, num_epochs=2)
    matplot_acc_loss(train_process)
