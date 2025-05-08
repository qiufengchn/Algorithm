import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. 定义一个简单的神经网络模型
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 2. 设置参数并生成随机数据
input_size = 10  # 输入特征数量
hidden_size = 20 # 隐藏层大小
output_size = 1  # 输出大小 (例如，回归问题)
num_samples = 100 # 样本数量
learning_rate = 0.01
num_epochs = 100 # 训练轮数

# 随机生成输入数据和目标标签
X_train = torch.randn(num_samples, input_size).float()
y_train = torch.randn(num_samples, output_size).float()

# 3. 实例化模型、损失函数和优化器
model = SimpleNet(input_size, hidden_size, output_size)
criterion = nn.MSELoss() # 均方误差损失，适用于回归问题
optimizer = optim.SGD(model.parameters(), lr=learning_rate) # 随机梯度下降

# 4. 创建存储训练过程数据的列表
loss_values = []
fc1_weights = []
fc2_weights = []

# 5. 训练循环
print("开始训练...")
plt.figure(figsize=(12, 8))

for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # 反向传播和优化
    optimizer.zero_grad() # 清除之前的梯度
    loss.backward()      # 计算梯度
    optimizer.step()     # 更新权重
    
    # 记录训练数据
    loss_values.append(loss.item())
    fc1_weights.append(model.fc1.weight[0, 0].item())
    fc2_weights.append(model.fc2.weight[0, 0].item())

    # 每隔10轮打印一次损失
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        # 每10轮绘制一次实时图表
        if (epoch+1) % 10 == 0:
            plt.clf()  # 清除之前的图
            
            # 1. 损失曲线
            plt.subplot(2, 2, 1)
            plt.plot(loss_values)
            plt.title('训练损失变化')
            plt.xlabel('训练轮次')
            plt.ylabel('损失值')
            plt.grid(True)
            
            # 2. FC1权重变化
            plt.subplot(2, 2, 2)
            plt.plot(fc1_weights)
            plt.title('第一层权重变化')
            plt.xlabel('训练轮次')
            plt.ylabel('权重值')
            plt.grid(True)
            
            # 3. FC2权重变化
            plt.subplot(2, 2, 3)
            plt.plot(fc2_weights)
            plt.title('第二层权重变化')
            plt.xlabel('训练轮次')
            plt.ylabel('权重值')
            plt.grid(True)
            
            # 4. 预测值与真实值对比
            with torch.no_grad():
                predictions = model(X_train[:20])
            
            plt.subplot(2, 2, 4)
            plt.scatter(range(20), y_train[:20].numpy(), color='blue', label='真实值')
            plt.scatter(range(20), predictions.numpy(), color='red', label='预测值')
            plt.legend()
            plt.title('预测值与真实值对比')
            plt.xlabel('样本索引')
            plt.ylabel('值')
            plt.grid(True)
            
            plt.tight_layout()
            plt.pause(0.1)  # 暂停一小段时间以便观察图形

print("训练完成!")

# 最终训练结果可视化
plt.figure(figsize=(12, 10))

# 1. 最终损失曲线
plt.subplot(2, 2, 1)
plt.plot(loss_values)
plt.title('最终训练损失曲线')
plt.xlabel('训练轮次')
plt.ylabel('损失值')
plt.grid(True)

# 2. 权重变化曲线
plt.subplot(2, 2, 2)
plt.plot(fc1_weights, label='FC1首个权重')
plt.plot(fc2_weights, label='FC2首个权重')
plt.title('模型权重变化')
plt.xlabel('训练轮次')
plt.ylabel('权重值')
plt.legend()
plt.grid(True)

# 3. 最终预测结果
with torch.no_grad():
    predictions = model(X_train)

plt.subplot(2, 2, 3)
plt.scatter(y_train.numpy(), predictions.numpy())
plt.plot([y_train.min().item(), y_train.max().item()], 
         [y_train.min().item(), y_train.max().item()], 'r--')
plt.title('预测值 vs 真实值')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.grid(True)

# 4. 预测误差直方图
plt.subplot(2, 2, 4)
errors = predictions - y_train
plt.hist(errors.numpy(), bins=20)
plt.title('预测误差分布')
plt.xlabel('预测误差')
plt.ylabel('频次')
plt.grid(True)

plt.tight_layout()
plt.show()

# 测试单个样例
with torch.no_grad():
    test_input = torch.randn(1, input_size).float()
    prediction = model(test_input)
    print(f"\n测试输入: {test_input}")
    print(f"模型预测输出: {prediction}")