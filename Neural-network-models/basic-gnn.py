# gnn.ipynb
import torch
import torch.nn as nn
import torch.optim as optim

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
# 使用 .float() 确保数据类型是 FloatTensor
X_train = torch.randn(num_samples, input_size).float()
# 假设目标是输入的某种线性组合加上一些噪声
# 这里简化为随机目标，实际应用中应根据问题定义
y_train = torch.randn(num_samples, output_size).float()

# 3. 实例化模型、损失函数和优化器
model = SimpleNet(input_size, hidden_size, output_size)
criterion = nn.MSELoss() # 均方误差损失，适用于回归问题
optimizer = optim.SGD(model.parameters(), lr=learning_rate) # 随机梯度下降

# 4. 训练循环
print("开始训练...")
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # 反向传播和优化
    optimizer.zero_grad() # 清除之前的梯度
    loss.backward()      # 计算梯度
    optimizer.step()     # 更新权重

    # 每隔10轮打印一次损失
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("训练完成!")

# (可选) 查看模型的部分预测结果
with torch.no_grad(): # 在评估/预测时不计算梯度
    test_input = torch.randn(1, input_size).float()
    prediction = model(test_input)
    print(f"\n测试输入: {test_input}")
    print(f"模型预测输出: {prediction}")