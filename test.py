import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 加载数据集
data = fetch_california_housing()
X, y = data.data, data.target

# 2. 数据预处理
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化输入特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为 PyTorch 的 Tensor 格式
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # 添加一个维度
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 3. 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # 输入特征数，输出为1（预测值）

    def forward(self, x):
        return self.linear(x)

# 4. 初始化模型
input_dim = X_train.shape[1]
model = LinearRegressionModel(input_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. 训练模型
epochs = 100
for epoch in range(epochs):
    # 前向传播
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 6. 测试模型
with torch.no_grad():
    y_pred = model(X_test_tensor)
    test_loss = criterion(y_pred, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")

# 7. 打印预测结果（部分）
print("\nSample Predictions:")
print(f"True Values: {y_test[:5]}")
print(f"Predicted Values: {y_pred[:5].squeeze().numpy()}")
