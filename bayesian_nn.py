# bayesian_nn.py

import torch
import torch.nn as nn
import torch.optim as optim
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

def analyze_data(df, x_cols, y_col):
    """分析数据分布"""
    print("\n数据分析:")
    print("\n特征统计:")
    print(df[x_cols + [y_col]].describe())
    
    # 检查是否需要标准化
    ranges = df[x_cols + [y_col]].max() - df[x_cols + [y_col]].min()
    print("\n数据范围:")
    for col in x_cols + [y_col]:
        print(f"{col}: [{df[col].min():.2f}, {df[col].max():.2f}], 范围: {ranges[col]:.2f}")
    
    return ranges.max() / ranges.min() > 10  # 如果最大范围/最小范围>10，建议标准化

# 一个简单的 MLP 用于回归
class SimpleRegressor(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[64, 128, 64]):  # 增大网络规模
        super(SimpleRegressor, self).__init__()
        
        # 构建多层网络
        layers = []
        prev_dim = input_dim
        
        # 添加多个隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.1))  # 添加dropout
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        # 将所有层组合成一个序列
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def model(net, x_data, y_data):
    # 放宽先验分布
    for name, param in net.named_parameters():
        loc = param.new_zeros(param.shape)
        scale = param.new_ones(param.shape) * 5.0  # 增大先验方差到5
        pyro.sample(name, dist.Normal(loc, scale).to_event(param.dim()))
    
    # 前向传播
    prediction = net(x_data)
    
    # 可学习的sigma
    sigma = pyro.param("sigma", 
                      torch.tensor(1.0),
                      constraint=dist.constraints.positive)
    
    with pyro.plate("data_plate", x_data.shape[0]):
        pyro.sample("obs", dist.Normal(prediction.squeeze(-1), sigma), obs=y_data)

def guide(net, x_data, y_data):
    for name, param in net.named_parameters():
        param_mean = pyro.param(f"{name}_mean", torch.randn_like(param))
        param_scale = pyro.param(f"{name}_scale",
                                 0.1 * torch.ones_like(param),
                                 constraint=dist.constraints.positive)
        pyro.sample(name,
                    dist.Normal(param_mean, param_scale).to_event(param.dim()))

def train_bnn(x_data, y_data, n_epochs=5000, lr=0.001):
    # 转换为torch的tensor
    x_tensor = torch.tensor(x_data.values, dtype=torch.float)
    y_tensor = torch.tensor(y_data.values, dtype=torch.float)
    
    # 初始化随机种子
    pyro.clear_param_store()
    net = SimpleRegressor(input_dim=x_data.shape[1], hidden_dims=[64, 128, 64])
    
    # 使用学习率调度器
    optimizer = ClippedAdam({"lr": lr})
    
    svi = SVI(lambda *args: model(net, *args),
              lambda *args: guide(net, *args),
              optimizer, loss=Trace_ELBO())
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    current_lr = lr
    
    # 训练
    for epoch in range(n_epochs):
        # 每1000个epoch将学习率减半
        if epoch > 0 and epoch % 1000 == 0:
            current_lr *= 0.5
            optimizer = ClippedAdam({"lr": current_lr})
            svi.optimizer = optimizer
            print(f"\n学习率调整为: {current_lr}")
        
        loss = svi.step(x_tensor, y_tensor)
        
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        if (epoch+1) % 200 == 0:
            sigma = pyro.param("sigma").item()
            print(f"Epoch [{epoch+1}/{n_epochs}] - Loss: {loss:.3f}, Sigma: {sigma:.3f}")
    
    return net

def train_baseline_models(X_train, y_train):
    """训练基线模型"""
    # 线性回归
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # 简单MLP
    mlp_model = MLPRegressor(hidden_layer_sizes=(64, 128, 64),
                            max_iter=1000,
                            early_stopping=True)
    mlp_model.fit(X_train, y_train)
    
    return lr_model, mlp_model

def evaluate_model(model_name, y_true, y_pred, std_pred=None):
    """评估模型性能"""
    mse = np.mean((y_pred - y_true)**2)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} 评估结果:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    if std_pred is not None:
        print(f"平均预测不确定性: {np.mean(std_pred):.4f} ± {np.std(std_pred):.4f}")

def predict_bnn(net, x_data, n_samples=100):
    """
    使用变分后验对网络参数进行采样，得到对同一输入的多次预测，实现不确定性估计
    """
    x_tensor = torch.tensor(x_data.values, dtype=torch.float)
    predictions = []
    for _ in range(n_samples):
        # 从变分分布guide中采样一次参数
        for name, param in net.named_parameters():
            param_mean = pyro.param(f"{name}_mean")
            param_scale = pyro.param(f"{name}_scale")
            sampled_param = dist.Normal(param_mean, param_scale).sample()
            # 用 sampled_param 替换 net中的权重
            param.data = sampled_param
        
        preds = net(x_tensor).detach().numpy().flatten()
        predictions.append(preds)
    
    predictions = np.array(predictions)  # shape = [n_samples, batch_size]
    mean_preds = predictions.mean(axis=0)
    std_preds = predictions.std(axis=0)
    return mean_preds, std_preds

if __name__ == "__main__":
    from data_generator import generate_synthetic_data
    from causal_augmentation import intervene_on_X1
    
    # 1. 生成原始数据
    print("生成数据...")
    df_original = generate_synthetic_data(n_samples=10000, seed=42)
    
    # 2. 做因果增广
    df_aug = intervene_on_X1(df_original, x1_new_values=[-2, -1, 0, 1, 2])
    
    # 3. 合并增广数据
    df_train = pd.concat([df_original, df_aug], ignore_index=True)
    
    x_cols = ['X1', 'X2', 'X3']
    y_col = 'Y'
    
    # 4. 数据分析和预处理
    need_scaling = analyze_data(df_train, x_cols, y_col)
    
    if need_scaling:
        print("\n执行数据标准化...")
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train = scaler_X.fit_transform(df_train[x_cols])
        y_train = scaler_y.fit_transform(df_train[[y_col]])
        
        X_test = scaler_X.transform(df_original[x_cols])
        y_test = scaler_y.transform(df_original[[y_col]])
        
        # 转换为DataFrame保持列名
        df_train_X = pd.DataFrame(X_train, columns=x_cols)
        df_train_y = pd.DataFrame(y_train, columns=[y_col])
    else:
        X_train = df_train[x_cols].values
        y_train = df_train[y_col].values
        X_test = df_original[x_cols].values
        y_test = df_original[y_col].values
        
        df_train_X = df_train[x_cols]
        df_train_y = df_train[y_col]
    
    # 5. 训练基线模型
    print("\n训练基线模型...")
    lr_model, mlp_model = train_baseline_models(X_train, y_train.ravel())
    
    # 6. 训练贝叶斯神经网络
    print("\n训练贝叶斯神经网络...")
    net_trained = train_bnn(df_train_X, df_train_y, n_epochs=5000, lr=0.001)
    
    # 7. 预测和评估
    print("\n模型评估...")
    
    # 基线模型评估
    lr_pred = lr_model.predict(X_test)
    mlp_pred = mlp_model.predict(X_test)
    
    # 贝叶斯神经网络预测
    mean_preds, std_preds = predict_bnn(net_trained, pd.DataFrame(X_test, columns=x_cols), n_samples=100)
    
    # 如果进行了标准化，需要将预测结果转换回原始尺度
    if need_scaling:
        lr_pred = scaler_y.inverse_transform(lr_pred.reshape(-1, 1)).ravel()
        mlp_pred = scaler_y.inverse_transform(mlp_pred.reshape(-1, 1)).ravel()
        mean_preds = scaler_y.inverse_transform(mean_preds.reshape(-1, 1)).ravel()
        std_preds = std_preds * scaler_y.scale_
    
    # 评估所有模型
    y_test = df_original[y_col].values
    evaluate_model("线性回归", y_test, lr_pred)
    evaluate_model("简单MLP", y_test, mlp_pred)
    evaluate_model("贝叶斯神经网络", y_test, mean_preds, std_preds)
    
    # 打印部分预测结果
    print("\n部分预测结果示例 (贝叶斯神经网络):")
    for i in range(5):
        print(f" TrueY={y_test[i]:.3f}, Pred={mean_preds[i]:.3f} ± {std_preds[i]:.3f}")
