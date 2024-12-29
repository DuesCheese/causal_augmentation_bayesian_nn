# data_generator.py

import numpy as np
import pandas as pd

def generate_synthetic_data(n_samples=200, seed=123):
    np.random.seed(seed)
    
    # 1. X1 ~ Uniform(-2, 2)
    X1 = np.random.uniform(-2, 2, size=n_samples)
    
    # 2. X2 = 0.5 * X1 + e2, e2 ~ Normal(0, 0.2)
    e2 = np.random.normal(loc=0.0, scale=0.2, size=n_samples)
    X2 = 0.5 * X1 + e2
    
    # 3. X3 ~ Normal(0, 1) (irrelevant noise)
    X3 = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
    
    # 4. Y = 2*X1 + 1.5*X2 + ey, ey ~ Normal(0, 0.2)
    ey = np.random.normal(loc=0.0, scale=0.2, size=n_samples)
    Y = 2.0 * X1 + 1.5 * X2 + ey
    
    # 组合到 DataFrame
    data = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'Y' : Y
    })
    return data

if __name__ == "__main__":
    # 简单测试
    df = generate_synthetic_data(n_samples=10)
    print(df.head(10))
