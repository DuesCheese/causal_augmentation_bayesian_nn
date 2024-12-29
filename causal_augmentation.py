# causal_augmentation.py

import numpy as np
import pandas as pd

def intervene_on_X1(original_df, x1_new_values=[-1.5, 0.0, 1.5]):
    """
    对原始数据集中每个样本，生成多个干预版本：
    - 将 X1 替换为 x1_new_value
    - 根据  X2 = 0.5 * X1 + e2 的因果机制重新采样 e2 并更新 X2
    - 最后根据  Y = 2*X1 + 1.5*X2 + ey 重新采样 ey 并更新 Y
    
    x1_new_values 可以是多个固定值，也可以是一批随机采样值。
    """
    augmented_rows = []
    
    for idx, row in original_df.iterrows():
        for x1_val in x1_new_values:
            # 干预：X1 <- x1_val
            x1_new = x1_val
            
            # 重新采样 e2
            e2 = np.random.normal(0.0, 0.2)
            x2_new = 0.5 * x1_new + e2
            
            # 对 X3 不做干预, 继续使用原值(或也可重新采样)
            x3_new = row['X3']
            
            # 重新采样 ey
            ey = np.random.normal(0.0, 0.2)
            y_new = 2.0 * x1_new + 1.5 * x2_new + ey
            
            augmented_rows.append({
                'X1': x1_new,
                'X2': x2_new,
                'X3': x3_new,
                'Y':  y_new
            })
    
    augmented_df = pd.DataFrame(augmented_rows)
    return augmented_df


if __name__ == "__main__":
    # 测试：从data_generator导入数据，做个干预增广
    from data_generator import generate_synthetic_data
    
    df_original = generate_synthetic_data(n_samples=5)
    print("Original Data:")
    print(df_original)
    
    df_aug = intervene_on_X1(df_original)
    print("\nAugmented Data (by intervening on X1):")
    print(df_aug.head(15))
