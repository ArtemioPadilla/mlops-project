import pandas as pd
import numpy as np

# ================================
# 1. DataLoader
# ================================
class DataLoader:
    def load_csv(self, path):
        return pd.read_csv(path)

    def save_csv(self, df, path):
        df.to_csv(path, index=False)
        print(f"💾 Guardado en {path} (shape={df.shape})")

