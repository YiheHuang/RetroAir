"""
baseline.py —— 传统机器学习基线模型
  - RandomForestModel (M1)
  - XGBoostModel      (M2)
每个模型实现统一的 fit / predict / save / load 接口
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor


class BaseModel:
    """所有模型的统一接口"""

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def evaluate(self, X, y, prefix="") -> dict:
        pred = self.predict(X)
        return {
            f"{prefix}r2": r2_score(y, pred),
            f"{prefix}rmse": np.sqrt(mean_squared_error(y, pred)),
        }


class RandomForestModel(BaseModel):
    """M1: Random Forest 回归器"""

    def __init__(self, n_estimators=200, max_depth=10, random_state=42):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_s = self.scaler.fit_transform(X_train)
        self.model.fit(X_s, y_train)
        return self

    def predict(self, X):
        X_s = self.scaler.transform(X)
        return self.model.predict(X_s)


class XGBoostModel(BaseModel):
    """M2: XGBoost 回归器"""

    def __init__(self, n_estimators=200, max_depth=6, lr=0.05, random_state=42):
        self.scaler = StandardScaler()
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_importances_ = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_s = self.scaler.fit_transform(X_train)
        eval_set = [(self.scaler.transform(X_val), y_val)] if X_val is not None else None
        self.model.fit(
            X_s, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X):
        X_s = self.scaler.transform(X)
        return self.model.predict(X_s)
