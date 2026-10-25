"""
Digital Twin Models for Active Learning Reproduction & Benchmarking.
Reproduces upstream Gradient Boosting, SVR, and GP models from Chen et al. 2026.
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DEFAULT_GB_PARAMS = {
    "Modulus": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 30,
    },
    "Stress Relaxation": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 30,
    },
}


class DigitalTwinModel:
    """
    Digital twin regressor wrapping Gradient Boosting, SVR, or GP.
    Includes input standard scaling as done in upstream DigitalTwin/digital_twin.py.
    """

    def __init__(
        self,
        model_type: str = "GB",
        params: Optional[Dict[str, Any]] = None,
        random_state: int = 30,
    ):
        self.model_type = model_type.upper()
        self.random_state = random_state
        self.params = params or {}
        self.scaler = StandardScaler()
        self.model = self._create_model()

    def _create_model(self):
        if self.model_type == "GB":
            p = {
                "n_estimators": int(self.params.get("n_estimators", 100)),
                "learning_rate": float(self.params.get("learning_rate", 0.1)),
                "max_depth": int(self.params.get("max_depth", 5)),
                "min_samples_split": int(self.params.get("min_samples_split", 2)),
                "min_samples_leaf": int(self.params.get("min_samples_leaf", 1)),
                "random_state": self.random_state,
            }
            return GradientBoostingRegressor(**p)
        elif self.model_type == "SVR":
            p = {
                "C": float(self.params.get("C", 100.0)),
                "epsilon": float(self.params.get("epsilon", 0.1)),
                "gamma": self.params.get("gamma", "scale"),
            }
            return SVR(**p)
        elif self.model_type == "GP":
            ls = float(self.params.get("length_scale", 1.0))
            alpha = float(self.params.get("alpha", 1e-2))
            kernel = C(1.0) * RBF(length_scale=ls)
            return GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DigitalTwinModel":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X_test)
        y_test = np.asarray(y_test, dtype=np.float64).ravel()
        return {
            "r2": float(r2_score(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
        }
