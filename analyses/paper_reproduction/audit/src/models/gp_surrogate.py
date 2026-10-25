"""
Gaussian Process Surrogate Models for Active Learning Reproduction & Audit.
Chen et al. (Adv. Funct. Mater. 2026, 36:e75639) Reproduction & Audit.
"""

from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood

# Standard formulation feature bounds from Table S1
DEFAULT_BOUNDS = np.array([
    [0.0, 600.0],  # Exposure time (s)
    [0.0, 6.0],    # GelMA-L (wt%)
    [0.0, 6.0],    # GelMA-H (wt%)
    [0.0, 6.0],    # aCNC (wt%)
    [0.0, 0.2],    # LAP (wt%)
], dtype=np.float64)


class GPSurrogate:
    """
    Unified Gaussian Process Surrogate wrapping BoTorch / GPyTorch SingleTaskGP.
    Supports 8 configurations:
      kernel: 'matern52' | 'rbf'
      ard: True | False
      noise: 'learned' | 'replicate'
    """

    def __init__(
        self,
        kernel: str = "matern52",
        ard: bool = True,
        noise_model: str = "learned",
        fixed_noise_std: Optional[float] = None,
        bounds: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None,
    ):
        self.kernel_type = kernel.lower()
        if self.kernel_type not in ("matern52", "rbf"):
            raise ValueError(f"Unknown kernel_type: {kernel}. Use 'matern52' or 'rbf'.")
        self.ard = ard
        self.noise_model = noise_model.lower()
        if self.noise_model not in ("learned", "replicate"):
            raise ValueError(f"Unknown noise_model: {noise_model}. Use 'learned' or 'replicate'.")
        self.fixed_noise_std = fixed_noise_std
        self.bounds = bounds if bounds is not None else DEFAULT_BOUNDS
        self.random_seed = random_seed

        self.model: Optional[SingleTaskGP] = None
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def normalize_x(self, X: np.ndarray) -> np.ndarray:
        """Normalize X to [0, 1]^d using parameter bounds."""
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        scale = np.where((upper - lower) == 0, 1.0, upper - lower)
        return (X - lower) / scale

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPSurrogate":
        """
        Fit GP model to training inputs X (N, d) and targets y (N,).
        """
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        self.X_train = X.copy()
        self.y_train = y.copy()
        N, d = X.shape

        # Standardize target
        self.y_mean = float(np.mean(y))
        std_val = float(np.std(y))
        self.y_std = std_val if std_val > 1e-8 else 1.0

        y_standardized = (y - self.y_mean) / self.y_std
        X_normalized = self.normalize_x(X)

        t_X = torch.tensor(X_normalized, dtype=torch.double)
        t_Y = torch.tensor(y_standardized, dtype=torch.double).unsqueeze(-1)

        ard_num_dims = d if self.ard else None

        if self.kernel_type == "matern52":
            base_covar = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
        else:
            base_covar = RBFKernel(ard_num_dims=ard_num_dims)

        covar_module = ScaleKernel(base_covar)

        if self.noise_model == "learned":
            likelihood = GaussianLikelihood()
            self.model = SingleTaskGP(
                train_X=t_X,
                train_Y=t_Y,
                likelihood=likelihood,
                covar_module=covar_module,
            )
        else:
            # Fixed replicate noise
            if self.fixed_noise_std is not None:
                noise_std_stdized = self.fixed_noise_std / self.y_std
            else:
                # Default replicate noise relative to y std (~10% relative standard error)
                noise_std_stdized = 0.10
            noise_var = max(noise_std_stdized ** 2, 1e-6)
            train_Yvar = torch.full_like(t_Y, noise_var)

            self.model = SingleTaskGP(
                train_X=t_X,
                train_Y=t_Y,
                train_Yvar=train_Yvar,
                covar_module=covar_module,
            )

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass

        return self

    def predict(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation for new inputs in physical units.
        Returns:
          mu: (M,) in physical target units
          sigma: (M,) in physical target units
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X_new = np.asarray(X_new, dtype=np.float64)
        X_norm = self.normalize_x(X_new)
        t_X = torch.tensor(X_norm, dtype=torch.double)

        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(t_X)
            mean_stdized = posterior.mean.detach().cpu().numpy().ravel()
            var_stdized = posterior.variance.detach().cpu().numpy().ravel()

        var_stdized = np.clip(var_stdized, 1e-12, None)
        std_stdized = np.sqrt(var_stdized)

        # Unstandardize to physical units
        mu_phys = mean_stdized * self.y_std + self.y_mean
        sigma_phys = std_stdized * self.y_std

        return mu_phys, sigma_phys

    def get_lengthscales(self) -> np.ndarray:
        """Return array of learned lengthscales in normalized [0, 1] unit space."""
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        ls = self.model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
        return ls

    def get_outputscale(self) -> float:
        """Return learned signal variance (outputscale)."""
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        return float(self.model.covar_module.outputscale.detach().cpu().item())

    def get_noise_variance(self) -> float:
        """Return noise variance in standardized space."""
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        if hasattr(self.model.likelihood, "noise"):
            noise = self.model.likelihood.noise.detach().cpu()
            return float(noise.mean().item())
        if hasattr(self.model, "train_Yvar") and self.model.train_Yvar is not None:
            return float(self.model.train_Yvar.detach().cpu().mean().item())
        return 0.0
