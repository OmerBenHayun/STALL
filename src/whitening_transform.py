import torch


class WhiteningTransform:
    def __init__(self, data=None, whitening_matrix=None, mean=None, n_components=None):
        # If data is numpy array, convert to torch tensor
        if data is not None and not isinstance(data, torch.Tensor):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            data = torch.tensor(data, dtype=torch.float32, device=device)

        self.n_components = n_components
        self.fitted = False
        self.truncated_ = False
        self.used_components_ = None
        self.fit(data)
        self.fitted = True

    def to(self, device):
        """Move the instance to the specified device."""
        self.mean_ = self.mean_.to(device)
        self.whitening_matrix_ = self.whitening_matrix_.to(device)
        if hasattr(self, 'eigenvalues_'):
            self.eigenvalues_ = self.eigenvalues_.to(device)
        if hasattr(self, 'eigenvectors_'):
            self.eigenvectors_ = self.eigenvectors_.to(device)
        return self

    def fit(self, X):
        if self.fitted:
            raise ValueError("Instance is already fitted.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("`X` must be a torch.Tensor")

        N, D = X.shape
        self.mean_ = X.mean(dim=0)
        X.sub_(self.mean_) # in-place to save memory (X can be large) [We will add back the mean later]

        cov = torch.cov(X.T)
        eigenvalues_complex, eigenvectors_complex = torch.linalg.eigh(cov)

        eigenvalues = eigenvalues_complex.real
        eigenvectors = eigenvectors_complex.real
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        max_rank = min(N - 1, D)
        self.truncated_ = max_rank < D
        eigenvalues = eigenvalues[:max_rank]
        eigenvectors = eigenvectors[:, :max_rank]

        if self.n_components is not None:
            r = min(self.n_components, max_rank)
            eigenvalues = eigenvalues[:r]
            eigenvectors = eigenvectors[:, :r]
        else:
            r = max_rank

        self.used_components_ = r

        # Filter out invalid (non-positive) eigenvalues before whitening
        # --------------------------------------------------------------
        # In theory, covariance matrices are positive semi-definite, so all eigenvalues λ ≥ 0.
        # However, in practice you can get small *negative* eigenvalues due to:
        #   • floating-point rounding errors (especially in float32 / float16),
        #   • tiny asymmetry in C = XᵀX / (N−1) if it isn’t perfectly symmetrized,
        #   • accumulated precision loss when using mixed-precision or AMP (automatic mixed precision),
        #   • or extremely small numerical noise in ill-conditioned data.
        #
        # If we try to take 1/√λ for λ ≤ 0, we’ll get NaN or Inf values in the whitening matrix.
        # To prevent this, we mask out all eigenvalues that are zero or negative before inversion.
        # Most of the time this won’t happen, but it’s a safeguard for rare edge cases.

        valid_mask = eigenvalues > 0
        if not torch.any(valid_mask):
            raise ValueError("All eigenvalues are too small or non-positive.")

        # Keep only stable eigenvectors/values
        eigenvalues = eigenvalues[valid_mask]
        eigenvectors = eigenvectors[:, valid_mask]


        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        diag_mat = torch.diag(1.0 / torch.sqrt(eigenvalues + 1e-5))

        self.whitening_matrix_ = eigenvectors @ diag_mat

        # Add again the mean to X
        X.add_(self.mean_)  # in-place to save memory [Here, we added back the mean]

    def transform_numpy(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32,device=self.mean_.device)
        return self.transform(X).numpy()

    def transform(self, X):
        if not self.fitted:
            raise ValueError("Call fit first")
        device = self.mean_.device
        X = X.to(device)
        X_centered = X - self.mean_
        return X_centered @ self.whitening_matrix_

    def get_eigenvalues(self):
        """
        Get the eigenvalues used in the whitening transformation.
        
        Returns:
            torch.Tensor: Eigenvalues in descending order Shape: (d,)
                         where d is the number of used components.
        
        Raises:
            ValueError: If the transform has not been fitted yet.
        """
        if not self.fitted:
            raise ValueError("Transform must be fitted before accessing eigenvalues")
        return self.eigenvalues_

    def get_eigenvectors(self):
        """
        Get the eigenvectors used in the whitening transformation.
        
        Returns:
            torch.Tensor: Eigenvectors in descending order.
                          Shape: (D, d)
                          where d is the number of used components.
        
        Raises:
            ValueError: If the transform has not been fitted yet.
        """
        if not self.fitted:
            raise ValueError("Transform must be fitted before accessing eigenvectors")
        return self.eigenvectors_

    def truncation_info(self):
        """
        Returns:
            truncated (bool): True if rank-truncation happened.
            used_components (int): final number of kept eigen-vectors.
        """
        return self.truncated_, self.used_components_
