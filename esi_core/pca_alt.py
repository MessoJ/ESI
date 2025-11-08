import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class PCAResult:
    def __init__(self, pc1_series: pd.Series, explained_variance_series: pd.Series, loadings_df: pd.DataFrame):
        self.pc1_series = pc1_series
        self.explained_variance_series = explained_variance_series
        self.loadings_df = loadings_df


def compute_pca_index(component_df: pd.DataFrame, n_components: int = 1, rolling_window: int | None = None) -> PCAResult:
    zcols = [c for c in component_df.columns if c.startswith('z_')]
    if not zcols:
        raise ValueError('no z_ columns present')
    Z = component_df[zcols]
    if rolling_window is None or rolling_window <= 0:
        Z_drop = Z.dropna()
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(Z_drop.values)
        pc1 = pd.Series(pcs[:, 0], index=Z_drop.index)
        # normalize pc1
        pc1 = (pc1 - pc1.mean()) / (pc1.std() + 1e-12)
        exp_var = pd.Series(np.repeat(pca.explained_variance_ratio_[0], len(pc1)), index=pc1.index)
        loadings = pd.DataFrame(pca.components_.T, index=zcols, columns=[f'PC{i+1}' for i in range(n_components)])
        return PCAResult(pc1_series=pc1.reindex(component_df.index), explained_variance_series=exp_var.reindex(component_df.index), loadings_df=loadings)
    else:
        # rolling PCA
        pc1_out = []
        exp_out = []
        idx = Z.index
        for i in range(len(idx)):
            start = max(0, i - rolling_window + 1)
            window_Z = Z.iloc[start:i+1].dropna()
            if len(window_Z) < max(20, len(zcols)):
                pc1_out.append(np.nan)
                exp_out.append(np.nan)
                continue
            pca = PCA(n_components=n_components)
            pcs = pca.fit_transform(window_Z.values)
            pc1_val = pcs[-1, 0]
            pc1_out.append(pc1_val)
            exp_out.append(pca.explained_variance_ratio_[0])
        pc1 = pd.Series(pc1_out, index=idx)
        pc1 = (pc1 - pc1.rolling(rolling_window, min_periods=20).mean()) / (pc1.rolling(rolling_window, min_periods=20).std() + 1e-12)
        exp_var = pd.Series(exp_out, index=idx)
        # last window loadings as reference
        loadings = pd.DataFrame(index=zcols)
        return PCAResult(pc1_series=pc1, explained_variance_series=exp_var, loadings_df=loadings)


