"""
FSAA (Feature Selection based on Augmentation and Asymmetry)
基于特征增强和非对称性的特征选择方法
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class FSAASelector:
    """
    FSAA特征选择器
    基于信息熵准则去除冗余特征，筛选保留因果关系的变量子集
    """

    def __init__(self,
                 threshold: float = 0.1,
                 alpha: float = 0.05,
                 max_lag: int = 3,
                 n_permutations: int = 100):
        """
        初始化FSAA选择器

        参数:
            threshold: 信息增益阈值
            alpha: 显著性水平
            max_lag: 最大滞后阶数
            n_permutations: 置换检验次数
        """
        self.threshold = threshold
        self.alpha = alpha
        self.max_lag = max_lag
        self.n_permutations = n_permutations
        self.selected_features_ = None
        self.feature_scores_ = None
        self.causal_scores_ = None

    def _calculate_entropy(self, x: np.ndarray) -> float:
        """计算时间序列的近似熵"""
        if len(x) < 2:
            return 0
        # 使用直方图方法估计熵
        hist, bin_edges = np.histogram(x, bins='auto', density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist)) * (bin_edges[1] - bin_edges[0])

    def _calculate_transfer_entropy(self, x: np.ndarray, y: np.ndarray, lag: int = 1) -> float:
        """
        计算转移熵 (Transfer Entropy)
        TE_{X->Y} = H(Y_t|Y_{t-1}) - H(Y_t|Y_{t-1}, X_{t-lag})
        """
        n = len(x) - lag

        if n <= 0:
            return 0

        # 创建条件概率的样本
        y_present = y[lag:]
        y_past = y[lag - 1:-1]
        x_past = x[:-(lag)]

        # 使用互信息近似转移熵
        # TE = I(Y_t; X_{t-lag} | Y_{t-1})
        # 使用条件互信息计算
        mi_conditional = self._conditional_mutual_information(y_present, x_past, y_past)

        return max(0, mi_conditional)

    def _conditional_mutual_information(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """计算条件互信息 I(X;Y|Z)"""
        # 使用k近邻方法近似计算
        from sklearn.neighbors import NearestNeighbors

        n_samples = len(x)
        if n_samples < 10:
            return 0

        # 创建联合向量
        xyz = np.column_stack([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)])
        xz = np.column_stack([x.reshape(-1, 1), z.reshape(-1, 1)])
        yz = np.column_stack([y.reshape(-1, 1), z.reshape(-1, 1)])

        # k值选择
        k = min(3, n_samples // 10)

        # 计算距离
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev').fit(xyz)
        distances, _ = nbrs.kneighbors(xyz)
        epsilon = distances[:, -1]

        # 计算各种情况下的计数
        n_xz = np.zeros(n_samples)
        n_yz = np.zeros(n_samples)
        n_z = np.zeros(n_samples)

        for i in range(n_samples):
            n_xz[i] = np.sum(np.max(np.abs(xz - xz[i]), axis=1) <= epsilon[i])
            n_yz[i] = np.sum(np.max(np.abs(yz - yz[i]), axis=1) <= epsilon[i])
            n_z[i] = np.sum(np.abs(z - z[i]) <= epsilon[i])

        # 计算条件互信息
        cmi = np.mean(np.log((n_z + 1e-10) * (n_samples + 1e-10) /
                             ((n_xz + 1e-10) * (n_yz + 1e-10) + 1e-10)))

        return max(0, cmi)

    def _feature_augmentation(self, data: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        特征增强：创建滞后特征和非线性变换
        """
        n_samples, n_features = data.shape
        augmented_features = []
        augmented_names = []

        # 原始特征
        augmented_features.append(data)
        augmented_names.extend(feature_names)

        # 滞后特征
        for lag in range(1, self.max_lag + 1):
            lagged_data = np.zeros((n_samples, n_features))
            lagged_data[lag:, :] = data[:-lag, :]
            augmented_features.append(lagged_data)
            augmented_names.extend([f"{name}_lag{lag}" for name in feature_names])

        # 非线性变换特征
        # 平方项
        squared_data = data ** 2
        augmented_features.append(squared_data)
        augmented_names.extend([f"{name}^2" for name in feature_names])

        # 差分特征
        diff_data = np.zeros_like(data)
        diff_data[1:, :] = data[1:, :] - data[:-1, :]
        augmented_features.append(diff_data)
        augmented_names.extend([f"Δ{name}" for name in feature_names])

        # 移动平均特征
        window_size = min(5, n_samples // 10)
        if window_size > 1:
            ma_data = np.zeros_like(data)
            for i in range(window_size, n_samples):
                ma_data[i, :] = np.mean(data[i - window_size:i, :], axis=0)
            augmented_features.append(ma_data)
            augmented_names.extend([f"MA{window_size}({name})" for name in feature_names])

        # 组合所有增强特征
        augmented_data = np.hstack(augmented_features)

        return augmented_data, augmented_names

    def _calculate_causal_asymmetry(self, data: np.ndarray, feature_names: List[str]) -> Dict[Tuple[str, str], float]:
        """
        计算特征间的因果非对称性分数
        """
        n_features = data.shape[1]
        causal_scores = {}

        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    continue

                # 计算两个方向的转移熵
                te_i_j = self._calculate_transfer_entropy(data[:, i], data[:, j], lag=1)
                te_j_i = self._calculate_transfer_entropy(data[:, j], data[:, i], lag=1)

                # 计算非对称性分数
                if te_i_j + te_j_i > 0:
                    asymmetry_score = abs(te_i_j - te_j_i) / (te_i_j + te_j_i)
                    causal_score = asymmetry_score * (te_i_j + te_j_i)  # 综合分数

                    causal_scores[(feature_names[i], feature_names[j])] = causal_score

        return causal_scores

    def _permutation_test(self, data: np.ndarray, feature_idx: int, target_idx: int) -> float:
        """
        置换检验计算p值
        """
        n_samples = data.shape[0]
        original_te = self._calculate_transfer_entropy(data[:, feature_idx],
                                                       data[:, target_idx],
                                                       lag=1)

        # 生成置换样本
        permuted_tes = []
        for _ in range(self.n_permutations):
            # 置换特征序列
            permuted_feature = np.random.permutation(data[:, feature_idx])
            permuted_te = self._calculate_transfer_entropy(permuted_feature,
                                                           data[:, target_idx],
                                                           lag=1)
            permuted_tes.append(permuted_te)

        # 计算p值
        p_value = np.mean(np.array(permuted_tes) >= original_te)

        return p_value

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        拟合FSAA模型并选择特征

        参数:
            X: 输入特征DataFrame
            y: 目标变量（可选，用于监督特征选择）
        """
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 特征增强
        print("进行特征增强...")
        X_augmented, augmented_names = self._feature_augmentation(X_scaled, X.columns.tolist())

        # 计算因果非对称性
        print("计算因果非对称性...")
        causal_scores = self._calculate_causal_asymmetry(X_augmented, augmented_names)
        self.causal_scores_ = causal_scores

        # 计算特征重要性分数
        n_augmented = X_augmented.shape[1]
        feature_scores = np.zeros(n_augmented)

        print("计算特征重要性...")
        for i in range(n_augmented):
            # 计算特征的信息增益
            if y is not None:
                # 监督模式：使用互信息
                mi = mutual_info_regression(X_augmented[:, i].reshape(-1, 1),
                                            y.values)
                feature_scores[i] = mi[0]
            else:
                # 无监督模式：使用自信息和因果强度
                # 计算自信息（熵）
                entropy = self._calculate_entropy(X_augmented[:, i])

                # 计算特征的因果影响力
                causal_influence = 0
                for j in range(n_augmented):
                    if i != j:
                        key = (augmented_names[i], augmented_names[j])
                        if key in causal_scores:
                            causal_influence += causal_scores[key]

                # 综合分数：高熵 + 高因果影响力
                feature_scores[i] = entropy * (1 + causal_influence / (n_augmented - 1))

        self.feature_scores_ = dict(zip(augmented_names, feature_scores))

        # 选择特征
        print("选择特征...")
        sorted_features = sorted(self.feature_scores_.items(),
                                 key=lambda x: x[1],
                                 reverse=True)

        # 基于阈值选择特征
        max_score = max(feature_scores)
        selected = [name for name, score in sorted_features
                    if score >= self.threshold * max_score]

        # 确保选择原始特征（去除增强特征）
        original_selected = []
        for feature in selected:
            base_name = feature.split('_')[0].split('^')[0].replace('Δ', '').replace('MA5(', '').replace(')', '')
            if base_name in X.columns and base_name not in original_selected:
                original_selected.append(base_name)

        self.selected_features_ = original_selected

        print(f"原始特征数: {len(X.columns)}")
        print(f"增强后特征数: {len(augmented_names)}")
        print(f"选择特征数: {len(self.selected_features_)}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换数据，只保留选择的特征"""
        if self.selected_features_ is None:
            raise ValueError("必须先调用fit方法")

        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """拟合并转换数据"""
        self.fit(X, y)
        return self.transform(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性"""
        if self.feature_scores_ is None:
            raise ValueError("必须先调用fit方法")

        return pd.DataFrame({
            'feature': list(self.feature_scores_.keys()),
            'importance': list(self.feature_scores_.values())
        }).sort_values('importance', ascending=False)