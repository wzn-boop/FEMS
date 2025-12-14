"""
FEMS (Feature Engineering based Multi-model Selection) for Time Series Anomaly Detection
"""
import numpy as np
import pandas as pd
from scipy import stats, fftpack
import pywt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import warnings

warnings.filterwarnings('ignore')


class FEMSFeatureExtractor:
    """FEMS特征提取器 - 提取统计、频域和时频域特征"""

    def __init__(self, window_size=100, stride=50, wavelet='db4', level=3):
        """
        初始化FEMS特征提取器

        Args:
            window_size: 滑动窗口大小
            stride: 滑动步长
            wavelet: 小波类型
            level: 小波分解层数
        """
        self.window_size = window_size
        self.stride = stride
        self.wavelet = wavelet
        self.level = level
        self.scaler = StandardScaler()
        self.feature_names = []

    def extract_statistical_features(self, window):
        """提取统计特征"""
        features = []
        names = []

        # 基本统计特征
        features.extend([np.mean(window), np.std(window), np.min(window), np.max(window)])
        names.extend(['mean', 'std', 'min', 'max'])

        # 高阶统计特征
        features.extend([stats.skew(window), stats.kurtosis(window)])
        names.extend(['skewness', 'kurtosis'])

        # 百分位数特征
        percentiles = [25, 50, 75, 90, 95]
        for p in percentiles:
            features.append(np.percentile(window, p))
            names.append(f'percentile_{p}')

        # 极值统计
        features.append(np.max(window) - np.min(window))
        names.append('range')

        # 变异系数
        features.append(np.std(window) / (np.mean(window) + 1e-10))
        names.append('cv')

        return np.array(features), names

    def extract_frequency_features(self, window):
        """提取频域特征"""
        features = []
        names = []

        # FFT变换
        fft_vals = np.abs(fftpack.fft(window))
        fft_freq = fftpack.fftfreq(len(window))

        # 只取正频率部分
        positive_freq = fft_freq > 0
        fft_vals = fft_vals[positive_freq]

        if len(fft_vals) > 0:
            # 频域统计特征
            features.extend([np.mean(fft_vals), np.std(fft_vals), np.max(fft_vals)])
            names.extend(['fft_mean', 'fft_std', 'fft_max'])

            # 主频特征
            dominant_freq_idx = np.argmax(fft_vals)
            features.append(dominant_freq_idx / len(window))
            names.append('dominant_freq_ratio')

            # 频谱能量
            total_energy = np.sum(fft_vals ** 2)
            features.append(total_energy)
            names.append('total_energy')

            # 频谱熵
            spectral_power = fft_vals ** 2
            spectral_power = spectral_power / (np.sum(spectral_power) + 1e-10)
            spectral_entropy = -np.sum(spectral_power * np.log(spectral_power + 1e-10))
            features.append(spectral_entropy)
            names.append('spectral_entropy')

        return np.array(features), names

    def extract_time_frequency_features(self, window):
        """提取时频域特征（小波变换）"""
        features = []
        names = []

        try:
            # 小波分解
            coeffs = pywt.wavedec(window, self.wavelet, level=self.level)

            # 各层系数特征
            for i, coeff in enumerate(coeffs):
                if len(coeff) > 0:
                    features.extend([np.mean(coeff), np.std(coeff), np.sum(coeff ** 2)])
                    names.extend([f'wavelet_{i}_mean', f'wavelet_{i}_std', f'wavelet_{i}_energy'])

            # 小波熵
            wavelet_energies = [np.sum(c ** 2) for c in coeffs if len(c) > 0]
            total_energy = np.sum(wavelet_energies)
            if total_energy > 0:
                energy_ratios = [e / total_energy for e in wavelet_energies]
                wavelet_entropy = -np.sum([r * np.log(r + 1e-10) for r in energy_ratios])
                features.append(wavelet_entropy)
                names.append('wavelet_entropy')

        except:
            # 如果小波分解失败，使用零填充
            features = [0] * (3 * (self.level + 1) + 1)
            names = [f'wavelet_feature_{i}' for i in range(len(features))]

        return np.array(features), names

    def sliding_window_extraction(self, time_series):
        """滑动窗口特征提取"""
        n_samples = len(time_series)
        all_features = []
        window_indices = []

        # 获取第一个窗口的特征以确定特征维度
        test_window = time_series[:self.window_size]
        stat_features, stat_names = self.extract_statistical_features(test_window)
        freq_features, freq_names = self.extract_frequency_features(test_window)
        tf_features, tf_names = self.extract_time_frequency_features(test_window)

        # 合并特征名称
        self.feature_names = stat_names + freq_names + tf_names

        # 滑动窗口提取
        start = 0
        while start + self.window_size <= n_samples:
            window = time_series[start:start + self.window_size]

            # 提取各类特征
            stat_features, _ = self.extract_statistical_features(window)
            freq_features, _ = self.extract_frequency_features(window)
            tf_features, _ = self.extract_time_frequency_features(window)

            # 合并特征
            window_features = np.concatenate([stat_features, freq_features, tf_features])
            all_features.append(window_features)
            window_indices.append(start)

            start += self.stride

        return np.array(all_features), np.array(window_indices)

    def extract_features(self, time_series):
        """主特征提取方法"""
        features, indices = self.sliding_window_extraction(time_series)

        # 特征标准化
        if len(features) > 0:
            features = self.scaler.fit_transform(features)

        return features, indices


class FEMSSubsequenceSelector:
    """FEMS子序列选择器 - 基于特征方差选择最具区分性的子序列"""

    def __init__(self, selection_ratio=0.3, method='variance'):
        """
        初始化子序列选择器

        Args:
            selection_ratio: 选择的子序列比例
            method: 选择方法 ('variance' or 'random')
        """
        self.selection_ratio = selection_ratio
        self.method = method
        self.selected_indices = []

    def select_by_variance(self, features):
        """基于特征方差选择子序列"""
        n_samples = features.shape[0]
        n_select = int(n_samples * self.selection_ratio)

        if n_select >= n_samples:
            return np.arange(n_samples)

        # 计算每个样本在所有特征上的方差
        sample_variances = np.var(features, axis=1)

        # 选择方差最大的子序列
        selected_indices = np.argsort(sample_variances)[-n_select:]
        selected_indices = np.sort(selected_indices)

        return selected_indices

    def select_subsequences(self, features, indices=None):
        """选择子序列"""
        if self.method == 'variance':
            self.selected_indices = self.select_by_variance(features)
        elif self.method == 'random':
            n_samples = features.shape[0]
            n_select = int(n_samples * self.selection_ratio)
            self.selected_indices = np.random.choice(n_samples, n_select, replace=False)
        else:
            self.selected_indices = np.arange(features.shape[0])

        selected_features = features[self.selected_indices]

        if indices is not None:
            selected_indices = indices[self.selected_indices]
            return selected_features, selected_indices

        return selected_features


class FEMSFeatureSelector:
    """FEMS特征选择器 - 基于PCA的特征选择"""

    def __init__(self, variance_threshold=0.95, n_components=None):
        """
        初始化特征选择器

        Args:
            variance_threshold: 保留的方差比例
            n_components: 主成分数量
        """
        self.variance_threshold = variance_threshold
        self.n_components = n_components
        self.pca = None
        self.variance_selector = VarianceThreshold()
        self.explained_variance_ratio_ = None

    def select_features(self, features):
        """特征选择"""
        # 方差阈值筛选
        features = self.variance_selector.fit_transform(features)

        # PCA降维
        self.pca = PCA(n_components=self.n_components)
        transformed = self.pca.fit_transform(features)

        # 如果未指定组件数，根据方差阈值确定
        if self.n_components is None:
            cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
            self.pca = PCA(n_components=n_components)
            transformed = self.pca.fit_transform(features)

        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_

        return transformed

    def reconstruct(self, transformed_features):
        """重构特征"""
        if self.pca is not None:
            return self.pca.inverse_transform(transformed_features)
        return transformed_features

    def get_reconstruction_error(self, original_features, reconstructed_features):
        """计算重构误差"""
        mse = np.mean((original_features - reconstructed_features) ** 2, axis=1)
        return mse


class FEMSModelSelector:
    """FEMS模型选择器 - 基于加权评分机制选择最佳模型"""

    def __init__(self, model_pool=None, consistency_weight=0.3):
        """
        初始化模型选择器

        Args:
            model_pool: 模型池
            consistency_weight: 预测一致性权重
        """
        self.model_pool = model_pool if model_pool is not None else []
        self.consistency_weight = consistency_weight
        self.selected_model = None
        self.model_scores = {}

    def compute_model_performance(self, model, X, y):
        """计算模型性能"""
        from sklearn.metrics import f1_score, precision_score, recall_score

        predictions = model.predict(X)

        # 计算各项指标
        f1 = f1_score(y, predictions, average='macro')
        precision = precision_score(y, predictions, average='macro')
        recall = recall_score(y, predictions, average='macro')

        # 综合性能得分
        performance_score = 0.5 * f1 + 0.3 * precision + 0.2 * recall

        return performance_score, predictions

    def compute_prediction_consistency(self, predictions_list):
        """计算预测一致性"""
        if not predictions_list:
            return 0

        # 将所有预测转换为二维数组
        predictions_array = np.array(predictions_list).T

        # 计算每行（样本）的一致性
        consistency_scores = []
        for row in predictions_array:
            # 计算该样本在不同模型上预测结果的一致性
            unique, counts = np.unique(row, return_counts=True)
            max_count = np.max(counts)
            consistency = max_count / len(row)
            consistency_scores.append(consistency)

        # 返回平均一致性
        return np.mean(consistency_scores)

    def select_model(self, X, y, X_val=None, y_val=None):
        """选择最佳模型"""
        if X_val is None or y_val is None:
            X_val, y_val = X, y

        best_score = -1
        best_model = None
        all_predictions = []

        for model_name, model in self.model_pool.items():
            try:
                # 训练模型
                model.fit(X, y)

                # 在验证集上评估
                performance_score, predictions = self.compute_model_performance(
                    model, X_val, y_val
                )

                all_predictions.append(predictions)

                # 计算该模型与其他模型的一致性（如果可能）
                consistency_score = 0
                if len(self.model_pool) > 1:
                    other_predictions = [p for i, p in enumerate(all_predictions)
                                         if i != len(all_predictions) - 1]
                    if other_predictions:
                        consistency_score = self.compute_prediction_consistency(
                            other_predictions + [predictions]
                        )

                # 加权总分
                total_score = (1 - self.consistency_weight) * performance_score + \
                              self.consistency_weight * consistency_score

                self.model_scores[model_name] = {
                    'performance': performance_score,
                    'consistency': consistency_score,
                    'total': total_score
                }

                if total_score > best_score:
                    best_score = total_score
                    best_model = model_name

            except Exception as e:
                print(f"Model {model_name} failed: {e}")
                continue

        self.selected_model = best_model
        return best_model, self.model_scores


class FEMSAnomalyDetector:
    """FEMS异常检测器 - 主类"""

    def __init__(self, window_size=100, stride=50, selection_ratio=0.3,
                 variance_threshold=0.95, n_components=None, consistency_weight=0.3):
        """
        初始化FEMS异常检测器

        Args:
            window_size: 窗口大小
            stride: 步长
            selection_ratio: 子序列选择比例
            variance_threshold: 方差阈值
            n_components: PCA组件数
            consistency_weight: 一致性权重
        """
        self.window_size = window_size
        self.stride = stride
        self.selection_ratio = selection_ratio
        self.variance_threshold = variance_threshold
        self.n_components = n_components
        self.consistency_weight = consistency_weight

        # 初始化组件
        self.feature_extractor = FEMSFeatureExtractor(
            window_size=window_size,
            stride=stride
        )
        self.subsequence_selector = FEMSSubsequenceSelector(
            selection_ratio=selection_ratio
        )
        self.feature_selector = FEMSFeatureSelector(
            variance_threshold=variance_threshold,
            n_components=n_components
        )
        self.model_selector = FEMSModelSelector(
            consistency_weight=consistency_weight
        )

        # 训练数据
        self.training_features = None
        self.selected_features = None
        self.transformed_features = None
        self.selected_model = None
        self.best_model_name = None

    def fit(self, time_series, model_pool=None):
        """
        训练FEMS异常检测器

        Args:
            time_series: 时间序列数据
            model_pool: 模型池
        """
        print("Step 1: Feature Extraction...")
        features, indices = self.feature_extractor.extract_features(time_series)
        self.training_features = features
        print(f"Extracted {features.shape[0]} windows with {features.shape[1]} features each")

        print("Step 2: Subsequence Selection...")
        selected_features, selected_indices = self.subsequence_selector.select_subsequences(
            features, indices
        )
        self.selected_features = selected_features
        print(f"Selected {selected_features.shape[0]} subsequences")

        print("Step 3: Feature Selection and Compression...")
        transformed_features = self.feature_selector.select_features(selected_features)
        self.transformed_features = transformed_features
        print(f"Compressed to {transformed_features.shape[1]} features")

        # 如果有模型池，进行模型选择
        if model_pool is not None:
            print("Step 4: Model Selection...")
            self.model_selector.model_pool = model_pool

            # 这里需要标签数据，在实际应用中可能需要调整
            # 为了演示，我们生成伪标签
            y_train = np.zeros(len(transformed_features))

            best_model, scores = self.model_selector.select_model(
                transformed_features, y_train,
                transformed_features, y_train
            )

            self.best_model_name = best_model
            self.selected_model = model_pool[best_model]

            print(f"Selected model: {best_model}")
            print("Model scores:")
            for model_name, score_info in scores.items():
                print(f"  {model_name}: performance={score_info['performance']:.3f}, "
                      f"consistency={score_info['consistency']:.3f}, "
                      f"total={score_info['total']:.3f}")

        return self

    def detect(self, time_series, model=None):
        """
        检测异常

        Args:
            time_series: 时间序列数据
            model: 使用的模型（如果为None则使用选择的模型）

        Returns:
            异常分数
        """
        # 提取特征
        features, indices = self.feature_extractor.extract_features(time_series)

        # 特征转换
        if self.feature_selector.pca is not None:
            features_transformed = self.feature_selector.pca.transform(features)
        else:
            features_transformed = features

        # 使用模型进行预测
        if model is None:
            if self.selected_model is not None:
                model = self.selected_model
            else:
                raise ValueError("No model available for detection")

        # 获取异常分数
        if hasattr(model, 'decision_function'):
            scores = model.decision_function(features_transformed)
        elif hasattr(model, 'predict_proba'):
            scores = model.predict_proba(features_transformed)[:, 1]
        else:
            predictions = model.predict(features_transformed)
            scores = predictions.astype(float)

        # 将窗口分数映射回原始时间点
        full_scores = np.zeros(len(time_series))
        count = np.zeros(len(time_series))

        for i, idx in enumerate(indices):
            end_idx = min(idx + self.window_size, len(time_series))
            full_scores[idx:end_idx] += scores[i]
            count[idx:end_idx] += 1

        # 避免除以零
        count[count == 0] = 1
        full_scores = full_scores / count

        return full_scores

    def bayesian_optimization(self, time_series, model, param_space, n_iter=50, cv_splits=5):
        """
        贝叶斯优化超参数

        Args:
            time_series: 时间序列数据
            model: 要优化的模型
            param_space: 参数空间
            n_iter: 迭代次数
            cv_splits: 交叉验证折数
        """
        from skopt import BayesSearchCV
        from sklearn.model_selection import TimeSeriesSplit

        # 准备数据
        features, _ = self.feature_extractor.extract_features(time_series)
        selected_features, _ = self.subsequence_selector.select_subsequences(features)
        transformed_features = self.feature_selector.select_features(selected_features)

        # 生成伪标签（实际应用中应使用真实标签）
        y = np.zeros(len(transformed_features))

        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=cv_splits)

        # 贝叶斯优化
        opt = BayesSearchCV(
            estimator=model,
            search_spaces=param_space,
            n_iter=n_iter,
            cv=tscv,
            n_jobs=-1,
            scoring='f1_macro'
        )

        opt.fit(transformed_features, y)

        print(f"Best parameters: {opt.best_params_}")
        print(f"Best score: {opt.best_score_:.3f}")

        return opt.best_estimator_, opt.best_params_