# FEMS: Feature Engineering based Multi-model Selection for Anomaly Detection

## 概述

FEMS (Feature Engineering based Multi-model Selection) 是一种先进的时间序列异常检测框架，它集成了特征工程、子序列选择、特征压缩、贝叶斯优化和智能模型选择技术。该框架基于MSAD代码库构建，提供了强大的异常检测能力。

## 主要特性

### 1. **多层次特征工程**
   - **统计特征**: 均值、标准差、偏度、峰度、百分位数等
   - **频域特征**: FFT变换、主频分析、频谱能量、谱熵等
   - **时频域特征**: 小波变换、小波系数统计、小波熵等

### 2. **智能子序列选择**
   - 基于特征空间方差评估子序列的区分性
   - 自动筛选最具代表性的子序列
   - 可配置的选择比例和策略

### 3. **自适应特征压缩**
   - 基于PCA的特征降维
   - 保留重构误差最小的特征
   - 可配置的方差保留阈值

### 4. **贝叶斯优化**
   - 自动化超参数调优
   - 支持时间序列交叉验证
   - 高效搜索最优参数组合

### 5. **加权模型选择**
   - 融合模型性能与预测一致性
   - 支持多模型集成
   - 自适应选择最佳检测模型

## 安装

```bash

# 安装依赖
pip install -r requirements.txt
```

主要依赖
numpy>=1.19.0

scipy>=1.6.0

scikit-learn>=0.24.0

pywt>=1.1.0

scikit-optimize>=0.9.0

matplotlib>=3.3.0

pandas>=1.2.1