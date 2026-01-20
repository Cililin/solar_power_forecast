"""
太阳能发电预测模型训练模块
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
import warnings
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置图表样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class SolarForecastModel:
    """太阳能预测模型类"""
    
    def __init__(self, data_path='processed_data/processed_solar_data.csv'):
        """
        初始化模型类
        
        Args:
            data_path: 处理后的数据路径
        """
        self.data_path = data_path
        self.data = None
        self.daily_data = None
        self.weekly_data = None
        
        # 模型存储
        self.models = {
            'hourly': None,
            'daily': None,
            'weekly': None
        }
        
        # 特征列
        self.feature_columns = {}
        
        # 评估结果
        self.evaluation_results = {}
        
        # 输出目录
        self.output_dir = 'model_output'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建模型保存目录
        self.model_dir = 'saved_models'
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 特征名称映射
        self.feature_name_mapping = {}
        
    def load_data(self):
        """加载预处理后的数据"""
        print("="*60)
        print("加载预处理数据")
        print("="*60)
        
        try:
            # 加载小时级数据
            self.data = pd.read_csv(self.data_path, index_col='timestamp', parse_dates=True)
            print(f"小时级数据加载成功，形状: {self.data.shape}")
            
            self._clean_feature_names()
            
            # 创建日级数据（按日聚合）
            self.daily_data = self.data.resample('D').agg({
                'generation_rate': 'mean',
                'solar_radiation_w_m2': 'mean',
                'temperature_c': 'mean',
                'humidity_percent': 'mean',
                'wind_speed_10m_mps': 'mean',
                'cloud_index': 'mean'
            }).dropna()
            
            # 添加日级时间特征
            self._add_daily_features()
            
            # 创建周级数据（按周聚合）
            self.weekly_data = self.data.resample('W').agg({
                'generation_rate': 'mean',
                'solar_radiation_w_m2': 'mean',
                'temperature_c': 'mean',
                'humidity_percent': 'mean',
                'wind_speed_10m_mps': 'mean',
                'cloud_index': 'mean'
            }).dropna()
            
            # 添加周级时间特征
            self._add_weekly_features()
            
            print(f"日级数据形状: {self.daily_data.shape}")
            print(f"周级数据形状: {self.weekly_data.shape}")
            
            # 显示发电率统计
            print(f"\n发电率统计:")
            print(f"小时级 - 均值: {self.data['generation_rate'].mean():.2f}%")
            print(f"日级   - 均值: {self.daily_data['generation_rate'].mean():.2f}%")
            print(f"周级   - 均值: {self.weekly_data['generation_rate'].mean():.2f}%")
            
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            raise
    
    def _clean_feature_names(self):
        """清理特征名称"""
        original_columns = self.data.columns.tolist()
        cleaned_columns = []
        
        for col in original_columns:
            cleaned = str(col).replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
            cleaned = cleaned.replace('[', '').replace(']', '').replace(',', '_')
            cleaned = cleaned.replace('.', '_').replace('%', 'pct').replace('/', '_')
            cleaned = cleaned.replace('+', 'plus').replace('*', 'star')
            
            # 如果以数字开头，添加前缀
            if cleaned[0].isdigit():
                cleaned = 'f_' + cleaned
            
            cleaned_columns.append(cleaned)
        
        # 更新列名
        self.data.columns = cleaned_columns
        
        # 记录映射关系
        for orig, clean in zip(original_columns, cleaned_columns):
            if orig != clean:
                self.feature_name_mapping[clean] = orig
        
        print(f"已清理 {len([c for c in original_columns if c != cleaned_columns])} 个特征名称")
    
    def _add_daily_features(self):
        """为日级数据添加特征"""
        if self.daily_data is not None:
            # 提取时间特征
            self.daily_data['year'] = self.daily_data.index.year
            self.daily_data['month'] = self.daily_data.index.month
            self.daily_data['day'] = self.daily_data.index.day
            self.daily_data['day_of_week'] = self.daily_data.index.dayofweek
            self.daily_data['day_of_year'] = self.daily_data.index.dayofyear
            self.daily_data['week_of_year'] = self.daily_data.index.isocalendar().week
            self.daily_data['quarter'] = self.daily_data.index.quarter
            
            # 是否周末
            self.daily_data['is_weekend'] = self.daily_data['day_of_week'].isin([5, 6])
            
            # 季节
            self.daily_data['season'] = self.daily_data['month'] % 12 // 3 + 1
            
            # 周期特征
            self.daily_data['month_sin'] = np.sin(2 * np.pi * self.daily_data['month'] / 12)
            self.daily_data['month_cos'] = np.cos(2 * np.pi * self.daily_data['month'] / 12)
            
            self.daily_data['day_of_year_sin'] = np.sin(2 * np.pi * self.daily_data['day_of_year'] / 365)
            self.daily_data['day_of_year_cos'] = np.cos(2 * np.pi * self.daily_data['day_of_year'] / 365)
            
            # 滞后特征（过去1, 2, 3, 7天）
            for lag in [1, 2, 3, 7]:
                self.daily_data[f'generation_rate_lag_{lag}d'] = self.daily_data['generation_rate'].shift(lag)
            
            # 滚动统计
            for window in [3, 7, 14]:
                self.daily_data[f'generation_rate_ma_{window}d'] = (
                    self.daily_data['generation_rate'].rolling(window=window, min_periods=1).mean()
                )
            
            # 填充缺失值（由于滞后特征）
            self.daily_data.fillna(method='bfill', inplace=True)
            self.daily_data.fillna(method='ffill', inplace=True)
    
    def _add_weekly_features(self):
        """为周级数据添加特征"""
        if self.weekly_data is not None:
            # 提取时间特征
            self.weekly_data['year'] = self.weekly_data.index.year
            self.weekly_data['week_of_year'] = self.weekly_data.index.isocalendar().week
            self.weekly_data['month'] = self.weekly_data.index.month
            
            # 季度
            self.weekly_data['quarter'] = (self.weekly_data['month'] - 1) // 3 + 1
            
            # 周期特征
            self.weekly_data['week_sin'] = np.sin(2 * np.pi * self.weekly_data['week_of_year'] / 52)
            self.weekly_data['week_cos'] = np.cos(2 * np.pi * self.weekly_data['week_of_year'] / 52)
            
            # 滞后特征（过去1, 2, 4周）
            for lag in [1, 2, 4]:
                self.weekly_data[f'generation_rate_lag_{lag}w'] = self.weekly_data['generation_rate'].shift(lag)
            
            # 滚动统计
            for window in [4, 8, 12]:
                self.weekly_data[f'generation_rate_ma_{window}w'] = (
                    self.weekly_data['generation_rate'].rolling(window=window, min_periods=1).mean()
                )
            
            # 填充缺失值
            self.weekly_data.fillna(method='bfill', inplace=True)
            self.weekly_data.fillna(method='ffill', inplace=True)
    
    def get_hourly_features(self):
        """获取小时级特征列表"""
        # 基础特征
        base_features = [
            'hour', 'month', 'day_of_week', 'day_of_year',
            'solar_radiation_w_m2', 'temperature_c', 'humidity_percent',
            'wind_speed_10m_mps', 'cloud_index',
            'is_night', 'has_sunlight', 'is_weekend'
        ]
        
        # 时间周期特征
        time_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
        
        # 工程特征
        engineering_features = [
            'generation_rate_lag_1h', 'generation_rate_lag_24h',
            'generation_rate_ma_3h', 'generation_rate_ma_24h',
            'generation_rate_diff_1h', 'generation_rate_diff_24h',
            'generation_rate_yesterday_same_hour',
            'radiation_ma_3h', 'radiation_ma_24h',
            'temperature_ma_3h', 'temperature_ma_24h',
            'radiation_temperature_interaction', 'hour_radiation_interaction'
        ]
        
        # 分类特征（查找实际存在的）
        categorical_features = []
        categorical_prefixes = ['time_of_day_', 'radiation_level_', 'temperature_level_', 'cloud_level_', 'season_']
        
        for col in self.data.columns:
            for prefix in categorical_prefixes:
                if col.startswith(prefix):
                    categorical_features.append(col)
                    break
        
        # 合并所有特征并确保存在
        all_candidates = base_features + time_features + engineering_features + categorical_features
        available_features = []
        
        for feature in all_candidates:
            # 检查原始名称或映射后的名称
            if feature in self.data.columns:
                available_features.append(feature)
            elif feature in self.feature_name_mapping.values():
                # 查找映射后的名称
                for clean_name, orig_name in self.feature_name_mapping.items():
                    if orig_name == feature:
                        available_features.append(clean_name)
                        break
        
        # 移除重复项
        available_features = list(set(available_features))
        
        print(f"小时级可用特征数量: {len(available_features)}")
        
        return available_features
    
    def get_daily_features(self):
        """获取日级特征列表"""
        if self.daily_data is None:
            return []
        
        # 日级特征
        features = [
            'month', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter',
            'solar_radiation_w_m2', 'temperature_c', 'humidity_percent',
            'wind_speed_10m_mps', 'cloud_index',
            'is_weekend', 'season',
            'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos',
            'generation_rate_lag_1d', 'generation_rate_lag_7d',
            'generation_rate_ma_3d', 'generation_rate_ma_7d',
        ]
        
        # 确保特征存在
        available_features = [f for f in features if f in self.daily_data.columns]
        
        print(f"日级可用特征数量: {len(available_features)}")
        
        return available_features
    
    def get_weekly_features(self):
        """获取周级特征列表"""
        if self.weekly_data is None:
            return []
        
        # 周级特征
        features = [
            'year', 'month', 'week_of_year', 'quarter',
            'solar_radiation_w_m2', 'temperature_c', 'humidity_percent',
            'wind_speed_10m_mps', 'cloud_index',
            'week_sin', 'week_cos',
            'generation_rate_lag_1w', 'generation_rate_lag_4w',
            'generation_rate_ma_4w', 'generation_rate_ma_12w',
        ]
        
        # 确保特征存在
        available_features = [f for f in features if f in self.weekly_data.columns]
        
        print(f"周级可用特征数量: {len(available_features)}")
        
        return available_features
    
    def prepare_data(self, data_type='hourly'):
        """
        准备训练数据
        
        Args:
            data_type: 数据类型，'hourly', 'daily', 或 'weekly'
        
        Returns:
            X_train, X_test, y_train, y_test, feature_names, train_idx, test_idx
        """
        print(f"\n准备 {data_type} 数据...")
        
        if data_type == 'hourly':
            data = self.data
            features = self.get_hourly_features()
        elif data_type == 'daily':
            data = self.daily_data
            features = self.get_daily_features()
        elif data_type == 'weekly':
            data = self.weekly_data
            features = self.get_weekly_features()
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")
        
        if len(features) == 0:
            raise ValueError(f"{data_type} 数据没有可用的特征")
        
        # 目标变量
        target = 'generation_rate'
        
        # 确保目标变量存在
        if target not in data.columns:
            raise ValueError(f"目标变量 '{target}' 不存在于数据中")
        
        # 移除有缺失值的行
        data_clean = data[features + [target]].dropna()
        
        if len(data_clean) == 0:
            raise ValueError(f"{data_type} 数据清洗后为空")
        
        print(f"清洗后数据形状: {data_clean.shape}")
        print(f"使用特征数量: {len(features)}")
        
        # 分离特征和目标
        X = data_clean[features].copy()
        y = data_clean[target].copy()
        
        # 时间序列分割（不能随机分割）
        split_idx = int(len(X) * 0.8)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        train_idx = X_train.index
        test_idx = X_test.index
        
        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")
        print(f"训练集时间范围: {train_idx.min()} 到 {train_idx.max()}")
        print(f"测试集时间范围: {test_idx.min()} 到 {test_idx.max()}")
        
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 保存特征列和标准化器
        self.feature_columns[data_type] = {
            'features': features,
            'scaler': scaler,
            'feature_names': list(X.columns)  # 保存原始特征名顺序
        }
        
        # 保存标准化器
        scaler_path = os.path.join(self.model_dir, f'scaler_{data_type}.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"标准化器已保存到: {scaler_path}")
        
        return (X_train_scaled, X_test_scaled, 
                y_train.values, y_test.values,
                features, train_idx, test_idx)
    
    def train_xgboost_model(self, data_type='hourly'):
        """
        训练XGBoost模型
        
        Args:
            data_type: 数据类型
        """
        print("\n" + "="*60)
        print(f"训练 {data_type} XGBoost模型")
        print("="*60)
        
        # 准备数据
        X_train, X_test, y_train, y_test, features, train_idx, test_idx = self.prepare_data(data_type)
        
        # 特征名是字符串
        feature_names = [str(f) for f in features]
        
        # XGBoost参数配置（严格控制过拟合）
        params = {
            'n_estimators': 200,           # 树的数量
            'max_depth': 2,                # 最大深度
            'learning_rate': 0.05,         # 学习率
            'subsample': 0.8,              # 样本采样比例
            'colsample_bytree': 0.8,       # 特征采样比例
            'reg_alpha': 0.1,              # L1正则化
            'reg_lambda': 1.0,             # L2正则化
            'min_child_weight': 5,         # 最小叶子节点样本权重和
            'random_state': 42,
            'n_jobs': -1,                  # 使用所有CPU核心
            'objective': 'reg:squarederror',
        }
        
        print(f"XGBoost参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # 创建DMatrix（使用正确的特征名）
        print(f"\n创建DMatrix，特征数量: {len(feature_names)}")
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
        
        # 训练模型
        print("\n开始训练模型...")
        
        # 使用scikit-learn API
        model = xgb.XGBRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            min_child_weight=params['min_child_weight'],
            random_state=params['random_state'],
            n_jobs=params['n_jobs'],
            verbosity=0  # 减少输出
        )
        
        # 训练模型
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric=['rmse', 'mae'],
            early_stopping_rounds=50,
            verbose=50
        )
        
        # 保存模型
        self.models[data_type] = model
        model_path = os.path.join(self.model_dir, f'xgboost_{data_type}.pkl')
        joblib.dump(model, model_path)
        print(f"模型已保存到: {model_path}")
        
        # 保存特征信息
        feature_info = {
            'feature_names': feature_names,
            'feature_importance': dict(zip(feature_names, model.feature_importances_))
        }
        
        feature_info_path = os.path.join(self.model_dir, f'features_{data_type}.pkl')
        joblib.dump(feature_info, feature_info_path)
        print(f"特征信息已保存到: {feature_info_path}")
        
        # 评估模型
        self.evaluate_model(model, X_test, y_test, data_type, test_idx, feature_names)
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, data_type, test_idx, feature_names):
        """
        评估模型性能
        
        Args:
            model: 训练好的模型
            X_test: 测试集特征
            y_test: 测试集真实值
            data_type: 数据类型
            test_idx: 测试集索引
            feature_names: 特征名称列表
        """
        print("\n" + "="*60)
        print(f"{data_type} 模型评估")
        print("="*60)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算评估指标
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # 避免除以0的情况计算MAPE
        mask = y_test != 0
        if mask.any():
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        else:
            mape = np.nan
        
        # 计算平均绝对百分比误差
        y_test_safe = np.maximum(y_test, 0.1)  # 避免0值
        y_pred_safe = np.maximum(y_pred, 0.1)
        mape_alt = np.mean(np.abs((y_test_safe - y_pred_safe) / y_test_safe)) * 100
        
        print(f"评估指标:")
        print(f"  MAE (平均绝对误差): {mae:.4f}%")
        print(f"  RMSE (均方根误差): {rmse:.4f}%")
        print(f"  R² (决定系数): {r2:.4f}")
        if not np.isnan(mape):
            print(f"  MAPE (平均绝对百分比误差): {mape:.2f}%")
        print(f"  MAPE (替代计算): {mape_alt:.2f}%")
        
        # 保存评估结果
        self.evaluation_results[data_type] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape) if not np.isnan(mape) else None,
            'mape_alt': float(mape_alt),
            'test_size': len(y_test),
            'test_date_range': {
                'start': test_idx.min().strftime('%Y-%m-%d %H:%M:%S'),
                'end': test_idx.max().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # 绘制预测结果图
        self.plot_predictions(y_test, y_pred, data_type, test_idx)
        
        # 绘制残差图
        self.plot_residuals(y_test, y_pred, data_type)
        
        # 分析特征重要性
        self.analyze_feature_importance(model, feature_names, data_type)
        
        # 保存评估结果到文件
        self.save_evaluation_results(data_type)
    
    def plot_predictions(self, y_true, y_pred, data_type, dates):
        """
        绘制预测结果图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            data_type: 数据类型
            dates: 日期索引
        """
        try:
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            
            # 图1: 完整时间序列
            axes[0].plot(dates, y_true, 'b-', label='Actual', alpha=0.7, linewidth=1)
            axes[0].plot(dates, y_pred, 'r-', label='Predicted', alpha=0.7, linewidth=1)
            axes[0].set_xlabel('Time', fontsize=12)
            axes[0].set_ylabel('Generation Rate (%)', fontsize=12)
            axes[0].set_title(f'{data_type.capitalize()} Solar Generation Forecast: Actual vs Predicted', fontsize=14, fontweight='bold')
            axes[0].legend(loc='best')
            axes[0].grid(True, alpha=0.3)
            
            # 图2: 最近100个点（细节）
            n_points = min(100, len(y_true))
            axes[1].plot(range(n_points), y_true[-n_points:], 'b-', label='Actual', alpha=0.7, linewidth=2)
            axes[1].plot(range(n_points), y_pred[-n_points:], 'r-', label='Predicted', alpha=0.7, linewidth=2)
            axes[1].set_xlabel('Time Steps', fontsize=12)
            axes[1].set_ylabel('Generation Rate (%)', fontsize=12)
            axes[1].set_title(f'Recent {n_points} Points (Zoomed In)', fontsize=14)
            axes[1].legend(loc='best')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, f'predictions_{data_type}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"预测结果图已保存到: {plot_path}")
            
        except Exception as e:
            print(f"绘制预测图失败: {str(e)}")
    
    def plot_residuals(self, y_true, y_pred, data_type):
        """
        绘制残差图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            data_type: 数据类型
        """
        try:
            residuals = y_true - y_pred
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 图1: 残差分布直方图
            axes[0, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
            axes[0, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
            axes[0, 0].set_xlabel('Residuals (%)', fontsize=12)
            axes[0, 0].set_ylabel('Frequency', fontsize=12)
            axes[0, 0].set_title('Residual Distribution', fontsize=14)
            axes[0, 0].grid(True, alpha=0.3)
            
            # 图2: 残差vs预测值
            axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
            axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axes[0, 1].set_xlabel('Predicted Values (%)', fontsize=12)
            axes[0, 1].set_ylabel('Residuals (%)', fontsize=12)
            axes[0, 1].set_title('Residuals vs Predicted Values', fontsize=14)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 图3: 残差时间序列
            axes[1, 0].plot(residuals, alpha=0.7)
            axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axes[1, 0].set_xlabel('Time Index', fontsize=12)
            axes[1, 0].set_ylabel('Residuals (%)', fontsize=12)
            axes[1, 0].set_title('Residuals Over Time', fontsize=14)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 图4: 预测vs实际散点图
            axes[1, 1].scatter(y_true, y_pred, alpha=0.5, s=10)
            axes[1, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                          'r--', linewidth=2, label='Perfect Prediction')
            axes[1, 1].set_xlabel('Actual Values (%)', fontsize=12)
            axes[1, 1].set_ylabel('Predicted Values (%)', fontsize=12)
            axes[1, 1].set_title('Actual vs Predicted Scatter Plot', fontsize=14)
            axes[1, 1].legend(loc='best')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, f'residuals_{data_type}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"残差分析图已保存到: {plot_path}")
            
        except Exception as e:
            print(f"绘制残差图失败: {str(e)}")
    
    def analyze_feature_importance(self, model, feature_names, data_type, top_n=20):
        """
        分析特征重要性
        
        Args:
            model: XGBoost模型
            feature_names: 特征名称列表
            data_type: 数据类型
            top_n: 显示前N个特征
        """
        print("\n" + "="*60)
        print(f"{data_type} 特征重要性分析")
        print("="*60)
        
        try:
            # 获取特征重要性
            importance = model.feature_importances_
            
            # 转换为DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })
            
            # 按重要性排序
            importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
            
            print(f"总特征数: {len(feature_names)}")
            print("\nTop 20 重要特征:")
            print(importance_df.head(20).to_string())
            
            # 保存特征重要性
            importance_path = os.path.join(self.output_dir, f'feature_importance_{data_type}.csv')
            importance_df.to_csv(importance_path, index=False)
            print(f"特征重要性已保存到: {importance_path}")
            
            # 绘制特征重要性图
            self.plot_feature_importance(importance_df, data_type, top_n)
            
        except Exception as e:
            print(f"特征重要性分析失败: {str(e)}")
    
    def plot_feature_importance(self, importance_df, data_type, top_n=20):
        """
        绘制特征重要性图
        
        Args:
            importance_df: 特征重要性DataFrame
            data_type: 数据类型
            top_n: 显示前N个特征
        """
        try:
            # 取前N个特征
            top_features = importance_df.head(top_n).copy()
            
            plt.figure(figsize=(12, 8))
            
            # 创建水平条形图
            bars = plt.barh(range(len(top_features)), top_features['importance'].values)
            
            # 设置y轴标签
            plt.yticks(range(len(top_features)), top_features['feature'].values, fontsize=10)
            
            # 添加数值标签
            for i, (bar, importance) in enumerate(zip(bars, top_features['importance'].values)):
                plt.text(importance, i, f' {importance:.4f}', va='center', fontsize=9)
            
            plt.xlabel('Feature Importance', fontsize=12)
            plt.title(f'Top {top_n} Feature Importance - {data_type.capitalize()} Model', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()  # 最重要的特征在顶部
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            
            plot_path = os.path.join(self.output_dir, f'feature_importance_plot_{data_type}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"特征重要性图已保存到: {plot_path}")
            
        except Exception as e:
            print(f"绘制特征重要性图失败: {str(e)}")
    
    def save_evaluation_results(self, data_type):
        """保存评估结果"""
        if data_type in self.evaluation_results:
            results_path = os.path.join(self.output_dir, f'evaluation_{data_type}.json')
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_results[data_type], f, indent=2, ensure_ascii=False)
            print(f"评估结果已保存到: {results_path}")
    
    def train_all_models(self):
        """训练所有模型（小时、日、周）"""
        print("="*80)
        print("开始训练所有预测模型")
        print("="*80)
        
        try:
            # 训练小时模型
            print("\n" + "="*80)
            print("1. 训练小时级预测模型")
            print("="*80)
            hourly_model = self.train_xgboost_model('hourly')
            
            # 训练日模型
            print("\n" + "="*80)
            print("2. 训练日级预测模型")
            print("="*80)
            daily_model = self.train_xgboost_model('daily')
            
            # 训练周模型
            print("\n" + "="*80)
            print("3. 训练周级预测模型")
            print("="*80)
            weekly_model = self.train_xgboost_model('weekly')
            
            # 综合评估报告
            self.generate_summary_report()
            
            return {
                'hourly': hourly_model,
                'daily': daily_model,
                'weekly': weekly_model
            }
            
        except Exception as e:
            print(f"训练过程中出现错误: {str(e)}")
            print("尝试继续训练其他模型...")
            
            # 尝试训练其他模型
            models = {}
            
            for data_type in ['hourly', 'daily', 'weekly']:
                if data_type not in models:
                    try:
                        print(f"\n尝试训练 {data_type} 模型...")
                        model = self.train_xgboost_model(data_type)
                        models[data_type] = model
                    except Exception as e2:
                        print(f"训练 {data_type} 模型失败: {str(e2)}")
            
            if len(models) > 0:
                self.generate_summary_report()
                return models
            else:
                raise
    
    def generate_summary_report(self):
        """生成综合评估报告"""
        print("\n" + "="*80)
        print("模型训练综合报告")
        print("="*80)
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models_trained': list(self.models.keys()),
            'evaluation_results': self.evaluation_results,
            'training_summary': {}
        }
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'training_summary.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("\n训练完成！模型和评估结果已保存。")
        print(f"详细报告: {report_path}")
        print(f"模型保存目录: {self.model_dir}")
        print(f"输出文件目录: {self.output_dir}")
        
        # 打印关键指标
        if self.evaluation_results:
            print("\n" + "="*80)
            print("关键性能指标汇总")
            print("="*80)
            
            for data_type, results in self.evaluation_results.items():
                print(f"\n{data_type.upper()} 模型:")
                print(f"  MAE:  {results['mae']:.4f}%")
                print(f"  RMSE: {results['rmse']:.4f}%")
                print(f"  R²:   {results['r2']:.4f}")
                print(f"  MAPE: {results['mape_alt']:.2f}%")
                print(f"  测试集大小: {results['test_size']} 个样本")
                print(f"  测试时间范围: {results['test_date_range']['start']} 到 {results['test_date_range']['end']}")


def install_required_packages():
    """安装必要的包"""
    print("检查并安装必要的包...")
    
    try:
        import subprocess
        import sys
        
        # 安装shap（如果不存在）
        try:
            import shap
            print("shap已安装")
        except ImportError:
            print("安装shap...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
            print("shap安装完成")
        
        # 安装其他可能需要的包
        required_packages = ['xgboost', 'scikit-learn', 'matplotlib', 'seaborn', 'pandas', 'numpy', 'joblib']
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"{package}已安装")
            except ImportError:
                print(f"安装{package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"{package}安装完成")
                
    except Exception as e:
        print(f"安装包时出错: {str(e)}")
        print("请手动安装缺少的包")


def main():
    """主函数"""
    print("太阳能发电预测模型训练")
    print("="*80)
    
    # 安装必要的包
    install_required_packages()
    
    # 初始化模型训练器
    trainer = SolarForecastModel('processed_data/processed_solar_data.csv')
    
    # 加载数据
    trainer.load_data()
    
    # 训练所有模型
    try:
        models = trainer.train_all_models()
        
        print("\n" + "="*80)
        print("模型训练流程完成！")
        print("="*80)
        
        print("1. 检查模型输出目录: model_output/")
        print("2. 检查保存的模型: saved_models/")
        print("3. 运行Flask应用: python app.py")
        
        return trainer
        
    except Exception as e:
        print(f"模型训练失败: {str(e)}")
        print("请检查数据预处理是否正确完成")
        return None


if __name__ == "__main__":
    # 运行模型训练
    trainer = main()