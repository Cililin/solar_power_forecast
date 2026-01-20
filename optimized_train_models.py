#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
太阳能发电预测模型v2.0
修复参数错误，优化模型性能
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class SolarForecastModelFinal:
    """太阳能预测模型"""
    
    def __init__(self, data_path='processed_data/processed_solar_data.csv'):
        self.data_path = data_path
        self.data = None
        self.daily_data = None
        self.weekly_data = None
        self.models = {}
        self.evaluation_results = {}
        self.output_dir = 'final_output'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """加载数据"""
        print("="*60)
        print("加载预处理数据")
        print("="*60)
        
        self.data = pd.read_csv(self.data_path, index_col='timestamp', parse_dates=True)
        print(f"小时级数据形状: {self.data.shape}")
        print(f"时间范围: {self.data.index.min()} 到 {self.data.index.max()}")
        
        # 确保有目标变量
        if 'generation_rate' not in self.data.columns:
            if 'power_output_mw' in self.data.columns:
                self.data['generation_rate'] = (self.data['power_output_mw'] / 50) * 100
                print("已创建 generation_rate 列")
        
        # 零值分析
        zero_count = (self.data['generation_rate'] == 0).sum()
        low_count = (self.data['generation_rate'] < 1).sum()
        print(f"\n零值分析:")
        print(f"  零值数量: {zero_count} ({zero_count/len(self.data)*100:.1f}%)")
        print(f"  低值(<1%)数量: {low_count} ({low_count/len(self.data)*100:.1f}%)")
        
        # 创建日级和周级数据
        if 'generation_rate' in self.data.columns:
            self.daily_data = self.data.resample('D').agg({
                'generation_rate': 'mean',
                'solar_radiation_w_m2': 'mean',
                'temperature_c': 'mean',
                'humidity_percent': 'mean',
                'wind_speed_10m_mps': 'mean',
                'cloud_index': 'mean'
            }).dropna()
            
            self.weekly_data = self.data.resample('W').agg({
                'generation_rate': 'mean',
                'solar_radiation_w_m2': 'mean',
                'temperature_c': 'mean',
                'humidity_percent': 'mean',
                'wind_speed_10m_mps': 'mean',
                'cloud_index': 'mean'
            }).dropna()
            
            print(f"\n日级数据形状: {self.daily_data.shape}")
            print(f"周级数据形状: {self.weekly_data.shape}")
            print(f"发电率统计 - 小时级: {self.data['generation_rate'].mean():.2f}%, "
                  f"日级: {self.daily_data['generation_rate'].mean():.2f}%, "
                  f"周级: {self.weekly_data['generation_rate'].mean():.2f}%")
    
    def calculate_weighted_mape(self, y_true, y_pred):
        """计算加权MAPE，解决零值问题"""
        # 忽略极低值（发电率<0.5%）
        mask = y_true > 0.5
        if mask.sum() == 0:
            return np.nan
        
        # 计算加权MAPE
        weights = y_true[mask] / y_true[mask].sum()
        errors = np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]
        wmape = np.sum(weights * errors) * 100
        return wmape
    
    def prepare_hourly_data(self, use_daytime_only=True):
        """准备小时级数据"""
        print("\n准备小时级数据...")
        
        # 基础特征
        base_features = [
            'hour', 'month', 'day_of_week', 'day_of_year',
            'solar_radiation_w_m2', 'temperature_c', 
            'humidity_percent', 'wind_speed_10m_mps',
            'cloud_index', 'is_night', 'has_sunlight', 'is_weekend'
        ]
        
        # 选择可用的特征
        features = [f for f in base_features if f in self.data.columns]
        
        # 添加重要特征
        important_features = [
            'generation_rate_lag_1h', 'generation_rate_lag_24h',
            'generation_rate_ma_3h', 'generation_rate_ma_24h',
            'radiation_temperature_interaction'
        ]
        
        for f in important_features:
            if f in self.data.columns:
                features.append(f)
        
        print(f"使用特征数量: {len(features)}")
        
        # 目标变量
        target = 'generation_rate'
        
        # 如果只使用白天数据
        if use_daytime_only:
            print("使用白天数据 (has_sunlight=1)")
            data_filtered = self.data[self.data['has_sunlight'] == 1]
        else:
            print("使用全部数据")
            data_filtered = self.data
        
        # 移除缺失值
        data_clean = data_filtered[features + [target]].dropna()
        print(f"清洗后数据形状: {data_clean.shape}")
        
        # 时间序列分割
        split_idx = int(len(data_clean) * 0.8)
        
        X = data_clean[features].values
        y = data_clean[target].values
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        train_idx = data_clean.index[:split_idx]
        test_idx = data_clean.index[split_idx:]
        
        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")
        print(f"训练时间范围: {train_idx.min()} 到 {train_idx.max()}")
        print(f"测试时间范围: {test_idx.min()} 到 {test_idx.max()}")
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, features, scaler, test_idx
    
    def prepare_daily_data(self):
        """准备日级数据"""
        print("\n准备日级数据...")
        
        # 添加时间特征到日级数据
        if self.daily_data is not None:
            self.daily_data['month'] = self.daily_data.index.month
            self.daily_data['day_of_week'] = self.daily_data.index.dayofweek
            self.daily_data['day_of_year'] = self.daily_data.index.dayofyear
            self.daily_data['week_of_year'] = self.daily_data.index.isocalendar().week
            self.daily_data['is_weekend'] = self.daily_data['day_of_week'].isin([5, 6])
        
        features = [
            'month', 'day_of_week', 'day_of_year', 'week_of_year', 'is_weekend',
            'solar_radiation_w_m2', 'temperature_c', 'humidity_percent',
            'wind_speed_10m_mps', 'cloud_index'
        ]
        
        # 选择可用的特征
        features = [f for f in features if f in self.daily_data.columns]
        
        # 添加滞后特征
        if 'generation_rate' in self.daily_data.columns:
            self.daily_data['generation_rate_lag_1d'] = self.daily_data['generation_rate'].shift(1)
            self.daily_data['generation_rate_lag_7d'] = self.daily_data['generation_rate'].shift(7)
            self.daily_data['generation_rate_ma_3d'] = self.daily_data['generation_rate'].rolling(3).mean()
            
            if 'generation_rate_lag_1d' in self.daily_data.columns:
                features.append('generation_rate_lag_1d')
            if 'generation_rate_lag_7d' in self.daily_data.columns:
                features.append('generation_rate_lag_7d')
            if 'generation_rate_ma_3d' in self.daily_data.columns:
                features.append('generation_rate_ma_3d')
        
        print(f"使用特征数量: {len(features)}")
        
        target = 'generation_rate'
        
        # 移除缺失值
        data_clean = self.daily_data[features + [target]].dropna()
        print(f"清洗后数据形状: {data_clean.shape}")
        
        # 时间序列分割
        split_idx = int(len(data_clean) * 0.8)
        
        X = data_clean[features].values
        y = data_clean[target].values
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        train_idx = data_clean.index[:split_idx]
        test_idx = data_clean.index[split_idx:]
        
        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, features, scaler, test_idx
    
    def prepare_weekly_data(self):
        """准备周级数据"""
        print("\n准备周级数据...")
        
        # 添加时间特征到周级数据
        if self.weekly_data is not None:
            self.weekly_data['month'] = self.weekly_data.index.month
            self.weekly_data['week_of_year'] = self.weekly_data.index.isocalendar().week
        
        features = [
            'month', 'week_of_year',
            'solar_radiation_w_m2', 'temperature_c', 'humidity_percent',
            'wind_speed_10m_mps', 'cloud_index'
        ]
        
        # 选择可用的特征
        features = [f for f in features if f in self.weekly_data.columns]
        
        # 添加滞后特征
        if 'generation_rate' in self.weekly_data.columns:
            self.weekly_data['generation_rate_lag_1w'] = self.weekly_data['generation_rate'].shift(1)
            self.weekly_data['generation_rate_lag_4w'] = self.weekly_data['generation_rate'].shift(4)
            self.weekly_data['generation_rate_ma_4w'] = self.weekly_data['generation_rate'].rolling(4).mean()
            
            if 'generation_rate_lag_1w' in self.weekly_data.columns:
                features.append('generation_rate_lag_1w')
            if 'generation_rate_lag_4w' in self.weekly_data.columns:
                features.append('generation_rate_lag_4w')
            if 'generation_rate_ma_4w' in self.weekly_data.columns:
                features.append('generation_rate_ma_4w')
        
        print(f"使用特征数量: {len(features)}")
        
        target = 'generation_rate'
        
        # 移除缺失值
        data_clean = self.weekly_data[features + [target]].dropna()
        print(f"清洗后数据形状: {data_clean.shape}")
        
        # 时间序列分割
        split_idx = int(len(data_clean) * 0.8)
        
        X = data_clean[features].values
        y = data_clean[target].values
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        train_idx = data_clean.index[:split_idx]
        test_idx = data_clean.index[split_idx:]
        
        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, features, scaler, test_idx
    
    def train_hourly_model(self):
        """训练小时级模型"""
        print("\n" + "="*60)
        print("训练小时级模型")
        print("="*60)
        
        # 准备数据 - 使用白天数据
        X_train, X_test, y_train, y_test, features, scaler, test_idx = self.prepare_hourly_data(use_daytime_only=True)
        
        # 模型参数
        params = {
            'n_estimators': 300,
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,  # 减少正则化
            'reg_lambda': 1.0,
            'min_child_weight': 3,
            'gamma': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'objective': 'reg:squarederror',
        }
        
        print("\nXGBoost参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # 训练模型
        print("\n开始训练...")
        model = xgb.XGBRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50,
            verbose=20
        )
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        wmape = self.calculate_weighted_mape(y_test, y_pred)
        
        print(f"\n评估结果:")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAE:  {mae:.4f}%")
        print(f"  RMSE: {rmse:.4f}%")
        print(f"  WMAPE: {wmape:.2f}%" if not np.isnan(wmape) else "  WMAPE: N/A")
        
        # 保存模型
        model_path = os.path.join(self.output_dir, 'xgboost_hourly.pkl')
        joblib.dump(model, model_path)
        print(f"\n模型已保存到: {model_path}")
        
        # 保存结果
        self.models['hourly'] = model
        self.evaluation_results['hourly'] = {
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse),
            'wmape': float(wmape) if not np.isnan(wmape) else None,
            'test_size': len(y_test),
            'test_date_range': {
                'start': test_idx.min().strftime('%Y-%m-%d %H:%M:%S'),
                'end': test_idx.max().strftime('%Y-%m-%d %H:%M:%S')
            },
            'params': params
        }
        
        # 绘制图表
        self.plot_predictions(y_test, y_pred, 'hourly', test_idx)
        self.plot_feature_importance(model, features, 'hourly')
        
        return model
    
    def train_daily_model(self):
        """训练日级模型"""
        print("\n" + "="*60)
        print("训练日级模型")
        print("="*60)
        
        # 准备数据
        X_train, X_test, y_train, y_test, features, scaler, test_idx = self.prepare_daily_data()
        
        # 模型参数
        params = {
            'n_estimators': 200,
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.15,
            'reg_lambda': 1.2,
            'min_child_weight': 5,
            'random_state': 42,
            'n_jobs': -1,
            'objective': 'reg:squarederror',
        }
        
        print("\nXGBoost参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # 训练模型
        print("\n开始训练...")
        model = xgb.XGBRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=30,
            verbose=20
        )
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        wmape = self.calculate_weighted_mape(y_test, y_pred)
        
        print(f"\n评估结果:")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAE:  {mae:.4f}%")
        print(f"  RMSE: {rmse:.4f}%")
        print(f"  WMAPE: {wmape:.2f}%" if not np.isnan(wmape) else "  WMAPE: N/A")
        
        # 保存模型
        model_path = os.path.join(self.output_dir, 'xgboost_daily.pkl')
        joblib.dump(model, model_path)
        print(f"\n模型已保存到: {model_path}")
        
        # 保存结果
        self.models['daily'] = model
        self.evaluation_results['daily'] = {
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse),
            'wmape': float(wmape) if not np.isnan(wmape) else None,
            'test_size': len(y_test),
            'test_date_range': {
                'start': test_idx.min().strftime('%Y-%m-%d'),
                'end': test_idx.max().strftime('%Y-%m-%d')
            },
            'params': params
        }
        
        # 绘制图表
        self.plot_predictions(y_test, y_pred, 'daily', test_idx)
        self.plot_feature_importance(model, features, 'daily')
        
        return model
    
    def train_weekly_model(self):
        """训练周级模型"""
        print("\n" + "="*60)
        print("训练周级模型")
        print("="*60)
        
        # 准备数据
        X_train, X_test, y_train, y_test, features, scaler, test_idx = self.prepare_weekly_data()
        
        # 模型参数
        params = {
            'n_estimators': 150,
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.2,
            'reg_lambda': 1.5,
            'min_child_weight': 8,
            'random_state': 42,
            'n_jobs': -1,
            'objective': 'reg:squarederror',
        }
        
        print("\nXGBoost参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # 训练模型
        print("\n开始训练...")
        model = xgb.XGBRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=20,
            verbose=20
        )
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        wmape = self.calculate_weighted_mape(y_test, y_pred)
        
        print(f"\n评估结果:")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAE:  {mae:.4f}%")
        print(f"  RMSE: {rmse:.4f}%")
        print(f"  WMAPE: {wmape:.2f}%" if not np.isnan(wmape) else "  WMAPE: N/A")
        
        # 保存模型
        model_path = os.path.join(self.output_dir, 'xgboost_weekly.pkl')
        joblib.dump(model, model_path)
        print(f"\n模型已保存到: {model_path}")
        
        # 保存结果
        self.models['weekly'] = model
        self.evaluation_results['weekly'] = {
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse),
            'wmape': float(wmape) if not np.isnan(wmape) else None,
            'test_size': len(y_test),
            'test_date_range': {
                'start': test_idx.min().strftime('%Y-%m-%d'),
                'end': test_idx.max().strftime('%Y-%m-%d')
            },
            'params': params
        }
        
        # 绘制图表
        self.plot_predictions(y_test, y_pred, 'weekly', test_idx)
        self.plot_feature_importance(model, features, 'weekly')
        
        return model
    
    def plot_predictions(self, y_true, y_pred, data_type, test_idx):
        """绘制预测结果"""
        plt.figure(figsize=(12, 6))
        
        # 取最近的100个点
        n_points = min(100, len(y_true))
        
        plt.plot(range(n_points), y_true[-n_points:], 'b-', label='Actual', alpha=0.7, linewidth=2)
        plt.plot(range(n_points), y_pred[-n_points:], 'r-', label='Predicted', alpha=0.7, linewidth=2)
        
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Generation Rate (%)', fontsize=12)
        plt.title(f'{data_type.capitalize()} Model: Actual vs Predicted (Last {n_points} Points)', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(self.output_dir, f'predictions_{data_type}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"预测图已保存到: {plot_path}")
    
    def plot_feature_importance(self, model, features, data_type, top_n=10):
        """绘制特征重要性"""
        importance = model.feature_importances_
        
        # 创建DataFrame
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importance_df)), importance_df['importance'].values)
        plt.yticks(range(len(importance_df)), importance_df['feature'].values)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'{data_type.capitalize()} Model: Top {top_n} Feature Importance', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        plot_path = os.path.join(self.output_dir, f'feature_importance_{data_type}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"特征重要性图已保存到: {plot_path}")
    
    def save_results(self):
        """保存所有结果"""
        # 保存评估结果
        results_path = os.path.join(self.output_dir, 'evaluation_results.json')
        
        # 转换参数为可序列化格式
        serializable_results = {}
        for data_type, results in self.evaluation_results.items():
            serializable_results[data_type] = {}
            for key, value in results.items():
                if key == 'params':
                    serializable_results[data_type][key] = str(value)
                else:
                    serializable_results[data_type][key] = value
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n评估结果已保存到: {results_path}")
        
        # 打印总结
        print("\n" + "="*80)
        print("训练完成！")
        print("="*80)
        
        all_r2_ok = True
        all_mape_ok = True
        all_mae_ok = True
        all_rmse_ok = True
        
        for data_type, results in self.evaluation_results.items():
            r2 = results['r2']
            wmape = results['wmape']
            mae = results['mae']
            rmse = results['rmse']
            
            r2_ok = 0.7 <= r2 <= 0.9
            mape_ok = wmape is not None and wmape < 10
            mae_ok = 5 <= mae <= 10
            rmse_ok = 5 <= rmse <= 10
            
            all_r2_ok &= r2_ok
            all_mape_ok &= mape_ok
            all_mae_ok &= mae_ok
            all_rmse_ok &= rmse_ok
       
    
    def train_all_models(self):
        """训练所有模型"""
        print("="*80)
        print("开始训练所有模型")
        print("="*80)
        
        try:
            # 训练小时模型
            self.train_hourly_model()
            
            # 训练日模型
            self.train_daily_model()
            
            # 训练周模型
            self.train_weekly_model()
            
            # 保存结果
            self.save_results()
            
            return self.models
            
        except Exception as e:
            print(f"训练过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    print("太阳能发电预测模型")
    print("="*80)
    
    # 创建模型实例
    model = SolarForecastModelFinal('processed_data/processed_solar_data.csv')
    
    # 加载数据
    model.load_data()
    
    # 训练所有模型
    models = model.train_all_models()
    
    if models:
        print("\n" + "="*80)
        print("所有模型训练完成！")
        print(f"结果保存在: {model.output_dir}/")
        print(f"使用 'python final_app.py' 启动应用查看结果")
        print("="*80)
    
    return model

if __name__ == "__main__":
    main()