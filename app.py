#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
太阳能发电预测Flask应用
主要功能：
1. 展示三个模型的预测效果
2. 显示评估指标（MAE, R², MAPE）
3. 提供实时预测接口
4. 可视化展示
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from flask import Flask, render_template, jsonify, request, send_from_directory
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import traceback
import platform
import sys

# 设置中文字体和图表样式
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文字体
    plt.rcParams['axes.unicode_minus'] = False
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 英文字体作为备选
    plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

class SolarForecastApp:
    """太阳能预测应用类"""
    
    def __init__(self):
        """初始化应用"""
        self.models = {}
        self.scalers = {}
        self.feature_info = {}
        self.evaluation_results = {}
        self.data_stats = {}
        
        # 加载模型和配置
        self.load_models()
        self.load_evaluation_results()
        self.load_data_stats()
    
    def load_models(self):
        """加载训练好的模型"""
        model_dir = 'saved_models'
        
        print("加载模型和配置...")
        
        try:
            # 加载小时模型
            hourly_model_path = os.path.join(model_dir, 'xgboost_hourly.pkl')
            hourly_scaler_path = os.path.join(model_dir, 'scaler_hourly.pkl')
            hourly_features_path = os.path.join(model_dir, 'features_hourly.pkl')
            
            if os.path.exists(hourly_model_path):
                self.models['hourly'] = joblib.load(hourly_model_path)
                self.scalers['hourly'] = joblib.load(hourly_scaler_path)
                self.feature_info['hourly'] = joblib.load(hourly_features_path)
                print(f"加载小时模型成功，特征数: {len(self.feature_info['hourly']['feature_names'])}")
        
        except Exception as e:
            print(f"加载小时模型失败: {str(e)}")
        
        try:
            # 加载日模型
            daily_model_path = os.path.join(model_dir, 'xgboost_daily.pkl')
            daily_scaler_path = os.path.join(model_dir, 'scaler_daily.pkl')
            daily_features_path = os.path.join(model_dir, 'features_daily.pkl')
            
            if os.path.exists(daily_model_path):
                self.models['daily'] = joblib.load(daily_model_path)
                self.scalers['daily'] = joblib.load(daily_scaler_path)
                self.feature_info['daily'] = joblib.load(daily_features_path)
                print(f"加载日模型成功，特征数: {len(self.feature_info['daily']['feature_names'])}")
        
        except Exception as e:
            print(f"加载日模型失败: {str(e)}")
        
        try:
            # 加载周模型
            weekly_model_path = os.path.join(model_dir, 'xgboost_weekly.pkl')
            weekly_scaler_path = os.path.join(model_dir, 'scaler_weekly.pkl')
            weekly_features_path = os.path.join(model_dir, 'features_weekly.pkl')
            
            if os.path.exists(weekly_model_path):
                self.models['weekly'] = joblib.load(weekly_model_path)
                self.scalers['weekly'] = joblib.load(weekly_scaler_path)
                self.feature_info['weekly'] = joblib.load(weekly_features_path)
                print(f"加载周模型成功，特征数: {len(self.feature_info['weekly']['feature_names'])}")
        
        except Exception as e:
            print(f"加载周模型失败: {str(e)}")
    
    def load_evaluation_results(self):
        """加载评估结果"""
        output_dir = 'model_output'

        
        print("加载评估结果...")
        
        for data_type in ['hourly', 'daily', 'weekly']:
            eval_path = os.path.join(output_dir, f'evaluation_{data_type}.json')
            if os.path.exists(eval_path):
                try:
                    with open(eval_path, 'r', encoding='utf-8') as f:
                        self.evaluation_results[data_type] = json.load(f)
                    print(f"加载{data_type}评估结果成功")
                except Exception as e:
                    print(f"加载{data_type}评估结果失败: {str(e)}")
    
    def load_data_stats(self):
        """加载数据统计信息"""
        stats_path = 'processed_data/data_statistics.json'
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r', encoding='utf-8') as f:
                    self.data_stats = json.load(f)
                print("加载数据统计信息成功")
            except Exception as e:
                print(f"加载数据统计信息失败: {str(e)}")
    
    def get_model_status(self):
        """获取模型状态"""
        status = {
            'hourly': 'hourly' in self.models,
            'daily': 'daily' in self.models,
            'weekly': 'weekly' in self.models,
            'total_models': len(self.models)
        }
        return status
    
    def create_comparison_chart(self):
        """创建模型对比图表"""
        try:
            # 准备数据
            model_names = []
            mae_values = []
            rmse_values = []
            r2_values = []
            mape_values = []
            
            for data_type in ['hourly', 'daily', 'weekly']:
                if data_type in self.evaluation_results:
                    model_names.append(data_type.upper())
                    results = self.evaluation_results[data_type]
                    mae_values.append(results.get('mae', 0))
                    rmse_values.append(results.get('rmse', 0))
                    r2_values.append(results.get('r2', 0))
                    mape_values.append(results.get('mape_alt', 0))
            
            if not model_names:
                return None
            
            # 创建对比图
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            
            # 图1: MAE对比
            colors = ['#3498db', '#2ecc71', '#e74c3c']
            bars1 = axes[0, 0].bar(model_names, mae_values, color=colors[:len(model_names)])
            axes[0, 0].set_title('模型对比: 平均绝对误差 (MAE)', fontsize=16, fontweight='bold', pad=20)
            axes[0, 0].set_ylabel('MAE (%)', fontsize=14)
            axes[0, 0].grid(True, alpha=0.3, linestyle='--')
            axes[0, 0].tick_params(axis='both', labelsize=12)
            for i, v in enumerate(mae_values):
                axes[0, 0].text(i, v, f'{v:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # 图2: RMSE对比
            bars2 = axes[0, 1].bar(model_names, rmse_values, color=colors[:len(model_names)])
            axes[0, 1].set_title('模型对比: 均方根误差 (RMSE)', fontsize=16, fontweight='bold', pad=20)
            axes[0, 1].set_ylabel('RMSE (%)', fontsize=14)
            axes[0, 1].grid(True, alpha=0.3, linestyle='--')
            axes[0, 1].tick_params(axis='both', labelsize=12)
            for i, v in enumerate(rmse_values):
                axes[0, 1].text(i, v, f'{v:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # 图3: R²对比
            bars3 = axes[1, 0].bar(model_names, r2_values, color=colors[:len(model_names)])
            axes[1, 0].set_title('模型对比: 决定系数 (R²)', fontsize=16, fontweight='bold', pad=20)
            axes[1, 0].set_ylabel('R²', fontsize=14)
            axes[1, 0].set_ylim(0, 1.1)
            axes[1, 0].grid(True, alpha=0.3, linestyle='--')
            axes[1, 0].tick_params(axis='both', labelsize=12)
            for i, v in enumerate(r2_values):
                axes[1, 0].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # 图4: MAPE对比
            bars4 = axes[1, 1].bar(model_names, mape_values, color=colors[:len(model_names)])
            axes[1, 1].set_title('模型对比: 平均绝对百分比误差 (MAPE)', fontsize=16, fontweight='bold', pad=20)
            axes[1, 1].set_ylabel('MAPE (%)', fontsize=14)
            axes[1, 1].grid(True, alpha=0.3, linestyle='--')
            axes[1, 1].tick_params(axis='both', labelsize=12)
            for i, v in enumerate(mape_values):
                axes[1, 1].text(i, v, f'{v:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            # 转换为base64
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            print(f"创建对比图失败: {str(e)}")
            return None
    
    def predict_hourly(self, features):
        """小时预测"""
        try:
            if 'hourly' not in self.models:
                return None
            
            model = self.models['hourly']
            scaler = self.scalers['hourly']
            feature_names = self.feature_info['hourly']['feature_names']
            
            # 准备特征向量
            feature_vector = []
            for feature in feature_names:
                if feature in features:
                    feature_vector.append(features[feature])
                else:
                    # 使用默认值或中位数
                    feature_vector.append(0.0)
            
            # 标准化
            feature_vector_scaled = scaler.transform([feature_vector])
            
            # 预测
            prediction = model.predict(feature_vector_scaled)[0]
            
            # 转换为kWh
            power_mw = prediction * 50 / 100  # 百分比转MW
            power_kwh = power_mw * 1000  # MW转kWh
            
            return {
                'generation_rate': float(prediction),
                'power_mw': float(power_mw),
                'power_kwh': float(power_kwh),
                'timestamp': features.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            }
        
        except Exception as e:
            print(f"小时预测失败: {str(e)}")
            return None
    
    def get_sample_features(self):
        """获取示例特征值"""
        sample_features = {
            'hourly': {
                'hour': 12,
                'month': 6,
                'day_of_week': 2,
                'day_of_year': 172,
                'solar_radiation_w_m2': 450.5,
                'temperature_c': 25.3,
                'humidity_percent': 65.2,
                'wind_speed_10m_mps': 3.2,
                'cloud_index': 0.3,
                'is_night': 0,
                'has_sunlight': 1,
                'is_weekend': 0,
                'hour_sin': 0.0,
                'hour_cos': -1.0,
                'month_sin': 0.0,
                'month_cos': -1.0,
                'generation_rate_lag_1h': 45.2,
                'generation_rate_lag_24h': 42.1,
                'generation_rate_ma_3h': 43.5,
                'generation_rate_ma_24h': 40.2,
                'generation_rate_diff_1h': 1.2,
                'generation_rate_diff_24h': 3.1,
                'generation_rate_yesterday_same_hour': 42.1,
                'radiation_ma_3h': 420.3,
                'radiation_ma_24h': 380.5,
                'temperature_ma_3h': 24.8,
                'temperature_ma_24h': 23.5,
                'radiation_temperature_interaction': 450.5 * 25.3,
                'hour_radiation_interaction': 12 * 450.5,
            },
            'daily': {
                'month': 6,
                'day_of_week': 2,
                'day_of_year': 172,
                'week_of_year': 25,
                'quarter': 2,
                'solar_radiation_w_m2': 350.5,
                'temperature_c': 24.3,
                'humidity_percent': 60.2,
                'wind_speed_10m_mps': 3.5,
                'cloud_index': 0.4,
                'is_weekend': 0,
                'season': 2,
                'month_sin': 0.0,
                'month_cos': -1.0,
                'day_of_year_sin': 0.0,
                'day_of_year_cos': -1.0,
                'generation_rate_lag_1d': 45.2,
                'generation_rate_lag_7d': 42.1,
                'generation_rate_ma_3d': 43.5,
                'generation_rate_ma_7d': 40.2,
            },
            'weekly': {
                'year': 2024,
                'month': 6,
                'week_of_year': 25,
                'quarter': 2,
                'solar_radiation_w_m2': 320.5,
                'temperature_c': 23.3,
                'humidity_percent': 58.2,
                'wind_speed_10m_mps': 3.8,
                'cloud_index': 0.5,
                'week_sin': 0.0,
                'week_cos': -1.0,
                'generation_rate_lag_1w': 44.2,
                'generation_rate_lag_4w': 41.1,
                'generation_rate_ma_4w': 42.5,
                'generation_rate_ma_12w': 39.2,
            }
        }
        
        # 添加分类特征
        for data_type in ['hourly', 'daily', 'weekly']:
            if data_type in sample_features:
                # 获取实际存在的特征前缀
                for prefix in ['time_of_day_', 'radiation_level_', 'temperature_level_', 'cloud_level_', 'season_']:
                    # 只检查小时模型的特征，其他模型可能没有这些特征
                    if data_type == 'hourly':
                        for feature in self.feature_info.get('hourly', {}).get('feature_names', []):
                            if feature.startswith(prefix):
                                sample_features[data_type][feature] = 0
                
                # 设置默认分类特征（仅小时模型需要）
                if data_type == 'hourly':
                    sample_features[data_type]['time_of_day_noon'] = 1
                    sample_features[data_type]['radiation_level_high'] = 1
                    sample_features[data_type]['temperature_level_warm'] = 1
                    sample_features[data_type]['cloud_level_clear'] = 1
                    sample_features[data_type]['season_2'] = 1
        
        return sample_features

# 初始化应用
solar_app = SolarForecastApp()

@app.route('/')
def index():
    """主页"""
    # 获取模型状态
    model_status = solar_app.get_model_status()
    
    # 获取评估结果
    evaluation_results = solar_app.evaluation_results
    
    # 创建对比图表
    comparison_chart = solar_app.create_comparison_chart()
    
    # 获取数据统计
    data_stats = solar_app.data_stats
    
    # 获取示例特征
    sample_features = solar_app.get_sample_features()
    
    # 加载现有图表
    model_images = {}
    for data_type in ['hourly', 'daily', 'weekly']:
        predictions_path = f'model_output/predictions_{data_type}.png'
        residuals_path = f'model_output/residuals_{data_type}.png'
        importance_path = f'model_output/feature_importance_plot_{data_type}.png'
        
        model_images[data_type] = {
            'predictions_exists': os.path.exists(predictions_path),
            'residuals_exists': os.path.exists(residuals_path),
            'importance_exists': os.path.exists(importance_path),
        }
    
    return render_template('index.html',
                         model_status=model_status,
                         evaluation_results=evaluation_results,
                         comparison_chart=comparison_chart,
                         data_stats=data_stats,
                         sample_features=sample_features,
                         model_images=model_images)

@app.route('/api/model_status')
def api_model_status():
    """API: 获取模型状态"""
    return jsonify(solar_app.get_model_status())

@app.route('/api/evaluation_results')
def api_evaluation_results():
    """API: 获取评估结果"""
    return jsonify(solar_app.evaluation_results)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API: 进行预测"""
    try:
        data = request.json
        data_type = data.get('data_type', 'hourly')
        
        if data_type == 'hourly':
            result = solar_app.predict_hourly(data.get('features', {}))
            if result:
                return jsonify({
                    'success': True,
                    'data_type': data_type,
                    'prediction': result
                })
            else:
                return jsonify({
                    'success': False,
                    'error': '预测失败'
                })
        else:
            return jsonify({
                'success': False,
                'error': f'预测类型 {data_type} 尚未实现'
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

@app.route('/api/sample_features')
def api_sample_features():
    """API: 获取示例特征"""
    return jsonify(solar_app.get_sample_features())

@app.route('/images/<data_type>/<image_type>')
def get_model_image(data_type, image_type):
    """获取模型图片"""
    if image_type == 'predictions':
        filename = f'predictions_{data_type}.png'
    elif image_type == 'residuals':
        filename = f'residuals_{data_type}.png'
    elif image_type == 'importance':
        filename = f'feature_importance_plot_{data_type}.png'
    else:
        return "图片类型不存在", 404
    
    filepath = os.path.join('model_output', filename)
    
    if os.path.exists(filepath):
        return send_from_directory('model_output', filename)
    else:
        return "图片未找到", 404

@app.route('/api/system_info')
def api_system_info():
    """API: 获取系统信息"""
    try:
        import psutil
        psutil_available = True
    except ImportError:
        psutil_available = False
    
    system_info = {
        'python_version': platform.python_version(),
        'system': platform.system(),
        'machine': platform.machine(),
        'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'app_uptime': str(datetime.now() - app_start_time),
        'models_loaded': len(solar_app.models),
        'psutil_available': psutil_available
    }
    
    if psutil_available:
        try:
            import psutil
            system_info.update({
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent
            })
        except:
            pass
    
    return jsonify(system_info)

# 创建必要的目录
os.makedirs('templates', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('static/images', exist_ok=True)

# 记录应用启动时间
app_start_time = datetime.now()

# 创建HTML模板
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write('''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>太阳能发电预测系统</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #3498db;
            --secondary-color: #2ecc71;
            --accent-color: #e74c3c;
            --dark-color: #2c3e50;
            --light-color: #ecf0f1;
        }
        
        body {
            background-color: #f8f9fa;
            font-family: 'Microsoft YaHei', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #333;
            line-height: 1.6;
        }
        
        .header {
            background: linear-gradient(135deg, var(--primary-color) 0%, #2980b9 100%);
            color: white;
            padding: 2.5rem 0;
            margin-bottom: 2rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        .header h1 {
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .header .lead {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .card {
            border-radius: 12px;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.08);
            margin-bottom: 2rem;
            border: none;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            overflow: hidden;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.12);
        }
        
        .card-header {
            background: linear-gradient(to right, #f8f9fa, #e9ecef);
            border-bottom: 2px solid var(--primary-color);
            font-weight: 600;
            padding: 1.2rem 1.5rem;
            font-size: 1.1rem;
        }
        
        .model-card {
            border-left: 6px solid var(--primary-color);
        }
        
        .model-card.daily {
            border-left-color: var(--secondary-color);
        }
        
        .model-card.weekly {
            border-left-color: var(--accent-color);
        }
        
        .metric-card {
            text-align: center;
            padding: 1.5rem 1rem;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
            height: 100%;
        }
        
        .metric-value {
            font-size: 2.2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            color: var(--dark-color);
        }
        
        .metric-label {
            color: #7f8c8d;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }
        
        .good {
            color: #27ae60;
        }
        
        .fair {
            color: #f39c12;
        }
        
        .poor {
            color: #e74c3c;
        }
        
        .chart-container {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        }
        
        .chart-container h6 {
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--dark-color);
            border-left: 4px solid var(--primary-color);
            padding-left: 10px;
        }
        
        .status-badge {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .status-active {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status-inactive {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .feature-table {
            font-size: 0.85rem;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .nav-tabs {
            border-bottom: 2px solid #dee2e6;
            margin-bottom: 1.5rem;
        }
        
        .nav-tabs .nav-link {
            border: none;
            color: #6c757d;
            font-weight: 500;
            padding: 0.8rem 1.5rem;
            border-radius: 8px 8px 0 0;
            margin-right: 5px;
            transition: all 0.3s;
        }
        
        .nav-tabs .nav-link:hover {
            color: var(--primary-color);
            background-color: rgba(52, 152, 219, 0.05);
        }
        
        .nav-tabs .nav-link.active {
            background-color: white;
            color: var(--primary-color);
            border-bottom: 3px solid var(--primary-color);
            font-weight: 600;
        }
        
        footer {
            background-color: var(--dark-color);
            color: white;
            padding: 2.5rem 0;
            margin-top: 3rem;
        }
        
        .last-updated {
            font-size: 0.85rem;
            color: #95a5a6;
        }
        
        .system-info {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 1.2rem;
            font-size: 0.9rem;
        }
        
        .prediction-result {
            background: linear-gradient(135deg, #e8f4fd 0%, #f0f7ff 100%);
            border-radius: 10px;
            padding: 1.5rem;
            margin-top: 1rem;
            border: 1px solid #d1e7ff;
        }
        
        .model-image {
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 10px;
            background-color: white;
            width: 100%;
            height: auto;
            margin-bottom: 1.5rem;
            transition: transform 0.3s ease;
        }
        
        .model-image:hover {
            transform: scale(1.02);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        }
        
        .image-gallery {
            margin-top: 1.5rem;
        }
        
        .image-gallery .col-md-6, .image-gallery .col-md-12 {
            margin-bottom: 1.5rem;
        }
        
        .github-link {
            color: #333;
            text-decoration: none;
            transition: color 0.3s;
        }
        
        .github-link:hover {
            color: var(--primary-color);
        }
        
        .btn-custom {
            background: linear-gradient(to right, var(--primary-color), #2980b9);
            color: white;
            border: none;
            padding: 0.7rem 1.5rem;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s;
        }
        
        .btn-custom:hover {
            background: linear-gradient(to right, #2980b9, var(--primary-color));
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
        }
        
        .alert-custom {
            border-radius: 10px;
            border-left: 5px solid var(--primary-color);
        }
        
        .feature-pill {
            display: inline-block;
            background-color: #e8f4fd;
            color: var(--primary-color);
            padding: 4px 12px;
            border-radius: 20px;
            margin: 3px;
            font-size: 0.85rem;
            border: 1px solid #d1e7ff;
        }
        
        @media (max-width: 768px) {
            .header {
                padding: 1.5rem 0;
            }
            
            .header h1 {
                font-size: 1.8rem;
            }
            
            .metric-value {
                font-size: 1.8rem;
            }
            
            .nav-tabs .nav-link {
                padding: 0.6rem 1rem;
                font-size: 0.9rem;
            }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <div class="header">
        <div class="container">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <h1><i class="fas fa-sun me-2"></i>太阳能发电预测系统</h1>
                    <p class="lead mb-0">基于XGBoost机器学习模型的太阳能发电量多时间尺度预测平台</p>
                    <p class="mb-0 mt-2"><small>甘肃酒泉50MW光伏电站 | 2020-2025年历史数据 | 三模型预测体系</small></p>
                </div>
                <div class="col-md-4 text-end">
                    <div class="system-info">
                        <div><i class="fas fa-microchip me-2"></i>模型状态: <span id="model-count">0</span>/3 已加载</div>
                        <div><i class="fas fa-clock me-2"></i>更新时间: <span id="last-updated">加载中...</span></div>
                        <div><i class="fas fa-server me-2"></i>服务器: 43.159.52.233:5002</div>
                        <div class="mt-2">
                            <a href="https://github.com/Cililin/solar_power_forecast" target="_blank" class="github-link">
                                <i class="fab fa-github me-1"></i>GitHub项目地址
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="container">
        <!-- 系统概览 -->
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0"><i class="fas fa-info-circle me-2"></i>系统概览</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-8">
                        <p>本系统基于XGBoost机器学习算法，提供太阳能发电量的多时间尺度预测，包含三个独立训练的模型：</p>
                        <ul>
                            <li><strong>小时级预测:</strong> 短期预测（未来1-24小时），适用于实时调度和运营</li>
                            <li><strong>日级预测:</strong> 中期预测（未来1-7天），适用于发电计划和设备维护</li>
                            <li><strong>周级预测:</strong> 长期预测（未来1-4周），适用于能源管理和投资分析</li>
                        </ul>
                        <p>所有模型均基于甘肃酒泉50MW光伏电站2020-2025年的历史数据训练，考虑了天气、时间、季节等多种因素。</p>
                    </div>
                    <div class="col-md-4">
                        <div class="alert alert-custom alert-info">
                            <h6><i class="fas fa-lightbulb me-2"></i>系统特色:</h6>
                            <ul class="mb-0">
                                <li>多时间尺度预测体系</li>
                                <li>基于XGBoost的机器学习模型</li>
                                <li>实时预测与可视化</li>
                                <li>完整的模型评估指标</li>
                                <li>RESTful API接口</li>
                                <li>响应式Web界面</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 模型性能对比 -->
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0"><i class="fas fa-chart-bar me-2"></i>模型性能对比</h5>
            </div>
            <div class="card-body">
                {% if comparison_chart %}
                <div class="text-center">
                    <img src="{{ comparison_chart }}" alt="模型对比图" class="img-fluid" style="max-width: 100%;">
                </div>
                {% else %}
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>对比图表不可用，请检查模型是否已正确加载。
                </div>
                {% endif %}
            </div>
        </div>

        <!-- 模型详情标签页 -->
        <ul class="nav nav-tabs" id="modelTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="hourly-tab" data-bs-toggle="tab" data-bs-target="#hourly" type="button" role="tab">
                    <i class="fas fa-clock me-1"></i>小时级模型
                </button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="daily-tab" data-bs-toggle="tab" data-bs-target="#daily" type="button" role="tab">
                    <i class="fas fa-calendar-day me-1"></i>日级模型
                </button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="weekly-tab" data-bs-toggle="tab" data-bs-target="#weekly" type="button" role="tab">
                    <i class="fas fa-calendar-week me-1"></i>周级模型
                </button>
            </li>
        </ul>

        <div class="tab-content" id="modelTabContent">
            {% for data_type in ['hourly', 'daily', 'weekly'] %}
            <div class="tab-pane fade show {{ 'active' if loop.first else '' }}" id="{{ data_type }}" role="tabpanel">
                <div class="card model-card {{ data_type }}">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">
                            {% if data_type == 'hourly' %}
                            <i class="fas fa-clock me-2"></i>小时级预测模型
                            {% elif data_type == 'daily' %}
                            <i class="fas fa-calendar-day me-2"></i>日级预测模型
                            {% else %}
                            <i class="fas fa-calendar-week me-2"></i>周级预测模型
                            {% endif %}
                        </h5>
                        <span class="status-badge status-{{ 'active' if model_status[data_type] else 'inactive' }}">
                            {{ '已加载' if model_status[data_type] else '未加载' }}
                        </span>
                    </div>
                    <div class="card-body">
                        <!-- 性能指标 -->
                        <div class="row mb-4">
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <div class="metric-value {{ 'good' if evaluation_results.get(data_type, {}).get('mae', 100) < 5 else 'fair' if evaluation_results.get(data_type, {}).get('mae', 100) < 10 else 'poor' }}">
                                        {{ "%.2f"|format(evaluation_results.get(data_type, {}).get('mae', 0)) }}
                                    </div>
                                    <div class="metric-label">平均绝对误差 (MAE)</div>
                                    <small class="text-muted">预测误差的平均值</small>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <div class="metric-value {{ 'good' if evaluation_results.get(data_type, {}).get('rmse', 100) < 5 else 'fair' if evaluation_results.get(data_type, {}).get('rmse', 100) < 10 else 'poor' }}">
                                        {{ "%.2f"|format(evaluation_results.get(data_type, {}).get('rmse', 0)) }}
                                    </div>
                                    <div class="metric-label">均方根误差 (RMSE)</div>
                                    <small class="text-muted">误差的平方平均根</small>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <div class="metric-value {{ 'good' if evaluation_results.get(data_type, {}).get('r2', 0) > 0.8 else 'fair' if evaluation_results.get(data_type, {}).get('r2', 0) > 0.5 else 'poor' }}">
                                        {{ "%.3f"|format(evaluation_results.get(data_type, {}).get('r2', 0)) }}
                                    </div>
                                    <div class="metric-label">决定系数 (R²)</div>
                                    <small class="text-muted">模型解释能力</small>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <div class="metric-value {{ 'good' if evaluation_results.get(data_type, {}).get('mape_alt', 100) < 10 else 'fair' if evaluation_results.get(data_type, {}).get('mape_alt', 100) < 20 else 'poor' }}">
                                        {{ "%.2f"|format(evaluation_results.get(data_type, {}).get('mape_alt', 0)) }}
                                    </div>
                                    <div class="metric-label">平均绝对百分比误差 (MAPE)</div>
                                    <small class="text-muted">相对误差百分比</small>
                                </div>
                            </div>
                        </div>

                        <!-- 模型信息和图表 -->
                        <div class="row">
                            <div class="col-md-4">
                                <div class="card mb-3">
                                    <div class="card-header">
                                        <h6 class="mb-0"><i class="fas fa-info-circle me-2"></i>模型信息</h6>
                                    </div>
                                    <div class="card-body">
                                        <p><strong>时间尺度:</strong> {{ '小时' if data_type == 'hourly' else '天' if data_type == 'daily' else '周' }}</p>
                                        <p><strong>测试样本数:</strong> {{ evaluation_results.get(data_type, {}).get('test_size', 0) }}</p>
                                        <p><strong>测试时间范围:</strong><br>
                                            {{ evaluation_results.get(data_type, {}).get('test_date_range', {}).get('start', 'N/A') }}<br>
                                            至 {{ evaluation_results.get(data_type, {}).get('test_date_range', {}).get('end', 'N/A') }}
                                        </p>
                                        <p><strong>XGBoost参数:</strong><br>
                                            <small>
                                                {% if data_type == 'hourly' %}
                                                树数量: 300 | 最大深度: 5 | 学习率: 0.05<br>
                                                正则化: L1=0.1, L2=1.0 | 采样: 80%
                                                {% elif data_type == 'daily' %}
                                                树数量: 200 | 最大深度: 4 | 学习率: 0.05<br>
                                                正则化: L1=0.15, L2=1.2 | 采样: 85%
                                                {% else %}
                                                树数量: 150 | 最大深度: 3 | 学习率: 0.05<br>
                                                正则化: L1=0.2, L2=1.5 | 采样: 90%
                                                {% endif %}
                                            </small>
                                        </p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-8">
                                <!-- 模型图表 -->
                                <div class="image-gallery">
                                    <div class="row">
                                        <div class="col-md-12">
                                            {% if model_images[data_type].predictions_exists %}
                                            <div class="chart-container">
                                                <h6><i class="fas fa-chart-line me-2"></i>预测值 vs 实际值对比</h6>
                                                <p class="text-muted mb-3">模型在测试集上的预测表现（蓝色为实际值，红色为预测值）</p>
                                                <img src="/images/{{ data_type }}/predictions" alt="预测对比图" class="model-image">
                                            </div>
                                            {% endif %}
                                        </div>
                                    </div>
                                    <div class="row">
                                        <div class="col-md-6">
                                            {% if model_images[data_type].residuals_exists %}
                                            <div class="chart-container">
                                                <h6><i class="fas fa-chart-area me-2"></i>残差分析</h6>
                                                <p class="text-muted mb-3">预测误差的统计分析</p>
                                                <img src="/images/{{ data_type }}/residuals" alt="残差分析图" class="model-image">
                                            </div>
                                            {% endif %}
                                        </div>
                                        <div class="col-md-6">
                                            {% if model_images[data_type].importance_exists %}
                                            <div class="chart-container">
                                                <h6><i class="fas fa-star me-2"></i>特征重要性</h6>
                                                <p class="text-muted mb-3">影响预测结果的关键特征</p>
                                                <img src="/images/{{ data_type }}/importance" alt="特征重要性图" class="model-image">
                                            </div>
                                            {% endif %}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>

        <!-- 实时预测演示 -->
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0"><i class="fas fa-bolt me-2"></i>实时预测演示</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <div class="mb-4">
                            <label class="form-label fw-bold">选择预测类型</label>
                            <select class="form-select" id="forecast-type">
                                <option value="hourly">小时级预测</option>
                            </select>
                        </div>
                        
                        <div class="mb-4">
                            <label class="form-label fw-bold">小时级模型</label>
                            <div class="feature-table">
                                <div class="mb-3">
                                    {% for feature, value in sample_features.hourly.items() %}
                                    {% if loop.index <= 15 %}
                                    <span class="feature-pill">{{ feature }}: {{ "%.1f"|format(value) if value is number else value }}</span>
                                    {% endif %}
                                    {% endfor %}
                                </div>
                                <button class="btn btn-custom" onclick="runPrediction()">
                                    <i class="fas fa-play me-1"></i>运行示例预测
                                </button>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div id="prediction-result" style="display: none;">
                            <h5><i class="fas fa-chart-line me-2"></i>预测结果</h5>
                            <div class="prediction-result">
                                <div class="row mb-4">
                                    <div class="col-md-6">
                                        <h1 id="prediction-rate" class="text-primary">0.00%</h1>
                                        <div class="metric-label">发电率预测</div>
                                        <small class="text-muted">占最大发电能力的百分比</small>
                                    </div>
                                    <div class="col-md-6">
                                        <h2 id="prediction-power" class="text-success">0.00 kWh</h2>
                                        <div class="metric-label">功率输出预测</div>
                                        <small class="text-muted">转换为电能单位</small>
                                    </div>
                                </div>
                                <div class="mt-3">
                                    <h6><i class="fas fa-lightbulb me-2"></i>结果解读</h6>
                                    <p id="prediction-interpretation">预测结果将显示在这里。</p>
                                    <div class="alert alert-info mt-3">
                                        <i class="fas fa-info-circle me-2"></i>
                                        <small>发电率表示实际发电量占电站最大发电能力（50MW）的百分比。功率输出是转换为电能单位后的预测值。</small>
                                    </div>
                                    <p class="last-updated mt-3">预测时间: <span id="prediction-time">-</span></p>
                                </div>
                            </div>
                        </div>
                        <div id="prediction-placeholder">
                            <div class="alert alert-info">
                                <i class="fas fa-info-circle me-2"></i>
                                点击"运行示例预测"按钮，使用小时级模型进行预测演示。
                                预测结果将显示预期的太阳能发电率和功率输出。
                            </div>
                            <div class="text-center mt-4">
                                <i class="fas fa-sun fa-4x text-warning"></i>
                                <p class="mt-3 text-muted">等待预测演示...</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 数据统计 -->
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0"><i class="fas fa-database me-2"></i>数据统计</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h6>数据集概览</h6>
                        <ul class="list-group list-group-flush">
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                总记录数
                                <span class="badge bg-primary rounded-pill">{{ data_stats.get('data_shape', [0, 0])[0]|default(0) }}</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                总特征数
                                <span class="badge bg-primary rounded-pill">{{ data_stats.get('data_shape', [0, 0])[1]|default(0) }}</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                时间范围
                                <span class="badge bg-info rounded-pill">{{ data_stats.get('time_range', {}).get('start', 'N/A') }} 至 {{ data_stats.get('time_range', {}).get('end', 'N/A') }}</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                电站容量
                                <span class="badge bg-success rounded-pill">50 MW</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                地理位置
                                <span class="badge bg-warning rounded-pill">酒泉, 甘肃 (39.70°N, 98.50°E)</span>
                            </li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h6>发电统计</h6>
                        <ul class="list-group list-group-flush">
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                平均发电率
                                <span class="badge bg-primary rounded-pill">{{ "%.2f"|format(data_stats.get('generation_rate_stats', {}).get('mean', 0)) }}%</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                标准差
                                <span class="badge bg-primary rounded-pill">{{ "%.2f"|format(data_stats.get('generation_rate_stats', {}).get('std', 0)) }}%</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                最小值
                                <span class="badge bg-success rounded-pill">{{ "%.2f"|format(data_stats.get('generation_rate_stats', {}).get('min', 0)) }}%</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                最大值
                                <span class="badge bg-danger rounded-pill">{{ "%.2f"|format(data_stats.get('generation_rate_stats', {}).get('max', 0)) }}%</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                模拟总发电量
                                <span class="badge bg-info rounded-pill">~{{ (data_stats.get('data_shape', [0, 0])[0] * 23.27 * 50000 / 100 / 1000)|round|int if data_stats.get('data_shape') else 0 }} MWh</span>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <!-- 系统信息 -->
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0"><i class="fas fa-server me-2"></i>系统信息</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h6>应用状态</h6>
                        <ul class="list-group list-group-flush">
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                Flask应用
                                <span class="badge bg-success rounded-pill">运行中 (端口: 5002)</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                模型加载状态
                                <span id="loaded-models" class="badge bg-primary rounded-pill">加载中...</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                运行时间
                                <span id="uptime" class="badge bg-info rounded-pill">加载中...</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                最后模型更新
                                <span id="model-update" class="badge bg-warning rounded-pill">加载中...</span>
                            </li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h6>技术栈</h6>
                        <div class="row">
                            <div class="col-6">
                                <div class="text-center mb-3">
                                    <i class="fas fa-python fa-2x text-primary mb-2"></i>
                                    <p class="mb-0">Python 3.8</p>
                                    <small class="text-muted">后端语言</small>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="text-center mb-3">
                                    <i class="fas fa-brain fa-2x text-success mb-2"></i>
                                    <p class="mb-0">XGBoost</p>
                                    <small class="text-muted">机器学习框架</small>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="text-center mb-3">
                                    <i class="fas fa-chart-line fa-2x text-danger mb-2"></i>
                                    <p class="mb-0">Flask</p>
                                    <small class="text-muted">Web框架</small>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="text-center mb-3">
                                    <i class="fab fa-bootstrap fa-2x text-purple mb-2"></i>
                                    <p class="mb-0">Bootstrap</p>
                                    <small class="text-muted">图表</small>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Footer -->
    <footer>
        <div class="container">
            <div class="row">
                <div class="col-md-6">
                    <h5>太阳能发电预测系统</h5>
                    <p>基于XGBoost的太阳能发电量多时间尺度预测平台，为可再生能源运营和研究提供数据支持。</p>
                    <p class="last-updated">系统版本: v1.0 | 最后训练: 2026-01-13</p>
                </div>
                <div class="col-md-3">
                    <h5>模型</h5>
                    <ul class="list-unstyled">
                        <li><a href="#hourly" class="text-light text-decoration-none">小时级预测</a></li>
                        <li><a href="#daily" class="text-light text-decoration-none">日级预测</a></li>
                        <li><a href="#weekly" class="text-light text-decoration-none">周级预测</a></li>
                    </ul>
                </div>
                <div class="col-md-3">
                    <h5>API接口</h5>
                    <ul class="list-unstyled">
                        <li><a href="/api/model_status" class="text-light text-decoration-none">/api/model_status</a></li>
                        <li><a href="/api/evaluation_results" class="text-light text-decoration-none">/api/evaluation_results</a></li>
                        <li><a href="/api/predict" class="text-light text-decoration-none">/api/predict</a></li>
                        <li><a href="/api/system_info" class="text-light text-decoration-none">/api/system_info</a></li>
                    </ul>
                </div>
            </div>
            <hr class="bg-light">
            <div class="row">
                <div class="col-md-12 text-center">
                    <p class="mb-0">
                        <i class="fas fa-sun me-1"></i> 太阳能发电预测系统 | 
                        <i class="fas fa-server ms-3 me-1"></i> 服务器: 43.159.52.233:5002 | 
                        <i class="fab fa-github ms-3 me-1"></i> GitHub: <a href="https://github.com/Cililin/solar_power_forecast" target="_blank" class="text-light text-decoration-none">项目地址</a> | 
                        <i class="fas fa-clock ms-3 me-1"></i> 最后更新: <span id="footer-updated">加载中...</span>
                    </p>
                </div>
            </div>
        </div>
    </footer>

    <!-- JavaScript -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // 初始化页面
        document.addEventListener('DOMContentLoaded', function() {
            updateSystemInfo();
            updateLastUpdated();
            
            // 每30秒更新系统信息
            setInterval(updateSystemInfo, 30000);
            // 每分钟更新时间
            setInterval(updateLastUpdated, 60000);
        });
        
        function updateSystemInfo() {
            fetch('/api/model_status')
                .then(response => response.json())
                .then(data => {
                    let loadedCount = 0;
                    if (data.hourly) loadedCount++;
                    if (data.daily) loadedCount++;
                    if (data.weekly) loadedCount++;
                    
                    document.getElementById('model-count').textContent = loadedCount;
                    document.getElementById('loaded-models').textContent = loadedCount + '/3 模型已加载';
                })
                .catch(error => console.error('获取模型状态失败:', error));
            
            fetch('/api/system_info')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('uptime').textContent = data.app_uptime;
                    document.getElementById('model-update').textContent = data.current_time;
                })
                .catch(error => console.error('获取系统信息失败:', error));
        }
        
        function updateLastUpdated() {
            const now = new Date();
            const timeStr = now.toLocaleString('zh-CN', {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
            
            document.getElementById('last-updated').textContent = timeStr;
            document.getElementById('footer-updated').textContent = timeStr;
        }
        
        function runPrediction() {
            const forecastType = document.getElementById('forecast-type').value;
            
            // 使用示例特征进行预测
            fetch('/api/sample_features')
                .then(response => response.json())
                .then(sampleFeatures => {
                    const features = sampleFeatures[forecastType] || sampleFeatures.hourly;
                    
                    return fetch('/api/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            data_type: forecastType,
                            features: features
                        })
                    });
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const prediction = data.prediction;
                        const rate = prediction.generation_rate.toFixed(2);
                        const powerKwh = prediction.power_kwh.toFixed(0);
                        const powerMw = prediction.power_mw.toFixed(2);
                        
                        document.getElementById('prediction-rate').textContent = rate + '%';
                        document.getElementById('prediction-power').textContent = powerKwh + ' kWh (' + powerMw + ' MW)';
                        document.getElementById('prediction-time').textContent = 
                            new Date().toLocaleString('zh-CN', {
                                year: 'numeric',
                                month: '2-digit',
                                day: '2-digit',
                                hour: '2-digit',
                                minute: '2-digit',
                                second: '2-digit'
                            });
                        
                        // 结果解读
                        let interpretation = '';
                        if (rate > 50) {
                            interpretation = '高发电率预期。太阳能发电条件良好，适合满负荷运行。';
                        } else if (rate > 20) {
                            interpretation = '中等发电率预期。正常发电条件，建议按计划运行。';
                        } else if (rate > 5) {
                            interpretation = '低发电率预期。可能是多云天气或早晚时段，发电效率较低。';
                        } else {
                            interpretation = '极低或无发电预期。夜间或恶劣天气条件，建议检查设备状态。';
                        }
                        
                        document.getElementById('prediction-interpretation').textContent = interpretation;
                        
                        // 显示结果
                        document.getElementById('prediction-placeholder').style.display = 'none';
                        document.getElementById('prediction-result').style.display = 'block';
                        
                        // 添加成功提示
                        showToast('预测成功', '模型预测完成，结果显示在右侧。', 'success');
                    } else {
                        showToast('预测失败', data.error || '未知错误', 'error');
                    }
                })
                .catch(error => {
                    console.error('预测请求失败:', error);
                    showToast('请求失败', '请检查网络连接或控制台错误信息。', 'error');
                });
        }
        
        function showToast(title, message, type) {
            // 创建toast元素
            const toastId = 'toast-' + Date.now();
            const toastHtml = `
                <div id="${toastId}" class="toast align-items-center text-white bg-${type === 'success' ? 'success' : 'danger'} border-0" role="alert" aria-live="assertive" aria-atomic="true">
                    <div class="d-flex">
                        <div class="toast-body">
                            <strong>${title}</strong>: ${message}
                        </div>
                        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
                    </div>
                </div>
            `;
            
            // 添加到toast容器
            let toastContainer = document.getElementById('toast-container');
            if (!toastContainer) {
                toastContainer = document.createElement('div');
                toastContainer.id = 'toast-container';
                toastContainer.className = 'position-fixed bottom-0 end-0 p-3';
                toastContainer.style.zIndex = '1050';
                document.body.appendChild(toastContainer);
            }
            
            toastContainer.innerHTML += toastHtml;
            
            // 显示toast
            const toastElement = document.getElementById(toastId);
            const toast = new bootstrap.Toast(toastElement, { delay: 3000 });
            toast.show();
            
            // 移除toast元素
            toastElement.addEventListener('hidden.bs.toast', function () {
                toastElement.remove();
            });
        }
        
        // 标签页功能
        document.querySelectorAll('#modelTab button').forEach(button => {
            button.addEventListener('click', event => {
                const tabId = event.target.getAttribute('data-bs-target').substring(1);
                localStorage.setItem('activeTab', tabId);
            });
        });
        
        // 恢复活动标签页
        const activeTab = localStorage.getItem('activeTab');
        if (activeTab) {
            const tabButton = document.querySelector(`[data-bs-target="#${activeTab}"]`);
            if (tabButton) {
                new bootstrap.Tab(tabButton).show();
            }
        }
    </script>
</body>
</html>
''')

# 创建CSS文件
with open('static/css/style.css', 'w', encoding='utf-8') as f:
    f.write('''
/* 自定义样式补充 */
.toast {
    border-radius: 10px;
}

.text-purple {
    color: #6f42c1 !important;
}

.badge.rounded-pill {
    font-size: 0.85rem;
    padding: 0.4em 0.8em;
}

.list-group-item {
    border: none;
    padding: 0.75rem 0;
}

.feature-pill {
    display: inline-block;
    background-color: #e8f4fd;
    color: #3498db;
    padding: 4px 12px;
    border-radius: 20px;
    margin: 3px;
    font-size: 0.85rem;
    border: 1px solid #d1e7ff;
}

@media (max-width: 768px) {
    .header h1 {
        font-size: 1.6rem;
    }
    
    .metric-value {
        font-size: 1.6rem;
    }
    
    .btn-custom {
        width: 100%;
    }
}
''')

if __name__ == '__main__':
    print("="*80)
    print("太阳能发电预测")
    print("="*80)
    print(f"访问地址: http://43.159.52.233:5002")
    print(f"本地访问: http://127.0.0.1:5002")
    print(f"或 http://10.3.0.7:5002")
    
    # 运行应用
    app.run(host='0.0.0.0', port=5002, debug=False)