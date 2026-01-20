#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
太阳能发电预测Flask应用
optimized_train_models.py 训练出来的模型结果
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from flask import Flask, render_template, jsonify, request, send_from_directory
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import traceback
import platform
import sys

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

class OptimizedSolarForecastApp:
    """太阳能预测应用类"""
    
    def __init__(self):
        """初始化应用"""
        self.models = {}
        self.scalers = {}
        self.feature_info = {}
        self.evaluation_results = {}
        self.data_stats = {}
        self.prediction_images = {}
        
        # 加载模型和配置
        self.load_models()
        self.load_evaluation_results()
        self.load_data_stats()
        self.load_prediction_images()
    
    def load_models(self):
        """加载训练好的模型"""
        model_dir = 'final_output'
        
        print("="*80)
        print("加载训练模型...")
        print("="*80)
        
        try:
            # 加载小时模型
            hourly_model_path = os.path.join(model_dir, 'xgboost_hourly.pkl')
            if os.path.exists(hourly_model_path):
                self.models['hourly'] = joblib.load(hourly_model_path)
                print(f"✓ 加载小时模型成功")
            else:
                print(f"✗ 小时模型文件不存在: {hourly_model_path}")
        
        except Exception as e:
            print(f"加载小时模型失败: {str(e)}")
        
        try:
            # 加载日模型
            daily_model_path = os.path.join(model_dir, 'xgboost_daily.pkl')
            if os.path.exists(daily_model_path):
                self.models['daily'] = joblib.load(daily_model_path)
                print(f"✓ 加载日模型成功")
            else:
                print(f"✗ 日模型文件不存在: {daily_model_path}")
        
        except Exception as e:
            print(f"加载日模型失败: {str(e)}")
        
        try:
            # 加载周模型
            weekly_model_path = os.path.join(model_dir, 'xgboost_weekly.pkl')
            if os.path.exists(weekly_model_path):
                self.models['weekly'] = joblib.load(weekly_model_path)
                print(f"✓ 加载周模型成功")
            else:
                print(f"✗ 周模型文件不存在: {weekly_model_path}")
        
        except Exception as e:
            print(f"加载周模型失败: {str(e)}")
        
        print(f"总计加载模型: {len(self.models)}/3")
        print("="*80)
    
    def load_evaluation_results(self):
        """加载模型的评估结果"""
        output_dir = 'final_output'
        eval_path = os.path.join(output_dir, 'evaluation_results.json')
        
        print(f"加载评估结果: {eval_path}")
        
        if os.path.exists(eval_path):
            try:
                with open(eval_path, 'r', encoding='utf-8') as f:
                    self.evaluation_results = json.load(f)
                print(f"✓ 加载评估结果成功")
                
                # 转换参数格式
                for data_type in self.evaluation_results:
                    if 'params' in self.evaluation_results[data_type]:
                        try:
                            # 将字符串参数转换回字典（如果可能）
                            params_str = self.evaluation_results[data_type]['params']
                            if isinstance(params_str, str) and params_str.startswith('{'):
                                import ast
                                self.evaluation_results[data_type]['params'] = ast.literal_eval(params_str)
                        except:
                            pass
                
                # 显示评估结果
                for data_type, results in self.evaluation_results.items():
                    print(f"\n{data_type.upper()} 模型评估:")
                    print(f"  R²:   {results.get('r2', 0):.4f}")
                    print(f"  MAE:  {results.get('mae', 0):.4f}%")
                    print(f"  RMSE: {results.get('rmse', 0):.4f}%")
                    if results.get('wmape'):
                        print(f"  MAPE: {results.get('wmape', 0):.2f}%")
            
            except Exception as e:
                print(f"加载评估结果失败: {str(e)}")
                traceback.print_exc()
        else:
            print(f"✗ 评估结果文件不存在: {eval_path}")
    
    def load_data_stats(self):
        """加载数据统计信息"""
        stats_path = 'processed_data/data_statistics.json'
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r', encoding='utf-8') as f:
                    self.data_stats = json.load(f)
                print(f"✓ 加载数据统计信息成功")
            except Exception as e:
                print(f"加载数据统计信息失败: {str(e)}")
        else:
            print(f"✗ 数据统计文件不存在: {stats_path}")
    
    def load_prediction_images(self):
        """加载预测图表"""
        output_dir = 'final_output'
        
        print("加载预测图表...")
        
        # 检查图表文件是否存在
        for data_type in ['hourly', 'daily', 'weekly']:
            predictions_path = os.path.join(output_dir, f'predictions_{data_type}.png')
            importance_path = os.path.join(output_dir, f'feature_importance_{data_type}.png')
            
            self.prediction_images[data_type] = {
                'predictions_exists': os.path.exists(predictions_path),
                'importance_exists': os.path.exists(importance_path),
            }
            
            if os.path.exists(predictions_path):
                print(f"✓ {data_type} 预测图存在")
            else:
                print(f"✗ {data_type} 预测图不存在: {predictions_path}")
    
    def get_model_status(self):
        """获取模型状态"""
        status = {
            'hourly': 'hourly' in self.models,
            'daily': 'daily' in self.models,
            'weekly': 'weekly' in self.models,
            'total_models': len(self.models)
        }
        return status
    
    def create_optimized_comparison_chart(self):
        """创建模型的对比图表"""
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
                    
                    # 获取指标，使用wmape作为MAPE
                    mae_values.append(results.get('mae', 0))
                    rmse_values.append(results.get('rmse', 0))
                    r2_values.append(results.get('r2', 0))
                    
                    # 使用wmape或计算替代值
                    wmape = results.get('wmape')
                    if wmape is not None:
                        mape_values.append(wmape)
                    else:
                        # 如果没有wmape，使用mae作为近似值
                        mape_values.append(results.get('mae', 0))
            
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
            traceback.print_exc()
            return None
    
    def get_model_parameters(self, data_type):
        """获取模型参数"""
        if data_type in self.evaluation_results:
            params = self.evaluation_results[data_type].get('params', {})
            
            # 返回格式化的参数
            param_text = ""
            if isinstance(params, dict):
                for key, value in params.items():
                    param_text += f"{key}: {value}\n"
            elif isinstance(params, str):
                param_text = params
            
            return param_text
        return "参数未找到"
    
    def get_sample_features_for_demo(self):
        """获取演示用示例特征"""
        # 根据模型的特征创建示例
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
                'generation_rate_lag_1h': 45.2,
                'generation_rate_lag_24h': 42.1,
                'generation_rate_ma_3h': 43.5,
                'generation_rate_ma_24h': 40.2,
                'radiation_temperature_interaction': 450.5 * 25.3,
            },
            'daily': {
                'month': 6,
                'day_of_week': 2,
                'day_of_year': 172,
                'week_of_year': 25,
                'is_weekend': 0,
                'solar_radiation_w_m2': 350.5,
                'temperature_c': 24.3,
                'humidity_percent': 60.2,
                'wind_speed_10m_mps': 3.5,
                'cloud_index': 0.4,
                'generation_rate_lag_1d': 45.2,
                'generation_rate_lag_7d': 42.1,
                'generation_rate_ma_3d': 43.5,
            },
            'weekly': {
                'month': 6,
                'week_of_year': 25,
                'solar_radiation_w_m2': 320.5,
                'temperature_c': 23.3,
                'humidity_percent': 58.2,
                'wind_speed_10m_mps': 3.8,
                'cloud_index': 0.5,
                'generation_rate_lag_1w': 44.2,
                'generation_rate_lag_4w': 41.1,
                'generation_rate_ma_4w': 42.5,
            }
        }
        
        return sample_features

# 初始化应用
solar_app = OptimizedSolarForecastApp()

@app.route('/')
def index():
    """主页 - 展示模型结果"""
    # 获取模型状态
    model_status = solar_app.get_model_status()
    
    # 获取评估结果
    evaluation_results = solar_app.evaluation_results
    
    # 创建对比图表
    comparison_chart = solar_app.create_optimized_comparison_chart()
    
    # 获取数据统计
    data_stats = solar_app.data_stats
    
    # 获取示例特征
    sample_features = solar_app.get_sample_features_for_demo()
    
    # 获取预测图表信息
    prediction_images = solar_app.prediction_images
    
    # 获取模型参数
    model_params = {}
    for data_type in ['hourly', 'daily', 'weekly']:
        model_params[data_type] = solar_app.get_model_parameters(data_type)
    
    return render_template('optimized_index.html',
                         model_status=model_status,
                         evaluation_results=evaluation_results,
                         comparison_chart=comparison_chart,
                         data_stats=data_stats,
                         sample_features=sample_features,
                         prediction_images=prediction_images,
                         model_params=model_params)

@app.route('/api/model_status')
def api_model_status():
    """API: 获取模型状态"""
    return jsonify(solar_app.get_model_status())

@app.route('/api/evaluation_results')
def api_evaluation_results():
    """API: 获取评估结果"""
    return jsonify(solar_app.evaluation_results)

@app.route('/api/model_params')
def api_model_params():
    """API: 获取模型参数"""
    params = {}
    for data_type in ['hourly', 'daily', 'weekly']:
        params[data_type] = solar_app.get_model_parameters(data_type)
    return jsonify(params)

@app.route('/images/<data_type>/<image_type>')
def get_optimized_image(data_type, image_type):
    """获取模型图片"""
    if image_type == 'predictions':
        filename = f'predictions_{data_type}.png'
    elif image_type == 'importance':
        filename = f'feature_importance_{data_type}.png'
    else:
        return "图片类型不存在", 404
    
    filepath = os.path.join('final_output', filename)
    
    if os.path.exists(filepath):
        return send_from_directory('final_output', filename)
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
        'app_name': '太阳能发电预测系统',
        'models_loaded': len(solar_app.models),
        'model_source': 'optimized_train_models.py',
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
            })
        except:
            pass
    
    return jsonify(system_info)

@app.route('/api/sample_features')
def api_sample_features():
    """API: 获取示例特征"""
    return jsonify(solar_app.get_sample_features_for_demo())

# 创建必要的目录
os.makedirs('templates', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('static/images', exist_ok=True)

# 创建HTML模板
with open('templates/optimized_index.html', 'w', encoding='utf-8') as f:
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
            --secondary-color: #27ae60;
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
            background: linear-gradient(135deg, var(--secondary-color) 0%, #229954 100%);
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
            border-bottom: 2px solid var(--secondary-color);
            font-weight: 600;
            padding: 1.2rem 1.5rem;
            font-size: 1.1rem;
        }

        .github-link {
            color: #333;
            text-decoration: none;
            transition: color 0.3s;
        }
                
        .github-link:hover {
            color: var(--primary-color);
        }
        
        .model-card {
            border-left: 6px solid var(--secondary-color);
        }
        
        .model-card.daily {
            border-left-color: #f39c12;
        }
        
        .model-card.weekly {
            border-left-color: #9b59b6;
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
            color: var(--secondary-color);
            background-color: rgba(39, 174, 96, 0.05);
        }
        
        .nav-tabs .nav-link.active {
            background-color: white;
            color: var(--secondary-color);
            border-bottom: 3px solid var(--secondary-color);
            font-weight: 600;
        }
        
        footer {
            background-color: var(--dark-color);
            color: white;
            padding: 2.5rem 0;
            margin-top: 3rem;
        }
        
        .system-info {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 1.2rem;
            font-size: 0.9rem;
        }
        
        .model-image {
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 10px;
            background-color: white;
            width: 100%;
            height: auto;
            margin-bottom: 1.5rem;
        }
        
        .btn-optimized {
            background: linear-gradient(to right, var(--secondary-color), #229954);
            color: white;
            border: none;
            padding: 0.7rem 1.5rem;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s;
        }
        
        .btn-optimized:hover {
            background: linear-gradient(to right, #229954, var(--secondary-color));
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(39, 174, 96, 0.3);
        }
        
        .target-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .target-achieved {
            background-color: #27ae60;
        }
        
        .target-not-achieved {
            background-color: #e74c3c;
        }
        
        .param-box {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            font-family: monospace;
            font-size: 0.85rem;
            max-height: 300px;
            overflow-y: auto;
            white-space: pre-wrap;
        }
        
        .optimization-info {
            background: linear-gradient(135deg, #e8f6f3 0%, #d1f2eb 100%);
            border-left: 5px solid var(--secondary-color);
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
                    <p class="lead mb-0">基于XGBoost模型展示</p>
                </div>
                <div class="col-md-4 text-end">
                    <div class="system-info">
                        <div><i class="fas fa-microchip me-2"></i>模型状态: <span id="model-count">0</span>/3 已加载</div>
                        <div><i class="fas fa-clock me-2"></i>更新时间: <span id="last-updated">加载中...</span></div>
                        <div class="mt-2">
                            <a href="/api/system_info" class="text-light text-decoration-none">
                                <i class="fas fa-server me-1"></i>系统信息
                            </a>
                        </div>
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
        <!-- 信息 -->
        <div class="card optimization-info">
            <div class="card-header">
                <h5 class="mb-0"><i class="fas fa-star me-2"></i>模型说明</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-8">
                        <p>本页面展示由 <code>optimized_train_models.py</code> 训练出的模型结果。主要包括：</p>
                        <ul>
                            <li><strong>参数调整:</strong> 使用更合理的XGBoost参数（max_depth=3, n_estimators=300, learning_rate=0.05）</li>
                            <li><strong>特征优化:</strong> 选择最重要的特征，减少过拟合</li>
                            <li><strong>数据筛选:</strong> 小时模型仅使用白天数据（has_sunlight=1）</li>
                            <li><strong>评估改进:</strong> 使用加权MAPE解决零值问题</li>
                        </ul>
                    </div>
                    <div class="col-md-4">
                        <div class="alert alert-info">
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
                                    <div class="metric-value {{ 'good' if evaluation_results.get(data_type, {}).get('mae', 100) < 10 else 'fair' if evaluation_results.get(data_type, {}).get('mae', 100) < 15 else 'poor' }}">
                                        {{ "%.2f"|format(evaluation_results.get(data_type, {}).get('mae', 0)) if evaluation_results.get(data_type) else 'N/A' }}
                                    </div>
                                    <div class="metric-label">平均绝对误差 (MAE)</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <div class="metric-value {{ 'good' if evaluation_results.get(data_type, {}).get('rmse', 100) < 10 else 'fair' if evaluation_results.get(data_type, {}).get('rmse', 100) < 15 else 'poor' }}">
                                        {{ "%.2f"|format(evaluation_results.get(data_type, {}).get('rmse', 0)) if evaluation_results.get(data_type) else 'N/A' }}
                                    </div>
                                    <div class="metric-label">均方根误差 (RMSE)</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <div class="metric-value {{ 'good' if evaluation_results.get(data_type, {}).get('r2', 0) > 0.7 else 'fair' if evaluation_results.get(data_type, {}).get('r2', 0) > 0.5 else 'poor' }}">
                                        {{ "%.3f"|format(evaluation_results.get(data_type, {}).get('r2', 0)) if evaluation_results.get(data_type) else 'N/A' }}
                                    </div>
                                    <div class="metric-label">决定系数 (R²)</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <div class="metric-value {{ 'good' if evaluation_results.get(data_type, {}).get('wmape', 100) < 10 else 'fair' if evaluation_results.get(data_type, {}).get('wmape', 100) < 20 else 'poor' }}">
                                        {{ "%.2f"|format(evaluation_results.get(data_type, {}).get('wmape', 0)) if evaluation_results.get(data_type) and evaluation_results.get(data_type, {}).get('wmape') is not none else 'N/A' }}
                                    </div>
                                    <div class="metric-label">加权MAPE</div>
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
                                        <p><strong>测试样本数:</strong> {{ evaluation_results.get(data_type, {}).get('test_size', 0) if evaluation_results.get(data_type) else 'N/A' }}</p>
                                        {% if evaluation_results.get(data_type, {}).get('test_date_range') %}
                                        <p><strong>测试时间范围:</strong><br>
                                            {{ evaluation_results.get(data_type, {}).get('test_date_range', {}).get('start', 'N/A') }}<br>
                                            至 {{ evaluation_results.get(data_type, {}).get('test_date_range', {}).get('end', 'N/A') }}
                                        </p>
                                        {% endif %}
                                        <p><strong>模型参数:</strong></p>
                                        <div class="param-box">
                                            {{ model_params[data_type] }}
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-8">
                                <!-- 模型图表 -->
                                <div class="row">
                                    <div class="col-md-12">
                                        {% if prediction_images[data_type].predictions_exists %}
                                        <div class="chart-container">
                                            <h6><i class="fas fa-chart-line me-2"></i>预测值 vs 实际值对比</h6>
                                            <p class="text-muted mb-3">模型在测试集上的预测表现（蓝色为实际值，红色为预测值）</p>
                                            <img src="/images/{{ data_type }}/predictions" alt="预测对比图" class="model-image">
                                        </div>
                                        {% else %}
                                        <div class="alert alert-warning">
                                            <i class="fas fa-exclamation-triangle me-2"></i>预测图表不可用
                                        </div>
                                        {% endif %}
                                    </div>
                                </div>
                                <div class="row">
                                    <div class="col-md-12">
                                        {% if prediction_images[data_type].importance_exists %}
                                        <div class="chart-container">
                                            <h6><i class="fas fa-star me-2"></i>特征重要性 (Top 10)</h6>
                                            <p class="text-muted mb-3">影响预测结果的关键特征</p>
                                            <img src="/images/{{ data_type }}/importance" alt="特征重要性图" class="model-image">
                                        </div>
                                        {% else %}
                                        <div class="alert alert-warning">
                                            <i class="fas fa-exclamation-triangle me-2"></i>特征重要性图表不可用
                                        </div>
                                        {% endif %}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {% endfor %}
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
                            {% if data_stats.get('time_range') %}
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                时间范围
                                <span class="badge bg-info rounded-pill">{{ data_stats.get('time_range', {}).get('start', 'N/A') }} 至 {{ data_stats.get('time_range', {}).get('end', 'N/A') }}</span>
                            </li>
                            {% endif %}
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                电站容量
                                <span class="badge bg-success rounded-pill">50 MW</span>
                            </li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h6>发电统计</h6>
                        {% if data_stats.get('generation_rate_stats') %}
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
                        </ul>
                        {% else %}
                        <div class="alert alert-warning">
                            <i class="fas fa-exclamation-triangle me-2"></i>发电统计信息不可用
                        </div>
                        {% endif %}
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
                                <span class="badge bg-success rounded-pill">运行中</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                模型加载状态
                                <span id="loaded-models" class="badge bg-primary rounded-pill">加载中...</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                当前时间
                                <span id="current-time" class="badge bg-info rounded-pill">加载中...</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                模型版本
                                <span class="badge bg-warning rounded-pill">v2.0</span>
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
                                    <small class="text-muted">模型</small>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="text-center mb-3">
                                    <i class="fab fa-bootstrap fa-2x text-purple mb-2"></i>
                                    <p class="mb-0">Bootstrap</p>
                                    <small class="text-muted">图表</small>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="text-center mb-3">
                                    <i class="fas fa-flask fa-2x text-danger mb-2"></i>
                                    <p class="mb-0">Flask</p>
                                    <small class="text-muted">Web框架</small>
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
                    <p>基于XGBoost模型展示平台</p>
                    <p class="last-updated">模型版本 : v2.0</p>
                </div>
                <div class="col-md-3">
                    <h5>API接口</h5>
                    <ul class="list-unstyled">
                        <li><a href="/api/model_status" class="text-light text-decoration-none">/api/model_status</a></li>
                        <li><a href="/api/evaluation_results" class="text-light text-decoration-none">/api/evaluation_results</a></li>
                        <li><a href="/api/model_params" class="text-light text-decoration-none">/api/model_params</a></li>
                        <li><a href="/api/system_info" class="text-light text-decoration-none">/api/system_info</a></li>
                    </ul>
                </div>
                <div class="col-md-3">
                    <h5>模型文件</h5>
                    <ul class="list-unstyled">
                        <li><a href="javascript:void(0)" class="text-light text-decoration-none">xgboost_hourly.pkl</a></li>
                        <li><a href="javascript:void(0)" class="text-light text-decoration-none">xgboost_daily.pkl</a></li>
                        <li><a href="javascript:void(0)" class="text-light text-decoration-none">xgboost_weekly.pkl</a></li>
                        <li><a href="javascript:void(0)" class="text-light text-decoration-none">evaluation_results.json</a></li>
                    </ul>
                </div>
            </div>
            <hr class="bg-light">
            <div class="row">
                <div class="col-md-12 text-center">
                    <p class="mb-0">
                        <i class="fas fa-sun me-1"></i> 太阳能发电预测系统 | 
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
            updateTime();
            
            // 每30秒更新系统信息
            setInterval(updateSystemInfo, 30000);
            // 每秒更新时间
            setInterval(updateTime, 1000);
            
            // 激活标签页
            const activeTab = localStorage.getItem('activeTab');
            if (activeTab) {
                const tabButton = document.querySelector(`[data-bs-target="#${activeTab}"]`);
                if (tabButton) {
                    new bootstrap.Tab(tabButton).show();
                }
            }
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
            
            fetch('/api/evaluation_results')
                .then(response => response.json())
                .then(data => {
                    // 检查目标达成情况
                    checkTargets(data);
                })
                .catch(error => console.error('获取评估结果失败:', error));
        }
        
        function updateTime() {
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
            document.getElementById('current-time').textContent = timeStr;
        }
        
        function checkTargets(evaluationResults) {
            // 更新目标指示器
            const targets = document.querySelectorAll('.target-indicator');
            
            // 检查每个模型的目标达成情况
            for (const target of targets) {
                // 这里可以根据实际评估结果动态更新目标达成状态
                // 简化处理：随机设置达成状态（实际应用中应根据评估结果判断）
                const random = Math.random();
                if (random > 0.5) {
                    target.classList.remove('target-not-achieved');
                    target.classList.add('target-achieved');
                } else {
                    target.classList.remove('target-achieved');
                    target.classList.add('target-not-achieved');
                }
            }
        }
        
        // 标签页功能
        document.querySelectorAll('#modelTab button').forEach(button => {
            button.addEventListener('click', event => {
                const tabId = event.target.getAttribute('data-bs-target').substring(1);
                localStorage.setItem('activeTab', tabId);
            });
        });
    </script>
</body>
</html>
''')

# 创建CSS文件
with open('static/css/optimized_style.css', 'w', encoding='utf-8') as f:
    f.write('''
/* 优化版样式补充 */
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

.param-box {
    font-family: 'Courier New', monospace;
    font-size: 0.85rem;
    line-height: 1.4;
}

@media (max-width: 768px) {
    .header h1 {
        font-size: 1.6rem;
    }
    
    .metric-value {
        font-size: 1.6rem;
    }
    
    .btn-optimized {
        width: 100%;
    }
    
    .param-box {
        font-size: 0.75rem;
    }
}
''')

if __name__ == '__main__':
    print("="*80)
    print("太阳能发电预测系统")
    print("="*80)
    print(f"展示 optimized_train_models.py 训练结果")
    print(f"访问地址: http://127.0.0.1:5003")
    print(f"或 http://localhost:5003")
    print(f"或 http://0.0.0.0:5003")
    print("="*80)
    
    # 运行应用
    app.run(host='0.0.0.0', port=5003, debug=True)