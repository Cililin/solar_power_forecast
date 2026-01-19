"""
配置文件
"""
import os

# 基础配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# 数据文件路径
DATA_FILE = os.path.join(DATA_DIR, 'solar_50MW_hourly_generation_data.csv')

# 模型配置
MODEL_CONFIG = {
    'xgb': {
        'max_depth': 3,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42
    }
}

# 电站配置
PLANT_CONFIG = {
    'plant_id': 'solar_50MW_001',
    'capacity_mw': 50,
    'location': 'Jiuquan, Gansu',
    'latitude': 39.70,
    'longitude': 98.50,
    'altitude': 1500
}

# Flask配置
FLASK_CONFIG = {
    'host': '0.0.0.0',
    'port': 5003,
    'debug': True,
    'threaded': True
}