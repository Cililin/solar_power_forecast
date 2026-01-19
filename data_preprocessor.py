"""
太阳能发电预测数据预处理模块
主要功能：
1. 数据加载和清洗
2. 单位转换（MW → kWh）
3. 处理夜间/无光照数据
4. 异常值处理
5. 计算发电率
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')

class SolarDataPreprocessor:
    """太阳能数据预处理类"""
    
    def __init__(self, data_path='solar_50MW_hourly_generation_data.csv'):
        """
        初始化预处理类
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        
        # 定义常数
        self.PLANT_CAPACITY = 50  # MW，电站装机容量
        self.MW_TO_KWH = 1000     # 1 MW = 1000 kWh
        
        # 定义夜间时间段
        self.NIGHT_HOURS = list(range(0, 7)) + list(range(19, 24))  # 晚7点-早7点
        
        # 定义雨天/阴天的辐射阈值（W/m²）
        self.LOW_RADIATION_THRESHOLD = 50
        
        # 定义异常值阈值
        self.POWER_OUTPUT_MAX = self.PLANT_CAPACITY * 1.2  # 最大输出功率的120%
        self.POWER_OUTPUT_MIN = -1  # 数据噪声
        
    def load_data(self):
        """加载数据"""
        print(f"正在加载数据: {self.data_path}")
        
        try:
            try:
                self.data = pd.read_csv(self.data_path, encoding='utf-8')
            except:
                self.data = pd.read_csv(self.data_path, encoding='gbk')
            
            print(f"数据加载成功，形状: {self.data.shape}")
            print(f"时间范围: {self.data['timestamp'].min()} 到 {self.data['timestamp'].max()}")
            
            # 确保时间列是datetime类型
            if 'timestamp' in self.data.columns:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                # 设置时间为索引
                self.data.set_index('timestamp', inplace=True)
                self.data.sort_index(inplace=True)
            
            # 显示基本信息
            self._display_basic_info()
            
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            self._create_sample_data()
    
    def _display_basic_info(self):
        """显示数据基本信息"""
        print("\n" + "="*50)
        print("数据基本信息:")
        print("="*50)
        print(f"总行数: {len(self.data)}")
        print(f"总列数: {len(self.data.columns)}")
        print(f"时间跨度: {self.data.index.min()} 到 {self.data.index.max()}")
        print(f"时间频率: {pd.infer_freq(self.data.index)}")
        
        # 检查关键列是否存在
        key_columns = ['power_output_mw', 'solar_radiation_w_m2', 'temperature_c']
        missing_cols = [col for col in key_columns if col not in self.data.columns]
        
        if missing_cols:
            print(f"警告: 以下关键列缺失: {missing_cols}")
        
        # 显示前几行
        print("\n前3行数据:")
        print(self.data.head(3))
        
        # 显示数据统计信息
        if 'power_output_mw' in self.data.columns:
            print("\n发电功率统计:")
            print(self.data['power_output_mw'].describe())
    
    def convert_units(self):
        """单位转换"""
        print("\n" + "="*50)
        print("单位转换:")
        print("="*50)
        
        # 复制数据
        self.processed_data = self.data.copy()
        
        # 1. 将发电功率从MW转换为kWh（1小时发电量）
        if 'power_output_mw' in self.processed_data.columns:
            self.processed_data['power_output_kwh'] = self.processed_data['power_output_mw'] * self.MW_TO_KWH
            print(f"发电功率转换: MW → kWh")
        
        # 2. 计算发电率（实际发电量 / 最大可能发电量）
        if 'power_output_mw' in self.processed_data.columns:
            self.processed_data['generation_rate'] = (self.processed_data['power_output_mw'] / self.PLANT_CAPACITY) * 100
            
            # 确保发电率在0-100%之间（允许轻微的负值和超发）
            self.processed_data['generation_rate'] = np.clip(
                self.processed_data['generation_rate'], 
                -1,  # 允许轻微负值（数据噪声）
                120   # 允许轻微超发
            )
            print(f"计算发电率完成，范围: [{self.processed_data['generation_rate'].min():.2f}%, {self.processed_data['generation_rate'].max():.2f}%]")
        
        # 3. 计算累计发电量（kWh）
        if 'power_output_kwh' in self.processed_data.columns:
            self.processed_data['cumulative_energy_kwh'] = self.processed_data['power_output_kwh'].cumsum()
            print(f"计算累计发电量完成")
        
        print("单位转换完成！")
    
    def handle_night_and_low_radiation_data(self):
        """处理夜间和低辐射数据"""
        print("\n" + "="*50)
        print("处理夜间和低辐射数据:")
        print("="*50)
        
        if self.processed_data is None:
            self.processed_data = self.data.copy()
        
        # 1. 识别夜间数据
        self.processed_data['is_night'] = self.processed_data.index.hour.isin(self.NIGHT_HOURS)
        
        # 2. 识别低辐射数据（阴天/雨天）
        if 'solar_radiation_w_m2' in self.processed_data.columns:
            self.processed_data['is_low_radiation'] = self.processed_data['solar_radiation_w_m2'] < self.LOW_RADIATION_THRESHOLD
            self.processed_data['is_no_radiation'] = self.processed_data['solar_radiation_w_m2'] <= 0
        else:
            self.processed_data['is_low_radiation'] = False
            self.processed_data['is_no_radiation'] = False
        
        # 3. 识别无光照情况（夜间或辐射为0）
        self.processed_data['has_sunlight'] = ~(self.processed_data['is_night'] | self.processed_data['is_no_radiation'])
        
        # 4. 统计各类情况
        night_count = self.processed_data['is_night'].sum()
        low_rad_count = self.processed_data['is_low_radiation'].sum()
        no_rad_count = self.processed_data['is_no_radiation'].sum()
        has_sunlight_count = self.processed_data['has_sunlight'].sum()
        
        print(f"夜间数据点: {night_count} ({night_count/len(self.processed_data)*100:.1f}%)")
        print(f"低辐射数据点: {low_rad_count} ({low_rad_count/len(self.processed_data)*100:.1f}%)")
        print(f"无辐射数据点: {no_rad_count} ({no_rad_count/len(self.processed_data)*100:.1f}%)")
        print(f"有光照数据点: {has_sunlight_count} ({has_sunlight_count/len(self.processed_data)*100:.1f}%)")
        
        # 5. 对于无光照的数据，确保发电功率为0或接近0
        if 'power_output_mw' in self.processed_data.columns:
            # 保存原始值用于分析
            self.processed_data['power_output_mw_original'] = self.processed_data['power_output_mw']
            
            # 对于无光照但仍有发电的数据，进行修正
            no_sunlight_mask = ~self.processed_data['has_sunlight']
            invalid_power_mask = no_sunlight_mask & (self.processed_data['power_output_mw'] > 0.1)
            
            invalid_count = invalid_power_mask.sum()
            if invalid_count > 0:
                print(f"发现 {invalid_count} 个无光照但发电功率>0.1MW的数据点，将其设为0")
                self.processed_data.loc[invalid_power_mask, 'power_output_mw'] = 0
                
                # 同时更新kWh和发电率
                if 'power_output_kwh' in self.processed_data.columns:
                    self.processed_data.loc[invalid_power_mask, 'power_output_kwh'] = 0
                if 'generation_rate' in self.processed_data.columns:
                    self.processed_data.loc[invalid_power_mask, 'generation_rate'] = 0
        
        print("夜间和低辐射数据处理完成！")
    
    def handle_outliers(self):
        """处理异常值"""
        print("\n" + "="*50)
        print("处理异常值:")
        print("="*50)
        
        # 1. 处理发电功率异常值
        if 'power_output_mw' in self.processed_data.columns:
            # 识别异常值
            original_count = len(self.processed_data)
            
            # 负功率异常值
            negative_mask = self.processed_data['power_output_mw'] < self.POWER_OUTPUT_MIN
            negative_count = negative_mask.sum()
            
            # 超发异常值
            over_production_mask = self.processed_data['power_output_mw'] > self.POWER_OUTPUT_MAX
            over_production_count = over_production_mask.sum()
            
            print(f"负功率异常值: {negative_count} 个")
            print(f"超发异常值: {over_production_count} 个")
            
            # 处理负功率异常值
            if negative_count > 0:
                print(f"将 {negative_count} 个负功率值设为0")
                self.processed_data.loc[negative_mask, 'power_output_mw'] = 0
            
            # 处理超发异常值
            if over_production_count > 0:
                print(f"将 {over_production_count} 个超发值限制在 {self.POWER_OUTPUT_MAX} MW")
                self.processed_data.loc[over_production_mask, 'power_output_mw'] = self.POWER_OUTPUT_MAX
            
            # 更新相关列
            if 'power_output_kwh' in self.processed_data.columns:
                self.processed_data['power_output_kwh'] = self.processed_data['power_output_mw'] * self.MW_TO_KWH
            
            if 'generation_rate' in self.processed_data.columns:
                self.processed_data['generation_rate'] = (self.processed_data['power_output_mw'] / self.PLANT_CAPACITY) * 100
        
        # 2. 处理温度异常值（假设温度在-30°C到50°C之间）
        if 'temperature_c' in self.processed_data.columns:
            temp_outliers = self.processed_data['temperature_c'].between(-30, 50)
            outlier_count = (~temp_outliers).sum()
            
            if outlier_count > 0:
                print(f"发现 {outlier_count} 个温度异常值，使用前向填充")
                self.processed_data['temperature_c'] = self.processed_data['temperature_c'].where(
                    temp_outliers, np.nan
                ).ffill().bfill()
        
        # 3. 处理辐射异常值（辐射应为非负值）
        if 'solar_radiation_w_m2' in self.processed_data.columns:
            rad_outliers = self.processed_data['solar_radiation_w_m2'] >= 0
            outlier_count = (~rad_outliers).sum()
            
            if outlier_count > 0:
                print(f"发现 {outlier_count} 个辐射异常值，设为0")
                self.processed_data.loc[~rad_outliers, 'solar_radiation_w_m2'] = 0
        
        print("异常值处理完成！")
    
    def handle_missing_values(self):
        """处理缺失值"""
        print("\n" + "="*50)
        print("处理缺失值:")
        print("="*50)
        
        # 检查缺失值
        missing_before = self.processed_data.isnull().sum().sum()
        print(f"缺失值总数（处理前）: {missing_before}")
        
        if missing_before > 0:
            # 显示各列缺失情况
            missing_cols = self.processed_data.isnull().sum()
            missing_cols = missing_cols[missing_cols > 0]
            
            if len(missing_cols) > 0:
                print("\n各列缺失值统计:")
                for col, count in missing_cols.items():
                    print(f"  {col}: {count} 个 ({count/len(self.processed_data)*100:.2f}%)")
            
            # 处理缺失值
            # 对于数值列，使用前向填充和线性插值
            numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if self.processed_data[col].isnull().sum() > 0:
                    # 首先尝试前向填充
                    self.processed_data[col] = self.processed_data[col].ffill()
                    # 然后尝试后向填充
                    self.processed_data[col] = self.processed_data[col].bfill()
                    # 如果还有缺失值，用列均值填充
                    if self.processed_data[col].isnull().sum() > 0:
                        self.processed_data[col].fillna(self.processed_data[col].mean(), inplace=True)
            
            missing_after = self.processed_data.isnull().sum().sum()
            print(f"缺失值总数（处理后）: {missing_after}")
            
            if missing_after > 0:
                print(f"仍有 {missing_after} 个缺失值，删除这些行")
                self.processed_data.dropna(inplace=True)
        else:
            print("没有发现缺失值")
        
        print("缺失值处理完成！")
    
    def create_time_features(self):
        """创建时间特征"""
        print("\n" + "="*50)
        print("创建时间特征:")
        print("="*50)
        
        # 提取时间特征
        self.processed_data['year'] = self.processed_data.index.year
        self.processed_data['month'] = self.processed_data.index.month
        self.processed_data['day'] = self.processed_data.index.day
        self.processed_data['hour'] = self.processed_data.index.hour
        self.processed_data['day_of_week'] = self.processed_data.index.dayofweek  # 周一=0, 周日=6
        self.processed_data['day_of_year'] = self.processed_data.index.dayofyear
        self.processed_data['week_of_year'] = self.processed_data.index.isocalendar().week
        self.processed_data['quarter'] = self.processed_data.index.quarter
        
        # 是否周末
        self.processed_data['is_weekend'] = self.processed_data['day_of_week'].isin([5, 6])
        
        # 季节（北半球）
        self.processed_data['season'] = self.processed_data['month'] % 12 // 3 + 1
        
        # 一天中的时间段
        def get_time_of_day(hour):
            if 5 <= hour < 11:
                return 'morning'
            elif 11 <= hour < 14:
                return 'noon'
            elif 14 <= hour < 18:
                return 'afternoon'
            elif 18 <= hour < 22:
                return 'evening'
            else:
                return 'night'
        
        self.processed_data['time_of_day'] = self.processed_data['hour'].apply(get_time_of_day)
        
        # 是否为工作日工作时间（假设9-17点为工作时间）
        self.processed_data['is_working_hour'] = (
            (self.processed_data['day_of_week'] < 5) &  # 周一到周五
            (self.processed_data['hour'] >= 9) & 
            (self.processed_data['hour'] < 17)
        )
        
        # 周期特征（使用正弦/余弦编码）
        self.processed_data['hour_sin'] = np.sin(2 * np.pi * self.processed_data['hour'] / 24)
        self.processed_data['hour_cos'] = np.cos(2 * np.pi * self.processed_data['hour'] / 24)
        
        self.processed_data['month_sin'] = np.sin(2 * np.pi * self.processed_data['month'] / 12)
        self.processed_data['month_cos'] = np.cos(2 * np.pi * self.processed_data['month'] / 12)
        
        self.processed_data['day_of_year_sin'] = np.sin(2 * np.pi * self.processed_data['day_of_year'] / 365)
        self.processed_data['day_of_year_cos'] = np.cos(2 * np.pi * self.processed_data['day_of_year'] / 365)
        
        print(f"创建了 {len([col for col in self.processed_data.columns if col not in self.data.columns])} 个时间特征")
        print("时间特征创建完成！")
    
    def create_weather_features(self):
        """创建天气相关特征"""
        print("\n" + "="*50)
        print("创建天气特征:")
        print("="*50)
        
        features_created = []
        
        # 1. 太阳辐射相关特征
        if 'solar_radiation_w_m2' in self.processed_data.columns:
            # 辐射等级
            def get_radiation_level(rad):
                if rad <= 0:
                    return 'no_radiation'
                elif rad < 100:
                    return 'low'
                elif rad < 300:
                    return 'medium'
                elif rad < 600:
                    return 'high'
                else:
                    return 'very_high'
            
            self.processed_data['radiation_level'] = self.processed_data['solar_radiation_w_m2'].apply(get_radiation_level)
            features_created.append('radiation_level')
            
            # 辐射变化率
            self.processed_data['radiation_change_rate'] = self.processed_data['solar_radiation_w_m2'].pct_change()
            features_created.append('radiation_change_rate')
            
            # 辐射滚动统计
            for window in [3, 6, 12, 24]:
                self.processed_data[f'radiation_ma_{window}h'] = (
                    self.processed_data['solar_radiation_w_m2'].rolling(window=window, min_periods=1).mean()
                )
                features_created.append(f'radiation_ma_{window}h')
        
        # 2. 温度相关特征
        if 'temperature_c' in self.processed_data.columns:
            # 温度等级
            def get_temperature_level(temp):
                if temp < 0:
                    return 'freezing'
                elif temp < 10:
                    return 'cold'
                elif temp < 20:
                    return 'cool'
                elif temp < 30:
                    return 'warm'
                else:
                    return 'hot'
            
            self.processed_data['temperature_level'] = self.processed_data['temperature_c'].apply(get_temperature_level)
            features_created.append('temperature_level')
            
            # 温度滚动统计
            for window in [3, 6, 12, 24]:
                self.processed_data[f'temperature_ma_{window}h'] = (
                    self.processed_data['temperature_c'].rolling(window=window, min_periods=1).mean()
                )
                features_created.append(f'temperature_ma_{window}h')
        
        # 3. 交互特征
        if 'solar_radiation_w_m2' in self.processed_data.columns and 'temperature_c' in self.processed_data.columns:
            self.processed_data['radiation_temperature_interaction'] = (
                self.processed_data['solar_radiation_w_m2'] * self.processed_data['temperature_c']
            )
            features_created.append('radiation_temperature_interaction')
        
        # 4. 天气状况特征
        if 'cloud_index' in self.processed_data.columns:
            # 云量等级
            def get_cloud_level(cloud_idx):
                if cloud_idx < 0.2:
                    return 'clear'
                elif cloud_idx < 0.5:
                    return 'partly_cloudy'
                elif cloud_idx < 0.8:
                    return 'cloudy'
                else:
                    return 'overcast'
            
            self.processed_data['cloud_level'] = self.processed_data['cloud_index'].apply(get_cloud_level)
            features_created.append('cloud_level')
        
        print(f"创建了 {len(features_created)} 个天气特征: {features_created}")
        print("天气特征创建完成！")
    
    def create_engineering_features(self):
        """创建工程特征"""
        print("\n" + "="*50)
        print("创建工程特征:")
        print("="*50)
        
        features_created = []
        
        if 'generation_rate' in self.processed_data.columns:
            # 1. 滞后特征（过去1, 2, 3, 6, 12, 24小时的发电率）
            for lag in [1, 2, 3, 6, 12, 24]:
                self.processed_data[f'generation_rate_lag_{lag}h'] = self.processed_data['generation_rate'].shift(lag)
                features_created.append(f'generation_rate_lag_{lag}h')
            
            # 2. 滚动统计特征
            for window in [3, 6, 12, 24]:
                # 均值
                self.processed_data[f'generation_rate_ma_{window}h'] = (
                    self.processed_data['generation_rate'].rolling(window=window, min_periods=1).mean()
                )
                features_created.append(f'generation_rate_ma_{window}h')
                
                # 标准差
                self.processed_data[f'generation_rate_std_{window}h'] = (
                    self.processed_data['generation_rate'].rolling(window=window, min_periods=1).std()
                )
                features_created.append(f'generation_rate_std_{window}h')
                
                # 最大值
                self.processed_data[f'generation_rate_max_{window}h'] = (
                    self.processed_data['generation_rate'].rolling(window=window, min_periods=1).max()
                )
                features_created.append(f'generation_rate_max_{window}h')
            
            # 3. 差分特征
            self.processed_data['generation_rate_diff_1h'] = self.processed_data['generation_rate'].diff(1)
            features_created.append('generation_rate_diff_1h')
            
            self.processed_data['generation_rate_diff_24h'] = self.processed_data['generation_rate'].diff(24)
            features_created.append('generation_rate_diff_24h')
            
            # 4. 同比特征（与昨天同一时间比较）
            self.processed_data['generation_rate_yesterday_same_hour'] = self.processed_data['generation_rate'].shift(24)
            features_created.append('generation_rate_yesterday_same_hour')
            
            self.processed_data['generation_rate_change_vs_yesterday'] = (
                self.processed_data['generation_rate'] - self.processed_data['generation_rate_yesterday_same_hour']
            )
            features_created.append('generation_rate_change_vs_yesterday')
            
            # 5. 周同比特征（与上周同一时间比较）
            self.processed_data['generation_rate_lastweek_same_hour'] = self.processed_data['generation_rate'].shift(24*7)
            features_created.append('generation_rate_lastweek_same_hour')
            
            self.processed_data['generation_rate_change_vs_lastweek'] = (
                self.processed_data['generation_rate'] - self.processed_data['generation_rate_lastweek_same_hour']
            )
            features_created.append('generation_rate_change_vs_lastweek')
        
        # 6. 时间交互特征
        if 'hour' in self.processed_data.columns and 'solar_radiation_w_m2' in self.processed_data.columns:
            self.processed_data['hour_radiation_interaction'] = (
                self.processed_data['hour'] * self.processed_data['solar_radiation_w_m2']
            )
            features_created.append('hour_radiation_interaction')
        
        print(f"创建了 {len(features_created)} 个工程特征")
        print("工程特征创建完成！")
    
    def prepare_for_modeling(self):
        """准备建模数据"""
        print("\n" + "="*50)
        print("准备建模数据:")
        print("="*50)
        
        # 1. 处理分类特征
        categorical_cols = []
        
        for col in ['time_of_day', 'radiation_level', 'temperature_level', 'cloud_level', 'season']:
            if col in self.processed_data.columns:
                categorical_cols.append(col)
        
        if categorical_cols:
            print(f"对以下分类特征进行独热编码: {categorical_cols}")
            self.processed_data = pd.get_dummies(
                self.processed_data, 
                columns=categorical_cols, 
                prefix=categorical_cols
            )
        
        # 2. 填充因滞后特征产生的缺失值
        self.processed_data.fillna(method='bfill', inplace=True)  # 后向填充
        self.processed_data.fillna(method='ffill', inplace=True)  # 前向填充
        self.processed_data.fillna(0, inplace=True)  # 剩余缺失值填0
        
        # 3. 选择特征列（排除目标列和标识列）
        exclude_cols = [
            'power_output_mw_original',  # 原始功率列
            'is_night', 'is_low_radiation', 'is_no_radiation',  # 标识列
            'cumulative_energy_kwh',  # 累计值
        ]
        
        # 保留实际存在的列
        exclude_cols = [col for col in exclude_cols if col in self.processed_data.columns]
        
        # 特征列（排除目标列和不需要的列）
        feature_cols = [col for col in self.processed_data.columns 
                       if col not in exclude_cols + ['power_output_mw', 'power_output_kwh']]
        
        print(f"总特征数: {len(feature_cols)}")
        print(f"前10个特征: {feature_cols[:10]}")
        
        # 4. 创建三个数据集
        # 小时级数据（原始数据）
        hourly_data = self.processed_data.copy()
        
        # 日级数据（按日聚合）
        daily_data = self.processed_data.resample('D').agg({
            'generation_rate': 'mean',  # 日均发电率
            'solar_radiation_w_m2': 'mean',
            'temperature_c': 'mean',
            'humidity_percent': 'mean',
            'wind_speed_10m_mps': 'mean',
        }).dropna()
        
        # 周级数据（按周聚合）
        weekly_data = self.processed_data.resample('W').agg({
            'generation_rate': 'mean',  # 周均发电率
            'solar_radiation_w_m2': 'mean',
            'temperature_c': 'mean',
            'humidity_percent': 'mean',
            'wind_speed_10m_mps': 'mean',
        }).dropna()
        
        print(f"小时级数据形状: {hourly_data.shape}")
        print(f"日级数据形状: {daily_data.shape}")
        print(f"周级数据形状: {weekly_data.shape}")
        
        return {
            'hourly': hourly_data,
            'daily': daily_data,
            'weekly': weekly_data,
            'feature_cols': feature_cols
        }
    
    def save_processed_data(self, output_dir='processed_data'):
        """保存处理后的数据"""
        print("\n" + "="*50)
        print("保存处理后的数据:")
        print("="*50)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存处理后的数据
        if self.processed_data is not None:
            output_path = os.path.join(output_dir, 'processed_solar_data.csv')
            self.processed_data.to_csv(output_path)
            print(f"处理后的数据已保存到: {output_path}")
            
            # 保存数据统计信息
            stats_path = os.path.join(output_dir, 'data_statistics.json')
            stats = {
                'data_shape': self.processed_data.shape,
                'time_range': {
                    'start': self.processed_data.index.min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': self.processed_data.index.max().strftime('%Y-%m-%d %H:%M:%S')
                },
                'generation_rate_stats': {
                    'mean': float(self.processed_data['generation_rate'].mean()),
                    'std': float(self.processed_data['generation_rate'].std()),
                    'min': float(self.processed_data['generation_rate'].min()),
                    'max': float(self.processed_data['generation_rate'].max())
                } if 'generation_rate' in self.processed_data.columns else {},
                'features_count': len(self.processed_data.columns)
            }
            
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            print(f"数据统计信息已保存到: {stats_path}")
        else:
            print("警告: 没有处理后的数据可保存")
    
    def run_full_preprocessing(self):
        """运行完整的预处理流程"""
        print("开始太阳能数据预处理...")
        print("="*60)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 单位转换
        self.convert_units()
        
        # 3. 处理夜间和低辐射数据
        self.handle_night_and_low_radiation_data()
        
        # 4. 处理异常值
        self.handle_outliers()
        
        # 5. 处理缺失值
        self.handle_missing_values()
        
        # 6. 创建时间特征
        self.create_time_features()
        
        # 7. 创建天气特征
        self.create_weather_features()
        
        # 8. 创建工程特征
        self.create_engineering_features()
        
        # 9. 准备建模数据
        modeling_data = self.prepare_for_modeling()
        
        # 10. 保存处理后的数据
        self.save_processed_data()
        
        print("\n" + "="*60)
        print("数据预处理完成！")
        print("="*60)
        
        return modeling_data


def main():
    """主函数"""
    # 初始化预处理器
    preprocessor = SolarDataPreprocessor('solar_50MW_hourly_generation_data.csv')
    
    # 运行完整预处理流程
    modeling_data = preprocessor.run_full_preprocessing()
    
    # 显示处理后的数据信息
    if preprocessor.processed_data is not None:
        print("\n处理后的数据信息:")
        print(f"数据形状: {preprocessor.processed_data.shape}")
        print(f"时间范围: {preprocessor.processed_data.index.min()} 到 {preprocessor.processed_data.index.max()}")
        
        if 'generation_rate' in preprocessor.processed_data.columns:
            print(f"发电率统计:")
            print(f"  均值: {preprocessor.processed_data['generation_rate'].mean():.2f}%")
            print(f"  标准差: {preprocessor.processed_data['generation_rate'].std():.2f}%")
            print(f"  最小值: {preprocessor.processed_data['generation_rate'].min():.2f}%")
            print(f"  最大值: {preprocessor.processed_data['generation_rate'].max():.2f}%")
        
        print(f"\n总特征数: {len(preprocessor.processed_data.columns)}")
        print(f"特征列示例: {list(preprocessor.processed_data.columns[:10])}")
    
    return preprocessor, modeling_data


if __name__ == "__main__":
    # 运行预处理
    preprocessor, modeling_data = main()