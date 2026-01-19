# 🌞 太阳能发电预测系统

基于机器学习的多时间尺度太阳能发电量预测平台，为可再生能源运营和电网调度提供智能预测支持。

## 📊 项目简介

本项目构建了一个完整的太阳能发电预测系统，从数据预处理、特征工程到模型训练和Web应用展示，实现了端到端的机器学习解决方案。系统基于甘肃酒泉50MW光伏电站的历史数据，采用XGBoost算法构建了小时、日、周三个时间尺度的预测模型。

## 📁 项目结构

```
solar-power-forecast/
├── app.py                          # Flask应用（version1.0）
├── app_opt.py                      # Flask应用（version2.0）
├── data_preprocessor.py            # 数据预处理模块
├── saved_models/                   # 原始模型保存目录
├── optimized_train_models.py       # 优化模型训练脚本
├── final_output/                   # 优化模型输出目录
│   ├── xgboost_hourly.pkl          # 优化小时模型
│   ├── xgboost_daily.pkl           # 优化日模型
│   ├── xgboost_weekly.pkl          # 优化周模型
│   ├── evaluation_results.json     # 优化评估结果
│   └── *.png                       # 预测图表
├── processed_data/                 # 预处理后的数据
├── model_output/                   # 模型输出文件
├── templates/                      # HTML模板
└── static/                         # 静态资源
```

## 🚀 核心功能

### 1. **数据预处理**
- 数据清洗与异常值处理
- 单位转换（MW → kWh）
- 时间特征工程
- 天气特征提取
- 滞后特征创建

### 2. **多时间尺度预测**
- **小时级预测**：短期预测（1-24小时），用于实时调度
- **日级预测**：中期预测（1-7天），用于发电计划
- **周级预测**：长期预测（1-4周），用于能源管理

### 3. **模型优化**
- 基于XGBoost的梯度提升树模型
- 参数调优与正则化
- 特征重要性分析
- 加权MAPE评估（解决零值问题）

### 4. **Web应用**
- 响应式Bootstrap界面
- 模型性能可视化对比
- 实时预测演示
- RESTful API接口
- 系统状态监控

## 🔧 技术栈

- **后端**: Python 3.8+, Flask
- **机器学习**: XGBoost, scikit-learn
- **数据处理**: pandas, NumPy
- **可视化**: Matplotlib, Seaborn
- **前端**: Bootstrap 5, JavaScript
- **部署**: 可部署于任何支持Python的服务器

## 📈 模型特征

### 特征重要性（小时模型）：
1. 太阳辐射强度（solar_radiation_w_m2）
2. 温度（temperature_c）
3. 前一小时发电率（generation_rate_lag_1h）
4. 24小时平均发电率（generation_rate_ma_24h）
5. 辐射-温度交互项（radiation_temperature_interaction）

## 🖥️ 快速开始

### 环境要求
```bash
Python >= 3.8
pip install -r requirements.txt
```

### 安装步骤
```bash
# 1. 克隆仓库
git clone https://github.com/Cililin/solar_power_forecast.git
cd solar-power-forecast

# 2. 安装依赖
# conda环境自行定义即可
pip install -r requirements.txt

# 3. 数据预处理
python data_preprocessor.py

# 4. 训练优化模型
python optimized_train_models.py

# 5. 启动Web应用
python app.py  # 原始版
# 或
python app_2.py  # 优化版（端口5002）
```

### 访问应用
- 原始版: http://43.159.52.233:5000/
- 优化版_v1: http://43.159.52.233:5002/
- 优化版_v2: http://43.159.52.233:5003/

## 📡 API接口

系统提供以下RESTful API接口：

```bash
GET  /api/model_status           # 获取模型加载状态
GET  /api/evaluation_results     # 获取模型评估指标
GET  /api/system_info           # 获取系统信息
GET  /api/sample_features       # 获取示例特征数据
POST /api/predict               # 执行实时预测
GET  /images/{type}/{chart}     # 获取预测图表
```

## 🎨 界面展示

### 主要页面
1. **系统概览** - 展示三个模型的整体状态
2. **模型对比** - 可视化对比各模型性能指标
3. **模型详情** - 分标签展示小时/日/周模型详细信息
4. **实时预测** - 提供交互式预测演示
5. **数据统计** - 展示数据集基本信息
6. **系统信息** - 显示运行状态和技术栈

### 可视化图表
- 模型性能对比柱状图
- 预测值 vs 实际值折线图
- 特征重要性条形图
- 残差分析图

## 📚 数据集

### 数据来源
- 甘肃酒泉50MW光伏电站
- 2020-2025年历史发电数据
- 小时级时间分辨率

### 主要特征
- **时间特征**: 年、月、日、时、季节、工作日/周末
- **天气特征**: 太阳辐射、温度、湿度、风速、云量
- **工程特征**: 滞后值、移动平均、差分、同比/环比
- **分类特征**: 时间段、辐射等级、温度等级、云量等级

## 🔍 模型优化亮点

1. **参数调优**: 调整XGBoost的max_depth、learning_rate等关键参数
2. **特征筛选**: 基于特征重要性选择最相关特征
3. **数据过滤**: 小时模型仅使用白天数据（has_sunlight=1）
4. **评估改进**: 使用加权MAPE解决零值问题
5. **正则化**: 应用L1/L2正则化防止过拟合
