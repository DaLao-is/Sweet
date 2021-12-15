import calendar
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使中文在图表中能正常显示
plt.rcParams['axes.unicode_minus'] = False  # 使负号可以正常显示
import numpy as np
import pandas as pd
import seaborn as sn
import missingno as ms
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

trainData = pd.read_csv("train.csv")  # 加入训练集
testData = pd.read_csv("test.csv")  # 加入测试集


trainData["date"] = trainData.datetime.apply(lambda x: x.split()[0])  # 将date分离出来
trainData["hour"] = trainData.datetime.apply(lambda x: x.split()[1].split(":")[0])  # 将hour分离出来
trainData["year"] = trainData.datetime.apply(lambda x: x.split()[0].split("-")[0])  # 将year分离出来
trainData["weekday"] = trainData.date.apply(
    lambda dateString: calendar.day_name[datetime.strptime(dateString, "%Y-%m-%d").weekday()])  # 将weekday分离出来
trainData["month"] = trainData.date.apply(
    lambda dateString: calendar.month_name[datetime.strptime(dateString, "%Y-%m-%d").month])  # 将month分离出来
trainData["season"] = trainData.season.map({1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"})  # 将season分离出来
trainData["weather"] = trainData.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",
                                              2: " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ",
                                              3: " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds",
                                              4: " Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog "})  # 将weather分离出来

categoryVariableList = ["hour", "weekday", "month", "season", "weather", "holiday", "workingday", "year"]
for cate in categoryVariableList:
    trainData[cate] = trainData[cate].astype("category")  # 将season,year,holiday,workingday,weather转换为类别型

trainData = trainData.drop(["datetime"], axis=1)  # 将datetime删除

ms.matrix(trainData, figsize=(20, 10))  # 使用missingno检查训练集中是否有缺失值
plt.show()
print(trainData.isnull().sum())  # 检查训练集中是否有缺失值

# 异常值分析
fig, axes = plt.subplots(nrows=2, ncols=2)  # 两行两列
fig.set_size_inches(12, 10)  # 尺寸
# orient用于控制图像使水平还是竖直显示
sn.boxplot(data=trainData, y="count", orient="v", ax=axes[0][0])  # 没有横坐标，看总体情况
sn.boxplot(data=trainData, y="count", x="season", orient="v", ax=axes[0][1])
sn.boxplot(data=trainData, y="count", x="hour", orient="v", ax=axes[1][0])
sn.boxplot(data=trainData, y="count", x="workingday", orient="v", ax=axes[1][1])

axes[0][0].set(ylabel='Count', title="总数")
axes[0][1].set(xlabel='Season', ylabel='Count', title="租赁数与季节")
axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count', title="每天各时段租赁数")
axes[1][1].set(xlabel='Working Day', ylabel='Count', title="非工作日与工作日")
plt.show()

trainDataWithoutOutliers = trainData[
    np.abs(trainData["count"] - trainData["count"].mean()) <= (3 * trainData["count"].std())]

# 从天气环境因素分析其对租借数量的影响
# 画出温度，体感温度，湿度，风速，租借数量之间的相关系数矩阵图
corrmat = trainData[["temp", "atemp", "humidity", "windspeed", "count"]].corr()
mask = np.array(corrmat)
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots()
fig.set_size_inches(12, 6)
sn.heatmap(corrmat, mask=mask, vmax=.8, square=True, annot=True, cmap="Blues")
plt.show()

# 对比2011和2012年租借数量的变化
fig.set_size_inches(12, 20)
sortOrder = ["2011", "2012"]
yearAggregated = pd.DataFrame(trainData.groupby("year")["count"].mean()).reset_index()
yearSorted = yearAggregated.sort_values(by="count", ascending=False)
sn.barplot(data=yearSorted, x="year", y="count", order=sortOrder)
plt.show()

# 处理异常值
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sn.distplot(trainData['count'])
ax.set_title('Distribution of Count')

trainData_withoutOutliers = trainData[(trainData['count'] - trainData['count'].mean()).abs()
                                      < 3 * trainData['count'].std()]
trainData_withoutOutliers.info()

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

sn.distplot(trainData['count'], ax=axes[0])
sn.distplot(trainData_withoutOutliers['count'], ax=axes[1])

axes[0].set_title('Count的分布')  # 分布有些偏斜，尾巴较长，count波动较大
axes[1].set_title(' 没有异常值的Count分布 ')  # 数据波动还是较大
plt.show()

trainDataWO_log = np.log(trainData_withoutOutliers['count'])  # 为了防止过拟合，我们看看count数据的对数分布
sn.distplot(trainDataWO_log)
plt.title(' 对数变换后的Count分布 ')  # 对数分布更稳定 我们将选择对数来建模
plt.show()

from ModelImprove import DataAnalysis
Bike_data=DataAnalysis()
# 相关性矩阵
corrDf = Bike_data.corr()

# ascending=False表示按降序排列
corrDf['count'].sort_values(ascending=False)

# 时段对租赁数量的影响
workingday_df = Bike_data[Bike_data['workingday'] == 1]
workingday_df = workingday_df.groupby(['hour'], as_index=True).agg({'casual': 'mean',
                                                                    'registered': 'mean',
                                                                    'count': 'mean'})

nworkingday_df = Bike_data[Bike_data['workingday'] == 0]
nworkingday_df = nworkingday_df.groupby(['hour'], as_index=True).agg({'casual': 'mean',
                                                                      'registered': 'mean',
                                                                      'count': 'mean'})
fig, axes = plt.subplots(1, 2, sharey=True)

workingday_df.plot(figsize=(15, 5), title='工作日平均每小时的租赁次数', ax=axes[0])
nworkingday_df.plot(figsize=(15, 5), title='非工作日内每小时的平均租赁次数', ax=axes[1])
plt.show()


# 温度对租赁数量的影响
# 数据按小时统计展示起来太麻烦，希望能够按天汇总取一天的气温中位数
temp_df = Bike_data.groupby(['date', 'weekday'], as_index=False).agg({'year': 'mean',
                                                                      'month': 'mean',
                                                                      'temp': 'median'})
# 由于测试数据集中没有租赁信息，会导致折线图有断裂，所以将缺失的数据丢弃
temp_df.dropna(axis=0, how='any', inplace=True)

# 预计按天统计的波动仍然很大，再按月取日平均值
temp_month = temp_df.groupby(['year', 'month'], as_index=False).agg({'weekday': 'min',
                                                                     'temp': 'median'})

# 将按天求和统计数据的日期转换成datetime格式
temp_df['date'] = pd.to_datetime(temp_df['date'])

# 将按月统计数据设置一列时间序列
temp_month.rename(columns={'weekday': 'day'}, inplace=True)
temp_month['date'] = pd.to_datetime(temp_month[['year', 'month', 'day']])

# 设置画框尺寸
fig = plt.figure(figsize=(18, 6))
ax = fig.add_subplot(1, 1, 1)

# 使用折线图展示总体租赁情况（count）随时间的走势
plt.plot(temp_df['date'], temp_df['temp'], linewidth=1.3, label='Daily average')
ax.set_title('两年日均气温变化趋势')
plt.plot(temp_month['date'], temp_month['temp'], marker='o', linewidth=1.3,
         label='Monthly average')
ax.legend()
plt.show()

# 按温度取租赁额平均值
temp_rentals = Bike_data.groupby(['temp'], as_index=True).agg({'casual': 'mean',
                                                               'registered': 'mean',
                                                               'count': 'mean'})
temp_rentals.plot(title='每小时平均租赁数量随温度变化')
plt.show()

# 使用线性回归来观察租借数量与温度，湿度，风速之间的关系。
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.set_size_inches(12, 5)
sn.regplot(x="temp", y="count", data=trainData, ax=ax1, color='green')
sn.regplot(x="windspeed", y="count", data=trainData, ax=ax2, color='orange')
sn.regplot(x="humidity", y="count", data=trainData, ax=ax3, color='red')
plt.show()


# 多变量图
sn.pairplot(trainData ,x_vars=['holiday','workingday','weather','season','weekday','hour','windspeed','humidity','temp','atemp'] ,y_vars=['casual','registered','count'] , plot_kws={'alpha': 0.1})
plt.show()
#各个季节，一天时段内，每星期，不同用户种类的租借数量
fig,(ax1,ax2,ax3)= plt.subplots(nrows=3)
fig.set_size_inches(12,20)
hueOrder = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

hourAggregated = pd.DataFrame(trainData.groupby(["hour","season"],sort=True)["count"].mean()).reset_index()
sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"],hue=hourAggregated["season"], data=hourAggregated, join=True,ax=ax1)
ax2.set(xlabel='时间', ylabel='租借数量',title="一年四季中平均每天按小时的用户数量",label='big')

hourAggregated = pd.DataFrame(trainData.groupby(["hour","weekday"],sort=True)["count"].mean()).reset_index()
sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"],hue=hourAggregated["weekday"],hue_order=hueOrder, data=hourAggregated, join=True,ax=ax2)
ax3.set(xlabel='时间', ylabel='租借数量',title="每周的平均用户数量",label='big')

hourTransformed = pd.melt(trainData[["hour","casual","registered"]], id_vars=['hour'], value_vars=['casual', 'registered'])
hourAggregated = pd.DataFrame(hourTransformed.groupby(["hour","variable"],sort=True)["value"].mean()).reset_index()
sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["value"],hue=hourAggregated["variable"],hue_order=["casual","registered"], data=hourAggregated, join=True,ax=ax3)
ax3.set(xlabel='时间', ylabel='租借数量',title="不同用户种类平均每天的数量",label='big')
plt.show()




