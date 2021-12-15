# 导入处理数据包
import numpy as np
import pandas as pd
from datetime import datetime

# 忽略警告提示
import warnings

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

dataTrain = pd.read_csv('train.csv')
dataTest = pd.read_csv('test.csv')
data = dataTrain.append(dataTest)  # 合并训练集和测试集
data.reset_index(inplace=True)
data.drop('index', inplace=True, axis=1)

data['date'] = data['datetime'].apply(lambda x: x.split()[0])  # 分离date
data['hour'] = data['datetime'].apply(lambda x: x.split()[1].split(':')[0]).astype('int')  # 分离hour
data['year'] = data['datetime'].apply(lambda x: x.split()[0].split('-')[0])  # 分离year
data['weekday'] = data['date'].apply(
    lambda dateString: datetime.strptime(dateString, "%Y-%m-%d").weekday())  # 分离weekday
data["month"] = data['date'].apply(lambda dateString: datetime.strptime(dateString, "%Y-%m-%d").month)  # 分离month

from sklearn.ensemble import RandomForestRegressor

dataWind0 = data[data["windspeed"] == 0]
dataWindNot0 = data[data["windspeed"] != 0]
rfModel_wind = RandomForestRegressor()
windColumns = ["season", "weather", "humidity", "month", "temp", "year", "atemp"]
rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])
# 预测风速
wind0Values = rfModel_wind.predict(X=dataWind0[windColumns])
dataWind0["windspeed"] = wind0Values
data = dataWindNot0.append(dataWind0)
data.reset_index(inplace=True)
data.drop('index', inplace=True, axis=1)

categoricalFeatureNames = ["season", "holiday", "workingday", "weather", "weekday", "month", "year", "hour"]
numericalFeatureNames = ["temp", "humidity", "windspeed", "atemp"]
dropFeatures = ['casual', "count", "datetime", "date", "registered"]
for var in categoricalFeatureNames:
    data[var] = data[var].astype('category')

# 把所有的“count”为空值的取出来，当做测试集，其他的的作为训练集
dataTrain = data[pd.notnull(data['count'])].sort_values(by=['datetime'])
# ~  按位取反运算符：对数据的每个二进制位取反,即把1变为0,把0变为1 。~x 类似于 -x-1
dataTest = data[~pd.notnull(data['count'])].sort_values(by=["datetime"])
# 将data重新拆分为训练集和测试集
datetimecol = dataTest['datetime']
yLabels = dataTrain['count']
yLabelsRegistered = dataTrain['registered']
yLabelsCasual = dataTrain['casual']

# 删除冗余变量
dataTrain = dataTrain.drop(dropFeatures, axis=1)
dataTest = dataTest.drop(dropFeatures, axis=1)


def rmsle(y, y_, convertExp=True):
    if convertExp:
        y = np.exp(y)
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))


import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 线性回归模型
# Linear_Model = LinearRegression()
# # 训练模型
# yLabelsLog = np.log1p(yLabels)
# Linear_Model.fit(X=dataTrain, y=yLabelsLog)
# # 预测
# preds = Linear_Model.predict(X=dataTrain)
# print('线性回归模型的RMSLE值： ', rmsle(np.exp(yLabelsLog), np.exp(preds), False))
# rf_preds=Linear_Model.predict(X=dataTest)
# # 导出数据 线性回归
# submission = pd.DataFrame({
#     'datetime': datetimecol,
#     'count': np.exp(rf_preds)})
# submission.to_csv('LR.csv', index=False)


# 随机森林
from sklearn.ensemble import RandomForestRegressor

rfModel = RandomForestRegressor(n_estimators=100)  # 树的数量
yLabelsLog = np.log1p(yLabels)
rfModel.fit(dataTrain, yLabelsLog)
preds = rfModel.predict(X=dataTrain)
print("随机森林的RMSLE值: ", rmsle(np.exp(yLabelsLog), np.exp(preds), False))
rf_preds = rfModel.predict(X=dataTest)
# 输出预测结果
submission = pd.DataFrame({
    "datetime": datetimecol,
    "count": rf_preds
})
# 导出数据 随机森林
submission.to_csv('RFR.csv', index=False)

# 集成模型-梯度提升
# from sklearn.ensemble import GradientBoostingRegressor
#
# gbm = GradientBoostingRegressor(n_estimators=4000, alpha=0.01)
# yLabelsLog = np.log1p(yLabels)
# gbm.fit(dataTrain, yLabelsLog)
# preds = gbm.predict(X=dataTrain)
# print("梯度提升的RMSLE值: ", rmsle(np.exp(yLabelsLog), np.exp(preds), False))
#
# predsTest = gbm.predict(X=dataTest)
# fig, (ax1, ax2) = plt.subplots(ncols=2)
# fig.set_size_inches(12, 5)
# sn.distplot(yLabels, ax=ax1, bins=50)
# sn.distplot(np.exp(predsTest), ax=ax2, bins=50)
# plt.plot()

# 导出数据 梯度提升
# submission = pd.DataFrame({
#     'datetime': datetimecol,
#     'count': [max(0, x) for x in np.exp(predsTest)]})
# submission.to_csv('GBR.csv', index=False)
