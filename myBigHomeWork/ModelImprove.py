import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使中文在图表中能正常显示
plt.rcParams['axes.unicode_minus'] = False  # 使负号可以正常显示
import numpy as np
import pandas as pd
import seaborn as sn
from datetime import datetime


# 下面特征工程参数优化参考网上资料
# 特征处理
def Featureprocessing():
    train = pd.read_csv('train.csv')
    # 查看测试集数据是否有缺失值
    test = pd.read_csv('test.csv')

    train_WithoutOutliers = train[np.abs(train['count'] -
                                         train['count'].mean()) <= (3 * train['count'].std())]

    Bike_data = pd.concat([train_WithoutOutliers, test], ignore_index=True)
    return Bike_data


# 为了方便查可视化数据，先把datetime拆分成成日期、时段、年份、月份、星期五列
def SplitDatetime():
    Bike_data = Featureprocessing()
    Bike_data['date'] = Bike_data.datetime.apply(lambda c: c.split()[0])
    Bike_data['hour'] = Bike_data.datetime.apply(lambda c: c.split()[1].split(':')[0]).astype('int')
    Bike_data['year'] = Bike_data.datetime.apply(lambda c: c.split()[0].split('-')[0]).astype('int')
    Bike_data['month'] = Bike_data.datetime.apply(lambda c: c.split()[0].split('-')[1]).astype('int')
    Bike_data['weekday'] = Bike_data.date.apply(lambda c: datetime.strptime(c, '%Y-%m-%d').isoweekday())
    return Bike_data


def ShowPlot():
    Bike_data = SplitDatetime()
    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(12, 10)

    axes[0, 0].set(xlabel='temp', title='Distribution of temp', )
    axes[0, 1].set(xlabel='atemp', title='Distribution of atemp')
    axes[1, 0].set(xlabel='humidity', title='Distribution of humidity')
    axes[1, 1].set(xlabel='windspeed', title='Distribution of windspeed')

    sn.distplot(Bike_data['temp'], ax=axes[0, 0])
    sn.distplot(Bike_data['atemp'], ax=axes[0, 1])
    sn.distplot(Bike_data['humidity'], ax=axes[1, 0])
    sn.distplot(Bike_data['windspeed'], ax=axes[1, 1])
    plt.show()
    return Bike_data


from sklearn.ensemble import RandomForestRegressor


def WindSpeedForecast():
    Bike_data = ShowPlot()
    # 采用随机森林填充风速
    Bike_data["windspeed_rfr"] = Bike_data["windspeed"]
    # 将数据分成风速等于0和不等于两部分
    dataWind0 = Bike_data[Bike_data["windspeed_rfr"] == 0]
    dataWindNot0 = Bike_data[Bike_data["windspeed_rfr"] != 0]
    # 选定模型
    rfModel_wind = RandomForestRegressor(n_estimators=1000, random_state=42)
    # 选定特征值
    windColumns = ["season", "weather", "humidity", "month", "temp", "year", "atemp"]
    # 将风速不等于0的数据作为训练集，fit到RandomForestRegressor之中
    rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed_rfr"])
    # 通过训练好的模型预测风速
    wind0Values = rfModel_wind.predict(X=dataWind0[windColumns])
    # 将预测的风速填充到风速为零的数据中
    dataWind0.loc[:, "windspeed_rfr"] = wind0Values
    # 连接两部分数据
    Bike_data = dataWindNot0.append(dataWind0)
    Bike_data.reset_index(inplace=True)
    Bike_data.drop('index', inplace=True, axis=1)
    return Bike_data


# 填充好再画图观察一下这四个特征值的密度分布
def ShowDensityDistribution():
    Bike_data = WindSpeedForecast()
    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(12, 10)

    axes[0, 0].set(xlabel='temp', title='Distribution of temp', )
    axes[0, 1].set(xlabel='atemp', title='Distribution of atemp')
    axes[1, 0].set(xlabel='humidity', title='Distribution of humidity')
    axes[1, 1].set(xlabel='windseed', title='Distribution of windspeed')

    sn.distplot(Bike_data['temp'], ax=axes[0, 0])
    sn.distplot(Bike_data['atemp'], ax=axes[0, 1])
    sn.distplot(Bike_data['humidity'], ax=axes[1, 0])
    sn.distplot(Bike_data['windspeed_rfr'], ax=axes[1, 1])
    plt.show()
    return Bike_data


def DataAnalysis():
    Bike_data = ShowDensityDistribution()
    # 将多类别型数据使用one-hot转化成多个二分型类别
    dummies_month = pd.get_dummies(Bike_data['month'], prefix='month')
    dummies_season = pd.get_dummies(Bike_data['season'], prefix='season')
    dummies_weather = pd.get_dummies(Bike_data['weather'], prefix='weather')
    dummies_year = pd.get_dummies(Bike_data['year'], prefix='year')
    # 把5个新的DF和原来的表连接起来
    Bike_data = pd.concat([Bike_data, dummies_month, dummies_season, dummies_weather, dummies_year], axis=1)

    #  分离训练集和测试集
    dataTrain = Bike_data[pd.notnull(Bike_data['count'])]
    dataTest = Bike_data[~pd.notnull(Bike_data['count'])].sort_values(by=['datetime'])
    datetimecol = dataTest['datetime']
    yLabels = dataTrain['count']
    yLabels_log = np.log(yLabels)

    # 把不要的列丢弃

    dropFeatures = ['casual', 'count', 'datetime', 'date', 'registered',
                    'windspeed', 'atemp', 'month', 'season', 'weather', 'year']

    dataTrain = dataTrain.drop(dropFeatures, axis=1)
    dataTest = dataTest.drop(dropFeatures, axis=1)

    # 选择模型、训练模型
    rfModel = RandomForestRegressor(n_estimators=1000, random_state=42)

    rfModel.fit(dataTrain, yLabels_log)

    preds = rfModel.predict(X=dataTrain)
    return datetimecol, dataTest, rfModel


def main():
    datetimecol, dataTest, rfModel = DataAnalysis()
    # predsTest = rfModel.predict(X=dataTest)
    #
    # submission = pd.DataFrame({'datetime': datetimecol, 'count': [max(0, x) for x in np.exp(predsTest)]})
    #
    # submission.to_csv('RF1.csv', index=False)


# 预测测试集数据

if __name__ == '__main__':
    main()
