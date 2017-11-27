import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from math import radians

# DATA PRE-PROCESSING

train = pd.read_csv("train.csv", parse_dates=['pickup_datetime'])
test = pd.read_csv("test.csv", parse_dates=['pickup_datetime'])

frames = [train, test]
df = pd.concat(frames, axis=0, join='outer')


"""
    Split the date into several columns
"""

df['year'] = df.pickup_datetime.dt.year  # train is only in 2016
df['month'] = df.pickup_datetime.dt.month
df['day'] = df.pickup_datetime.dt.day
df['weekday'] = df.pickup_datetime.dt.dayofweek
df['hour'] = df.pickup_datetime.dt.hour
df['date'] = df.pickup_datetime.dt.date


def get_daytime(hour, weekday):
    daytime = []  # np.zeros(hour.__len__())  # init vector
    hour_weekday = list(zip(hour,weekday))
    for (i,j) in hour_weekday:
        # rush hours: http://www.nytimes.com/1987/09/09/nyregion/new-york-rush-hours-grow-earlier-and-later.html?pagewanted=all
        if i in [7, 8, 9, 16, 17, 18, 19] and j < 6:
            daytime.append("rush_hours")
        elif i in np.arange(0, 6):
            daytime.append("early_morning")
        else:
            daytime.append("other")
    return daytime

df['daytime'] = get_daytime(df['hour'], df['weekday'])

"""
    Add weather data
"""

weather = pd.read_csv("Weather.csv", parse_dates=['pickup_datetime'])
weather['date'] = weather.pickup_datetime.dt.date
new_weather = weather.groupby(['date'], as_index=False)['tempm', 'rain', 'snow'].mean()

# join weather to train dataset
df = df.merge(new_weather,on='date',how='left')

# subset dataframe to get train and test separately:
train_df = df[0:train.__len__()]
test_df = df[train_df.__len__():]

"""
    Delete geographical outliers
"""
# let's see the rides in the map
longitude = list(train_df.pickup_longitude) + list(train_df.dropoff_longitude)
latitude = list(train_df.pickup_latitude) + list(train_df.dropoff_latitude)
plt.figure()
plt.plot(longitude,latitude,'.', alpha = 0.6, markersize = 0.09)
plt.show()
# after looking at this plot, we decided to only use the coordinates that are most populated
# as the other ones are outside NYC

# only select observation within geographical limits
# we will delete observations so we can only do this for train dataset
xlim = [-74.1, -73.7]
ylim = [40.6, 40.9]
train_df = train_df[(train_df.pickup_longitude> xlim[0]) & (train_df.pickup_longitude < xlim[1])]
train_df = train_df[(train_df.dropoff_longitude> xlim[0]) & (train_df.dropoff_longitude < xlim[1])]
train_df = train_df[(train_df.pickup_latitude> ylim[0]) & (train_df.pickup_latitude < ylim[1])]
train_df = train_df[(train_df.dropoff_latitude> ylim[0]) & (train_df.dropoff_latitude < ylim[1])]

"""
    Cluster by pickup location
"""
# re-calculate long and lat with cleaned data
longitude = list(train_df.pickup_longitude) + list(train_df.dropoff_longitude)
latitude = list(train_df.pickup_latitude) + list(train_df.dropoff_latitude)
# now we cluster the
k_means = KMeans(n_clusters=6)
k_means.fit(pd.DataFrame({'lat':latitude,'lon':longitude}))
# see means in map
plt.figure()
plt.plot(longitude,latitude,'.', alpha = 0.6, markersize = 0.09)
plt.plot(k_means.cluster_centers_[:,1], k_means.cluster_centers_[:,0],'.', markersize = 15)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
# create two new columns with the cluster of each location
train_df['pickup_cluster'] = k_means.labels_[0:train_df.__len__()]
train_df['dropoff_cluster'] = k_means.labels_[train_df.__len__():]

"""
    Manhattan distance
"""


# manhattan distance in kilometers
def distance_manhattan(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    md = abs(lon1 - lon2) + abs(lat1 - lat2)
    return md * 6373

train_df['manhattan_distance'] = train_df.apply(lambda r:
                                                distance_manhattan(r['pickup_latitude'], r['pickup_longitude'], r['dropoff_latitude'], r['dropoff_longitude']), axis=1)

"""
    Delete unused columns in train and test
"""

# now, get rid of unnecesary columns
drop_columns = ['id', 'dropoff_datetime', 'dropoff_latitude', 'dropoff_longitude', 'date', 'day', 'hour',
                'pickup_datetime', 'pickup_latitude', 'pickup_longitude', 'year']
train_df.drop(drop_columns, 1, inplace=True)
test_df.drop(drop_columns, 1, inplace=True)
# see columns left
list(train_df)

"""
    Convert to categorical some that are numerical
"""

train_df["month"] = train_df["month"].astype('category')
train_df["weekday"] = train_df["weekday"].astype('category')
train_df["pickup_cluster"] = train_df["pickup_cluster"].astype('category')
train_df["dropoff_cluster"] = train_df["dropoff_cluster"].astype('category')
train_df['snow'] = train_df["snow"].astype('category')

train_df.to_pickle('data_train_preprocessed.pkl')


# df = train_df
# cat_columns = train_df.select_dtypes(['object']).columns
# train_df[cat_columns] = train_df[cat_columns].apply(lambda x: x.astype('category').cat.codes)
#
# train_df = df


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import haversine
#dataset loading
#loading pickup_datetime as date dtype
train_df = pd.read_csv('train.csv', parse_dates=['pickup_datetime'])
test_df = pd.read_csv('test.csv', parse_dates=['pickup_datetime'])
train_df.dtypes
# missing values
train_df.isnull().sum()
sns.set(style="darkgrid", palette="deep", color_codes=True)
f, axes = plt.subplots(1, 1, figsize=(11, 7), sharex=True)
sns.despine(left=True)
sns.distplot(np.log(train_df['trip_duration'])+1, axlabel = 'Log(trip_duration)', label = 'log(trip_duration)', bins = 50, color="b")
plt.setp(axes, yticks=[])
plt.tight_layout()
plt.show()

#data summary
pd.set_option('display.float_format', lambda x: '%.3f' % x)
train_df.describe()

#in above summary i fwe look at the trip duration minimum is 1 second, and the maximum is 3.5million seconds approx 950 hour
#so it's not possible to travel for those many hours, it clearly shows that there are some outliers
#removing outliers
mean = np.mean(train_df.trip_duration)
sd = np.std(train_df.trip_duration)
train_df = train_df[train_df['trip_duration'] <= mean + 2*sd]
train_df = train_df[train_df['trip_duration'] >= mean - 2*sd]

plt.hist(train_df['trip_duration'].values, bins=100)
plt.xlabel('trip_duration')
plt.ylabel('number of train records')
plt.show()

#transform the target trip duration to logarithemic form i.e x ==> log(x+1)
train_y = np.log1p(train_df.trip_duration)

# Add some features.. like distance between pick up and drop off coordinates
#simplify pickupdate feature  into more specific features like month, day, weekday etc
train_df['distance'] = train_df.apply(lambda r: haversine.haversine((r['pickup_latitude'], r['pickup_longitude']), (r['dropoff_latitude'], r['dropoff_longitude'])), axis=1)
train_df['month'] = train_df.pickup_datetime.dt.month
train_df['day'] = train_df.pickup_datetime.dt.day
train_df['weekday'] = train_df.pickup_datetime.dt.dayofweek
train_df['hour'] = train_df.pickup_datetime.dt.hour
train_df['store_and_fwd_flag'] = train_df['store_and_fwd_flag'].map(lambda x: 0 if x == 'N' else 1)
train_df = train_df.drop(['id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration'], axis=1)

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor

xgb_model = XGBRegressor(objective='reg:linear', n_estimators=500, subsample=0.75, max_depth=5)
rf_model = RandomForestRegressor(n_estimators=25, min_samples_leaf=25, min_samples_split=25)
tree_model = DecisionTreeRegressor(min_samples_leaf=25, min_samples_split=25)

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

train_x, Test_X,Train_y, test_y =  train_test_split(train_df, train_y, test_size = 0.2)

param_grid = {"learning_rate": np.random.uniform(0.001, 0.1, 5),'max_depth': np.arange(2,8)}

tree_model.fit(train_x, Train_y)

