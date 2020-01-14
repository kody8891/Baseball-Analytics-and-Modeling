# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC #Predicting OPS Using Batter Tendencies
# MAGIC 
# MAGIC 
# MAGIC After the 2011 Academy Award nominated film *Moneyball* chronicled the rise of the 2001 Oakland Athletics and their use of analytics, every Major League baseball organization started investing into research and development.
# MAGIC 
# MAGIC ### Problem Statement
# MAGIC We are looking to use 2015-2018 baseball season data to predict and validate the 2019 season’s statistics. This is due to our interest in predictive modeling in the sports sector and our curiosity to see how well wins can be predicted based on various factors such as hits and pitch velocity.
# MAGIC 
# MAGIC ### References 
# MAGIC 
# MAGIC Data obtained data from:
# MAGIC 
# MAGIC * <a href="https://baseballsavant.mlb.com" rel="noopener noreferrer" target="_blank">Baseball Savant</a>
# MAGIC * <a href="https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=y&type=8&season=2015&month=0&season1=2015&ind=0" rel="noopener noreferrer" target="_blank">Fangraphs</a>
# MAGIC 
# MAGIC ### ETL
# MAGIC 
# MAGIC After we etracted the data from our sources it was pretty simple to load thanks to Spark. We simply imported our csv datasets into Databricks using Spark via the data tab on the lefthand side of the website. It is also worth noting that when we finished our cleansing that we converted all of the spark dataframes to pandas dataframes to easily utilize visualization libraries such as seaborn, matplotlib as well as the popular machine learning library scikitlearn.
# MAGIC 
# MAGIC ### Data wrangling
# MAGIC 
# MAGIC Wrangling the data was a bit more complicated. We applied several different techniques to obtain the data desired.
# MAGIC 
# MAGIC 
# MAGIC - Joins via Spark SQL
# MAGIC - Removal of missing/na values
# MAGIC - Data aggregation via Spark
# MAGIC - Data concentation for training models 
# MAGIC - Data type manipulation 
# MAGIC - Filtering/subsetting 
# MAGIC 
# MAGIC 
# MAGIC ### Data Quality/Completeness
# MAGIC 
# MAGIC The fangraphs datasets include up to date observations from all qualified batters (A qualified batter is a batter that attains at least 502 plate appearances over a 162 game season, or 3.1 PA per game.) The dataset was curated by the MLB so data is very complete,high quality, accurate, and up to date.
# MAGIC   

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Pre-processing and Cleaning
# MAGIC 
# MAGIC #### Raw Data
# MAGIC 
# MAGIC * <b>sv_id</b>: unique pitch id
# MAGIC * <b>game_date</b>: date of game
# MAGIC * <b>pitch_type</b>: type of pitch the pitcher threw (fastball = FF, curveball = CU, etc.)
# MAGIC * <b>description</b>: description of the pitch (ball, strike, hit_into_play, etc.)
# MAGIC * <b>events</b>: description of the end of the plate appearance (single, home_run, etc.) This is different from description because this is NA for pitches that do not end an at bat
# MAGIC   - For example, a home run would have an event `home_run` and a description of `hit_into_play_score`
# MAGIC   - For example, a called strike would have an event `NA` and a description of `called_strike`
# MAGIC   - For example, a swinging strikeout would have an event `swinging_strikeout` and a description of `swinging_strike`
# MAGIC * <b>launch_speed</b>: how hard the batter hit the ball, in mph
# MAGIC * <b>launch_angle</b>: the vertical angle at which the ball left the bat, in degrees
# MAGIC * <b>hit_distance_sc</b>: distance of batted ball, in feet
# MAGIC * <b>player_name</b>: batter name
# MAGIC * <b>batter</b>: batter id
# MAGIC * <b>pitcher</b>: pitcher id
# MAGIC * <b>home_team</b>: home team
# MAGIC * <b>away_team</b>: away team
# MAGIC * <b>inning_topbot</b>: the top or bottom of the inning
# MAGIC   - If it’s the top of the inning, the away team is batting
# MAGIC   - If it’s the bottom of the inning, the home team is batting
# MAGIC 
# MAGIC 
# MAGIC #### Modeling Predictors
# MAGIC 
# MAGIC * <b>Swing%</b>: how many times the batter swings divided by pitches seen
# MAGIC * <b>Contact%</b>: how many times the batter makes contact divided by the number of swings
# MAGIC * <b>BB%</b>: walk percentage
# MAGIC * <b>K%</b>: strikeout percentage
# MAGIC * <b>Average Exit Velocity</b>: average launch speed
# MAGIC * <b>Average Launch Angle</b>: average launch angle
# MAGIC * <b>Ground Ball%</b>: how many times the batter hits a ground ball divided by number of balls in play
# MAGIC * <b>Fly Ball%</b>: how many times the batter hits a fly ball divided by number of balls in play
# MAGIC 
# MAGIC #### Response
# MAGIC 
# MAGIC * <b>OBP</b>: on base percentage (How many times the batter got on base divided by opportunities)
# MAGIC   - \> .400: great
# MAGIC   - \> .350: good
# MAGIC   - \> .300: bad
# MAGIC   - < .300: terrible
# MAGIC * <b>SLG</b>: slugging percentage (How many bases the batter acquired divided by at bats)
# MAGIC   - \> .500: great
# MAGIC   - \> .400: good
# MAGIC   - \> .300: bad
# MAGIC   - < .300: terrible
# MAGIC * <b>OPS</b>: the sum of OBP and SLG
# MAGIC   - \> 1.000: excellent (MVP)
# MAGIC   - \> .900: great
# MAGIC   - \> .800: good
# MAGIC   - \> .700: average
# MAGIC   - \> .600: bad
# MAGIC   - < .600: terrible

# COMMAND ----------

from functools import reduce
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import substring, length, expr
from pyspark.sql import functions as F
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import random as rand
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# COMMAND ----------

# Helper functions

def read_pitches(year):
  pitches = spark.read.format("csv") \
  .option("inferSchema", "false") \
  .option("header", "true") \
  .option("sep", ",") \
  .load("/FileStore/tables/pitches_" + year + ".csv")
  
  return(pitches)

def join_fangraphs(year, df):
  fangraphs = spark.read.format("csv") \
    .option("inferSchema", "false") \
    .option("header", "true") \
    .option("sep", ",") \
    .load("/FileStore/tables/fangraphs_" + year + ".csv") \
    .withColumnRenamed("Name", "player_name") \
    .withColumnRenamed("BB%", "bb_pct") \
    .withColumnRenamed("K%", "k_pct")

  fangraphs = fangraphs.withColumn("OPS", fangraphs.OBP + fangraphs.SLG)
  
  fangraphs = fangraphs.withColumn("bb_pct", expr("substring(bb_pct, 1, length(bb_pct)-2)"))
  fangraphs = fangraphs.withColumn("bb_pct", fangraphs.bb_pct.cast("float") / 100)
  
  fangraphs = fangraphs.withColumn("k_pct", expr("substring(k_pct, 1, length(k_pct)-2)"))
  fangraphs = fangraphs.withColumn("k_pct", fangraphs.k_pct.cast("float") / 100)
  
  df = df.join(fangraphs, on="player_name").toPandas()
  
  df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
  df = df.rename(columns={"BB%": "bb_pct", "K%": "k_pct"})
  
  return(df)

def count_stat(pitches, name):
  pitches_agg = pitches \
    .groupby(["player_name", "batter"]) \
    .count() \
    .withColumnRenamed("count", name)
  
  return(pitches_agg)

def group_pitches(pitches):
  pitches_agg = pitches \
    .groupby(["player_name", "batter"]) \
    .agg({"launch_speed": "mean",
          "launch_angle": "mean",
          "events": "count"}) \
    .withColumnRenamed("avg(launch_speed)", "avg_launch_speed") \
    .withColumnRenamed("avg(launch_angle)", "avg_launch_angle") \
    .withColumnRenamed("count(events)", "pitches")
  
  swings = pitches[(pitches.description.isin([
    "swinging_strike", 
    "foul", 
    "hit_into_play", 
    "hit_into_play_no_out", 
    "foul_tip", 
    "swinging_strike_blocked",
    "hit_into_play_score"
  ]))]
  
  contacts = swings[(swings.description.isin([
    "foul", 
    "hit_into_play", 
    "hit_into_play_no_out", 
    "foul_tip", 
    "hit_into_play_score"
  ]))]
  
  balls_in_play = contacts[(contacts.description.isin([
    "hit_into_play", 
    "hit_into_play_no_out", 
    "hit_into_play_score"
  ]))]

  plate_apps = pitches[pitches.events != "NA"]
  ground_balls = pitches[pitches.launch_angle < 10]
  fly_balls = pitches[(pitches.launch_angle > 25) & (pitches.launch_angle < 50)]
  
  swings_agg = count_stat(swings, "swings")
  contacts_agg = count_stat(contacts, "contacts")
  balls_in_play_agg = count_stat(balls_in_play, "bip")
  plate_apps_agg = count_stat(plate_apps, "pa")
  ground_balls_agg = count_stat(ground_balls, "gb")
  fly_balls_agg = count_stat(fly_balls, "fb")
  
  dfs = [pitches_agg, swings_agg, contacts_agg, plate_apps_agg, ground_balls_agg, fly_balls_agg, balls_in_play_agg]
  pitches_agg = reduce(lambda left,right: left.join(right, on=["player_name", "batter"]), dfs)
  pitches_agg = pitches_agg[(pitches_agg.pa > 100)]
  
  pitches_agg = pitches_agg.withColumn("swing_pct", pitches_agg.swings / pitches_agg.pitches)
  pitches_agg = pitches_agg.withColumn("contact_pct", pitches_agg.contacts / pitches_agg.swings)
  pitches_agg = pitches_agg.withColumn("gb_pct", pitches_agg.gb / pitches_agg.bip)
  pitches_agg = pitches_agg.withColumn("fb_pct", pitches_agg.fb / pitches_agg.bip)

  return(pitches_agg)

def calc_rmse(actual, predicted):
  error = actual - predicted
  se = (error)**2 
  mean_se = np.mean(se)
  rmse = np.sqrt(mean_se)
  return rmse

cols = [
  'RBI',
  'SB',
  'G',
  'PA',
  'HR',
  'R',
  'ISO',
  'BABIP',
  'AVG',
  'OBP',
  'SLG',
  'wOBA',
  'wRC+',
  'BsR',
  'Off',
  'Def',
  'WAR',
  'avg_launch_angle',
  'avg_launch_speed'
]

# COMMAND ----------

# Load Data

pitches_2015 = read_pitches("2015")
df_2015 = group_pitches(pitches_2015)
df_2015 = join_fangraphs("2015", df_2015)
pitches_2015 = pitches_2015.toPandas().dropna()

pitches_2016 = read_pitches("2016")
df_2016 = group_pitches(pitches_2016)
df_2016 = join_fangraphs("2016", df_2016)
pitches_2016 = pitches_2016.toPandas().dropna()

pitches_2017 = read_pitches("2017")
df_2017 = group_pitches(pitches_2017)
df_2017 = join_fangraphs("2017", df_2017)
pitches_2017 = pitches_2017.toPandas().dropna()

pitches_2018 = read_pitches("2018")
df_2018 = group_pitches(pitches_2018)
df_2018 = join_fangraphs("2018", df_2018)
pitches_2018 = pitches_2018.toPandas().dropna()

pitches_2019 = read_pitches("2019")
df_2019 = group_pitches(pitches_2019)
df_2019 = join_fangraphs("2019", df_2019)
pitches_2019 = pitches_2019.toPandas().dropna()

# COMMAND ----------

# MAGIC %md #Exploratory Data Visualizations

# COMMAND ----------

p = sns.countplot(y='player_name',data=pitches_2019,order=pitches_2019.player_name.value_counts().iloc[:10].index)
plt.title("Most Pitches Seen by Batters in 2019")
plt.xlabel("Batter")
plt.ylabel("Number of Pitches")
plt.tight_layout()
display(p)


# COMMAND ----------

hr18=df_2019.sort_values('WAR',ascending=False).head(10)
p = sns.barplot(x='player_name',y='WAR',data= hr18)
plt.title("Wins Above Replacement in 2019")
plt.xlabel("Player")
plt.xticks(rotation=50)
plt.ylabel("Wins Above Replacement")
plt.tight_layout()
display(p)

# COMMAND ----------

l18=df_2019.sort_values('HR',ascending=False).head(10)
p = sns.barplot(x='player_name',y='HR',data=l18)
plt.title("Home Run Leaders in 2019")
plt.xlabel("Batter")
plt.xticks(rotation=50)
plt.ylabel("Home Runs")
plt.tight_layout()
display(p)

# COMMAND ----------

s=sns.regplot(x=df_2019["avg_launch_speed"], y=df_2019["HR"])
plt.title('Home Runs by Launch Speed')
plt.xlabel('Average Launch Speed')
plt.ylabel('Total Home Runs')
display(s)

# COMMAND ----------

s=sns.regplot(x=df_2019["OPS"], y=df_2019["R"])
plt.title('Runs by On-base plus slugging (OPS)')
plt.xlabel('Average OPS')
plt.ylabel('Total Runs')
display(s)

# COMMAND ----------

s=sns.regplot(x=df_2019["swings"], y=df_2019["AVG"])
plt.title('Batting Average by # of Swings')
plt.xlabel('Swings')
plt.ylabel('Batting Average')
display(s)

# COMMAND ----------

k=sns.distplot(df_2019['AVG'], bins=20)
plt.title("Batting Average Distribuiton 2019")
plt.xlabel("Batting Average ")
plt.ylabel("Batting AVG")
display(k)

# COMMAND ----------

L=sns.distplot(df_2019['OPS'], bins=20)
plt.title("OPS Distribuiton 2019")
plt.xlabel("Distribuition")
plt.ylabel("OPS")
display(L)

# COMMAND ----------

# MAGIC %md #Model Building

# COMMAND ----------

# MAGIC %md #### Multiple Linear Regression

# COMMAND ----------

# Dividing into 80-20 training and testing set for data 2015-2018 - primary testing
train = pd.concat([df_2015, df_2016, df_2017, df_2018])
test = df_2019
length_train = range(0, len(train))
train_80_size = int(.8*len(train))

train_80_rows = rand.sample(length_train, k=train_80_size)
train_80_rows.sort()
train_20_rows = []
for x in length_train:
  if x not in train_80_rows:
    train_20_rows.append(x)
    train_20_rows.sort()
    
train_80 = train.iloc[train_80_rows]
train_20 = train.iloc[train_20_rows]

# COMMAND ----------

# MAGIC %md https://xavierbourretsicotte.github.io/subset_selection.html - forward selection python algorithm

# COMMAND ----------

def fit_linear_reg(X,Y):
    #Fit linear regression model and return RSS and R squared values
    model_k = LinearRegression(fit_intercept = True)
    model_k.fit(X,Y)
    RSS = mean_squared_error(Y,model_k.predict(X)) * len(Y)
    R_squared = model_k.score(X,Y)
    return RSS, R_squared

#Initialization variables
Y = train_80["OPS"]
X = train_80[['fb','gb','avg_launch_angle','avg_launch_speed','k_pct','bb_pct','contact_pct','swing_pct']]
k = 8

remaining_features = list(X.columns.values)
features = []
RSS_list, R_squared_list = [np.inf], [np.inf] #Due to 1 indexing of the loop...
features_list = dict()

for i in range(1,k+1):
    best_RSS = np.inf
    
    for combo in itertools.combinations(remaining_features,1):

            RSS = fit_linear_reg(X[list(combo) + features],Y)   #Store temp result 

            if RSS[0] < best_RSS:
                best_RSS = RSS[0]
                best_R_squared = RSS[1] 
                best_feature = combo[0]

    #Updating variables for next loop
    features.append(best_feature)
    remaining_features.remove(best_feature)
    
    #Saving values for plotting
    RSS_list.append(best_RSS)
    R_squared_list.append(best_R_squared)
    features_list[i] = features.copy()

# COMMAND ----------

df1 = pd.concat([pd.DataFrame({'features':features_list}),pd.DataFrame({'RSS':RSS_list, 'R_squared': R_squared_list})], axis=1, join='inner')
df1['numb_features'] = df1.index

#Initializing useful variables
m = len(Y)
p = 11
hat_sigma_squared = (1/(m - p -1)) * min(df1['RSS'])

#Computing
df1['C_p'] = (1/m) * (df1['RSS'] + 2 * df1['numb_features'] * hat_sigma_squared )
df1['AIC'] = (1/(m*hat_sigma_squared)) * (df1['RSS'] + 2 * df1['numb_features'] * hat_sigma_squared )
df1['BIC'] = (1/(m*hat_sigma_squared)) * (df1['RSS'] +  np.log(m) * df1['numb_features'] * hat_sigma_squared )
df1['R_squared_adj'] = 1 - ( (1 - df1['R_squared'])*(m-1)/(m-df1['numb_features'] -1))
df1['RMSE'] = np.sqrt(df1['RSS']/len(df1['RSS']))
df1

#including all predictors has greated AIC - therefore using all 8 predictors works!

# COMMAND ----------

ops_array = np.array(train_80["OPS"])
i = 0
predictors_array = np.zeros((len(train_80["OPS"]), 8))
for fb, gb, angle, speed, k, bb, cont, swing  in zip(train_80['fb'], train_80['gb'], train_80['avg_launch_angle'], train_80['avg_launch_speed'], train_80['k_pct'], train_80['bb_pct'], train_80['contact_pct'], train_80['swing_pct']):
  predictors_array[i, 0] = fb
  predictors_array[i, 1] = gb
  predictors_array[i, 2] = angle
  predictors_array[i, 3] = speed
  predictors_array[i, 4] = k
  predictors_array[i, 5] = bb
  predictors_array[i, 6] = cont
  predictors_array[i, 7] = swing
  i += 1
  
model = LinearRegression().fit(predictors_array, ops_array)

# COMMAND ----------

print('intercept:', model.intercept_)
print('slope:', model.coef_)

# COMMAND ----------

predictors20_array = np.zeros((len(train_20["OPS"]), 8))
i = 0
for fb, gb, angle, speed, k, bb, cont, swing  in zip(train_20['fb'], train_20['gb'], train_20['avg_launch_angle'], train_20['avg_launch_speed'], train_20['k_pct'], train_20['bb_pct'], train_20['contact_pct'], train_20['swing_pct']):
  predictors20_array[i, 0] = fb
  predictors20_array[i, 1] = gb
  predictors20_array[i, 2] = angle
  predictors20_array[i, 3] = speed
  predictors20_array[i, 4] = k
  predictors20_array[i, 5] = bb
  predictors20_array[i, 6] = cont
  predictors20_array[i, 7] = swing
  i += 1
  
predict20 = model.predict(predictors20_array)

# COMMAND ----------

#Plotting the OPS values - actual vs. predicted, using the training data
train_20["predicted_values"] = model.predict(train_20[['fb', 'gb', 'avg_launch_angle', 'avg_launch_speed', 'k_pct', 'bb_pct', 'contact_pct', 'swing_pct']])
display(train_20.plot.scatter(x='predicted_values',
                          y='OPS',
                          c='Pink'))

# COMMAND ----------

predictors_test_array = np.zeros((len(test["OPS"]), 8))
i = 0
for fb, gb, angle, speed, k, bb, cont, swing  in zip(test['fb'], test['gb'], test['avg_launch_angle'], test['avg_launch_speed'], test['k_pct'], test['bb_pct'], test['contact_pct'], test['swing_pct']):
  predictors_test_array[i, 0] = fb
  predictors_test_array[i, 1] = gb
  predictors_test_array[i, 2] = angle
  predictors_test_array[i, 3] = speed
  predictors_test_array[i, 4] = k
  predictors_test_array[i, 5] = bb
  predictors_test_array[i, 6] = cont
  predictors_test_array[i, 7] = swing
  i += 1
  
predict_test = model.predict(predictors_test_array)

# COMMAND ----------

#Plotting the OPS values - actual vs. predicted, using the testing data
test['predicted_values'] = model.predict(test[['fb', 'gb', 'avg_launch_angle', 'avg_launch_speed', 'k_pct', 'bb_pct', 'contact_pct', 'swing_pct']])
display(test.plot.scatter(x='predicted_values',
                          y='OPS',
                          c='Pink',))

# COMMAND ----------

tst_rmse = calc_rmse(test['OPS'], test['predicted_values'])
print(tst_rmse)

# COMMAND ----------

# MAGIC %md ### Random Forest Regression

# COMMAND ----------

train = pd.concat([df_2015, df_2016, df_2017, df_2018])
test = df_2019

cat_features = []
for col in train.columns:
    if train[col].dtype =='object':
        cat_features.append(col)

train_clean = train.drop(cat_features, axis=1)
test_clean = test.drop(cat_features, axis=1)

cols = ['fb', 'gb', 'avg_launch_angle', 'avg_launch_speed', 'k_pct', 'bb_pct', 'contact_pct', 'swing_pct']
x = train_clean.loc[:, cols]
y = train['OPS']

test_x = test_clean.loc[:, cols]

#test_x = test_clean.loc[:, test_clean.columns.difference(['OPS', 'predicted_values'])]
test_y = test_clean['OPS']

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(x, y)

predicted_values = regressor.predict(test_x)




figure = plt.scatter(test_y, predicted_values, color = 'blue')
plt.xlabel("Actual OPS")
plt.ylabel("Predicted OPS")
plt.title("Actual vs Predicted OPS")
plt.ylim(0.6,1.2)
plt.xlim(0.6,1.2)
display(plt.show(figure))

# COMMAND ----------

tst_rmse = calc_rmse(test_y, predicted_values)
print(tst_rmse)

# COMMAND ----------

def rfr_model(X, y):
# Perform Grid-Search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(3,8),
            'n_estimators': (10, 50, 200, 500),
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    
    rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"], random_state=False, verbose=False)
# Perform K-Fold CV
    #scores = cross_val_score(rfr, X, y, cv=10, scoring='neg_mean_absolute_error')

    return rfr

  

# COMMAND ----------

rfr = rfr_model(train_clean.loc[:, cols], train_clean['OPS'])
rfr.fit(train_clean.loc[:, cols], train_clean['OPS'])
predicted_values_CV = rfr.predict(test_x)
plt.clf()
figure = plt.scatter(test_y, predicted_values_CV, color = 'blue')
plt.title("Actual vs Predicted OPS")
plt.xlabel("Actual OPS")
plt.ylabel("Predicted OPS")
plt.ylim(0.6,1.2)
plt.xlim(0.6,1.2)
display(plt.show(figure))

# COMMAND ----------

tst_rmse_cv = calc_rmse(test_y, predicted_values_CV)
print(tst_rmse_cv)

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} '.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
  
evaluate(rfr,test_x, test_y)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Model Evaluation
# MAGIC 
# MAGIC * The linear regression performed with an acceptable RMSE and predicted higher caliber players somewhat often
# MAGIC * The random forest performed with a better RMSE, but was hesitant to predict higher caliber players
# MAGIC * The cross-validated random forest performed with the best RMSE, but ran into the same problem as the original random forest