#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

import os
import pdb
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA    # 找不到明顯的降維方式
import xgboost as xgb
from xgboost import XGBRegressor
import warnings
from model_helper import custom_loss

# 忽略所有類型的警告
warnings.filterwarnings('ignore')





#%%
#============================建立訓練數據============================
#載入模型
LookBackNum = 12     # 參考前12筆
ForecastNum = 48     # 一天要預測48筆
features_columns = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']
target_column = ['Power(mW)']

# 載入1~17 所有 location 的 AVG(完整)，但要加上sensor_id
SourceData = pd.DataFrame()

for i in range(1,18):
  s = str(i)
  if i < 10: 
    s = '0' + s
  Avg_DataName = os.getcwd() + f'\\ExampleTrainData(AVG)\\AvgDATA_{s}.csv'
  Incomplete_Avg_DataName = os.getcwd() + f'\\ExampleTrainData(IncompleteAVG)\\IncompleteAvgDATA_{s}.csv'

  if os.path.exists(Avg_DataName):
    Avg_temp_data = pd.read_csv(Avg_DataName, encoding='utf-8')
    Incomplete_Avg_temp_data = pd.read_csv(Incomplete_Avg_DataName, encoding='utf-8')
    # 先結合AvgDATA和IncompleteAvgDATA
    combined_df = pd.concat([Avg_temp_data, Incomplete_Avg_temp_data])

    combined_df['Serial'] = combined_df['Serial'].astype(str)
    #為了排序先加一個Timestamp(之後要刪掉)
    combined_df['Timestamp'] = pd.to_datetime(combined_df['Serial'].str[:12], format='%Y%m%d%H%M')
    #為了計算是不是每一天都是60筆資料先加一個Date(之後要刪掉)
    combined_df['Date'] = combined_df['Timestamp'].dt.date  # Extract date only for grouping

    # 將同一天的資料整理成一組，計算裡面是否有60筆。最後篩選出有60筆資料的天數
    grouped = combined_df.groupby('Date').size()
    valid_dates_60 = grouped[grouped == 60].index 
    
    filtered_df_60 = combined_df[combined_df['Date'].isin(valid_dates_60)]

    filtered_df_60_sorted = filtered_df_60.sort_values(by='Timestamp').reset_index(drop=True)
    
    filtered_df_60_sorted['sensor_id'] = i

    SourceData = pd.concat([SourceData, filtered_df_60_sorted], axis=0)

SourceData = SourceData.drop(columns=['Timestamp']).drop(columns=['Date']).reset_index(drop=True)
print(SourceData.describe())
# 透過 describe 發現 WindSpeed 過於不平均，pressure 幾乎都一樣 
print(SourceData.tail(n=2))
print(SourceData.shape[0])
print(SourceData.count())
print(F"{len(SourceData)}筆資料")


#%%
#============================特徵工程函數============================
def feature_engineering(data, scaler=None, fit_scaler=True):
    # 1. 替換離群值為平均，並且不採用 Windspeed
    features_columns = ['Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']
    xgb_X_train = data[features_columns]
    Q1 = xgb_X_train.quantile(0.25)
    Q3 = xgb_X_train.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # 替換離群值為各自特徵的平均值
    for column in features_columns:
        mean_value = xgb_X_train[column].mean()
        xgb_X_train.loc[(xgb_X_train[column] < lower_bound[column]) | (xgb_X_train[column] > upper_bound[column]), column] = mean_value

    # 2. 添加特徵：時間特徵
    # xgb_X_train['Day'] = data['Serial'].astype(str).str[6:8].astype(int)
    # xgb_X_train['Hour'] = data['Serial'].astype(str).str[8:10].astype(int)
    # xgb_X_train['Minute'] = data['Serial'].astype(str).str[10:12].astype(int)
    # features_columns = [
    #     'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)',
    # ]

    # 4. 資料正規化：LSTM (僅用來預測特徵值) + 迴歸分析 (預測最終結果)
    Regression_X_train = xgb_X_train[features_columns].values
    if scaler is None:
        scaler = MinMaxScaler()
    # 創一個新的用 fit_transform，拿現有的則用 transform
    if fit_scaler:
        AllOutPut_MinMax = scaler.fit_transform(Regression_X_train)
    else:
        AllOutPut_MinMax = scaler.transform(Regression_X_train)
    AllOutPut_MinMax_df = pd.DataFrame(AllOutPut_MinMax, columns=features_columns)

    # 5. 補回 sensor_id (從 data 補就可以)
    AllOutPut_MinMax_df['sensor_id'] = data['sensor_id'].values

    return AllOutPut_MinMax_df, scaler, features_columns





#%%
#============================應用特徵工程============================

# 對 TrainData 進行特徵工程
AllOutPut_MinMax_df, LSTM_MinMaxModel, features_columns = feature_engineering(SourceData)
AllOutPut_MinMax_df.describe()
# print(features_columns)


#%%
# 共線性檢查(每個特徵建議不超過10)，避免多重共線性
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

vif_data = pd.DataFrame()
vif_data["Feature"] = AllOutPut_MinMax_df[features_columns].columns
vif_data["VIF"] = [variance_inflation_factor(AllOutPut_MinMax_df[features_columns].values, i) for i in range(len(AllOutPut_MinMax_df[features_columns].columns))]
print(vif_data)





#%%
#============================建置&訓練「LSTM模型」============================
# 建置LSTM模型 input layer, hidden layer
early_stopping = EarlyStopping(
    monitor='val_loss',   # 監控指標（例如驗證損失）
    patience=10,          # 在指標沒有改善時等待的 epoch 次數
    restore_best_weights=True  # 恢復訓練過程中性能最好的權重
)

regressor = Sequential()
regressor.add(LSTM(units=128, return_sequences=True, input_shape=(LookBackNum, len(features_columns))))
regressor.add(BatchNormalization())
regressor.add(LSTM(units=64))
regressor.add(BatchNormalization())  # 在每個 LSTM layer 後面都加上正則化
regressor.add(Dropout(0.2))
regressor.add(Dense(units=len(features_columns)))
regressor.compile(optimizer = 'adam', loss = custom_loss)

unique_ids = SourceData['sensor_id'].unique()
for sid in unique_ids:
  print(f"Training model for sensor_id: {sid}")
  # 相同的 sensor_id 一起訓練，但 sensor_id 本人要拿掉
  Avg_temp_data = AllOutPut_MinMax_df[AllOutPut_MinMax_df['sensor_id'] == sid].drop(columns=['sensor_id']).values
  temp_X_train = []
  temp_y_train = []
  #設定每i-12筆資料(X_train)就對應到第i筆資料(y_train)
  for i in range(LookBackNum, len(Avg_temp_data)):
    if i % 60 == 0:
       i = i +12
    temp_X_train.append(Avg_temp_data[i-LookBackNum:i, :])
    temp_y_train.append(Avg_temp_data[i, :])
  temp_X_train = np.array(temp_X_train)
  temp_y_train = np.array(temp_y_train)
  #Reshaping for LSTM
  #(samples 是訓練樣本數量,timesteps 是每個樣本的時間步長,features 是每個時間步的特徵數量)
  temp_X_train = np.reshape(temp_X_train, (temp_X_train.shape[0], temp_X_train.shape[1], len(features_columns)))
  # 開始訓練
  # breakpoint()
  regressor.fit(temp_X_train, temp_y_train, callbacks=[early_stopping], epochs=10, batch_size=32)
  
# saved model
regressor.save('WeatherLSTM_all.h5')
print('Model trained and saved successfully.')






#%%
#============================建置&訓練「回歸模型」========================
# 開始迴歸分析(對發電量做迴歸)，拿之前的scaler做特徵工程
Regression_X_train = AllOutPut_MinMax_df.drop(columns=['sensor_id']).values
Regression_y_train = SourceData[target_column].values
RegressionModel = LinearRegression()
RegressionModel.fit(Regression_X_train, Regression_y_train)
# breakpoint()
# RandomForestModel = RandomForestRegressor(n_estimators=100, random_state=42)
# RandomForestModel.fit(Regression_X_train, Regression_y_train)

# XGBModel = XGBRegressor(objective='reg:squarederror', n_estimators=100)
# XGBModel.fit(Regression_X_train, Regression_y_train)
# features = XGBModel.feature_importances_
# for i, im in enumerate(features):
#   print(f'{features_columns[i]}\t\t: {im}')

# 保存模型
# joblib.dump(RandomForestModel, 'RandomForest_all')
# joblib.dump(XGBModel, 'XGBoost_all')
joblib.dump(RegressionModel, f'WeatherRegression_all')

print('截距: ',RegressionModel.intercept_)
print('係數 : ', RegressionModel.coef_)
print('R squared: ',RegressionModel.score(Regression_X_train, Regression_y_train))






#%%
#============================預測數據============================
regressor = load_model('WeatherLSTM_all.h5', custom_objects={'custom_loss': custom_loss})
RegressionModel = joblib.load(f'WeatherRegression_all')
RandomForestModel = joblib.load('RandomForest_all')
XGBModel = joblib.load('XGBoost_all')

#載入答案
Avg_DataName = os.getcwd()+'\\ExampleTestData\\upload_short.csv'
HeaderName = os.getcwd()+'\\ExampleTestData\\Title.txt'
header = []
with open(HeaderName, 'r', encoding='utf-8') as f:
  header = f.readline().strip().split(',')
  
SourceData = pd.read_csv(Avg_DataName, header=None, encoding='utf-8')
SourceData.columns = header
target = ['序號']
EXquestion = SourceData[target].values     # 所有題目的14碼序號(9600行)
# 預測
inputs = []   # LSTM 預測時要參考的 LookBackNum筆 的資料
PredictOutput = []   # LSTM 預測值(5個特徵)
PredictPower = []    # Regression 預測值(1個發電量值)

count = 0
while(count < len(EXquestion)):
  print('count : ', count)
  LocationCode = int(EXquestion[count])
  strLocationCode = str(LocationCode)[-2:]
  if LocationCode < 10:
    strLocationCode = '0'+LocationCode
    
  # 載入此題對應的 IncompleteAVG (因為題目的7:00~8:59 都在 IncompleteAVG 裡面)
  Avg_DataName = os.getcwd()+'\\ExampleTrainData(IncompleteAVG)\\IncompleteAvgDATA_'+ strLocationCode +'.csv'
  SourceData = pd.read_csv(Avg_DataName, encoding='utf-8')
  SourceData['sensor_id'] = LocationCode
  
  # 在TestData上面套用一樣的特徵工程，取出序號欄和其他特徵欄
  TestOutPut_MinMax_df, _, features_columns = feature_engineering(SourceData, scaler=LSTM_MinMaxModel, fit_scaler=False)
  ReferTitle = SourceData[['Serial']].values
  ReferData = TestOutPut_MinMax_df[features_columns].values
  
  #找到相同的一天(在同個裝置碼下)，把12個資料都加進inputs(因為是IncompleteAVG)
  for DaysCount in range(len(ReferTitle)):
    if str(int(ReferTitle[DaysCount]))[:8] == str(int(EXquestion[count]))[:8]:
      TempData = ReferData[DaysCount].reshape(1,-1)   # 1列，若干欄
      inputs.append(TempData)
  # 此時 input 長度為 (LookBackNum,特徵數)
      
  # 參考 LookBackNum筆 資料的 sliding window，總共預測 ForecastNum筆
  for i in range(ForecastNum):
    if i > 0:   # 把上一輪預測的加進來(sliding window)
      inputs.append(PredictOutput[i-1].reshape(1, len(features_columns)))
    
    X_test = []
    X_test.append(inputs[0+i:LookBackNum+i])      # X_test.shape = (1, 12, 5)
    #Reshaping for LSTM
    NewTest = np.array(X_test)
    NewTest = np.reshape(NewTest, (NewTest.shape[0], NewTest.shape[1], len(features_columns)))
    
    predicted = regressor.predict(NewTest)
    PredictOutput.append(predicted)
    # 再拿LSTM預測的特徵，集成學習預測發電量
    # 使用集成模型進行預測
    linear_pred = RegressionModel.predict(predicted)
    # rf_pred = RandomForestModel.predict(predicted)
    # xgb_pred = XGBModel.predict(predicted)
    # final_prediction = (linear_pred + rf_pred + xgb_pred) / 3
    final_prediction = linear_pred
    PredictPower.append(np.round(final_prediction, 2).flatten())
    
  # 每天有48行預測，+48 換到下一天
  count += 48

PredictPower = [p[0] if isinstance(p, (list, np.ndarray)) else p for p in PredictPower]
df = pd.DataFrame({
    '序號': EXquestion.flatten(),  # 將 EXquestion 展平為一維數組
    '答案': PredictPower
})

df.to_csv('output.csv', index=False, encoding='utf-8', header=False)
print('Output CSV File Saved')

#%%