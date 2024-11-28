
#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model

import numpy as np
import pandas as pd
import os

#設定LSTM往前看的筆數和預測筆數
LookBackNum = 12 #LSTM往前看的筆數
ForecastNum = 48 #預測筆數

#載入訓練資料
# DataName = os.getcwd()+'\ExampleTrainData(AVG)\AvgDATA_17.csv'
# SourceData = pd.read_csv(DataName, encoding='utf-8')
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
    # combined_df = Avg_temp_data
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

print(SourceData.head(n=20))
print(SourceData.tail(n=10))
print(SourceData.shape)
#選擇要留下來的資料欄位(發電量)
target = ['Power(mW)']
AllOutPut = SourceData[target].astype(float).values

X_train = []
y_train = []

#設定每i-12筆資料(X_train)就對應到第i筆資料(y_train)
for i in range(LookBackNum,len(AllOutPut)):
  if i % 60 == 0:
       i = i +12
  X_train.append(AllOutPut[i-LookBackNum:i, 0])
  y_train.append(AllOutPut[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping
X_train = np.reshape(X_train,(X_train.shape [0], X_train.shape [1], 1))

print(X_train[0:10])
print(y_train[0:10])


#%%
#============================建置&訓練模型============================
#建置LSTM模型

regressor = Sequential ()

regressor.add(LSTM(units = 128, return_sequences = True, input_shape = (X_train.shape[1], 1)))

regressor.add(LSTM(units = 64))

regressor.add(Dropout(0.2))

# output layer
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#開始訓練
regressor.fit(X_train, y_train, epochs = 100, batch_size = 128)

#保存模型
# from datetime import datetime
# NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
regressor.save('WheatherLSTM.h5')
print('Model Saved')











# %%
#============================預測數據============================

#載入模型
regressor = load_model('WheatherLSTM.h5')

#載入測試資料
# DataName = os.getcwd()+r'\short.csv'
DataName = os.getcwd()+r'\upload(no answer).csv'
SourceData = pd.read_csv(DataName, encoding='utf-8')
target = ['序號']
EXquestion = SourceData[target].values

inputs = []#存放參考資料
PredictOutput = [] #存放預測值
temp_PredictOutput = []

count = 0
while(count < len(EXquestion)):
  print('count : ',count)
  LocationCode = int(EXquestion[count])
  strLocationCode = str(LocationCode)[-2:]
  if LocationCode < 10 :
    strLocationCode = '0'+LocationCode
  DataName = os.getcwd()+r'\data_process\Train\IncompleteAvgDATA_'+ strLocationCode +'.csv'
  SourceData = pd.read_csv(DataName, encoding='utf-8')
  ReferTitle = SourceData[['Serial']].values
  ReferData = SourceData[['Power(mW)']].astype(float).values
  
  inputs = []#重置存放參考資料
  temp_PredictOutput = []

  #找到相同的一天，把12個資料都加進inputs
  for DaysCount in range(len(ReferTitle)):
    if(str(int(ReferTitle[DaysCount]))[:8] == str(int(EXquestion[count]))[:8]):
      inputs = np.append(inputs, ReferData[DaysCount])
         
  #用迴圈不斷使新的預測值塞入參考資料，並預測下一筆資料
  for i in range(ForecastNum) :

    #print(i)
    
    #將新的預測值加入參考資料(用自己的預測值往前看)
    if i > 0 :
      inputs = np.append(inputs, temp_PredictOutput[i-1])
    
    #切出新的參考資料12筆(往前看12筆)
    X_test = []
    X_test.append(inputs[0+i:LookBackNum+i])
    
    #Reshaping
    NewTest = np.array(X_test)
    NewTest = np.reshape(NewTest, (NewTest.shape[0], NewTest.shape[1], 1))
    predicted = regressor.predict(NewTest)
    temp_PredictOutput.append(round(predicted[0,0], 2))
  
  #每次預測都要預測48個，因此加48個會切到下一天
  #0~47,48~95,96~143...
  count += 48
  PredictOutput.extend(temp_PredictOutput)

#寫預測結果寫成新的CSV檔案
# 將陣列轉換為 DataFrame
df = pd.DataFrame(PredictOutput, columns=['答案'])

# 將 DataFrame 寫入 CSV 檔案
df_q = pd.read_csv('upload(no answer).csv')
output_df = pd.concat([df_q['序號'], df], axis=1)
output_df.to_csv('output.csv', index=False)
print('Output CSV File Saved')

# %%
