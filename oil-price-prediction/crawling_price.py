from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
import time
import json
from datetime import datetime, timedelta
import pandas as pd
# 獲取今天的日期
today = datetime.today()
# 計算昨天的日期
yesterday = today - timedelta(days=1)
# 格式化為月/日/年
formatted_yesterday = yesterday.strftime("%m/%d/%Y")
print(formatted_yesterday)

# 設定 Chrome 選項 (禁用彈出式廣告)
chrome_options = Options()
chrome_options.add_argument("--disable-popup-blocking")  # 禁用彈出式廣告
chrome_options.add_argument("--disable-notifications")   # 禁用通知
driver = webdriver.Chrome(options=chrome_options)
driver.set_window_size(1920, 1080)
driver.get('https://www2.moeaea.gov.tw/oil111/CrudeOil/Price#list_item2')



last_height = driver.execute_script("return document.body.scrollHeight")
time.sleep(2)
while True:
    # 滾動到頁面底部
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    # 計算新的頁面高度
    new_height = driver.execute_script("return document.body.scrollHeight")
    # 如果頁面高度沒有變化，說明加載完成
    if new_height == last_height:
        break
    last_height = new_height
    
    
# 滾動回目標位置（以 CSS Selector 為例）
target_element = driver.find_element(By.ID, "list_item2")
# driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", target_element)
driver.execute_script("""
    arguments[0].scrollIntoView(true);  // 滾動到元素頂部
    window.scrollBy(0, -100);  // 往上偏移 100 像素
""", target_element)
print("已加載整個頁面")
time.sleep(2)



# 設置目標日期
date_input = "2020/01/01"

# 定位 <input> 元素
time_button = driver.find_element(By.CLASS_NAME, "form-control")
time_button.click()

datetimepicker = driver.find_element(By.CLASS_NAME, "datetimepicker-days")
datetimepicker = datetimepicker.find_element(By.CLASS_NAME, "switch")
datetimepicker.click()
datetimepicker = driver.find_element(By.CLASS_NAME, "datetimepicker-months")
datetimepicker = datetimepicker.find_element(By.CLASS_NAME, "switch")
datetimepicker.click()

# year
year_start = driver.find_element(By.CLASS_NAME, "datetimepicker-years")
year_start = year_start.find_element(By.TAG_NAME, "tbody")
year_start = year_start.find_element(By.TAG_NAME, "tr")
year_start = year_start.find_elements(By.TAG_NAME, "span")
year_start[1].click()       # 2020年

# month
month_start = driver.find_element(By.CLASS_NAME, "datetimepicker-months")
month_start = month_start.find_element(By.TAG_NAME, "tbody")
month_start = month_start.find_element(By.TAG_NAME, "tr")
month_start = month_start.find_elements(By.TAG_NAME, "span")
print(month_start[0].text)
month_start[0].click()      # 1月

# day
day_start = driver.find_element(By.CLASS_NAME, "datetimepicker-days")
day_start = day_start.find_element(By.TAG_NAME, "tbody")
day_start = day_start.find_element(By.TAG_NAME, "tr")
day_start = day_start.find_elements(By.TAG_NAME, "td")
day_start[3].click()        # 1日

print("已選擇日期")
time.sleep(10)


# 開始爬資料
data_table = driver.find_element(By.ID, "datadetail")
header_row = data_table.find_element(By.CLASS_NAME, "tr_title")
header_row = [header.text for header in header_row.find_elements(By.TAG_NAME, "th") if header.text != ""]
data_dict = {header: [] for header in header_row}

for row in data_table.find_elements(By.TAG_NAME, "tr"):
    if row.get_attribute("class") == "tr_title":
        continue
    # 滾動到當前行 (row)
    driver.execute_script("arguments[0].scrollIntoView(true);", row)
    columns = [column.text for column in row.find_elements(By.TAG_NAME, "td") if column.text != ""]
    for header, value in zip(header_row, columns):
        data_dict[header].append(value)
        
data_df = pd.DataFrame(data_dict)
data_df.to_csv("price.csv", index=False)
driver.quit()
