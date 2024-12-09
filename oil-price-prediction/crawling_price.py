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
driver.get('https://www.investing.com/commodities/crude-oil-historical-data')


# time.sleep(40)
# viewport_width = driver.execute_script("return window.innerWidth")
# viewport_height = driver.execute_script("return window.innerHeight")
# # 關閉彈出式的登入頁面(隨便點一個地方)
# center_width = viewport_width // 2
# center_height = viewport_height - 50
# action = ActionChains(driver)
# action.move_by_offset(center_width, center_height).click().perform()




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
target_element = driver.find_element(By.XPATH, "//div[contains(@class, 'shadow-select') and contains(@class, 'gap-3.5')]")
driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", target_element)
print("已加載整個頁面")
time.sleep(2)


# 點開時間
try:
    target_element.click()
except Exception as e:
    print(f"click date failed: {e}")
    driver.quit()
    exit()



# 更改開始時間
try:
    # 設置目標日期
    date_input = "2020-02-01"
    
    # 定位 <input> 元素
    price_date = driver.find_elements(By.CSS_SELECTOR, ".absolute.left-0.top-0.h-full.w-full.opacity-0")
    driver.execute_script("arguments[0].value = arguments[1];", price_date[0], date_input)

    price_date[0].click()
    # 等待日曆彈出框生成
    calendar = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".calendar-class"))  # 替換為日曆框的實際選擇器
    )
    print("日曆彈出框已顯示")
    
    # 觸發 input 和 change 事件
    # driver.execute_script("""
    #     arguments[0].dispatchEvent(new Event('input', { bubbles: true }));
    #     arguments[0].dispatchEvent(new Event('change', { bubbles: true }));
    # """, price_date[0])

    print(price_date[0].get_attribute("value"))  # 驗證更新後的值

    # 定位 <span> 元素並同步更新
    # span_element = driver.find_element(By.CSS_SELECTOR, "span.whitespace-nowrap")
    # driver.execute_script("arguments[0].textContent = arguments[1];", span_element, "02/01/2020")
    # span_element = target_element.find_element(By.TAG_NAME, "div")
    # driver.execute_script("arguments[0].textContent = arguments[1];", span_element, f"02/01/2020 - {formatted_yesterday}")    
    # print(span_element.text)
    time.sleep(10)
    # 點擊更改時間按鈕
    # change_time_button = driver.find_element(By.CSS_SELECTOR, ".text-center.text-sm.font-bold.leading-5.text-white")
    # print(change_time_button.text)
    # change_time_button.click()
    # time.sleep(2)

except Exception as e:
    print(f"select date failed: {e}")
    driver.quit()
    exit()
    
    
    
time.sleep(30)
    
    

# try:
#     price_list = []

#     # 抓取所有價格的資料
#     rows = driver.find_elements(By.CSS_SELECTOR, '.grid_data tr')
#     for row in rows:
#         try:
#             # 滾動到該列頂部
#             driver.execute_script("arguments[0].scrollIntoView({block: 'start', inline: 'nearest'});", row)
#             time.sleep(1)  # 等待滾動穩定

#             # 點擊檢視按鈕
#             view_button = row.find_element(By.XPATH, ".//button[contains(text(), '檢視')]")
#             view_button.click()

#             # 等待彈窗顯示
#             WebDriverWait(driver, 10).until(
#                 EC.visibility_of_element_located((By.ID, "detail_div"))
#             )

#             # 抓取資料
#             data_table = driver.find_element(By.CLASS_NAME, "gray_table")
#             data_rows = data_table.find_elements(By.TAG_NAME, 'tr')
#             club = dict()
#             for dr in data_rows:
#                 key = dr.find_element(By.TAG_NAME, 'th').text
#                 value = dr.find_element(By.TAG_NAME, 'td').text if dr.find_elements(By.TAG_NAME, 'td') else ""
#                 if key:
#                     club[key] = value
#             club_list.append(club)
#         except Exception as e:
#             print(f"Data Table failed: {e}")
#             continue

#         # 嘗試模擬滑鼠點擊關閉按鈕
#         try:
#             close_button = driver.find_element(By.XPATH, "//div[@id='detail_div']//button[@class='close']")
#             actions = ActionChains(driver)
#             actions.move_to_element(close_button).click().perform()

#             # 確認彈窗是否關閉
#             WebDriverWait(driver, 10).until(
#                 lambda d: d.find_element(By.ID, 'detail_div').get_attribute("style") == "display: none;")
#         except Exception as e:
#             print(f"Close View failed with ActionChains: {e}")
#             continue

#         time.sleep(2)

#     # 寫入 JSON 檔案
#     with open(f'club_{category}.json', 'w', encoding='utf-8') as json_file:
#         json.dump(club_list, json_file, ensure_ascii=False, indent=4)

#     print(f"Successfully written to club_{category}.json")
# except Exception as e:
#     print(f"Crawl failed: {e}")

driver.quit()
