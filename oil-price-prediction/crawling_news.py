from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
import time
import re
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

 # 設定 Chrome 瀏覽器選項（例如：模擬正常瀏覽器行為）
chrome_options = Options()
chrome_options.add_argument("--start-maximized")  # 啟動時最大化
chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # 隱藏自動化標識
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")
# 初始化瀏覽器驅動
driver = webdriver.Chrome(options=chrome_options)
print("瀏覽器已啟動！")
# lists
datas = []
first_time = True


def clear_text(text):
    #如果頭尾有”"“ 
    if text[0] == "\"" and text[-1] == "\"":
        text = text[1:-1]
    return text

def save_to_csv(data, filename):
    df = pd.DataFrame(data, columns=["title","date"])
    df.to_csv(filename, index=False, encoding="utf-8-sig", mode="a", header=not pd.io.common.file_exists(filename))

def parse_date(date_text):
    date_patterns = [
        r"\b\d{2} [A-Za-z]{4} \d{4}\b",     # 07 June 2023
        r"\b\d{2} [A-Za-z]{3} \d{4}\b",    # June 07, 2023
    ]
    for pattern in date_patterns:
        date_match = re.search(pattern, date_text)
        if date_match:
            date = date_match.group()  # 提取匹配到的日期字符串
            return date
    
    return date_text
def crawl(star_year, end_year):
    # Step 1: 訪問指定網址
    url = "https://www.proquest.com/advanced?accountid=12719"
    driver.get(url)
    time.sleep(10)  # 等待頁面加載
    
    # Step 2: 接受 Cookies（如有彈出窗口）
    try:
        accept_cookies_btn = driver.find_element(By.ID, "onetrust-accept-btn-handler")  # 假設有 Cookies 彈窗
        accept_cookies_btn.click()
        print("Cookies 接受成功！")
    except Exception as e:
        print("沒有發現 Cookies 彈窗或自動接受失敗！")

    # Step 3: 定位目標 <textarea> 框框
    textarea = driver.find_element(By.ID, "queryTermField")
    print("找到 textarea 元素！")

    # Step 4: 輸入文字 "war OR conflict OR oil"
    textarea.clear()  # 清空預設內容
    textarea.send_keys("title(war OR conflict OR oil)")
    print("輸入文字成功！")

    # Step 5: 按下按來源，選擇新聞，
    
   # 等待複選框元素出現
    wait = WebDriverWait(driver, 10)
    checkbox = wait.until(EC.presence_of_element_located((By.ID, "SourceType_Newspapers")))

    # 使用 JavaScript 點擊複選框
    driver.execute_script("arguments[0].click();", checkbox)
    print("成功勾選「報紙」選項！")

    # 確保元素可點擊並滾動到視野中
    driver.execute_script("arguments[0].scrollIntoView(true);", checkbox)

    # 勾選複選框
    if not checkbox.is_selected():
        checkbox.click()
        print("成功勾選「報紙」選項！")
    # 等待複選框元素出現
    wait = WebDriverWait(driver, 10)
    checkbox = wait.until(EC.presence_of_element_located((By.ID, "SourceType_Wire_Feeds")))
    # 使用 JavaScript 點擊複選框
    driver.execute_script("arguments[0].click();", checkbox)
    print("成功勾選「報紙」選項！")

    # 確保元素可點擊並滾動到視野中
    driver.execute_script("arguments[0].scrollIntoView(true);", checkbox)
    # 勾選複選框
    if not checkbox.is_selected():
        checkbox.click()
        print("成功勾選「電報」選項！")

    # Step 6: 選擇「特定日期範圍」選項（value="RANGE"）
    # 定位到下拉選單元素
    dropdown = driver.find_element("id", "select_multiDateRange")
    # 使用 Select 物件操作下拉選單
    select = Select(dropdown)
    select.select_by_value("RANGE")
    print("已選擇「特定日期範圍」選項！")
    
    date_text = driver.find_element(By.ID, "year2")     #year2是start year
    date_text.clear()
    date_text.send_keys(star_year)

    date_text = driver.find_element(By.ID, "year2_0")     #year2_0是end year
    date_text.clear()
    date_text.send_keys(end_year)
    print(f"輸入日期範圍{star_year} to {end_year}成功！")
    time.sleep(5)  # 等待頁面加載

    # step 7 選擇日期排序
    #關閉彈出視窗
    if(first_time):
        try:
            # 等待按鈕出現，但如果超時會拋出 TimeoutException
            close_btn = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "button._pendo-close-guide")))
            
            # 檢查按鈕是否可見並可交互
            if close_btn.is_displayed() and close_btn.is_enabled():
                close_btn.click()
                print("成功關閉彈出guide視窗！")
            else:
                print("按鈕不可見或不可交互。")

            more_option_btn = wait.until(EC.presence_of_element_located((By.ID, "showResultPageOptions")))
            more_option_btn.click()
            print("成功點擊「更多選項」按鈕！")
            
            # 使用 JavaScript 選擇下拉選項
            dropdown = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "select.form-control.float_left.form-control")))
            driver.execute_script("arguments[0].value = 'DateAsc'; arguments[0].dispatchEvent(new Event('change'))", dropdown)
            print("已成功選擇「日期升序」選項！")
        except TimeoutException:
            print("彈出guide視窗按鈕超時，視窗不存在。")
        except Exception as e:
            print(f"發生其他錯誤：{e}")

    # step 8:模擬按下 Enter 鍵來提交搜尋
    textarea.send_keys(Keys.RETURN)
    print("成功提交搜尋！")

##------------------------------------------進入資料頁面------------------------------------------
   
    
    # 開始爬取資料
    
    # 翻頁
    for page in range(1, 101):
        print(f"正在處理第 {page} 頁...")

        # 等待頁面元素加載
        # 等待標題元素加載完成
        wait = WebDriverWait(driver, 10)

        try:
            # 等待標題元素加載完成
            wait.until(EC.presence_of_element_located((By.XPATH, "//a[contains(@class, 'previewTitle')]")))
        
            # 抓取所有標題元素
            # 找到所有標題和日期元素
            title_elements = driver.find_elements(
                By.XPATH, 
                "//a[contains(@class, 'previewTitle Topicsresult ') and not(contains(@class, 'citationSourceTypeIconLink'))]"
            )

            date_elements = driver.find_elements(By.XPATH, "//span[contains(@class, 'newspaperArticle')]")

            # 確保標題和日期數量一致
            num_titles = len(title_elements)
            num_dates = len(date_elements)
            if num_titles != num_dates:
                print(f"標題數量 {num_titles} 與日期數量 {num_dates} 不一致！")
                break
            # 遍歷所有標題和日期
            for i in range(min(num_titles, num_dates)):
                text = title_elements[i].get_attribute('title')  # 獲取標題文本
                # url = title.get_attribute('href')   # 獲取標題鏈接
                date_text = date_elements[i].text
                date = parse_date(date_text)
                print(f"{text}, {date}")
                
                if(text):
                    text = clear_text(text)
                    datas.append([text,date])

            #strore the data to csv
            if datas:
                save_to_csv(datas,"titles.csv")
                print(f"第 {page}頁，標題已保存到 titles.csv!")
                datas.clear()

            # 找到“下一頁”按鈕並點擊
            next_btn = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//a[@title="下一頁"]'))
            )
            # 點擊「下一頁」按鈕
            next_btn.click()

        except Exception as e:
            print("無法點擊下一頁按鈕，翻頁結束")
            break

if __name__ == "__main__":
    #initailize .csv file
    df = pd.DataFrame(columns=["title","date"])
    df.to_csv("titles.csv", index=False, encoding="utf-8-sig")
    try:

        crawl("2020","2021")
        first_time = False
        crawl("2022","2024")
    
    except Exception as e:
        print(f"出現錯誤：{e}")

    finally:
        if datas:
            save_to_csv(datas, "titles.csv")
            print("標題已保存到 titles.csv!")
        else:
            print("運行結束，沒有剩餘標題數據可保存。")
        # 關閉瀏覽器
        driver.quit()
        print("瀏覽器已關閉！")
