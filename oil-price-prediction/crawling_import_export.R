install.packages("eia")
eia_set_key("acHvs7QlaWuJJtyp9o9jGN0P8rUPsAclmyoa6aZQ")

# 初始化一個空的 DataFrame 來儲存所有結果
all_data <- data.frame()

# 設置分頁參數
limit <- 5000  # 每次請求返回的最大行數
offset <- 0    # 起始位置，第一次查詢從第 0 行開始

repeat {
    crude_oil_data <- eia_data(
    dir = "crude-oil-imports",  # 資料來源目錄
    data = "quantity",          # 要查詢的數據類型，這裡是數量
    facets = list(originType = "CTY"),  # 篩選條件：來源類型為國家級別
    freq = "monthly",           # 設定查詢的資料頻率為月度
    start = "2020-01",          # 設定查詢的開始日期
    end = "2024-09",            # 設定查詢的結束日期
    sort = list(cols = "period", order = "asc"),  # 按照時間升序排序
    offset = offset,            # 設定偏移量（分頁）
    )

    if (length(crude_oil_data) == 0) {
    break  # 若沒有返回數據，則退出循環
  }

  df <- as.data.frame(crude_oil_data)
  all_data <- rbind(all_data, df)
  
  offset <- offset + limit
}

# 檢查數據結構
str(all_data)

# 將數據轉換為 DataFrame
df <- as.data.frame(all_data)

# 存儲數據為 CSV 檔案
write.csv(df, "eia_crude_oil_imports.csv", row.names = FALSE)

# 顯示成功訊息
print("數據已成功存儲為 CSV 檔案！")
