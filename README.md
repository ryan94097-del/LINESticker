# LINE 貼圖分割處理器

自動從貼圖合集大圖中分割出獨立貼圖，並處理成 LINE 規格。

## 功能

- 🔍 自動識別並分割每個獨立貼圖
- 🎨 AI 智慧去背 (使用 rembg)
- 📐 自動調整為 LINE 規格 (370 x 320 px)
- 📦 一鍵打包下載 ZIP

## 部署到 Streamlit Cloud

### 步驟 1: 上傳至 GitHub

1. 在 GitHub 建立新儲存庫 (例如: `line-sticker-splitter`)
2. 上傳以下檔案:
   - `app.py`
   - `requirements.txt`

### 步驟 2: 部署

1. 前往 [Streamlit Cloud](https://streamlit.io/cloud)
2. 使用 GitHub 帳號登入
3. 點擊 **New app**
4. 選擇你的儲存庫和 `app.py`
5. 點擊 **Deploy**

> ⚠️ 首次部署可能需要 5-10 分鐘安裝依賴套件

## 使用方式

1. 上傳包含多個貼圖的合集圖片 (JPG/PNG)
2. 調整「最小輪廓面積」過濾雜訊
3. 點擊「開始處理」
4. 等待 AI 去背和分割完成
5. 預覽結果並下載 ZIP

## 技術規格

- **輸出尺寸**: 370 x 320 px (LINE 貼圖最大規格)
- **輸出格式**: PNG (透明背景)
- **邊距**: 約 10px 透明邊框
