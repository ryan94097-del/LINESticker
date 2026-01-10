"""
LINE 貼圖合集圖片分割處理器
==========================
此應用程式可自動從貼圖合集大圖中分割出每個獨立貼圖，
並處理成符合 LINE 規範的格式 (W370 x H320 px, PNG 透明背景)。

處理流程：
1. 上傳合集大圖
2. 使用 rembg 對整張圖去背
3. 使用形態學膨脹將相近區域連接
4. 使用 RETR_EXTERNAL 找最外層輪廓
5. 裁剪並個別處理每個貼圖
6. 打包成 ZIP 下載
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io
import zipfile
from typing import List, Tuple

# ============================================================
# 常數設定
# ============================================================
LINE_STICKER_MAX_WIDTH = 370   # LINE 貼圖最大寬度
LINE_STICKER_MAX_HEIGHT = 320  # LINE 貼圖最大高度
STICKER_MARGIN = 10            # 貼圖四周透明邊距


# ============================================================
# 核心處理函式
# ============================================================

def remove_background_full(image: Image.Image) -> Image.Image:
    """
    對整張圖片執行 rembg 去背處理。
    
    Args:
        image: PIL Image 物件
        
    Returns:
        去背後的 PIL Image (RGBA 格式)
    """
    return remove(image)


def find_sticker_contours(image_rgba: Image.Image, 
                          dilation_size: int = 15,
                          min_area_percent: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """
    使用增強的形態學操作找出貼圖邊界框。
    
    核心邏輯：
    1. 從 Alpha 通道取得前景 Mask
    2. 高斯模糊去除噪點
    3. 形態學膨脹將相近區域（角色+配件）連接成一體
    4. 使用 RETR_EXTERNAL 只抓最外層輪廓
    5. 依面積過濾雜訊
    
    Args:
        image_rgba: 已去背的 RGBA 圖片
        dilation_size: 膨脹核心大小（越大越能連接遠距離物件）
        min_area_percent: 最小面積百分比（相對於圖片總面積）
        
    Returns:
        邊界框列表 [(x, y, w, h), ...]，已按位置排序
    """
    # 轉換為 numpy 陣列
    img_array = np.array(image_rgba)
    img_height, img_width = img_array.shape[:2]
    total_area = img_height * img_width
    
    # 計算最小輪廓面積閾值
    min_area = int(total_area * min_area_percent / 100)
    
    # 取得 Alpha 通道作為前景 Mask
    alpha_channel = img_array[:, :, 3]
    
    # 步驟 1: 高斯模糊去除噪點
    blurred = cv2.GaussianBlur(alpha_channel, (5, 5), 0)
    
    # 步驟 2: 二值化
    _, binary = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    
    # 步驟 3: 形態學膨脹 - 將相近區域連接成一體
    # 使用較大的核心將角色與其配件（驚嘆號、文字等）黏在一起
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)
    
    # 步驟 4: 閉運算填補內部空隙
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    
    # 步驟 5: 使用 RETR_EXTERNAL 只找最外層輪廓（忽略內部細節如眼睛）
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 步驟 6: 過濾雜訊 - 只保留面積足夠大的輪廓
    bounding_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            # 確保邊界框有合理的長寬比（過濾掉太細長的線條）
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 999
            if aspect_ratio < 10:  # 長寬比不超過 10:1
                bounding_boxes.append((x, y, w, h))
    
    # 步驟 7: 由上而下、由左而右排序
    # 使用動態的行高閾值進行分組排序
    if bounding_boxes:
        # 計算平均貼圖高度作為行高基準
        avg_height = sum(box[3] for box in bounding_boxes) / len(bounding_boxes)
        row_threshold = avg_height * 0.5  # 高度差在平均高度 50% 內視為同一行
        
        # 排序：先按 Y 座標分組（同一行），再按 X 座標排序
        bounding_boxes.sort(key=lambda box: (box[1] // int(row_threshold) if row_threshold > 0 else box[1], box[0]))
    
    return bounding_boxes


def crop_stickers(original_image: Image.Image, 
                  bounding_boxes: List[Tuple[int, int, int, int]],
                  padding: int = 10) -> List[Image.Image]:
    """
    根據邊界框從原始圖片裁剪出子圖像。
    
    Args:
        original_image: 原始上傳的圖片
        bounding_boxes: 邊界框列表
        padding: 裁剪時額外的邊距
        
    Returns:
        裁剪後的子圖像列表
    """
    cropped_images = []
    for x, y, w, h in bounding_boxes:
        # 裁剪時稍微擴大範圍，避免邊緣被切掉
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(original_image.width, x + w + padding)
        y2 = min(original_image.height, y + h + padding)
        
        cropped = original_image.crop((x1, y1, x2, y2))
        cropped_images.append(cropped)
    
    return cropped_images


def process_single_sticker(image: Image.Image) -> Image.Image:
    """
    處理單張貼圖：去背 + 縮放 + 置中。
    
    Args:
        image: 裁剪後的子圖像
        
    Returns:
        處理完成的 LINE 規格貼圖
    """
    # 再次執行 rembg 確保邊緣乾淨
    image_nobg = remove(image)
    
    # 計算可用的畫布尺寸（扣除邊距）
    canvas_width = LINE_STICKER_MAX_WIDTH
    canvas_height = LINE_STICKER_MAX_HEIGHT
    usable_width = canvas_width - (STICKER_MARGIN * 2)
    usable_height = canvas_height - (STICKER_MARGIN * 2)
    
    # 等比例縮放以 fit 進可用區域
    img_width, img_height = image_nobg.size
    if img_width == 0 or img_height == 0:
        return Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    
    scale = min(usable_width / img_width, usable_height / img_height)
    
    new_width = max(1, int(img_width * scale))
    new_height = max(1, int(img_height * scale))
    
    # 使用高品質縮放
    resized = image_nobg.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 建立透明畫布並置中貼上
    canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    paste_x = (canvas_width - new_width) // 2
    paste_y = (canvas_height - new_height) // 2
    canvas.paste(resized, (paste_x, paste_y), resized)
    
    return canvas


def create_zip_download(stickers: List[Image.Image]) -> bytes:
    """
    將所有貼圖打包成 ZIP 檔案。
    
    Args:
        stickers: 處理完成的貼圖列表
        
    Returns:
        ZIP 檔案的 bytes
    """
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, sticker in enumerate(stickers, 1):
            img_buffer = io.BytesIO()
            sticker.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            zip_file.writestr(f'sticker_{i:02d}.png', img_buffer.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ============================================================
# Streamlit UI
# ============================================================

def main():
    """主程式進入點"""
    
    # 頁面設定
    st.set_page_config(
        page_title="LINE 貼圖分割處理器",
        page_icon="✂️",
        layout="wide"
    )
    
    # 標題與說明
    st.title("✂️ LINE 貼圖合集分割處理器")
    st.markdown("""
    上傳一張貼圖合集大圖，自動分割並處理成符合 LINE 規範的格式。
    
    **功能特色：**
    - 🔍 自動識別並分割每個獨立貼圖（包含角色與配件）
    - 🎨 AI 智慧去背 (使用 rembg)
    - 📐 自動調整為 LINE 規格 (370 x 320 px)
    - 📦 一鍵打包下載 ZIP
    """)
    
    st.divider()
    
    # 檔案上傳
    uploaded_file = st.file_uploader(
        "上傳貼圖合集圖片",
        type=['png', 'jpg', 'jpeg'],
        help="支援 PNG、JPG 格式的貼圖合集圖片"
    )
    
    if uploaded_file is not None:
        # 載入圖片
        original_image = Image.open(uploaded_file).convert('RGBA')
        
        # 顯示原始圖片
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("📷 原始圖片")
            st.image(original_image, use_container_width=True)
            st.caption(f"尺寸: {original_image.width} x {original_image.height} px")
        
        # 處理設定
        with col2:
            st.subheader("⚙️ 處理設定")
            
            with st.expander("進階參數調整", expanded=False):
                dilation_size = st.slider(
                    "膨脹核心大小",
                    min_value=5,
                    max_value=50,
                    value=20,
                    step=5,
                    help="越大越能將角色與配件連接在一起。如果貼圖被分割成多個部分，請增大此值。"
                )
                
                min_area_percent = st.slider(
                    "最小面積百分比 (%)",
                    min_value=0.1,
                    max_value=5.0,
                    value=0.5,
                    step=0.1,
                    help="小於此比例的區域會被視為雜訊而忽略。如果偵測到太多小碎片，請增大此值。"
                )
            
            if st.button("🚀 開始處理", type="primary", use_container_width=True):
                process_stickers(original_image, dilation_size, min_area_percent)


def process_stickers(original_image: Image.Image, dilation_size: int, min_area_percent: float):
    """
    執行貼圖分割與處理的主要流程。
    
    Args:
        original_image: 原始上傳的圖片
        dilation_size: 膨脹核心大小
        min_area_percent: 最小面積百分比
    """
    
    # 建立進度容器
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 步驟 1: 整張圖去背
        status_text.text("⏳ 步驟 1/4: 對大圖進行 AI 去背處理（這可能需要一些時間）...")
        progress_bar.progress(10)
        
        try:
            image_nobg = remove_background_full(original_image)
        except Exception as e:
            st.error(f"❌ 去背處理失敗: {str(e)}")
            return
        
        progress_bar.progress(30)
        
        # 步驟 2: 找出輪廓
        status_text.text("⏳ 步驟 2/4: 使用形態學膨脹連接相近區域，尋找貼圖輪廓...")
        bounding_boxes = find_sticker_contours(image_nobg, dilation_size, min_area_percent)
        progress_bar.progress(40)
        
        if len(bounding_boxes) == 0:
            st.error("❌ 無法偵測到任何貼圖！請嘗試調整進階參數（減少最小面積百分比或調整膨脹核心大小）。")
            return
        
        st.success(f"✅ 成功偵測到 **{len(bounding_boxes)}** 個貼圖區域")
        
        # 步驟 3: 裁剪子圖像
        status_text.text("⏳ 步驟 3/4: 裁剪子圖像...")
        cropped_images = crop_stickers(original_image, bounding_boxes)
        progress_bar.progress(50)
        
        # 步驟 4: 個別處理每張貼圖
        status_text.text("⏳ 步驟 4/4: 處理每張貼圖（去背 + 縮放）...")
        processed_stickers = []
        
        for i, cropped in enumerate(cropped_images):
            try:
                processed = process_single_sticker(cropped)
                processed_stickers.append(processed)
                # 更新進度
                progress = 50 + int((i + 1) / len(cropped_images) * 45)
                progress_bar.progress(progress)
                status_text.text(f"⏳ 步驟 4/4: 處理第 {i + 1}/{len(cropped_images)} 張貼圖...")
            except Exception as e:
                st.warning(f"⚠️ 第 {i + 1} 張貼圖處理失敗: {str(e)}")
        
        progress_bar.progress(100)
        status_text.text("✅ 處理完成！")
    
    # 顯示結果
    st.divider()
    st.subheader(f"🎉 處理結果：共 {len(processed_stickers)} 張貼圖")
    
    # 網格顯示預覽
    if processed_stickers:
        # 每行顯示 5 張
        cols_per_row = 5
        for row_start in range(0, len(processed_stickers), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, sticker_idx in enumerate(range(row_start, min(row_start + cols_per_row, len(processed_stickers)))):
                with cols[col_idx]:
                    st.image(processed_stickers[sticker_idx], caption=f"sticker_{sticker_idx + 1:02d}.png")
        
        # ZIP 下載按鈕
        st.divider()
        zip_data = create_zip_download(processed_stickers)
        
        st.download_button(
            label="📦 下載所有貼圖 (ZIP)",
            data=zip_data,
            file_name="line_stickers.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True
        )
        
        st.info(f"📐 所有貼圖尺寸: {LINE_STICKER_MAX_WIDTH} x {LINE_STICKER_MAX_HEIGHT} px (PNG 格式)")


if __name__ == "__main__":
    main()
