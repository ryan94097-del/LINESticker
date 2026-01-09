"""
LINE 貼圖合集圖片分割處理器
==========================
此應用程式可自動從貼圖合集大圖中分割出每個獨立貼圖，
並處理成符合 LINE 規範的格式 (W370 x H320 px, PNG 透明背景)。

處理流程：
1. 上傳合集大圖
2. 使用 rembg 對整張圖去背
3. 分析 Alpha 通道找出每個貼圖的輪廓
4. 裁剪並個別處理每個貼圖
5. 打包成 ZIP 下載
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
MIN_CONTOUR_AREA = 1000        # 最小輪廓面積（過濾雜訊用）


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


def find_sticker_contours(image_rgba: Image.Image, min_area: int = MIN_CONTOUR_AREA) -> List[Tuple[int, int, int, int]]:
    """
    分析 Alpha 通道，找出所有非透明區域的邊界框。
    使用增強的形態學操作和邊界框合併來避免一個貼圖被分成多個部分。
    
    Args:
        image_rgba: 已去背的 RGBA 圖片
        min_area: 最小輪廓面積，小於此值視為雜訊
        
    Returns:
        邊界框列表 [(x, y, w, h), ...]
    """
    # 轉換為 numpy 陣列並取得 Alpha 通道
    img_array = np.array(image_rgba)
    alpha_channel = img_array[:, :, 3]
    
    # 二值化 Alpha 通道
    _, binary = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)
    
    # 增強形態學操作：使用較大的核心進行閉運算，連接相鄰區域
    # 根據圖片大小動態調整核心尺寸
    img_height, img_width = binary.shape
    kernel_size = max(15, min(img_width, img_height) // 50)  # 動態核心大小
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 先膨脹再侵蝕（閉運算），填補貼圖內部的空隙
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 額外的膨脹操作，確保相近的區域能連接在一起
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size // 2, kernel_size // 2))
    binary = cv2.dilate(binary, dilate_kernel, iterations=2)
    binary = cv2.erode(binary, dilate_kernel, iterations=2)
    
    # 找出輪廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 過濾並取得邊界框
    bounding_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))
    
    # 合併重疊或相近的邊界框
    bounding_boxes = merge_overlapping_boxes(bounding_boxes, img_width, img_height)
    
    # 按照位置排序（先上後下，先左後右）
    # 使用較大的行高閾值來正確分組
    row_height = img_height // 10 if img_height > 0 else 50
    bounding_boxes.sort(key=lambda box: (box[1] // row_height, box[0]))
    
    return bounding_boxes


def merge_overlapping_boxes(boxes: List[Tuple[int, int, int, int]], 
                            img_width: int, img_height: int) -> List[Tuple[int, int, int, int]]:
    """
    合併重疊或相近的邊界框。
    
    Args:
        boxes: 邊界框列表 [(x, y, w, h), ...]
        img_width: 圖片寬度
        img_height: 圖片高度
        
    Returns:
        合併後的邊界框列表
    """
    if not boxes:
        return boxes
    
    # 設定合併距離閾值（圖片較小邊的 5%）
    merge_threshold = max(20, min(img_width, img_height) // 20)
    
    merged = True
    while merged:
        merged = False
        new_boxes = []
        used = [False] * len(boxes)
        
        for i in range(len(boxes)):
            if used[i]:
                continue
                
            x1, y1, w1, h1 = boxes[i]
            # 擴大邊界框用於重疊檢測
            expanded_x1 = x1 - merge_threshold
            expanded_y1 = y1 - merge_threshold
            expanded_x2 = x1 + w1 + merge_threshold
            expanded_y2 = y1 + h1 + merge_threshold
            
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                    
                x2, y2, w2, h2 = boxes[j]
                
                # 檢查擴大後的邊界框是否重疊
                if (expanded_x1 < x2 + w2 and expanded_x2 > x2 and
                    expanded_y1 < y2 + h2 and expanded_y2 > y2):
                    # 合併兩個邊界框
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_x2 = max(x1 + w1, x2 + w2)
                    new_y2 = max(y1 + h1, y2 + h2)
                    x1, y1, w1, h1 = new_x, new_y, new_x2 - new_x, new_y2 - new_y
                    # 更新擴大範圍
                    expanded_x1 = x1 - merge_threshold
                    expanded_y1 = y1 - merge_threshold
                    expanded_x2 = x1 + w1 + merge_threshold
                    expanded_y2 = y1 + h1 + merge_threshold
                    used[j] = True
                    merged = True
            
            new_boxes.append((x1, y1, w1, h1))
            used[i] = True
        
        boxes = new_boxes
    
    return boxes


def crop_stickers(original_image: Image.Image, bounding_boxes: List[Tuple[int, int, int, int]]) -> List[Image.Image]:
    """
    根據邊界框從原始圖片裁剪出子圖像。
    
    Args:
        original_image: 原始上傳的圖片
        bounding_boxes: 邊界框列表
        
    Returns:
        裁剪後的子圖像列表
    """
    cropped_images = []
    for x, y, w, h in bounding_boxes:
        # 裁剪時稍微擴大範圍，避免邊緣被切掉
        padding = 5
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
    scale = min(usable_width / img_width, usable_height / img_height)
    
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    
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
    - 🔍 自動識別並分割每個獨立貼圖
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
        
        # 處理按鈕
        with col2:
            st.subheader("⚙️ 處理設定")
            
            min_area = st.slider(
                "最小輪廓面積（過濾雜訊）",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                help="小於此面積的區域會被視為雜訊而忽略"
            )
            
            if st.button("🚀 開始處理", type="primary", use_container_width=True):
                process_stickers(original_image, min_area)


def process_stickers(original_image: Image.Image, min_area: int):
    """
    執行貼圖分割與處理的主要流程。
    
    Args:
        original_image: 原始上傳的圖片
        min_area: 最小輪廓面積
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
        status_text.text("⏳ 步驟 2/4: 分析 Alpha 通道，尋找貼圖輪廓...")
        bounding_boxes = find_sticker_contours(image_nobg, min_area)
        progress_bar.progress(40)
        
        if len(bounding_boxes) == 0:
            st.error("❌ 無法偵測到任何貼圖！請確認圖片內容或調整最小輪廓面積設定。")
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
