"""
LINE 貼圖合集圖片分割處理器
==========================
此應用程式可自動從貼圖合集大圖中分割出每個獨立貼圖，
並處理成符合 LINE 規範的格式 (W370 x H320 px, PNG 透明背景)。

支援兩種分割模式：
1. 網格分割模式（推薦）：指定欄數和列數，平均分割圖片
2. 自動偵測模式：使用 AI 去背 + 輪廓偵測
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

# LINE 主要圖片與標籤圖片尺寸
LINE_MAIN_WIDTH = 240          # 主要圖片寬度
LINE_MAIN_HEIGHT = 240         # 主要圖片高度
LINE_TAB_WIDTH = 96            # 聊天室標籤圖片寬度
LINE_TAB_HEIGHT = 74           # 聊天室標籤圖片高度


# ============================================================
# 核心處理函式
# ============================================================

def grid_split(image: Image.Image, cols: int, rows: int) -> List[Image.Image]:
    """
    使用網格方式分割圖片。
    
    Args:
        image: 原始圖片
        cols: 欄數
        rows: 列數
        
    Returns:
        分割後的子圖像列表（由左到右、由上到下排序）
    """
    img_width, img_height = image.size
    cell_width = img_width // cols
    cell_height = img_height // rows
    
    cropped_images = []
    for row in range(rows):
        for col in range(cols):
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            
            cropped = image.crop((x1, y1, x2, y2))
            cropped_images.append(cropped)
    
    return cropped_images


def remove_background_full(image: Image.Image) -> Image.Image:
    """
    對整張圖片執行 rembg 去背處理。
    """
    return remove(image)


def find_sticker_contours(image_rgba: Image.Image, 
                          dilation_size: int = 20,
                          min_area_percent: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """
    使用形態學操作找出貼圖邊界框（自動偵測模式用）。
    """
    img_array = np.array(image_rgba)
    img_height, img_width = img_array.shape[:2]
    total_area = img_height * img_width
    min_area = int(total_area * min_area_percent / 100)
    
    alpha_channel = img_array[:, :, 3]
    blurred = cv2.GaussianBlur(alpha_channel, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 999
            if aspect_ratio < 10:
                bounding_boxes.append((x, y, w, h))
    
    if bounding_boxes:
        avg_height = sum(box[3] for box in bounding_boxes) / len(bounding_boxes)
        row_threshold = avg_height * 0.5
        bounding_boxes.sort(key=lambda box: (box[1] // int(row_threshold) if row_threshold > 0 else box[1], box[0]))
    
    return bounding_boxes


def crop_stickers_by_boxes(original_image: Image.Image, 
                           bounding_boxes: List[Tuple[int, int, int, int]],
                           padding: int = 10) -> List[Image.Image]:
    """
    根據邊界框裁剪圖片。
    """
    cropped_images = []
    for x, y, w, h in bounding_boxes:
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(original_image.width, x + w + padding)
        y2 = min(original_image.height, y + h + padding)
        cropped = original_image.crop((x1, y1, x2, y2))
        cropped_images.append(cropped)
    return cropped_images


def process_single_sticker(image: Image.Image, apply_rembg: bool = True) -> Image.Image:
    """
    處理單張貼圖：去背 + 縮放 + 置中。
    
    Args:
        image: 裁剪後的子圖像
        apply_rembg: 是否執行 rembg 去背
        
    Returns:
        處理完成的 LINE 規格貼圖
    """
    if apply_rembg:
        image_nobg = remove(image)
    else:
        image_nobg = image.convert('RGBA')
    
    canvas_width = LINE_STICKER_MAX_WIDTH
    canvas_height = LINE_STICKER_MAX_HEIGHT
    usable_width = canvas_width - (STICKER_MARGIN * 2)
    usable_height = canvas_height - (STICKER_MARGIN * 2)
    
    img_width, img_height = image_nobg.size
    if img_width == 0 or img_height == 0:
        return Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    
    scale = min(usable_width / img_width, usable_height / img_height)
    new_width = max(1, int(img_width * scale))
    new_height = max(1, int(img_height * scale))
    
    resized = image_nobg.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    paste_x = (canvas_width - new_width) // 2
    paste_y = (canvas_height - new_height) // 2
    canvas.paste(resized, (paste_x, paste_y), resized)
    
    return canvas


def create_zip_download(stickers: List[Image.Image]) -> bytes:
    """
    將所有貼圖打包成 ZIP 檔案。
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


def resize_to_main(image: Image.Image, apply_rembg: bool = True) -> Image.Image:
    """
    將圖片調整為主要圖片尺寸 (240 x 240)。
    圖片會等比例縮放並置中於畫布。
    
    Args:
        image: 原始圖片
        apply_rembg: 是否執行 rembg 去背
        
    Returns:
        調整後的主要圖片
    """
    if apply_rembg:
        image_nobg = remove(image)
    else:
        image_nobg = image.convert('RGBA')
    
    canvas_width = LINE_MAIN_WIDTH
    canvas_height = LINE_MAIN_HEIGHT
    margin = 5  # 主要圖片邊距較小
    usable_width = canvas_width - (margin * 2)
    usable_height = canvas_height - (margin * 2)
    
    img_width, img_height = image_nobg.size
    if img_width == 0 or img_height == 0:
        return Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    
    scale = min(usable_width / img_width, usable_height / img_height)
    new_width = max(1, int(img_width * scale))
    new_height = max(1, int(img_height * scale))
    
    resized = image_nobg.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    paste_x = (canvas_width - new_width) // 2
    paste_y = (canvas_height - new_height) // 2
    canvas.paste(resized, (paste_x, paste_y), resized)
    
    return canvas


def resize_to_tab(image: Image.Image, apply_rembg: bool = True) -> Image.Image:
    """
    將圖片調整為聊天室標籤圖片尺寸 (96 x 74)。
    圖片會等比例縮放並置中於畫布。
    
    Args:
        image: 原始圖片
        apply_rembg: 是否執行 rembg 去背
        
    Returns:
        調整後的聊天室標籤圖片
    """
    if apply_rembg:
        image_nobg = remove(image)
    else:
        image_nobg = image.convert('RGBA')
    
    canvas_width = LINE_TAB_WIDTH
    canvas_height = LINE_TAB_HEIGHT
    margin = 3  # 標籤圖片邊距更小
    usable_width = canvas_width - (margin * 2)
    usable_height = canvas_height - (margin * 2)
    
    img_width, img_height = image_nobg.size
    if img_width == 0 or img_height == 0:
        return Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    
    scale = min(usable_width / img_width, usable_height / img_height)
    new_width = max(1, int(img_width * scale))
    new_height = max(1, int(img_height * scale))
    
    resized = image_nobg.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    paste_x = (canvas_width - new_width) // 2
    paste_y = (canvas_height - new_height) // 2
    canvas.paste(resized, (paste_x, paste_y), resized)
    
    return canvas


# ============================================================
# Streamlit UI
# ============================================================

def main():
    """主程式進入點"""
    
    st.set_page_config(
        page_title="LINE 貼圖處理器",
        page_icon="✂️",
        layout="wide"
    )
    
    st.title("✂️ LINE 貼圖處理器")
    st.markdown("""
    上傳圖片，自動處理成符合 LINE 規範的格式。
    """)
    
    # 使用 tabs 分隔不同功能
    tab1, tab2 = st.tabs(["📐 貼圖分割", "🖼️ 主要圖片/標籤圖片"])
    
    # ========================================
    # Tab 1: 貼圖分割功能（原有功能）
    # ========================================
    with tab1:
        st.subheader("貼圖合集分割處理")
        st.caption("將貼圖合集大圖分割成單張貼圖 (370 x 320 px)")
        
        st.divider()
        
        # 檔案上傳
        uploaded_file = st.file_uploader(
            "上傳貼圖合集圖片",
            type=['png', 'jpg', 'jpeg'],
            help="支援 PNG、JPG 格式的貼圖合集圖片",
            key="sticker_uploader"
        )
        
        if uploaded_file is not None:
            original_image = Image.open(uploaded_file).convert('RGBA')
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("📷 原始圖片")
                st.image(original_image, use_container_width=True)
                st.caption(f"尺寸: {original_image.width} x {original_image.height} px")
            
            with col2:
                st.subheader("⚙️ 分割設定")
                
                # 選擇分割模式
                split_mode = st.radio(
                    "選擇分割模式",
                    ["📐 網格分割（推薦）", "🔍 自動偵測"],
                    help="網格分割適用於整齊排列的貼圖；自動偵測適用於不規則排列"
                )
                
                if "網格分割" in split_mode:
                    st.info("💡 請輸入貼圖的排列方式（欄數 × 列數）")
                    
                    grid_col1, grid_col2 = st.columns(2)
                    with grid_col1:
                        cols = st.number_input("欄數（橫向）", min_value=1, max_value=20, value=4)
                    with grid_col2:
                        rows = st.number_input("列數（縱向）", min_value=1, max_value=20, value=7)
                    
                    total_stickers = cols * rows
                    st.success(f"預計分割出 **{total_stickers}** 張貼圖")
                    
                    apply_rembg = st.checkbox("對每張貼圖執行 AI 去背", value=True, 
                                              help="勾選後會使用 rembg 移除每張貼圖的背景")
                    
                    if st.button("🚀 開始處理", type="primary", use_container_width=True, key="grid_btn"):
                        process_grid_mode(original_image, cols, rows, apply_rembg)
                
                else:
                    with st.expander("進階參數調整", expanded=False):
                        dilation_size = st.slider("膨脹核心大小", 5, 50, 20, 5)
                        min_area_percent = st.slider("最小面積百分比 (%)", 0.1, 5.0, 0.5, 0.1)
                    
                    if st.button("🚀 開始處理", type="primary", use_container_width=True, key="auto_btn"):
                        process_auto_mode(original_image, dilation_size, min_area_percent)
    
    # ========================================
    # Tab 2: 主要圖片/標籤圖片轉換功能（新功能）
    # ========================================
    with tab2:
        st.subheader("主要圖片/聊天室標籤圖片轉換")
        st.caption("將圖片調整為 LINE 貼圖所需的主要圖片 (main) 或聊天室標籤圖片 (tab) 尺寸")
        
        st.divider()
        
        # 顯示尺寸說明
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.info("🖼️ **主要圖片 (main)**\n\n尺寸：240 x 240 px")
        with info_col2:
            st.info("💬 **聊天室標籤圖片 (tab)**\n\n尺寸：96 x 74 px")
        
        st.divider()
        
        # 檔案上傳
        uploaded_icon = st.file_uploader(
            "上傳要轉換的圖片",
            type=['png', 'jpg', 'jpeg'],
            help="支援 PNG、JPG 格式的圖片",
            key="icon_uploader"
        )
        
        if uploaded_icon is not None:
            icon_image = Image.open(uploaded_icon).convert('RGBA')
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("📷 原始圖片")
                st.image(icon_image, use_container_width=True)
                st.caption(f"尺寸: {icon_image.width} x {icon_image.height} px")
            
            with col2:
                st.subheader("⚙️ 轉換設定")
                
                # 選擇輸出類型
                output_type = st.radio(
                    "選擇輸出類型",
                    ["🖼️ 主要圖片 (240 x 240)", "💬 聊天室標籤圖片 (96 x 74)", "📦 兩種都輸出"],
                    help="選擇要轉換的圖片類型"
                )
                
                apply_rembg_icon = st.checkbox("執行 AI 去背", value=True, 
                                               help="勾選後會使用 rembg 移除圖片背景",
                                               key="icon_rembg")
                
                if st.button("🚀 開始轉換", type="primary", use_container_width=True, key="icon_btn"):
                    process_icon_conversion(icon_image, output_type, apply_rembg_icon)


def process_icon_conversion(image: Image.Image, output_type: str, apply_rembg: bool):
    """
    處理主要圖片/標籤圖片轉換。
    """
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        
        if "主要圖片" in output_type or "兩種都輸出" in output_type:
            status_text.text("⏳ 轉換主要圖片 (240 x 240)...")
            progress_bar.progress(30)
            main_image = resize_to_main(image, apply_rembg)
            results['main'] = main_image
        
        if "聊天室標籤" in output_type or "兩種都輸出" in output_type:
            status_text.text("⏳ 轉換聊天室標籤圖片 (96 x 74)...")
            progress_bar.progress(60)
            tab_image = resize_to_tab(image, apply_rembg)
            results['tab'] = tab_image
        
        progress_bar.progress(100)
        status_text.text("✅ 轉換完成！")
    
    # 顯示結果
    st.divider()
    st.subheader("🎉 轉換結果")
    
    result_cols = st.columns(len(results))
    
    for idx, (key, img) in enumerate(results.items()):
        with result_cols[idx]:
            if key == 'main':
                st.markdown("**🖼️ 主要圖片 (main.png)**")
                st.caption(f"尺寸: {LINE_MAIN_WIDTH} x {LINE_MAIN_HEIGHT} px")
            else:
                st.markdown("**💬 聊天室標籤圖片 (tab.png)**")
                st.caption(f"尺寸: {LINE_TAB_WIDTH} x {LINE_TAB_HEIGHT} px")
            
            st.image(img, use_container_width=True)
            
            # 下載按鈕
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            st.download_button(
                label=f"📥 下載 {key}.png",
                data=img_buffer.getvalue(),
                file_name=f"{key}.png",
                mime="image/png",
                use_container_width=True
            )


def process_grid_mode(original_image: Image.Image, cols: int, rows: int, apply_rembg: bool):
    """
    網格分割模式處理流程。
    """
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 步驟 1: 網格分割
        status_text.text("⏳ 步驟 1/2: 按網格分割圖片...")
        progress_bar.progress(10)
        
        cropped_images = grid_split(original_image, cols, rows)
        progress_bar.progress(20)
        
        st.success(f"✅ 已分割出 **{len(cropped_images)}** 個區塊")
        
        # 步驟 2: 處理每張貼圖
        status_text.text("⏳ 步驟 2/2: 處理每張貼圖...")
        processed_stickers = []
        
        for i, cropped in enumerate(cropped_images):
            try:
                processed = process_single_sticker(cropped, apply_rembg)
                processed_stickers.append(processed)
                progress = 20 + int((i + 1) / len(cropped_images) * 75)
                progress_bar.progress(progress)
                status_text.text(f"⏳ 步驟 2/2: 處理第 {i + 1}/{len(cropped_images)} 張貼圖...")
            except Exception as e:
                st.warning(f"⚠️ 第 {i + 1} 張貼圖處理失敗: {str(e)}")
        
        progress_bar.progress(100)
        status_text.text("✅ 處理完成！")
    
    display_results(processed_stickers)


def process_auto_mode(original_image: Image.Image, dilation_size: int, min_area_percent: float):
    """
    自動偵測模式處理流程。
    """
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 步驟 1: 去背
        status_text.text("⏳ 步驟 1/3: 對大圖進行 AI 去背處理...")
        progress_bar.progress(10)
        
        try:
            image_nobg = remove_background_full(original_image)
        except Exception as e:
            st.error(f"❌ 去背處理失敗: {str(e)}")
            return
        
        progress_bar.progress(30)
        
        # 步驟 2: 找輪廓
        status_text.text("⏳ 步驟 2/3: 尋找貼圖輪廓...")
        bounding_boxes = find_sticker_contours(image_nobg, dilation_size, min_area_percent)
        progress_bar.progress(40)
        
        if len(bounding_boxes) == 0:
            st.error("❌ 無法偵測到任何貼圖！建議改用「網格分割」模式。")
            return
        
        st.success(f"✅ 偵測到 **{len(bounding_boxes)}** 個貼圖區域")
        
        # 步驟 3: 處理每張貼圖
        cropped_images = crop_stickers_by_boxes(original_image, bounding_boxes)
        status_text.text("⏳ 步驟 3/3: 處理每張貼圖...")
        processed_stickers = []
        
        for i, cropped in enumerate(cropped_images):
            try:
                processed = process_single_sticker(cropped, apply_rembg=True)
                processed_stickers.append(processed)
                progress = 40 + int((i + 1) / len(cropped_images) * 55)
                progress_bar.progress(progress)
                status_text.text(f"⏳ 步驟 3/3: 處理第 {i + 1}/{len(cropped_images)} 張貼圖...")
            except Exception as e:
                st.warning(f"⚠️ 第 {i + 1} 張貼圖處理失敗: {str(e)}")
        
        progress_bar.progress(100)
        status_text.text("✅ 處理完成！")
    
    display_results(processed_stickers)


def display_results(processed_stickers: List[Image.Image]):
    """
    顯示處理結果與下載按鈕。
    """
    st.divider()
    st.subheader(f"🎉 處理結果：共 {len(processed_stickers)} 張貼圖")
    
    if processed_stickers:
        cols_per_row = 5
        for row_start in range(0, len(processed_stickers), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                sticker_idx = row_start + col_idx
                if sticker_idx < len(processed_stickers):
                    with cols[col_idx]:
                        st.image(processed_stickers[sticker_idx], caption=f"sticker_{sticker_idx + 1:02d}.png")
        
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
