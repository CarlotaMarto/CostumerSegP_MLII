import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import base64
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_csv_data(filename):
    return pd.read_csv(BASE_DIR / "datasets" / filename)
IMAGENS_DIR = BASE_DIR / "imagens"

def load_image_as_base64(path):
    if path.exists():
        with open(path, "rb") as img_file:
            data = base64.b64encode(img_file.read()).decode("utf-8")
            ext = path.suffix[1:].lower()
            if ext == "jpg":
                ext = "jpeg"
            return f"data:image/{ext};base64,{data}"
    return ""

BAR_URI = load_image_as_base64(IMAGENS_DIR / "bar.png")
PRODUTOS_TECH_URI = load_image_as_base64(IMAGENS_DIR / "produtos" / "image (1).png")
PRODUTOS_MEAT_URI = load_image_as_base64(IMAGENS_DIR / "produtos" / "image (2).png")
INITIAL_BG_URI = load_image_as_base64(IMAGENS_DIR / "initial.png")
SEG_IMG_URI = load_image_as_base64(IMAGENS_DIR / "customer_segmentation_imagem.png")
GROCERY_BASKET_URI = load_image_as_base64(IMAGENS_DIR / "grocery_basket.jpg")
CESTO_URI = load_image_as_base64(IMAGENS_DIR / "cesto.png")
PRODUCT_GRID_URI = load_image_as_base64(IMAGENS_DIR / "product_grid.jpg")

ICON_INTRO_URI = load_image_as_base64(IMAGENS_DIR / "icon_intro.png")
ICON_ANALYSIS_URI = load_image_as_base64(IMAGENS_DIR / "icon_analysis.png")
ICON_PREPROCESS_URI = load_image_as_base64(IMAGENS_DIR / "icon_preprocess.png")
ICON_GEOGRAPHY_URI = load_image_as_base64(IMAGENS_DIR / "icon_geography.png")
ICON_CLUSTERING_URI = load_image_as_base64(IMAGENS_DIR / "icon_clustering.png")
BRAND_ICON_URI = load_image_as_base64(IMAGENS_DIR / "brand_icon.png")

SYMBOL_URIS = [load_image_as_base64(IMAGENS_DIR / f"symbol_{i}.png") for i in range(1, 8)]
SLICES_URIS = [load_image_as_base64(IMAGENS_DIR / "slices_simbolos" / f"symbol_{i}.png") for i in range(1, 8)]
FAMILIES_URI = load_image_as_base64(IMAGENS_DIR / "families.png")
VEGETARIANS_URI = load_image_as_base64(IMAGENS_DIR / "vegetarians.png")
WELLNESS_URI = load_image_as_base64(IMAGENS_DIR / "wellness.png")
PROMOTERS_URI = load_image_as_base64(IMAGENS_DIR / "promoters.png")
TECHIES_URI = load_image_as_base64(IMAGENS_DIR / "techies.png")
LOYALISTS_URI = load_image_as_base64(IMAGENS_DIR / "loyalists.png")
ECONOMIZERS_URI = load_image_as_base64(IMAGENS_DIR / "economizers.png")
REGULARS_URI = load_image_as_base64(IMAGENS_DIR / "regulars.png")

st.set_page_config(page_title="Customer Segmentation Project", layout="wide", page_icon=str(Path(__file__).resolve().parent / "imagens" / "brand_icon.png"))

css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, p, span, div, h1, h2, h3, h4, h5, h6, li, a, label, button, input {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}
h1, h2, h3 {
    font-family: 'Playfair Display', Georgia, Cambria, "Times New Roman", Times, serif !important;
    font-weight: 700 !important;
}
html {
    background-color: #fafaf8 !important;
}
body {
    background-color: #fafaf8 !important;
    color: #1a1a1a;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMainContainer"], [data-testid="stAppViewBlockContainer"], main, .main, div.block-container {
    background: transparent !important;
    background-color: transparent !important;
    backdrop-filter: none !important;
    padding-bottom: 0rem !important;
}
section[role="main"] {
    padding-top: 0px !important;
}
header, header[role="banner"], div[data-testid="stToolbar"] {
    position: relative !important;
    width: 100% !important;
    z-index: 50 !important;
    background: rgba(255, 255, 255, 0.8) !important;
    backdrop-filter: blur(8px) !important;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05) !important;
    border-bottom: 1px solid rgba(0, 0, 0, 0.05) !important;
}
[data-testid="stSidebar"] {
    background: #ede8df !important;
    border-right: 1px solid rgba(0,0,0,0.06) !important;
}

.css-1d391kg {
    background-color: #f1f5f9;
}
.stButton>button {
    background-color: #000000 !important;
    color: white !important;
    border-radius: 999px !important;
    border: none !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    transition: background-color 0.2s ease, transform 0.1s ease !important;
}
.stButton>button:hover {
    background-color: #1a1a1a !important;
    transform: translateY(-1px);
}
.st-bb {
    border-radius: 12px;
}
.stSidebarNav label {
    font-size: 16px;
    font-weight: 700;
    line-height: 1.3;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.7);
}
div[role="radiogroup"] label {
    font-size: 16px !important;
    font-weight: 600 !important;
    line-height: 1.3;
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: keep-all;
    letter-spacing: -0.01em;
}
.stAppViewContainer, .main > div, div.block-container {
    max-width: 100% !important;
    width: 100% !important;
}
div.block-container, [data-testid="stAppViewBlockContainer"] {
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-top: 1rem !important;
    padding-bottom: 8rem !important;
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    max-width: 100% !important;
}
.stMarkdown, .stText, .css-1d391kg {
    color: #1a1a1a;
}
div.block-container p, div.block-container div[data-testid="stMarkdownContainer"] > p {
    text-align: justify !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] > p, div[role="radiogroup"] p {
    text-align: left !important;
}
div[data-testid="stImage"] img {
    border-radius: 16px;
}
div[data-testid="stImage"] {
    border: 1px solid rgba(0, 0, 0, 0.05);
    padding: 8px;
    border-radius: 20px;
    background: #ffffff;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.01);
}
[data-testid="stArrowVegaLiteChart"],
[data-testid="stVegaLiteChart"],
[data-testid="stPlotlyChart"],
.stVegaLiteChart,
.stPlotlyChart,
div[class*="stVegaLiteChart"],
div[class*="stPlotlyChart"],
iframe[title="plotly"] {
    border: 1px solid rgba(111, 79, 53, 0.15) !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    padding: 0px !important;
    margin-bottom: 16px !important;
}
[data-testid="stArrowVegaLiteChart"] svg,
[data-testid="stVegaLiteChart"] svg,
.stVegaLiteChart svg,
div[class*="stVegaLiteChart"] svg,
[data-testid="stArrowVegaLiteChart"] svg rect.background,
[data-testid="stVegaLiteChart"] svg rect.background,
.stVegaLiteChart svg rect.background,
div[class*="stVegaLiteChart"] svg rect.background,
[data-testid="stPlotlyChart"] .js-plotly-plot,
[data-testid="stPlotlyChart"] .plot-container,
[data-testid="stPlotlyChart"] .main-svg,
[data-testid="stPlotlyChart"] rect.bg,
[data-testid="stPlotlyChart"] rect.plotbg {
    fill: transparent !important;
    background: transparent !important;
    background-color: transparent !important;
}
div[role="radiogroup"] label {
    background: #ffffff !important;
    padding: 12px 16px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0, 0, 0, 0.05) !important;
    transition: all 0.2s ease !important;
}
div[role="radiogroup"] label:hover {
    background: #fcfbfa !important;
    border-color: #000000 !important;
}
.page-shell {
    background: #ffffff !important;
    border: 1px solid rgba(0, 0, 0, 0.05);
    border-radius: 16px;
    padding: 28px 36px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.015);
    margin-bottom: 20px;
    width: 100%;
    box-sizing: border-box;
}
.page-title-box {
    display: block;
    width: 100%;
    background: transparent !important;
    color: #000000 !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    font-size: 28px;
    font-weight: 800;
    margin-top: 48px;
    margin-bottom: 24px;
    letter-spacing: -0.03em;
}
.page-text {
    color: #4a4a4a;
    line-height: 1.8;
    text-align: justify;
    font-size: 16px;
}
.page-next {
    margin-top: 32px;
    display: flex;
    justify-content: flex-end;
}
.page-next button {
    background: #ffffff !important;
    color: #000000 !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
    padding: 12px 24px !important;
    border-radius: 999px !important;
    font-weight: 700;
}

/* Sidebar Radio Navigation Override Styles */
[data-testid="stSidebar"] div[role="radiogroup"] {
    background-color: transparent !important;
    padding: 0 !important;
    gap: 0 !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label [role="presentation"],
[data-testid="stSidebar"] div[role="radiogroup"] label div[dir="ltr"],
[data-testid="stSidebar"] div[role="radiogroup"] label input[type="radio"],
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label {
    background-color: transparent !important;
    color: #4a3f2f !important;
    border: none !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    line-height: 1.5 !important;
    border-radius: 10px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    margin-bottom: 2px !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    color: #1a1208 !important;
    background-color: rgba(0,0,0,0.07) !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    color: #c94f38 !important;
    background-color: #f7e6e1 !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p,
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) span {
    color: #c94f38 !important;
    font-weight: 700 !important;
}

/* SVG icons for sidebar radio */
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(1)::before {
    content: "";
    display: inline-block;
    width: 16px;
    height: 16px;
    margin-right: 12px;
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="rgba(74,63,47,0.9)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>');
    background-size: contain;
    background-repeat: no-repeat;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(1):has(input:checked)::before {
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23c94f38" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>');
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(2)::before {
    content: "";
    display: inline-block;
    width: 16px;
    height: 16px;
    margin-right: 12px;
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="rgba(74,63,47,0.9)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>');
    background-size: contain;
    background-repeat: no-repeat;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(2):has(input:checked)::before {
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23c94f38" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>');
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(3)::before {
    content: "";
    display: inline-block;
    width: 16px;
    height: 16px;
    margin-right: 12px;
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="rgba(74,63,47,0.9)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>');
    background-size: contain;
    background-repeat: no-repeat;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(3):has(input:checked)::before {
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23c94f38" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>');
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(4)::before {
    content: "";
    display: inline-block;
    width: 16px;
    height: 16px;
    margin-right: 12px;
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="rgba(74,63,47,0.9)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/></svg>');
    background-size: contain;
    background-repeat: no-repeat;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(4):has(input:checked)::before {
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23c94f38" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/></svg>');
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(5)::before {
    content: "";
    display: inline-block;
    width: 16px;
    height: 16px;
    margin-right: 12px;
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="rgba(74,63,47,0.9)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>');
    background-size: contain;
    background-repeat: no-repeat;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(5):has(input:checked)::before {
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23c94f38" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>');
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(6)::before {
    content: "";
    display: inline-block;
    width: 16px;
    height: 16px;
    margin-right: 12px;
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="rgba(74,63,47,0.9)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>');
    background-size: contain;
    background-repeat: no-repeat;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(6):has(input:checked)::before {
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23c94f38" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>');
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(7)::before {
    content: "";
    display: inline-block;
    width: 16px;
    height: 16px;
    margin-right: 12px;
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="rgba(74,63,47,0.9)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"/><path d="m9 12 2 2 4-4"/></svg>');
    background-size: contain;
    background-repeat: no-repeat;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(7):has(input:checked)::before {
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23c94f38" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"/><path d="m9 12 2 2 4-4"/></svg>');
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(8)::before {
    content: "";
    display: inline-block;
    width: 16px;
    height: 16px;
    margin-right: 12px;
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="rgba(74,63,47,0.9)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>');
    background-size: contain;
    background-repeat: no-repeat;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(8):has(input:checked)::before {
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23c94f38" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>');
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(9)::before {
    content: "";
    display: inline-block;
    width: 16px;
    height: 16px;
    margin-right: 12px;
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="rgba(74,63,47,0.9)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="21" r="1"/><circle cx="20" cy="21" r="1"/><path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"/></svg>');
    background-size: contain;
    background-repeat: no-repeat;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(9):has(input:checked)::before {
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23c94f38" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="21" r="1"/><circle cx="20" cy="21" r="1"/><path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"/></svg>');
}

.communities-grid {
    display: grid;
    gap: 12px;
    grid-template-columns: repeat(8, 1fr);
    margin-top: 20px;
    margin-bottom: 32px;
}
@media (max-width: 1400px) {
    .communities-grid { grid-template-columns: repeat(4, 1fr); }
}
@media (max-width: 900px) {
    .communities-grid { grid-template-columns: repeat(2, 1fr); }
}
.community-card {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.07);
    border-radius: 14px;
    padding: 16px 16px 14px 16px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 320px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: pointer;
}
.community-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    border-color: rgba(0,0,0,0.15);
}
.community-card-icon-container {
    height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 12px;
}
.community-card-icon { display: inline-block; }
.community-card-img {
    max-height: 200px;
    max-width: 100%;
    object-fit: contain;
}
.community-card-title {
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
    font-size: 16px;
    font-weight: 700;
    color: #111827;
    margin: 0 0 6px 0 !important;
}
.community-card-value {
    font-size: 28px;
    font-weight: 800;
    color: #000000;
    margin-bottom: 1px;
    line-height: 1;
}
.community-card-sub {
    font-size: 11px;
    color: #9ca3af;
    margin-bottom: 8px;
}
.community-card-desc {
    font-size: 12px;
    color: #6b7280;
    line-height: 1.5;
    flex-grow: 1;
    margin-bottom: 10px;
}
.community-card-arrow {
    align-self: flex-end;
    font-size: 18px;
    color: #9ca3af;
}

.metrics-row-grid {
    border-top: 1px solid rgba(0, 0, 0, 0.08);
    padding-top: 28px;
    margin-top: 40px;
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 24px;
}
@media (max-width: 1000px) {
    .metrics-row-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    .metrics-row-item {
        border-right: none !important;
        border-bottom: 1px solid rgba(0, 0, 0, 0.08);
        padding-bottom: 16px;
    }
}
@media (max-width: 600px) {
    .metrics-row-grid {
        grid-template-columns: 1fr;
    }
}
/* Hide heading anchor/link icons */
.anchor-link, 
.header-anchor, 
[data-testid="stHeaderActionElements"],
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
    display: none !important;
    visibility: hidden !important;
}

/* Restore Streamlit's native Material Icons */
.material-icons,
[class*="material-symbols"],
[data-testid="stIconMaterial"],
span[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Outlined', 'Material Symbols Rounded', 'Material Symbols Sharp', 'Material Icons' !important;
}
</style>
""".replace('{INITIAL_BG_URI}', INITIAL_BG_URI)

st.markdown(css, unsafe_allow_html=True)

SEGMENT_COLORS = {
    0: "#b76563",  # Vegetarians -> Dusty Rose
    1: "#bc7933",  # Regulars -> Ochre
    2: "#687643",  # Wellness -> Olive Green
    3: "#b64828",  # Promoters -> Rust Red
    4: "#c88d40",  # Loyalists -> Mustard
    5: "#368689",  # Families -> Teal
    6: "#36668d",  # Economizers -> Dark Blue
    7: "#6c4d36"   # Techies -> Dark Brown
}

SEGMENT_NAME_COLORS = {
    "Regulars": SEGMENT_COLORS[0],
    "Families": SEGMENT_COLORS[1],
    "Economizers": SEGMENT_COLORS[2],
    "Vegetarians": SEGMENT_COLORS[3],
    "Loyalists": SEGMENT_COLORS[4],
    "Techies": SEGMENT_COLORS[5],
    "Wellness": SEGMENT_COLORS[6],
    "Promoters": SEGMENT_COLORS[7],
}


_page_labels = {
    "Overview":                    "Overview",
    "Data Analysis":               "**NB0** - Data Analysis",
    "Data Preprocessing":          "**NB1** - EDA & Preprocessing",
    "Data in Geography":           "**NB2** - Geographic Analysis",
    "NB3 Clustering":             "**NB3** - Clustering",
    "NB4 Characterisation":       "**NB4** - Cluster Characterisation",
    "Targeter Promotion":          "**NB5** - Association Rules",
    "Conclusion & Recommendations":"Conclusion & Recommendations",
    "Customer Simulator":          "Customer Simulator",
}

st.sidebar.markdown("<div style='font-size: 32px; font-weight: 800; color: #000000; margin-top: 15px; margin-bottom: 25px; font-family: \"Plus Jakarta Sans\", \"Inter\", sans-serif !important; letter-spacing: -0.03em; line-height: 1.1; padding-left: 14px;'>Costumer<br/>Segmentation</div>", unsafe_allow_html=True)
if "current_page" not in st.session_state:
    st.session_state.current_page = "Overview"

page_list = list(_page_labels.keys())
current_index = page_list.index(st.session_state.current_page) if st.session_state.current_page in page_list else 0

selected_page = st.sidebar.radio(
    label="Index",
    options=page_list,
    index=current_index,
    format_func=lambda x: _page_labels[x],
    label_visibility="collapsed"
)
st.session_state.current_page = selected_page

if "previous_page" not in st.session_state:
    st.session_state.previous_page = "Overview"

if st.session_state.previous_page != selected_page:
    st.session_state.previous_page = selected_page
    st.session_state.scroll_trigger = True

def render_footer():
    st.markdown(
        f"""
        <div style='margin-top: 24px; font-family:"Plus Jakarta Sans", "Inter", sans-serif; background-color: #fcfbf8; padding: 32px; border-radius: 16px; border: 1px solid #f0ece1;'>
          <div style='width:100%; display:grid; grid-template-columns:repeat(4, 1fr); gap:24px; align-items:start;'>
            <div>
              <div style='text-transform:uppercase; font-size:11px; letter-spacing:0.15em; color:#8c8c8c; margin-bottom:16px; font-weight:700;'>Navigation</div>
              <p style='color:#3b2720; margin:0; line-height:1.6; font-size:14px;'>Use the navigation bar on the left side of the screen.</p>
            </div>
            <div>
              <div style='text-transform:uppercase; font-size:11px; letter-spacing:0.15em; color:#8c8c8c; margin-bottom:16px; font-weight:700;'>Work done by</div>
              <div style='display:grid; gap:12px;'>
                <a href='https://github.com/CarlotaMarto' target='_blank' rel='noreferrer noopener' style='display:flex; gap:12px; align-items:center; text-decoration:none; color:#0f172a; cursor:pointer; padding:0;'>
                  <img src='https://github.com/CarlotaMarto.png' alt='Carlota Marto GitHub' style='width:36px; height:36px; border-radius:50%; object-fit:cover; flex-shrink:0;'/>
                  <div>
                    <div style='font-weight:700; font-size:14px; color:#3b2720;'>Carlota Marto</div>
                    <div style='font-size:12px; color:#8c8c8c;'>20241729</div>
                  </div>
                </a>
                <a href='https://github.com/Franciscaveigateixeira' target='_blank' rel='noreferrer noopener' style='display:flex; gap:12px; align-items:center; text-decoration:none; color:#0f172a; cursor:pointer; padding:0;'>
                  <img src='https://github.com/Franciscaveigateixeira.png' alt='Francisca Teixeira GitHub' style='width:36px; height:36px; border-radius:50%; object-fit:cover; flex-shrink:0;'/>
                  <div>
                    <div style='font-weight:700; font-size:14px; color:#3b2720;'>Francisca Teixeira</div>
                    <div style='font-size:12px; color:#8c8c8c;'>20241702</div>
                  </div>
                </a>
                <a href='https://github.com/Gouveia316' target='_blank' rel='noreferrer noopener' style='display:flex; gap:12px; align-items:center; text-decoration:none; color:#0f172a; cursor:pointer; padding:0;'>
                  <img src='https://github.com/Gouveia316.png' alt='Pedro GitHub' style='width:36px; height:36px; border-radius:50%; object-fit:cover; flex-shrink:0;'/>
                  <div>
                    <div style='font-weight:700; font-size:14px; color:#3b2720;'>Pedro Gouveia</div>
                    <div style='font-size:12px; color:#8c8c8c;'>20231657</div>
                  </div>
                </a>
              </div>
            </div>
            <div>
              <div style='text-transform:uppercase; font-size:11px; letter-spacing:0.15em; color:#8c8c8c; margin-bottom:16px; font-weight:700;'>Teacher</div>
              <div style='font-weight:700; color:#3b2720; margin-bottom:4px; font-size:14px;'>Ivo Bernardo</div>
              <div style='color:#8c8c8c; font-size:18px;'>Machine Learning II</div>
            </div>
            <div style='text-align: left;'>
              <div style='color:#64748b; line-height:1.6; margin:0; font-size:14px; padding-top: 27px; text-align: left !important;'>This project is optimized for executive-level business intelligence and strategic decision making.</div>
            </div>
          </div>
          <div style='margin-top: 32px; width: 100%; overflow: hidden; border-radius: 12px;'>
            <img src='{BAR_URI}' style='width: 100%; display: block; border-radius: 12px 12px 0 0;' />
          </div>
          <div style='background-color: #f7e6e1; color: #a36154; padding: 16px 32px; display: flex; justify-content: space-between; align-items: center; font-size: 12px; width: 100%; border-radius: 0 0 12px 12px; box-sizing: border-box;'>
            <span>&copy; 2026 Customer Segmentation Project - Academic Use Only</span>
            <span>Built for Machine Learning II</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get('scroll_trigger', False):
        import streamlit.components.v1 as components
        components.html(
            '''
            <script>
                function forceScroll() {
                    var selectors = [
                        window.parent,
                        window.parent.document.documentElement,
                        window.parent.document.body,
                        window.parent.document.querySelector("[data-testid='stAppViewContainer']"),
                        window.parent.document.querySelector(".main"),
                        window.parent.document.querySelector(".block-container")
                    ];
                    selectors.forEach(function(c) {
                        if (c && typeof c.scrollTo === 'function') {
                            c.scrollTo({top: 0, behavior: 'instant'});
                        }
                    });
                }
                forceScroll();
                setTimeout(forceScroll, 50);
                setTimeout(forceScroll, 200);
                setTimeout(forceScroll, 500);
                setTimeout(forceScroll, 1000);
            </script>
            ''',
            height=0
        )
        st.session_state.scroll_trigger = False

# Page conditional routing
if selected_page == "Overview":
    st.markdown(f"""
    <div style='display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 32px; margin-top: 0px; margin-bottom: 56px; font-family: "Plus Jakarta Sans", "Inter", sans-serif; width: 100%;'>
        <div style='flex: 1.5; min-width: 320px;'>
            <h1 style='font-size: clamp(2rem, 4vw, 3.2rem); font-weight: 800; color: #000000; line-height: 1.05; margin: 0 0 20px 0; letter-spacing: -0.04em;'>Understand every customer.<br/>Grow with purpose.</h1>
            <p style='font-size: 16px; color: #5f6368; line-height: 1.7; margin: 0; max-width: 520px;'>A machine learning project that segments 33,038 supermarket customers into 8 distinct communities, uncovering who they are, how they shop, and what drives their decisions.</p>
        </div>
        <div style='flex: 1; min-width: 300px; display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 0px;'>
            <img src='{CESTO_URI}' style='max-height: 250px; width: auto; max-width: 100%; object-fit: contain;' />
            <div style='display: flex; gap: 40px; text-align: center; margin-top: -10px;'>
                <div>
                    <div style='font-size: clamp(1.6rem, 2.5vw, 2.2rem); font-weight: 800; color: #000000; line-height: 1;'>33,038</div>
                    <div style='font-size: 18px; color: #5f6368; margin-top: 4px;'>customers analyzed</div>
                </div>
                <div>
                    <div style='font-size: clamp(1.6rem, 2.5vw, 2.2rem); font-weight: 800; color: #000000; line-height: 1;'>8</div>
                    <div style='font-size: 18px; color: #5f6368; margin-top: 4px;'>communities discovered</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    def _go_to_comm():
        st.session_state.current_page = "NB4 Characterisation"
        
    st.button("Explore your communities →", key="btn_explore", on_click=_go_to_comm)

    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 20px; margin-top:16px; margin-bottom:8px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Scientific reference</div>
  <p style='font-size:16px; color:#374151; line-height:1.7; margin:0 0 8px 0;'>This project follows the methodology supported by recent literature on unsupervised machine learning applied to retail segmentation. Spiegel et al. (2022) demonstrate that K-Means with silhouette-based cluster selection consistently outperforms alternative unsupervised methods when the goal is actionable, interpretable customer profiles for targeted marketing, a finding directly aligned with the approach adopted here.</p>
  <p style='font-size:16px; color:#9ca3af; margin:0;'>Spiegel, R. et al. (2022). K-Means Clustering Approach for Intelligent Customer Segmentation Using Customer Purchase Behavior Data. <em>Sustainability</em>, 14(12), 7243. MDPI. https://doi.org/10.3390/su14127243</p>
</div>
""", unsafe_allow_html=True)

    # Mini Community Cards — Overview preview
    import pandas as _pd
    _seg = load_csv_data("id_and_cluster.csv")
    _total = len(_seg)
    _counts = _seg.groupby(["cluster","cluster_name"]).size().reset_index(name="n")
    _counts["pct"] = (_counts["n"] / _total * 100).round(1)

    # cluster → URI mapping
    _cluster_images = {
        0: REGULARS_URI,
        1: FAMILIES_URI,
        2: ECONOMIZERS_URI,
        3: VEGETARIANS_URI,
        4: LOYALISTS_URI,
        5: TECHIES_URI,
        6: WELLNESS_URI,
        7: PROMOTERS_URI,
    }
    for _i, _row in _counts.iterrows():
        _c = _row["cluster"]
        if _c not in _cluster_images:
            _cluster_images[_c] = SLICES_URIS[_i % len(SLICES_URIS)]

    _cluster_descs = {
        0: "Active but newer, deal-aware shoppers. Strong targets for loyalty program onboarding.",
        1: "Large households. Loyal without needing promotions. Target with bulk-buying bundles.",
        2: "Restrained, low-friction spenders. Value baseline pricing over deals.",
        3: "Full-price, promotion-resistant shoppers. Lead with curation and quality framing.",
        4: "Highest LTV and tenure. Loyal for 13+ years. Reward and protect.",
        5: "Small households buying high-value tech. Best electronics cross-sell audience.",
        6: "Quiet, low-maintenance shoppers. Low friction and buy full price.",
        7: "The ultimate deal-seekers. Perfect for price-led campaign stacking.",
    }

    _cards_html = ""
    for _, _row in _counts.sort_values("cluster").iterrows():
        _c = int(_row["cluster"])
        _name = _row["cluster_name"]
        _pct = _row["pct"]
        _n = int(_row["n"])
        _img = _cluster_images.get(_c, "")
        _desc = _cluster_descs.get(_c, "")
        _cards_html += f"""
        <div class='ov-card'>
            <div class='ov-img-wrap'><img src='{_img}' /></div>
            <div style='margin-top:10px;'>
                <div class='ov-name'>{_name}</div>
                <div class='ov-pct'>{_pct}%</div>
                <div class='ov-sub'>{_n:,} customers</div>
                <div class='ov-desc'>{_desc}</div>
            </div>
            <div class='ov-arrow'>→</div>
        </div>"""

    import streamlit.components.v1 as _components
    _overview_cards_html = f"""
    <style>
      body {{ margin:0; padding:0; background:transparent; font-family:'Plus Jakarta Sans','Inter',sans-serif; }}
      .ov-grid {{
          display: flex;
          flex-wrap: nowrap;
          overflow-x: auto;
          gap: 12px;
          width: 100%;
          box-sizing: border-box;
          padding-bottom: 12px;
          scroll-behavior: smooth;
      }}
      .ov-grid::-webkit-scrollbar {{
          height: 6px;
      }}
      .ov-grid::-webkit-scrollbar-track {{
          background: #f3f4f6;
          border-radius: 99px;
      }}
      .ov-grid::-webkit-scrollbar-thumb {{
          background: #e5e7eb;
          border-radius: 99px;
      }}
      .ov-grid::-webkit-scrollbar-thumb:hover {{
          background: #d1d5db;
      }}
      .ov-card {{
          background:#fff;
          border:1px solid #e5e7eb;
          border-radius:14px;
          padding:14px 12px;
          display:flex;
          flex-direction:column;
          justify-content:space-between;
          min-height:290px;
          flex: 0 0 200px;
          box-sizing: border-box;
      }}
      .ov-img-wrap {{ height:120px; display:flex; align-items:center; justify-content:center; overflow:hidden; }}
      .ov-img-wrap img {{ max-height:120px; max-width:100%; object-fit:contain; }}
      .ov-name {{ font-size:14px; font-weight:700; color:#111827; margin-bottom:4px; }}
      .ov-pct {{ font-size:26px; font-weight:800; color:#111827; line-height:1; }}
      .ov-sub {{ font-size:11px; color:#9ca3af; margin-top:2px; margin-bottom:6px; }}
      .ov-desc {{ font-size:11px; color:#6b7280; line-height:1.4; }}
      .ov-arrow {{ font-size:14px; color:#9ca3af; margin-top:8px; }}
      .ov-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }}
      .ov-header-title {{ font-size:16px; font-weight:700; color:#111827; }}
      .ov-header-link {{ font-size:18px; color:#6b7280; }}
    </style>
    <div class='ov-header'>
      <div class='ov-header-title'>Your 8 customer communities</div>
      <div class='ov-header-link'>View all communities →</div>
    </div>
    <div class='ov-grid'>
      {_cards_html}
    </div>"""
    _components.html(_overview_cards_html, height=340, scrolling=False)

    # Key Metrics Bar
    st.markdown("""
    <div style='display: grid; grid-template-columns: repeat(5, 1fr); gap: 0; border: 1px solid #e5e7eb; border-radius: 14px; overflow: hidden; margin-top: 24px; background: #ffffff; width: 100%; box-sizing: border-box;'>
        <div style='padding: 20px 24px; border-right: 1px solid #e5e7eb;'>
            <div style='font-size: 11px; font-weight: 600; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;'>Total Revenue</div>
            <div style='font-size: 26px; font-weight: 800; color: #111827; line-height: 1;'>€835.6M</div>
            <div style='font-size: 12px; color: #6b7280; margin-top: 4px;'>lifetime customer value</div>
        </div>
        <div style='padding: 20px 24px; border-right: 1px solid #e5e7eb;'>
            <div style='font-size: 11px; font-weight: 600; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;'>Avg. Lifetime Spend</div>
            <div style='font-size: 26px; font-weight: 800; color: #111827; line-height: 1;'>€26,100</div>
            <div style='font-size: 12px; color: #6b7280; margin-top: 4px;'>per customer</div>
        </div>
        <div style='padding: 20px 24px; border-right: 1px solid #e5e7eb;'>
            <div style='font-size: 11px; font-weight: 600; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;'>Loyalty Card Rate</div>
            <div style='font-size: 26px; font-weight: 800; color: #111827; line-height: 1;'>60.5%</div>
            <div style='font-size: 12px; color: #6b7280; margin-top: 4px;'>of customers enrolled</div>
        </div>
        <div style='padding: 20px 24px; border-right: 1px solid #e5e7eb;'>
            <div style='font-size: 11px; font-weight: 600; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;'>Avg. Tenure</div>
            <div style='font-size: 26px; font-weight: 800; color: #111827; line-height: 1;'>11.0 yrs</div>
            <div style='font-size: 12px; color: #6b7280; margin-top: 4px;'>customer relationship</div>
        </div>
        <div style='padding: 20px 24px;'>
            <div style='font-size: 11px; font-weight: 600; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;'>Distinct Products</div>
            <div style='font-size: 26px; font-weight: 800; color: #111827; line-height: 1;'>146</div>
            <div style='font-size: 12px; color: #6b7280; margin-top: 4px;'>avg. per customer</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Project Presentation
    st.markdown("""
    <div style='margin-top: 32px; width: 100%; box-sizing: border-box;'>

      <!-- Section: Objective -->
      <div style='margin-bottom: 48px;'>
        <div style='font-size: 11px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #9ca3af; margin-bottom: 12px;'>Project Objective</div>
        <h2 style='font-size: 28px; font-weight: 800; color: #111827; margin: 0 0 16px 0; letter-spacing: -0.02em; line-height: 1.2;'>Why segment customers at all?</h2>
        <p style='font-size: 16px; color: #374151; line-height: 1.8; margin: 0 0 14px 0;'>
          Traditional retail analytics treat the customer base as a single, homogeneous mass, averaging out behaviors that are fundamentally different from one another. This project challenges that assumption. The central objective of this work is to apply unsupervised machine learning to a real-world supermarket dataset and discover whether meaningful, stable, and actionable customer groups exist within the data.
        </p>
        <p style='font-size: 16px; color: #374151; line-height: 1.8; margin: 0;'>
          Rather than relying on intuition or demographic proxies, this analysis is driven entirely by observed behavior: spending patterns across product categories, promotional sensitivity, household composition, loyalty card usage, shopping frequency, and geographic footprint. The goal is to surface communities that are genuinely distinct and that carry real implications for marketing, product, and retention strategy.
        </p>
      </div>

      <!-- Divider -->
      <div style='border-top: 1px solid #e5e7eb; margin-bottom: 48px;'></div>

      <!-- Section: Scope -->
      <div style='margin-bottom: 48px;'>
        <div style='font-size: 11px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #9ca3af; margin-bottom: 12px;'>Scope & Dataset</div>
        <h2 style='font-size: 28px; font-weight: 800; color: #111827; margin: 0 0 16px 0; letter-spacing: -0.02em; line-height: 1.2;'>What data was used?</h2>
        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 20px;'>
          <div style='background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px 22px;'>
            <div style='font-size: 32px; font-weight: 800; color: #111827; line-height: 1;'>33,038</div>
            <div style='font-size: 13px; color: #6b7280; margin-top: 4px;'>unique customers</div>
          </div>
          <div style='background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px 22px;'>
            <div style='font-size: 32px; font-weight: 800; color: #111827; line-height: 1;'>39</div>
            <div style='font-size: 13px; color: #6b7280; margin-top: 4px;'>behavioral & demographic features</div>
          </div>
          <div style='background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px 22px;'>
            <div style='font-size: 32px; font-weight: 800; color: #111827; line-height: 1;'>11 yrs</div>
            <div style='font-size: 13px; color: #6b7280; margin-top: 4px;'>average customer tenure</div>
          </div>
        </div>
        <p style='font-size: 16px; color: #374151; line-height: 1.8; margin: 0 0 14px 0;'>
          The dataset covers two linked tables: <strong>customer_info</strong>, containing demographic attributes, geographic coordinates, loyalty flag, and lifetime spend broken down by 10 product categories; and <strong>customer_basket</strong>, containing transaction-level purchase logs used for association rule mining.
        </p>
        <p style='font-size: 16px; color: #374151; line-height: 1.8; margin: 0;'>
          Prior to modelling, the data required substantial cleaning: removal of negative spend values introduced by return corrections, filtering of future-dated transactions, and outlier capping at the 99th percentile to prevent extreme spenders from distorting cluster centroids. Feature engineering added tenure (years since first transaction), annual spend rates, total children, a technology spend aggregate, and cyclical encodings of typical shopping hour.
        </p>
      </div>

      <!-- Divider -->
      <div style='border-top: 1px solid #e5e7eb; margin-bottom: 48px;'></div>

      <!-- Section: Methodology -->
      <div style='margin-bottom: 48px;'>
        <div style='font-size: 11px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #9ca3af; margin-bottom: 12px;'>Methodology</div>
        <h2 style='font-size: 28px; font-weight: 800; color: #111827; margin: 0 0 16px 0; letter-spacing: -0.02em; line-height: 1.2;'>How were the communities found?</h2>
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px;'>
          <div style='border-left: 3px solid #111827; padding-left: 16px;'>
            <div style='font-size: 13px; font-weight: 700; color: #111827; margin-bottom: 6px;'>1. Exploratory Analysis</div>
            <div style='font-size: 13px; color: #6b7280; line-height: 1.6;'>Distribution checks, missing value mapping, spend category breakdowns, and geographic density analysis across the customer base.</div>
          </div>
          <div style='border-left: 3px solid #111827; padding-left: 16px;'>
            <div style='font-size: 13px; font-weight: 700; color: #111827; margin-bottom: 6px;'>2. Preprocessing</div>
            <div style='font-size: 13px; color: #6b7280; line-height: 1.6;'>Outlier capping, negative value correction, feature engineering of annual rates, tenure, and technology spend. StandardScaler normalization before clustering.</div>
          </div>
          <div style='border-left: 3px solid #111827; padding-left: 16px;'>
            <div style='font-size: 13px; font-weight: 700; color: #111827; margin-bottom: 6px;'>3. K-Means Clustering</div>
            <div style='font-size: 13px; color: #6b7280; line-height: 1.6;'>K=8 selected via the Elbow Method and Silhouette Score analysis. Multiple random seeds tested for centroid stability. Final model validated against key behavioral axes.</div>
          </div>
          <div style='border-left: 3px solid #111827; padding-left: 16px;'>
            <div style='font-size: 13px; font-weight: 700; color: #111827; margin-bottom: 6px;'>4. Association Rules</div>
            <div style='font-size: 13px; color: #6b7280; line-height: 1.6;'>Apriori algorithm applied per cluster to customer basket data, extracting high-confidence cross-sell rules for targeted campaign design.</div>
          </div>
        </div>
      </div>

      <!-- Divider -->
      <div style='border-top: 1px solid #e5e7eb; margin-bottom: 48px;'></div>

      <!-- Section: Navigate -->
      <div style='margin-bottom: 48px;'>
        <div style='font-size: 11px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #9ca3af; margin-bottom: 12px;'>How to use this dashboard</div>
        <h2 style='font-size: 28px; font-weight: 800; color: #111827; margin: 0 0 16px 0; letter-spacing: -0.02em; line-height: 1.2;'>Navigate the findings</h2>
        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;'>
          <div style='background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px 22px;'>
            <div style='font-size: 13px; font-weight: 700; color: #111827; margin-bottom: 6px;'>Data Analysis</div>
            <div style='font-size: 13px; color: #6b7280; line-height: 1.6;'>Raw dataset exploration: spend distributions, category breakdowns, and customer demographic profiles.</div>
          </div>
          <div style='background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px 22px;'>
            <div style='font-size: 13px; font-weight: 700; color: #111827; margin-bottom: 6px;'>Customer Communities</div>
            <div style='font-size: 13px; color: #6b7280; line-height: 1.6;'>The 8 discovered segments with full behavioral profiles, spend radars, and cluster-level KPIs.</div>
          </div>
          <div style='background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px 22px;'>
            <div style='font-size: 13px; font-weight: 700; color: #111827; margin-bottom: 6px;'>Opportunities & Simulator</div>
            <div style='font-size: 13px; color: #6b7280; line-height: 1.6;'>Cross-sell campaign rules per segment and a real-time simulator to classify hypothetical customers.</div>
          </div>
        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)
    render_footer()

elif selected_page == "Data Analysis":
    st.markdown("""
    <div style='margin-top: 0px; margin-bottom: 24px;'>
        <h2 style='font-size: 56px; font-weight: 800; color: #000000; margin: 0; letter-spacing: -0.03em;'>Data Analysis</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='display:flex; gap:24px; margin-bottom:32px; align-items:flex-start; flex-wrap:wrap;'>
      <div style='flex:1; min-width:300px;'>
    <div style='margin-bottom:32px;'>
      <div>
        <div style='font-size:11px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#9ca3af; margin-bottom:10px;'>Notebook 0: Data Analysis</div>
        <p style='font-size:18px; color:#374151; line-height:1.9; margin:0 0 24px 0; text-align: justify;'>
          This notebook establishes the baseline understanding of the raw customer dataset. The objective is to identify data quality issues: missing values, erroneous entries, outliers, and skews, before any modelling takes place. No data is modified or removed in this notebook; it serves purely as a diagnostic stage to inform the decisions made in Notebook 1.
        </p>
        <div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:16px; margin-bottom:28px;'>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:18px 20px;'>
            <div style='font-size:28px; font-weight:800; color:#c94f38; line-height:1;'>33,038</div>
            <div style='font-size:18px; color:#7a6454; margin-top:4px;'>unique customers in raw dataset</div>
          </div>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:18px 20px;'>
            <div style='font-size:28px; font-weight:800; color:#c94f38; line-height:1;'>21</div>
            <div style='font-size:18px; color:#7a6454; margin-top:4px;'>numerical variables identified</div>
          </div>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:18px 20px;'>
            <div style='font-size:28px; font-weight:800; color:#c94f38; line-height:1;'>30%</div>
            <div style='font-size:18px; color:#7a6454; margin-top:4px;'>missing value threshold for feature exclusion</div>
          </div>
        </div>
      </div>

      <div style='background:linear-gradient(145deg, #ffffff, #f9fafb); border:1px solid #e5e7eb; border-radius:12px; padding:20px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05);'>
        <div style='font-size:14px; font-weight:800; color:#111827; margin-bottom:16px; border-bottom:2px solid #f3eee6; padding-bottom:8px;'>Notebook 0 Index</div>
        <div style='display:flex; flex-wrap:wrap; gap:12px; align-items:center;'>
          <a href="#nb0-1" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>1. Imports and Data Loading</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb0-1-1" style="text-decoration:none;"><div style='font-size:18px; color:#6b7280;'>1.1 Initial Data Analysis</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb0-1-2" style="text-decoration:none;"><div style='font-size:18px; color:#6b7280;'>1.2 Duplicate Rows Analysis</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb0-1-2-1" style="text-decoration:none;"><div style='font-size:18px; color:#6b7280;'>1.2.1 Surname Repetition Check</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb0-1-3" style="text-decoration:none;"><div style='font-size:18px; color:#6b7280;'>1.3 Missing Values Analysis</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb0-1-4" style="text-decoration:none;"><div style='font-size:18px; color:#6b7280;'>1.4 Numerical and Categorical Columns</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb0-1-4-1" style="text-decoration:none;"><div style='font-size:18px; color:#6b7280;'>1.4.1 Findings in Categorical Columns</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb0-1-5" style="text-decoration:none;"><div style='font-size:18px; color:#6b7280;'>1.5 Statistical Summary</div></a>
        </div>
      </div>
    </div>

    <div id="nb0-1"></div>
    <div id="nb0-1-1" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">1.1 Initial Data Analysis</h2></div>
    <div style='border-left:3px solid #111827; padding-left:20px; margin-bottom:20px;'>
      <div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>Dataset Overview</div>
      <div style='font-size:16px; color:#6b7280; line-height:1.8;'>The dataset contains demographic and transactional records for a subset of the customer base. This initial pass focuses on identifying data quality issues, structural problems, and general feature distributions before any preprocessing or scaling is applied.</div>
    </div>

    <div id="nb0-1-2" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">1.2 Duplicate Rows Analysis & <span id="nb0-1-2-1">1.2.1 Surname Repetition Check</span></h2></div>
    <div style='border-left:3px solid #111827; padding-left:20px; margin-bottom:20px;'>
      <div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>Duplicate check</div>
      <div style='font-size:16px; color:#6b7280; line-height:1.8;'>No exact duplicate rows were found. A logical duplicate check (matching on customer name AND birthdate) was also performed. A surname-only proximity test was also run but produced too many false positives due to common surnames. The conclusion is that the dataset does not contain systematic duplicate records requiring removal.</div>
    </div>

    <div id="nb0-1-3" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">1.3 Missing Values Analysis</h2></div>
    <div style='border-left:3px solid #111827; padding-left:20px; margin-bottom:20px;'>
      <div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>Missing value strategy</div>
      <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Features with more than 30% missing values were flagged as too sparse to impute reliably. The inspection confirmed that missing values are concentrated in a limited group of behavioural and spend variables, supporting imputation over row-dropping: the customer base does not need to be reduced.</div>
    </div>
    """, unsafe_allow_html=True)

    customer_info = load_csv_data("customer_info.csv")

    # Chart 1: Missing values per feature
    st.markdown("""
<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Missing values per feature (%)</div>
</div>
""", unsafe_allow_html=True)
    missing_pct = (customer_info.isnull().mean() * 100).reset_index()
    missing_pct.columns = ["feature", "missing_pct"]
    missing_pct = missing_pct.sort_values("missing_pct", ascending=False)
    missing_pct = missing_pct[missing_pct["missing_pct"] > 0]
    base_missing = alt.Chart(missing_pct).mark_bar(color=SEGMENT_COLORS[7], cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        y=alt.Y("feature:N", sort="-x", title="Feature"),
        x=alt.X("missing_pct:Q", title="Missing (%)"),
        tooltip=["feature", alt.Tooltip("missing_pct:Q", format=".2f", title="Missing %")]
    ).properties(height=max(200, len(missing_pct) * 22))
    threshold_line = alt.Chart(pd.DataFrame({"threshold": [30]})).mark_rule(color="#ef4444", strokeDash=[6, 3], strokeWidth=2).encode(
        x="threshold:Q"
    )
    st.altair_chart((base_missing + threshold_line), use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The 30% threshold (red dashed line) was chosen as the boundary above which imputation is judged unreliable: reconstructing more than three out of ten values for a given feature would introduce more noise than signal into the dataset. Features below this threshold retain sufficient observed data to support KNN imputation, which leverages the similarity structure of the customer base. The chart confirms that missing values are concentrated in a small number of behavioural variables, and that no feature exceeds the threshold by a large margin, making row-dropping unnecessary. The majority of the 33,038 customers remain usable across all features.</p>
</div>
""", unsafe_allow_html=True)

    # 1.4 Numerical and Categorical Columns
    st.markdown("""
<div id="nb0-1-4" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">1.4 Numerical and Categorical Columns</h2></div>
<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Distribution of key numerical features</div>
</div>
""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown("<div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:8px;'>Lifetime spend: Groceries</div>", unsafe_allow_html=True)
    c2.markdown("<div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:8px;'>Lifetime spend: Electronics</div>", unsafe_allow_html=True)
    c3.markdown("<div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:8px;'>Total distinct products</div>", unsafe_allow_html=True)

    hist_groceries = alt.Chart(customer_info.dropna(subset=["lifetime_spend_groceries"])).mark_bar(color=SEGMENT_COLORS[3], opacity=0.85).encode(
        x=alt.X("lifetime_spend_groceries:Q", bin=alt.Bin(maxbins=30), title="Groceries spend"),
        y=alt.Y("count():Q", title="Customers"),
        tooltip=[alt.Tooltip("count():Q", title="Customers")]
    ).properties(height=280)
    hist_electronics = alt.Chart(customer_info.dropna(subset=["lifetime_spend_electronics"])).mark_bar(color=SEGMENT_COLORS[5], opacity=0.85).encode(
        x=alt.X("lifetime_spend_electronics:Q", bin=alt.Bin(maxbins=30), title="Electronics spend"),
        y=alt.Y("count():Q", title="Customers"),
        tooltip=[alt.Tooltip("count():Q", title="Customers")]
    ).properties(height=280)
    hist_products = alt.Chart(customer_info.dropna(subset=["lifetime_total_distinct_products"])).mark_bar(color=SEGMENT_COLORS[4], opacity=0.85).encode(
        x=alt.X("lifetime_total_distinct_products:Q", bin=alt.Bin(maxbins=30), title="Distinct products"),
        y=alt.Y("count():Q", title="Customers"),
        tooltip=[alt.Tooltip("count():Q", title="Customers")]
    ).properties(height=280)
    c1.altair_chart(hist_groceries, use_container_width=True)
    c2.altair_chart(hist_electronics, use_container_width=True)
    c3.altair_chart(hist_products, use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>All three distributions exhibit pronounced right-skew: the mass of customers clusters near the lower end of the scale, with a progressively thinner tail extending toward high-spending or high-variety individuals. This asymmetry has two direct implications for modelling. First, standard Euclidean distance in clustering is sensitive to scale differences, meaning that a small group of high-spending customers could disproportionately pull cluster centroids if the data are not scaled. Second, the long tail is precisely where the consensus outlier separation strategy intervenes: rather than capping values, the most extreme multivariate observations are separated into a dedicated outlier dataset before the clustering model is fitted, preserving the shape of the majority distribution while removing undue influence from the periphery.</p>
</div>
""", unsafe_allow_html=True)

    # 1.4.1 Findings in Categorical Columns
    st.markdown("""
<div id="nb0-1-4-1" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">1.4.1 Findings in Categorical Columns</h2></div>
<div style='border-left:3px solid #111827; padding-left:20px; margin-bottom:20px;'>
  <div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>Education level as a proxy feature</div>
  <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Customer names contain academic prefixes (BSc., MSc., PhD.) across all 33,038 unique names. These prefixes are flagged as an education-level proxy and earmarked for feature engineering in Notebook 1. Surname repetition alone was found to be too common to be a useful household signal; it was not carried into the modelling feature set.</div>
</div>
<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Gender distribution</div>
</div>
""", unsafe_allow_html=True)
    gender_counts = customer_info["customer_gender"].value_counts().reset_index()
    gender_counts.columns = ["customer_gender", "count"]
    gender_bar = alt.Chart(gender_counts).mark_bar(color=SEGMENT_COLORS[6], cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("customer_gender:N", title="Gender"),
        y=alt.Y("count:Q", title="Number of customers"),
        tooltip=["customer_gender", alt.Tooltip("count:Q", title="Customers", format=",")]
    ).properties(height=300)
    st.altair_chart(gender_bar, use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The gender distribution across the customer base is approximately balanced between male and female customers, with no category representing an extreme minority. This near-parity is relevant for segmentation methodology: a heavily skewed gender distribution would risk producing segments that reflect gender composition artefacts rather than genuine behavioural differences. The approximate balance observed here supports the interpretation that the eight clusters recovered by the model reflect spending behaviour and lifestyle patterns rather than demographic overrepresentation of one group. Gender is retained as a profiling variable for segment characterisation but is not included in the clustering distance matrix.</p>
</div>
""", unsafe_allow_html=True)

    # 1.5 Statistical Summary
    st.markdown("""
<div id="nb0-1-5" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">1.5 Statistical Summary</h2></div>
<div style='border-left:3px solid #111827; padding-left:20px; margin-bottom:8px;'>
  <div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>Impossible values detected</div>
  <div style='font-size:16px; color:#6b7280; line-height:1.8;'><code>percentage_of_products_bought_promotion</code> was found to contain values outside the valid [0, 1] range: both below 0.0 and above 1.0, indicating data entry errors. These are flagged here and corrected in preprocessing. Spending variables show strong right-skew, confirming that a small group of customers spends disproportionately more than the majority.</div>
</div>
<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Promotion ratio distribution (valid range only)</div>
</div>
""", unsafe_allow_html=True)
    promo_valid = customer_info[
        (customer_info["percentage_of_products_bought_promotion"] >= 0) &
        (customer_info["percentage_of_products_bought_promotion"] <= 1)
    ].copy()
    promo_hist = alt.Chart(promo_valid).mark_bar(color=SEGMENT_COLORS[1], opacity=0.85).encode(
        x=alt.X("percentage_of_products_bought_promotion:Q", bin=alt.Bin(maxbins=30), title="Proportion of products bought on promotion"),
        y=alt.Y("count():Q", title="Customers"),
        tooltip=[alt.Tooltip("count():Q", title="Customers")]
    ).properties(height=300)
    st.altair_chart(promo_hist, use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The raw dataset contains entries for <code>percentage_of_products_bought_promotion</code> that fall outside the physically valid interval [0, 1], including both negative values and values exceeding 1.0. These are impossible by definition: a proportion cannot be negative or greater than unity, confirming data entry errors rather than extreme but valid behaviour. The chart above is restricted to the valid range only. Within [0, 1], the distribution is roughly bimodal: a concentration of customers near 0.4 to 0.6 suggests a moderately promotion-responsive majority, while a second mass near 1.0 identifies a distinct group of near-exclusively promotional buyers. This heterogeneity in promotional sensitivity later becomes one of the most discriminating variables in the clustering model, most clearly visible in the Promoters segment.</p>
</div>
""", unsafe_allow_html=True)

    # Chart 5: Skewness table
    st.markdown("""
<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Skewness of spend variables</div>
</div>
""", unsafe_allow_html=True)
    spend_cols_skew = [c for c in customer_info.columns if c.startswith("lifetime_spend_")]
    skew_df = customer_info[spend_cols_skew].skew().reset_index()
    skew_df.columns = ["Feature", "Skewness"]
    skew_df["Feature"] = skew_df["Feature"].str.replace("lifetime_spend_", "", regex=False).str.replace("_", " ").str.title()
    skew_df = skew_df.sort_values("Skewness", ascending=False)
    skew_df["Skewness"] = skew_df["Skewness"].round(2)
    st.dataframe(skew_df.reset_index(drop=True), use_container_width=True, hide_index=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>All spend categories exhibit positive skewness, confirming that the right-tailed pattern observed in the histograms above is not isolated to groceries and electronics but is systematic across the entire spend feature space. High positive skewness (values substantially above 1.0) indicates that a small number of customers account for a disproportionate share of category-level spending. This has two direct consequences for preprocessing: first, lifetime spend values are converted to annual rates by dividing by tenure, to reduce the contribution of long-tenure customers to the skew; second, the consensus outlier separation rule is applied before clustering to set aside the most extreme multivariate observations. Skewness alone does not justify removing observations, but it confirms the need for careful outlier treatment before distance-based clustering is applied.</p>
</div>
""", unsafe_allow_html=True)

    # Chart: Numeric boxplots (from NB0)
    st.markdown("""
<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Numerical variable distributions: boxplots</div>
</div>
""", unsafe_allow_html=True)
    nb00_boxplot_path = IMAGENS_DIR / "charts" / "numeric_boxplots.png"
    if nb00_boxplot_path.exists():
        st.image(str(nb00_boxplot_path), use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The boxplots confirm the systematic right-skew identified in the histograms: every numerical variable shows a compact interquartile box close to the lower end of its range and a long upper whisker or visible outlier cloud extending far to the right. Median values across spending categories are low relative to the scale maximum, meaning that the majority of customers spend modestly while a thin tail of high-value customers accounts for most of the category-level variance. Variables such as total lifetime spend and electronics show particularly extreme upper outliers, reinforcing the need for the consensus outlier separation step prior to scaling. Age and tenure, by contrast, are more symmetric but still show a modest right tail in tenure, consistent with a customer base accumulated over multiple years of varying acquisition rates. These shapes collectively justify the MinMaxScaler choice over standard z-score normalisation: min-max compression bounds all features to [0, 1] while preserving the relative magnitude ordering within each variable, which is important when inter-feature comparisons are made in UMAP visualisations.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
        <div style='padding:24px; border-radius:16px; background:#f9fafb; border:1px solid #e5e7eb; margin-top:12px;'>
          <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:10px;'>Notebook 0: Conclusions</div>
          <p style='font-size:16px; color:#374151; line-height:1.9; margin:0 0 10px 0;'>The raw customer data contains missing values, skewed spending variables, identifier columns, date fields and several variables that require type conversion before modelling. These findings motivate the preprocessing notebook, where invalid values are handled, categorical variables are encoded and outliers are addressed.</p>
          <p style='font-size:16px; color:#374151; line-height:1.9; margin:0 0 10px 0;'>The distribution plots confirm that spending variables are highly skewed. This supports two later decisions: separating the most atypical customers before fitting the clustering model, and leaving scaling as a modelling choice.</p>
          <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The missing value inspection shows that the dataset is usable without dropping large parts of the customer base. No cleaning is applied in this notebook; all transformations are deferred to NB1.</p>
        </div>
    """, unsafe_allow_html=True)
    render_footer()

elif selected_page == "Data Preprocessing":
    st.markdown("""
    <div style='margin-top: 0px; margin-bottom: 24px;'>
        <h2 style='font-size: 56px; font-weight: 800; color: #000000; margin: 0; letter-spacing: -0.03em;'>Data Preprocessing</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-bottom:32px;'>
      <div>
        <div style='font-size:11px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#9ca3af; margin-bottom:10px;'>Notebook 1: Data Preprocessing</div>
        <p style='font-size:18px; color:#374151; line-height:1.9; margin:0 0 24px 0; text-align: justify;'>
          This notebook applies all transformations identified during the exploratory analysis. The goal is to produce a clean, analysis-ready dataset without losing customers unnecessarily. Every decision is justified by domain logic or statistical evidence; no arbitrary removals are made.
        </p>
      </div>

      <div style='background:linear-gradient(145deg, #ffffff, #f9fafb); border:1px solid #e5e7eb; border-radius:12px; padding:20px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05);'>
        <div style='font-size:14px; font-weight:800; color:#111827; margin-bottom:16px; border-bottom:2px solid #f3eee6; padding-bottom:8px;'>Notebook 1 Index</div>
        <div style='display:flex; flex-wrap:wrap; gap:12px; align-items:center;'>
          <a href="#nb1-1" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>1. Imports and Data Loading</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb1-2" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>2. Data Cleaning</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb1-2-1" style="text-decoration:none;"><div style='font-size:18px; color:#6b7280;'>2.1 Detecting Invalid Future Years</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb1-2-2" style="text-decoration:none;"><div style='font-size:18px; color:#6b7280;'>2.2 Detecting Negative Values</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb1-2-3" style="text-decoration:none;"><div style='font-size:18px; color:#6b7280;'>2.3 Fixing Negative Percentages</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb1-2-4" style="text-decoration:none;"><div style='font-size:18px; color:#6b7280;'>2.4 Fixing Data Types</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb1-2-5" style="text-decoration:none;"><div style='font-size:18px; color:#6b7280;'>2.5 Missing Values Treatment</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb1-3" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>3. Aggregation Feature Engineering</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb1-4" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>4. Consensus Outlier Separation</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb1-5" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>5. Transformation Feature Engineering</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb1-6" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>6. Outlier Diagnostics</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb1-7" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>7. Multivariate Analysis</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb1-8" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>8. Feature Selection</div></a>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
      <div id="nb1-1" style='border-top:1px solid #e5e7eb; padding-top:24px; margin-bottom:20px;'>
        <h2 style='font-size:24px; font-weight:800; color:#111827; margin:0 0 10px 0;'>1. Imports and Data Loading</h2>
        <p style='font-size:16px; color:#374151; line-height:1.9; margin:0 0 16px 0;'>
          The dataset is loaded and prepared for cleaning. Initial inspection of data types and general structure sets the foundation for the preprocessing pipeline.
        </p>
      </div>
      <!-- Stage 2: Data Cleaning -->
      <div id="nb1-2" style='border-top:1px solid #e5e7eb; padding-top:24px; margin-bottom:20px;'>
        <h2 style='font-size:24px; font-weight:800; color:#111827; margin:0 0 10px 0;'>2. Data Cleaning</h2>
        <p style='font-size:16px; color:#374151; line-height:1.9; margin:0 0 16px 0;'>
          Data cleaning and missing value treatment are crucial initial steps to prevent downstream model bias and mathematical distance calculation errors. In this phase, logical inconsistencies are corrected, raw types are structured, and missing entries are addressed using custom domain logic and KNN imputation.
        </p>
      </div>

      <div style='display:grid; grid-template-columns:repeat(2,1fr); gap:16px; margin-bottom:28px;'>
        <div id="nb1-2-1" style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:18px 20px;'>
          <div style='font-size:18px; font-weight:700; color:#c94f38; margin-bottom:4px;'>2.1 Detecting Invalid Future Years</div>
          <div style='font-size:16px; color:#7a6454; line-height:1.5;'>Future transaction years exceeding 2026 are logical errors. They are set to NaN rather than dropping the customer row.</div>
        </div>
        <div id="nb1-2-2" style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:18px 20px;'>
          <div style='font-size:18px; font-weight:700; color:#c94f38; margin-bottom:4px;'>2.2 Detecting Negative Values</div>
          <div style='font-size:16px; color:#7a6454; line-height:1.5;'>Numerical columns are scanned for negative values. Valid ones (e.g. longitude) are kept, but logical errors are set to NaN.</div>
        </div>
        <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:18px 20px;'>
          <div style='font-size:18px; font-weight:700; color:#c94f38; margin-bottom:4px;'>Zero-imputation for counts</div>
          <div style='font-size:16px; color:#7a6454; line-height:1.5;'>Missing values in kids_home, teens_home, and number_complaints are filled with 0, assuming lack of entry implies zero count.</div>
        </div>
        <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:18px 20px;'>
          <div style='font-size:18px; font-weight:700; color:#c94f38; margin-bottom:4px;'>Loyalty indicator flag</div>
          <div style='font-size:16px; color:#7a6454; line-height:1.5;'>Rather than discarding the 38% missing loyalty card numbers, a binary flag is engineered (1 if card exists, 0 otherwise).</div>
        </div>
      </div>

      <div style='margin-bottom:28px;'>
      <div style='margin-bottom:28px;'>
        <p id="nb1-2-3" style='font-size:16px; color:#374151; line-height:1.9; margin:0 0 12px 0;'>
          <strong>2.3 Fixing Negative Percentages:</strong> Promotion ratios outside the valid [0, 1] range are set to NaN, as they represent invalid measurements (e.g., negative values or values above 1).
        </p>
        <p id="nb1-2-4" style='font-size:16px; color:#374151; line-height:1.9; margin:0 0 12px 0;'>
          <strong>2.4 Fixing Data Types:</strong> Mixed types in object columns are coerced. For instance, <code>customer_birthdate</code> is parsed to <code>datetime64</code> to compute age, and counts like <code>kids_home</code> and <code>number_complaints</code> are cast to nullable integers to handle noise.
        </p>
        <p id="nb1-2-5" style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>
          <strong>2.5 Missing Values Treatment:</strong> Remaining missing values in variables such as typical hour, age, and spend categories are handled using a K-Nearest Neighbors (KNN) model with $k=5$. Imputation is only executed after outliers are separated to ensure that extreme multivariate observations do not bias the imputations of the regular customer base.
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    customer_info_pre = load_csv_data("customer_info.csv")
    info_unscaled = load_csv_data("info_clustering_unscaled.csv")
    outlier_df = load_csv_data("outlier_dataset.csv")

    current_year = 2026
    # Calculate counts of anomalies on raw data (kept for interpretation text reference)
    future_years = customer_info_pre[customer_info_pre["year_first_transaction"] > current_year]
    future_year_count = len(future_years)
    invalid_promo = customer_info_pre[(customer_info_pre["percentage_of_products_bought_promotion"] < 0) | (customer_info_pre["percentage_of_products_bought_promotion"] > 1)]
    invalid_promo_count = len(invalid_promo)

    # Raw missing values bar chart
    raw_missing = customer_info_pre.isna().sum().reset_index(name="missing_count")
    raw_missing.columns = ["Column", "Missing Count"]
    raw_missing["Missing %"] = (raw_missing["Missing Count"] / len(customer_info_pre)) * 100
    raw_missing = raw_missing[raw_missing["Missing Count"] > 0].sort_values("Missing %", ascending=False)
    
    st.markdown("""
      <div style='border-top:1px solid #e5e7eb; padding-top:24px; margin-bottom:10px;'>
        <h2 style='font-size:24px; font-weight:800; color:#111827; margin:0 0 10px 0;'>2.5 Missing Values Treatment</h2>
      </div>
<div style='margin-top:10px; margin-bottom:12px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Missing values percentage by feature before imputation (raw dataset)</div>
</div>
""", unsafe_allow_html=True)

    missing_chart = alt.Chart(raw_missing).mark_bar(color=SEGMENT_COLORS[3], cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        y=alt.Y("Column:N", sort="-x", title="Feature"),
        x=alt.X("Missing %:Q", title="Missing Percentage (%)"),
        tooltip=["Column", alt.Tooltip("Missing Count:Q", format=","), alt.Tooltip("Missing %:Q", format=".2f")]
    ).properties(height=280)
    st.altair_chart(missing_chart, use_container_width=True)

    # 2. Visual Heatmap of Missing Values (Plotly Heatmap with custom palette)
    df_missing_sample = customer_info_pre.sample(min(2500, len(customer_info_pre)), random_state=42).sort_index()
    null_matrix = df_missing_sample.isnull().astype(int)
    missing_cols = null_matrix.sum()[null_matrix.sum() > 0].index.tolist()
    
    if missing_cols:
        null_matrix_filtered = null_matrix[missing_cols]
        clean_cols = [col.replace("_", " ").title() for col in missing_cols]
        
        import plotly.graph_objects as go
        fig_heat = go.Figure(data=go.Heatmap(
            z=null_matrix_filtered.values,
            x=clean_cols,
            colorscale=[[0, '#fff8f2'], [1, '#c94f38']],
            showscale=False,
            ygap=0,
            xgap=0
        ))
        fig_heat.update_layout(
            margin=dict(l=100, r=40, t=15, b=40),
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis={'title': "Features with Missing Values", 'tickangle': 45},
            yaxis={'title': "Customers", 'showticklabels': False}
        )
        
        st.markdown("""
<div style='margin-top:20px; margin-bottom:12px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Visual map of missing values distribution (raw dataset sample)</div>
</div>
""", unsafe_allow_html=True)
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("""
<div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#7a6454; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The data cleaning shows specific logical errors (such as future transaction years and negative promotion percentages) that were successfully resolved. The missing value map confirms that <code>loyalty_card_number</code> contains the largest volume of missing values (approx 38%). Rather than removing these records or imputing card numbers, a binary loyalty indicator is engineered. Features like spend categories (meat, fish, vegetables) and typical hour have less than 3% missing rates. These are cleanly handled later by KNN Imputation (k=5) inside the regular base, preserving customer rows without losing significant volume.</p>
</div>
""", unsafe_allow_html=True)

    # Stage 2: Aggregation Feature Engineering
    st.markdown("""
      <!-- Stage 3: Aggregation Feature Engineering -->
      <div style='border-top:1px solid #e5e7eb; padding-top:24px; margin-bottom:20px;'>
        <div id="nb1-3" style='font-size:18px; font-weight:700; color:#111827; margin-bottom:10px;'>3. Aggregation Feature Engineering</div>
        <p style='font-size:16px; color:#374151; line-height:1.9; margin:0 0 16px 0;'>
          Before outlier separation and final transformation, broader demographic and loyalty features are created by parsing raw identifiers and dates:
        </p>
      </div>

      <div style='display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-bottom:28px;'>
        <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:18px 20px;'>
          <div style='font-size:18px; font-weight:700; color:#c94f38; margin-bottom:4px;'>Age calculation</div>
          <div style='font-size:16px; color:#7a6454; line-height:1.5;'>Age is calculated dynamically relative to the 2026 temporal baseline. Implausible ages (under 16 or over 100) are set to NaN.</div>
        </div>
        <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:18px 20px;'>
          <div style='font-size:18px; font-weight:700; color:#c94f38; margin-bottom:4px;'>Education level proxy</div>
          <div style='font-size:16px; color:#7a6454; line-height:1.5;'>Academic titles (BSc., MSc., PhD.) are extracted from names as a proxy for education level (years of study), and names are cleaned.</div>
        </div>
        <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:18px 20px;'>
          <div style='font-size:18px; font-weight:700; color:#c94f38; margin-bottom:4px;'>Gender binary mapping</div>
          <div style='font-size:16px; color:#7a6454; line-height:1.5;'>The raw customer_gender text field (male/female) is mapped to a binary indicator (is_male: 1 for male, 0 for female).</div>
        </div>
      </div>
    """, unsafe_allow_html=True)

    # Engineered categorical features summary (Gender, Loyalty, Education)
    gender_df = customer_info_pre["customer_gender"].value_counts().reset_index()
    gender_df.columns = ["Gender", "Customers"]
    gender_df["Gender"] = gender_df["Gender"].astype(str).str.title()
    
    gender_chart = alt.Chart(gender_df).mark_bar(color=SEGMENT_COLORS[7], cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Gender:N", title="Gender"),
        y=alt.Y("Customers:Q", title="Customers"),
        tooltip=["Gender", alt.Tooltip("Customers:Q", format=",")]
    ).properties(height=220)

    loyalty_df = info_unscaled["customer_loyalty_flag"].value_counts().reset_index()
    loyalty_df.columns = ["Loyalty", "Customers"]
    loyalty_df["Loyalty"] = loyalty_df["Loyalty"].map({1: "Loyal", 0: "Non-Loyal"})
    
    loyalty_chart = alt.Chart(loyalty_df).mark_bar(color=SEGMENT_COLORS[1], cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Loyalty:N", title="Loyalty Card Flag"),
        y=alt.Y("Customers:Q", title="Customers"),
        tooltip=["Loyalty", alt.Tooltip("Customers:Q", format=",")]
    ).properties(height=220)

    def get_edu_label(name):
        if pd.isna(name):
            return "High School (12y)"
        name = str(name).strip().lower()
        if name.startswith("bsc."):
            return "BSc (15y)"
        elif name.startswith("msc."):
            return "MSc (17y)"
        elif name.startswith("phd."):
            return "PhD (22y)"
        return "High School (12y)"

    edu_series = customer_info_pre["customer_name"].apply(get_edu_label)
    edu_df = edu_series.value_counts().reset_index()
    edu_df.columns = ["Education", "Customers"]
    
    edu_chart = alt.Chart(edu_df).mark_bar(color=SEGMENT_COLORS[3], cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Education:N", title="Education Level", sort=["High School (12y)", "BSc (15y)", "MSc (17y)", "PhD (22y)"]),
        y=alt.Y("Customers:Q", title="Customers"),
        tooltip=["Education", alt.Tooltip("Customers:Q", format=",")]
    ).properties(height=220)

    st.markdown("""
<div style='margin-top:10px; margin-bottom:12px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Distribution of engineered categorical features</div>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.altair_chart(gender_chart, use_container_width=True)
    with col2:
        st.altair_chart(loyalty_chart, use_container_width=True)
    with col3:
        st.altair_chart(edu_chart, use_container_width=True)

    st.markdown("""
<div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:28px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#7a6454; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The gender distribution is approximately balanced, with a slight male majority, indicating no significant gender-driven sampling bias. The loyalty flag distribution reveals that approximately 38% of customers had a missing loyalty card number; these are retained as a distinct binary class (no card) rather than excluded, preserving data volume. The education distribution, proxied from name prefixes, shows a broad range of qualification levels: BSc and MSc holders represent the largest shares, while PhD holders constitute a small but identifiable minority. These demographic features are not used as clustering inputs but serve as post-hoc segment validation variables in Notebook 4.</p>
</div>
""", unsafe_allow_html=True)

    # 3. Household distributions
    kids_df = info_unscaled["kids_home"].value_counts().reset_index()
    kids_df.columns = ["Kids", "Customers"]
    kids_chart = alt.Chart(kids_df).mark_bar(color=SEGMENT_COLORS[1], cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Kids:N", title="Kids at Home"),
        y=alt.Y("Customers:Q", title="Customers"),
        tooltip=["Kids", alt.Tooltip("Customers:Q", format=",")]
    ).properties(height=220)

    teens_df = info_unscaled["teens_home"].value_counts().reset_index()
    teens_df.columns = ["Teens", "Customers"]
    teens_chart = alt.Chart(teens_df).mark_bar(color=SEGMENT_COLORS[7], cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Teens:N", title="Teens at Home"),
        y=alt.Y("Customers:Q", title="Customers"),
        tooltip=["Teens", alt.Tooltip("Customers:Q", format=",")]
    ).properties(height=220)

    complaints_df = info_unscaled["number_complaints"].value_counts().reset_index()
    complaints_df.columns = ["Complaints", "Customers"]
    complaints_chart = alt.Chart(complaints_df).mark_bar(color=SEGMENT_COLORS[3], cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Complaints:N", title="Number of Complaints"),
        y=alt.Y("Customers:Q", title="Customers"),
        tooltip=["Complaints", alt.Tooltip("Customers:Q", format=",")]
    ).properties(height=220)

    st.markdown("""
<div style='margin-top:20px; margin-bottom:12px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Distribution of household and complaints variables (after zero-filling)</div>
</div>
""", unsafe_allow_html=True)

    col1_hh, col2_hh, col3_hh = st.columns(3)
    with col1_hh:
        st.altair_chart(kids_chart, use_container_width=True)
    with col2_hh:
        st.altair_chart(teens_chart, use_container_width=True)
    with col3_hh:
        st.altair_chart(complaints_chart, use_container_width=True)

    st.markdown("""
<div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:28px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#7a6454; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The kids and teens distributions are right-skewed, with most customers having zero or one child at home. The majority of the customer base has no children (approximately 60%), which is consistent with a broad urban supermarket customer profile. The complaints distribution is strongly concentrated at zero, with very few customers reporting more than one complaint, confirming that this is a low-complaint dataset and that the complaints variable will have limited discriminative power in clustering. Zero-filling was applied to all three variables where missing values resulted from data entry gaps rather than genuine absence.</p>
</div>
""", unsafe_allow_html=True)

    # Customer age distribution
    st.markdown("""
<div style='margin-top:20px; margin-bottom:12px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Engineered customer age distribution</div>
</div>
""", unsafe_allow_html=True)
    customer_info_pre["customer_birthdate"] = pd.to_datetime(customer_info_pre["customer_birthdate"], errors="coerce")
    customer_info_pre["customer_age"] = 2026 - customer_info_pre["customer_birthdate"].dt.year
    age_clean = customer_info_pre.dropna(subset=["customer_age"])
    age_hist = alt.Chart(age_clean).mark_bar(color=SEGMENT_COLORS[1], opacity=0.85).encode(
        x=alt.X("customer_age:Q", bin=alt.Bin(maxbins=25), title="Age (years)"),
        y=alt.Y("count():Q", title="Customers"),
        tooltip=[alt.Tooltip("count():Q", title="Customers")]
    ).properties(height=280)
    st.altair_chart(age_hist, use_container_width=True)

    st.markdown("""
<div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#7a6454; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The age distribution covers a wide spread (centering on 30 to 50 years). Using birthdates directly would introduce raw date formats that distance calculations cannot interpret. Calculating age dynamically relative to the 2026 temporal baseline resolves this. The categorical summaries also confirm that gender is almost evenly split, loyalty is map-encoded for 62% of customers, and name parsing successfully identifies academic prefixes (BSc, MSc, PhD) as education level proxies. Zero-filling kids, teens and complaints handles the missing values under the assumption that missing counts represent zero counts.</p>
</div>
""", unsafe_allow_html=True)

    # Stage 3: Outlier Separation — The Consensus Rule
    st.markdown("""
      <!-- Stage 4: Outlier Separation -->
      <div style='border-top:1px solid #e5e7eb; padding-top:24px; margin-bottom:20px;'>
        <div id="nb1-4" style='font-size:18px; font-weight:700; color:#111827; margin-bottom:10px;'>4. Consensus Outlier Separation and KNN Imputation</div>
        <p style='font-size:16px; color:#374151; line-height:1.9; margin:0 0 16px 0;'>
          Rather than capping or removing outliers based on a single method, a <strong>conservative consensus rule</strong> is applied: a customer is set aside only when simultaneously flagged as an outlier by all three of the following methods:
        </p>
      </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:12px;'>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:10px; padding:14px 16px; text-align:center;'>
            <div style='font-size:18px; font-weight:700; color:#c94f38;'>IQR</div>
            <div style='font-size:24px; font-weight:800; color:#c94f38; margin:4px 0;'>k = 2.0</div>
            <div style='font-size:12px; color:#7a6454;'>multiplier</div>
          </div>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:10px; padding:14px 16px; text-align:center;'>
            <div style='font-size:18px; font-weight:700; color:#c94f38;'>DBSCAN</div>
            <div style='font-size:24px; font-weight:800; color:#c94f38; margin:4px 0;'>eps = 1.0</div>
            <div style='font-size:12px; color:#7a6454;'>neighbourhood radius</div>
          </div>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:10px; padding:14px 16px; text-align:center;'>
            <div style='font-size:18px; font-weight:700; color:#c94f38;'>3rd method</div>
            <div style='font-size:24px; font-weight:800; color:#c94f38; margin:4px 0;'>&cap;</div>
            <div style='font-size:12px; color:#7a6454;'>all three must agree</div>
          </div>
        </div>
        <div style='display:grid; grid-template-columns:repeat(2,1fr); gap:14px; margin:16px 0;'>
          <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:14px 16px;'>
            <div style='font-size:14px; font-weight:700; color:#111827; margin-bottom:5px;'>Why a consensus rule instead of a single method?</div>
            <div style='font-size:16px; color:#374151; line-height:1.6;'>Any single outlier detector has blind spots. IQR flags univariate extremes; DBSCAN flags multivariate density gaps. A customer with unusually high electronics spend but otherwise normal behaviour would be caught by IQR but not DBSCAN. The consensus requirement means only genuine multivariate extremes, flagged by all methods, are removed, keeping the separation rate conservative (≈3%).</div>
          </div>
          <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:14px 16px;'>
            <div style='font-size:14px; font-weight:700; color:#111827; margin-bottom:5px;'>Why separate instead of cap or remove?</div>
            <div style='font-size:16px; color:#374151; line-height:1.6;'>Capping compresses extreme values to the 99th percentile, distorting genuine behaviour. Permanent removal loses the customer. Separation keeps the outlier intact in its own dataset and reattaches it to the nearest cluster centroid after model fitting, so every customer gets a segment label and no information is lost.</div>
          </div>
          <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:14px 16px;'>
            <div style='font-size:14px; font-weight:700; color:#111827; margin-bottom:5px;'>Why KNN imputation (k=5)?</div>
            <div style='font-size:16px; color:#374151; line-height:1.6;'>KNN imputation replaces missing values with the mean of the five most similar customers in feature space, preserving the local correlation structure. It outperforms mean/median imputation for behavioural data because a missing typical-hour for an electronics-heavy customer should be inferred from other electronics-heavy customers, not from the global average. Imputation is run after outlier separation to prevent extreme customers from distorting neighbour lookups.</div>
          </div>
          <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:14px 16px;'>
            <div style='font-size:14px; font-weight:700; color:#111827; margin-bottom:5px;'>Why not log-transform spending?</div>
            <div style='font-size:16px; color:#374151; line-height:1.6;'>Log transforms collapse the tail of the spend distribution, reducing the distance between moderate and high spenders. For this dataset, the goal is precisely to preserve spending magnitude differences: Techies should be far from Economizers in feature space. MinMaxScaler on the post-separation distribution retains those differences while controlling scale, making log transformation unnecessary.</div>
          </div>
        </div>
        <p style='font-size:16px; color:#374151; line-height:1.9; margin:14px 0;'>
          Customers flagged by all three methods are exported to <code>outlier_dataset.csv</code>. The regular base is then processed with KNN imputation. This approach ensures that only multivariate extremes are removed: customers with one extreme variable but otherwise normal behaviour are retained. Outliers are later <strong>reattached to their nearest cluster centroid</strong> after the model is fitted.
        </p>
    """, unsafe_allow_html=True)

    # Dataset split after consensus outlier separation
    st.markdown("""
<div style='margin-top:20px; margin-bottom:12px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Dataset split after consensus outlier separation</div>
</div>
""", unsafe_allow_html=True)
    split_df = pd.DataFrame({
        "Dataset": ["Regular base", "Outlier dataset"],
        "Customers": [len(info_unscaled), len(outlier_df)]
    })
    split_chart = alt.Chart(split_df).mark_bar(color=SEGMENT_COLORS[1], cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Dataset:N", title=""),
        y=alt.Y("Customers:Q", title="Number of customers"),
        tooltip=["Dataset", alt.Tooltip("Customers:Q", format=",")]
    ).properties(height=280)
    st.altair_chart(split_chart, use_container_width=True)

    st.markdown("""
<div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#7a6454; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The chart demonstrates the conservative nature of the consensus outlier rule: the overwhelming majority of customers remain in the regular base, with only 982 customers (approx. 3.0%) separated. This conservative threshold avoids over-excluding customers who are unusual on a single variable but unremarkable in the multivariate space. Outliers are reattached to their nearest cluster centroid after model fitting.</p>
</div>
""", unsafe_allow_html=True)

    # Stage 4: Transformation Feature Engineering
    st.markdown("""
      <!-- Stage 5: Transformation Feature Engineering -->
      <div style='border-top:1px solid #e5e7eb; padding-top:24px; margin-bottom:20px;'>
        <div id="nb1-5" style='font-size:18px; font-weight:700; color:#111827; margin-bottom:10px;'>5. Transformation Feature Engineering</div>
        <p style='font-size:16px; color:#374151; line-height:1.9; margin:0 0 16px 0;'>
          After outlier separation, mathematical transformations are applied to prepare the variables for modelling, ensuring distance metrics operate correctly across circular dimensions and spending volume:
        </p>
      </div>

      <div style='display:grid; grid-template-columns:repeat(2,1fr); gap:16px; margin-bottom:28px;'>
        <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:18px 20px;'>
          <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#b07060; margin-bottom:6px;'>Lifecycle feature</div>
          <div style='font-size:16px; font-weight:700; color:#c94f38; margin-bottom:6px;'>tenure (years of relationship)</div>
          <div style='font-size:16px; color:#374151; line-height:1.6;'>Raw <code>year_first_transaction</code> is an absolute year label (e.g. 2014), meaninglessss as a distance metric. Converting it to years-since-first-purchase produces a continuous, interpretable feature (e.g. 9.3 years) that captures customer lifecycle stage. Longer tenure is associated with higher loyalty, so this feature is a direct input to loyalty-based profiling.</div>
        </div>
        <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:18px 20px;'>
          <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#b07060; margin-bottom:6px;'>Circular encoding</div>
          <div style='font-size:16px; font-weight:700; color:#c94f38; margin-bottom:6px;'>Cyclic typical shopping hour (sin/cos)</div>
          <div style='font-size:16px; color:#374151; line-height:1.6;'>Hour is circular: 23:00 and 00:00 are adjacent but numerically 23 apart. Using raw hours would tell K-Means that a morning shopper (hour 8) and a late-night shopper (hour 22) are closer than midnight shoppers on opposite sides of midnight. Sin/cos encoding maps the 24-hour clock onto a unit circle so that all adjacent hours are equidistant in Euclidean space.</div>
        </div>
        <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:18px 20px;'>
          <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#b07060; margin-bottom:6px;'>Tenure normalisation</div>
          <div style='font-size:16px; font-weight:700; color:#c94f38; margin-bottom:6px;'>Annual spend rates (spend ÷ tenure)</div>
          <div style='font-size:16px; color:#374151; line-height:1.6;'>A customer who joined 15 years ago will have higher lifetime spend than an identical-behaviour customer who joined 2 years ago, purely because they have had more time to accumulate transactions. Dividing lifetime spend by tenure normalises for relationship length, making the spend features reflect behavioural intensity rather than longevity. Without this, long-tenure customers would cluster together even if their annual behaviour is unremarkable.</div>
        </div>
        <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:18px 20px;'>
          <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#b07060; margin-bottom:6px;'>Dimensionality reduction</div>
          <div style='font-size:16px; font-weight:700; color:#c94f38; margin-bottom:6px;'>Technology spend aggregate</div>
          <div style='font-size:16px; color:#374151; line-height:1.6;'><code>lifetime_spend_electronics</code> and <code>lifetime_spend_videogames</code> are highly correlated (both driven by tech-oriented customers) and individually sparse, as most customers spend zero on each. Aggregating them into a single <code>lifetime_spend_technology</code> feature reduces noise and creates one informative dimension that separates tech-oriented customers cleanly in the clustering space.</div>
        </div>
      </div>
    """, unsafe_allow_html=True)

    # Typical hour cyclic encoding scatter plot
    st.markdown("""
<div style='margin-top:20px; margin-bottom:12px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Cyclic typical shopping hour scatter plot</div>
</div>
""", unsafe_allow_html=True)
    hour_scatter = alt.Chart(info_unscaled.sample(min(5000, len(info_unscaled)), random_state=42)).mark_circle(color=SEGMENT_COLORS[3], opacity=0.4, size=15).encode(
        x=alt.X("typical_hour_sin:Q", title="sin(hour)", scale=alt.Scale(domain=[-1.1, 1.1])),
        y=alt.Y("typical_hour_cos:Q", title="cos(hour)", scale=alt.Scale(domain=[-1.1, 1.1])),
        tooltip=["typical_hour_sin", "typical_hour_cos"]
    ).properties(height=340)
    st.altair_chart(hour_scatter, use_container_width=True)

    st.markdown("""
<div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#7a6454; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Using raw hours directly (0 to 23) introduces a boundary error where 23:00 and 00:00 appear maximally distant (difference of 23). The cyclic encoding resolves this by plotting them on a circle, ensuring hour 23 and hour 0 are correctly recognized as adjacent.</p>
</div>
""", unsafe_allow_html=True)

    # Stage 5: Outlier Diagnostics
    st.markdown("""
      <!-- Stage 6: Outlier Diagnostics -->
      <div style='border-top:1px solid #e5e7eb; padding-top:24px; margin-bottom:20px;'>
        <div id="nb1-6" style='font-size:18px; font-weight:700; color:#111827; margin-bottom:10px;'>6. Outlier Diagnostics</div>
        <p style='font-size:16px; color:#374151; line-height:1.9; margin:0 0 16px 0;'>
          The plots below are used as a final visual check of the main skewed variables after preprocessing. At this point, the extreme consensus subset has already been separated, so these diagnostics help confirm whether the regular customer base is more stable before the clustering stage.
        </p>
      </div>
    """, unsafe_allow_html=True)

    # Spend boxplots after preprocessing
    st.markdown("""
<div style='margin-top:20px; margin-bottom:12px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Lifetime spend per category after preprocessing</div>
</div>
""", unsafe_allow_html=True)
    annual_cols = ["lifetime_spend_groceries", "lifetime_spend_electronics", "lifetime_spend_vegetables", "lifetime_spend_meat", "lifetime_spend_fish", "lifetime_spend_hygiene"]
    spend_melt = info_unscaled[annual_cols].melt(var_name="category", value_name="annual_spend")
    spend_melt["category"] = spend_melt["category"].str.replace("lifetime_spend_", "", regex=False).str.replace("_", " ").str.title()
    spend_melt = spend_melt.dropna(subset=["annual_spend"])
    box_chart_pre = alt.Chart(spend_melt).mark_boxplot(color=SEGMENT_COLORS[7], outliers={"size": 4, "opacity": 0.2}).encode(
        x=alt.X("category:N", title="Category"),
        y=alt.Y("annual_spend:Q", title="Annual spend"),
        tooltip=["category"]
    ).properties(height=360)
    st.altair_chart(box_chart_pre, use_container_width=True)

    st.markdown("""
<div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#7a6454; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Some high values remain visible in the boxplots after preprocessing. This is expected because lifetime spending variables are naturally skewed. The goal was not to remove every univariate extreme value, but to keep valid high spending customers while separating only the most atypical multivariate cases. This confirms that the regular customer base is more stable before the clustering stage.</p>
</div>
""", unsafe_allow_html=True)

    # Stage 6: Multivariate Analysis: Feature Correlation
    st.markdown("""
      <!-- Stage 7: Multivariate Analysis: Feature Correlation -->
      <div style='border-top:1px solid #e5e7eb; padding-top:24px; margin-bottom:20px;'>
        <div id="nb1-7" style='font-size:18px; font-weight:700; color:#111827; margin-bottom:10px;'>7. Multivariate Analysis: Feature Correlation</div>
        <p style='font-size:16px; color:#374151; line-height:1.9; margin:0 0 16px 0;'>
          Highly redundant features can distort distance metrics by double-weighting similar signals. The correlation matrix is inspected to ensure no features exceed a collinearity threshold of 0.7.
        </p>
      </div>
    """, unsafe_allow_html=True)

    corr_features = [
        "lifetime_spend_groceries", "lifetime_spend_electronics", "lifetime_spend_vegetables",
        "lifetime_spend_meat", "lifetime_spend_fish", "lifetime_spend_hygiene",
        "lifetime_spend_videogames", "percentage_of_products_bought_promotion",
        "tenure", "total_children"
    ]
    corr_matrix = info_unscaled[corr_features].corr()
    corr_labels = [f.replace("lifetime_spend_", "").replace("_", " ").title() for f in corr_features]
    custom_corr_scale = ["#6B7D7D", "#fcfbf8", "#9D5C4A"]
    corr_fig = px.imshow(
        corr_matrix.values,
        x=corr_labels,
        y=corr_labels,
        color_continuous_scale=custom_corr_scale,
        color_continuous_midpoint=0,
        zmin=-1,
        zmax=1,
        text_auto=".2f"
    )
    corr_fig.update_layout(margin=dict(l=60, r=20, t=60, b=60), height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(corr_fig, use_container_width=True)

    st.markdown("""
<div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#7a6454; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Most spend categories correlate weakly, confirming they carry distinct signals. The strongest positive correlation is between meat and fish (0.42), well below the 0.7 feature-exclusion threshold. Promotional sensitivity shows near-zero correlation with spend categories, supporting its use as an independent clustering variable.</p>
</div>
""", unsafe_allow_html=True)

    # Full correlation heatmap from raw data
    st.markdown("""
<div style='margin-top:20px; margin-bottom:12px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Full correlation heatmap (raw features)</div>
</div>
""", unsafe_allow_html=True)
    import numpy as np
    numeric_df = info_unscaled.select_dtypes(include=['number'])
    if 'customer_id' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['customer_id'])
    full_corr = numeric_df.corr()
    mask = np.triu(np.ones_like(full_corr, dtype=bool))
    full_corr_masked = full_corr.mask(mask)
    
    full_corr_labels = [c.replace('_', ' ').title() for c in full_corr.columns]
    
    custom_corr_scale = ["#6B7D7D", "#fcfbf8", "#9D5C4A"]
    full_corr_fig = px.imshow(
        full_corr_masked.values,
        x=full_corr_labels,
        y=full_corr_labels,
        color_continuous_scale=custom_corr_scale,
        color_continuous_midpoint=0,
        zmin=-1,
        zmax=1
    )
    full_corr_fig.update_layout(margin=dict(l=60, r=20, t=60, b=100), height=800, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(full_corr_fig, use_container_width=True)
        
    st.markdown("""
<div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#7a6454; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The full-feature correlation heatmap covers all variables in the raw dataset. It validates that spending variables are relatively independent, and that demographic age/tenure do not share a strong linear association, preventing redundancy in model features.</p>
</div>
""", unsafe_allow_html=True)

    # Stage 7: Feature Selection and Final Export
    st.markdown("""
      <!-- Stage 8: Feature Selection and Final Export -->
      <div style='border-top:1px solid #e5e7eb; padding-top:24px; margin-bottom:20px;'>
        <div id="nb1-8" style='font-size:18px; font-weight:700; color:#111827; margin-bottom:10px;'>8. Feature Selection and Final Export</div>
        <p style='font-size:16px; color:#374151; line-height:1.9; margin:0 0 16px 0;'>
          This final step prepares the dataset that will be used in the clustering notebook. Keeping the export step separate makes the transition from preprocessing to clustering clear and reproducible.
        </p>
      </div>

      <div style='display:grid; grid-template-columns:repeat(2,1fr); gap:16px; margin-bottom:20px;'>
        <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:18px 20px;'>
          <div style='font-size:18px; font-weight:700; color:#c94f38; margin-bottom:4px;'>info_clustering_unscaled.csv</div>
          <div style='font-size:16px; color:#7a6454; line-height:1.5;'>The main unscaled clustering dataset (32,056 customers) containing absolute spend features, tenure, and cyclic typical-hour components.</div>
        </div>
        <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:18px 20px;'>
          <div style='font-size:18px; font-weight:700; color:#c94f38; margin-bottom:4px;'>outlier_dataset.csv</div>
          <div style='font-size:16px; color:#7a6454; line-height:1.5;'>The separated extreme customer subset (982 customers, approx 3.0% of base). These are kept aside during training and later reattached to their nearest cluster centroid.</div>
        </div>
      </div>
    """, unsafe_allow_html=True)

    # Why export unscaled card
    st.markdown("""
      <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:20px 24px; margin-top:16px; margin-bottom:32px;'>
        <div style='font-size:18px; font-weight:700; color:#c94f38; margin-bottom:6px;'>Why export unscaled?</div>
        <div style='font-size:16px; color:#7a6454; line-height:1.8;'>The final dataset is exported <strong>without scaling</strong>. Scaling is treated as a modelling choice: MinMaxScaler and RobustScaler are both evaluated in Notebook 3 against each other. Keeping the export unscaled ensures that the same raw dataset can be tested against any scaling strategy without re-running preprocessing.</div>
      </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='padding:24px; border-radius:16px; background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); margin-top:12px;'>
          <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#7a6454; margin-bottom:10px;'>Notebook 1: Conclusions</div>
          <p style='font-size:16px; color:#374151; line-height:1.9; margin:0 0 10px 0;'>Before imputing missing values, the most atypical customers are separated into an outlier dataset. The rule is conservative: a customer is only kept aside when it is simultaneously flagged by three methods.</p>
          <p style='font-size:16px; color:#374151; line-height:1.9; margin:0 0 10px 0;'>After the conservative outlier separation, the regular customer base still contains natural variation, but the most extreme observations are kept aside. This reduces the risk that a small number of atypical customers dominate the clustering distances.</p>
          <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>In the final exported dataset, <code>year_first_transaction</code> is replaced by <code>tenure</code>, and <code>typical_hour</code> is replaced by its cyclic components. Some high values remain visible in the boxplots after preprocessing. This is expected because lifetime spending variables are naturally skewed.</p>
        </div>
    """, unsafe_allow_html=True)
    render_footer()

elif selected_page == "Data in Geography":
    st.markdown("""
    <div style='margin-top: 0px; margin-bottom: 24px;'>
        <h2 style='font-size: 56px; font-weight: 800; color: #000000; margin: 0; letter-spacing: -0.03em;'>Data In Geography</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='display:flex; gap:24px; margin-bottom:32px; align-items:flex-start; flex-wrap:wrap;'>
      <div style='flex:1; min-width:300px;'>
    <div style='margin-bottom:32px;'>
      <div>
        <div style='font-size:11px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#9ca3af; margin-bottom:10px;'>Notebook 2: Geographic Analysis</div>
        <p style='font-size:18px; color:#374151; line-height:1.9; margin:0 0 24px 0; text-align: justify;'>
          With the data cleaned, Notebook 2 isolates the two geographic variables, latitude and longitude, to understand the spatial distribution of the customer base. The primary objective is to detect spatial patterns and concentrations that might correlate with distinct spending behaviours or demographics.
        </p>
        <div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:16px; margin-bottom:28px;'>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:18px 20px;'>
            <div style='font-size:18px; font-weight:700; color:#c94f38; margin-bottom:4px;'>Core spatial pattern</div>
            <div style='font-size:16px; color:#7a6454; line-height:1.6;'>Strong concentration along a specific linear band, interpreted as a major coastal or river-adjacent urban corridor (consistent with Lisbon/Cascais profile).</div>
          </div>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:18px 20px;'>
            <div style='font-size:18px; font-weight:700; color:#c94f38; margin-bottom:4px;'>Dominant age group</div>
            <div style='font-size:16px; color:#7a6454; line-height:1.6;'>The <strong>25–34</strong> age band dominates the hotspot profile.</div>
          </div>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:18px 20px;'>
            <div style='font-size:18px; font-weight:700; color:#c94f38; margin-bottom:4px;'>Hotspot radius</div>
            <div style='font-size:16px; color:#7a6454; line-height:1.6;'><strong>0.006 decimal degrees</strong>, defined programmatically via grid-based analysis.</div>
          </div>
        </div>
      </div>

      <div style='background:linear-gradient(145deg, #ffffff, #f9fafb); border:1px solid #e5e7eb; border-radius:12px; padding:20px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05); margin-top:20px;'>
        <div style='font-size:14px; font-weight:800; color:#111827; margin-bottom:16px; border-bottom:2px solid #f3eee6; padding-bottom:8px;'>Notebook 2 Index</div>
        <div style='display:flex; flex-wrap:wrap; gap:12px; align-items:center;'>
          <a href="#nb2-1" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>1) Imports & 2) Data loading</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb2-3" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>3) Basic geographic statistics</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb2-4" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>4) Customer geographic distribution</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb2-insights" style="text-decoration:none;"><div style='font-size:18px; color:#6b7280;'>Insights from Geospatial Mapping</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb2-5" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>5) Customer density map</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb2-6" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>6) Hotspot profile</div></a>
        </div>
      </div>
    </div>

    <div id="nb2-1" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">1) Imports & 2) Data loading</h2></div>
    <div style='border-left:3px solid #111827; padding-left:20px; margin-bottom:20px;'>
      <div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>Data Preparation</div>
      <div style='font-size:16px; color:#6b7280; line-height:1.8;'>The geographic coordinates (latitude and longitude) are loaded from the customer information dataset. Customers with missing geographic data are excluded from this specific spatial analysis to ensure mapping integrity.</div>
    </div>

    <div id="nb2-3" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">3) Basic geographic statistics</h2></div>
    <div style='border-left:3px solid #111827; padding-left:20px; margin-bottom:20px;'>
      <div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>Coordinate range</div>
      <div style='font-size:16px; color:#6b7280; line-height:1.8;'>The spatial bounding box confirms that all valid customer coordinates fall within the expected region. There are no erroneous locations (e.g. coordinates in the ocean) that require cleaning.</div>
    </div>

    <div id="nb2-insights" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">Insights from Geospatial Mapping</h2></div>

    <div style='border-left:3px solid #111827; padding-left:20px; margin-bottom:20px;'>
      <div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>Behavioural differences: hotspot vs. rest of base</div>
      <div style='font-size:16px; color:#6b7280; line-height:1.8;'>The hotspot shows a distinct behavioural profile even before clustering labels are applied. The strongest differences are in <strong>age, product diversity, number of complaints, store visits, total spend, and promotion usage</strong>. Hotspot customers are younger, more active, and more variety-seeking, consistent with a younger urban population, though the data does not confirm student status directly.</div>
    </div>

    <div style='border-left:3px solid #111827; padding-left:20px;'>
      <div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>Why geography is excluded from clustering</div>
      <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Including geographic coordinates in the clustering distance would create spatially-defined groups: customers near each other in space would be forced into the same cluster regardless of their spending behaviour. The objective is to discover <em>behavioural</em> communities, not geographic ones. Geography is kept as a profiling tool: after clusters are fitted, the geographic distribution of each cluster is inspected as a validation and characterisation layer.</div>
    </div>
    """, unsafe_allow_html=True)

    customer_info = load_csv_data("customer_info.csv")
    customer_info = customer_info.dropna(subset=["latitude", "longitude"]).copy()
    customer_info["promo_ratio"] = customer_info["percentage_of_products_bought_promotion"] * 100
    customer_info["size_spend"] = customer_info["lifetime_total_distinct_products"].fillna(0) / 50

    st.markdown("""<div id='nb2-4' style='margin-top:64px; margin-bottom:16px; border-top:2px solid #e5e7eb; padding-top:32px;'><h2 style='font-size:24px; font-weight:800; color:#111827; margin:0;'>4) Customer geographic distribution</h2></div>
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:14px 18px; margin-bottom:20px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:6px;'>Why this chart?</div>
  <div style='font-size:16px; color:#374151; line-height:1.7;'>Each dot is one customer, positioned at their coordinates, sized by total lifetime spend and coloured by promotion usage rate. This reveals whether high-spend or promo-heavy customers concentrate in particular zones, a finding that the density map alone cannot show because density aggregates all customers equally regardless of value.</div>
</div>""", unsafe_allow_html=True)
    scatter_map = px.scatter_mapbox(
        customer_info,
        lat="latitude",
        lon="longitude",
        color="promo_ratio",
        size="size_spend",
        size_max=8,
        zoom=9,
        mapbox_style="open-street-map",
        color_continuous_scale=list(SEGMENT_COLORS.values()),
        hover_data={"customer_gender":True, "promo_ratio":True, "lifetime_total_distinct_products":True},
    )
    scatter_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=dict(title="Promo %"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(scatter_map, width=1080, config={"scrollZoom": True})

    st.markdown("""
<div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:28px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#7a6454; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The scatter map confirms that the customer base is geographically concentrated in the Lisbon Metropolitan Area, with a visible urban core and a sparser suburban periphery. Point size encodes lifetime spend, where larger points indicate higher-value customers, while colour encodes promotion sensitivity, revealing that higher promotion usage is not exclusively an urban or suburban phenomenon but is distributed across the full geographic footprint. No distinct geographic cluster of uniformly high- or low-promotion customers is visible, supporting the decision to exclude geography from the clustering feature set.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""<div id='nb2-5' style='margin-top:40px; margin-bottom:16px; border-top:2px solid #e5e7eb; padding-top:32px;'><h2 style='font-size:24px; font-weight:800; color:#111827; margin:0 0 12px 0;'>5) Customer density map</h2>
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:14px 18px; margin-bottom:20px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:6px;'>Why this chart?</div>
  <div style='font-size:16px; color:#374151; line-height:1.7;'>The density map aggregates individual coordinates into a continuous heatmap, making the urban concentration visible even where individual dots overlap and become illegible. It confirms the Lisbon Metropolitan Area hotspot and shows the suburban periphery thinning out, supporting the geographic bounding-box and hotspot definitions used in subsequent analysis.</div>
</div></div>""", unsafe_allow_html=True)
    density_map = px.density_mapbox(
        customer_info,
        lat="latitude",
        lon="longitude",
        radius=15,
        zoom=9,
        mapbox_style="open-street-map",
        color_continuous_scale=list(SEGMENT_COLORS.values()),
        hover_data={"promo_ratio":True},
    )
    density_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(density_map, width=1080, config={"scrollZoom": True})

    st.markdown("""
<div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:28px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#7a6454; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The density map reveals a single dominant hotspot centred on central Lisbon, with a secondary concentration along the northern corridor of the metropolitan area. The gradient from peak density (core) to near-zero density (outer periphery) is steep, indicating that the majority of the customer base shops from a limited geographic radius. This concentration pattern is consistent with a single or small number of store locations rather than a geographically dispersed retail network, and it justifies the use of a city-level bounding box for geographic analysis.</p>
</div>
""", unsafe_allow_html=True)

    customer_info["lat_bin"] = (customer_info["latitude"] * 20).round(0) / 20
    customer_info["lon_bin"] = (customer_info["longitude"] * 20).round(0) / 20
    grid = customer_info.groupby(["lat_bin", "lon_bin"]).size().reset_index(name="count")
    hotspot = grid.sort_values("count", ascending=False).head(1).iloc[0]
    hotspot_customers = customer_info[(customer_info.lat_bin == hotspot.lat_bin) & (customer_info.lon_bin == hotspot.lon_bin)]
    outside_customers = customer_info[~((customer_info.lat_bin == hotspot.lat_bin) & (customer_info.lon_bin == hotspot.lon_bin))]
    hotspot_summary = pd.DataFrame({
        "metric": ["Promo ratio", "Distinct products", "Stores visited"],
        "Hotspot": [
            hotspot_customers["promo_ratio"].mean(),
            hotspot_customers["lifetime_total_distinct_products"].mean(),
            hotspot_customers["distinct_stores_visited"].mean(),
        ],
        "Outside": [
            outside_customers["promo_ratio"].mean(),
            outside_customers["lifetime_total_distinct_products"].mean(),
            outside_customers["distinct_stores_visited"].mean(),
        ],
    })
    hotspot_compare = hotspot_summary.melt(id_vars=["metric"], var_name="group", value_name="value")

    compare_chart = alt.Chart(hotspot_compare).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("metric:N", title="Metric"),
        y=alt.Y("value:Q", title="Value"),
        color=alt.Color("group:N", title="Group", scale=alt.Scale(range=[SEGMENT_COLORS[0], SEGMENT_COLORS[1]])),
        column=alt.Column("metric:N", header=alt.Header(labelAngle=0, labelAlign="left", title=""))
    ).properties(height=260)

    st.markdown("<div id='nb2-6'></div>", unsafe_allow_html=True)
    st.subheader("6) Hotspot profile vs outside")
    st.altair_chart(compare_chart, use_container_width=True)

    st.markdown("""
<div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:28px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#7a6454; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The hotspot-vs-outside comparison reveals that customers located within the highest-density geographic cell show modestly higher promotion sensitivity and purchase a broader range of distinct products on average. The number of distinct stores visited is also slightly higher in the hotspot, consistent with a more urban, multi-channel shopping pattern. These differences are present but not large enough to constitute a distinct behavioural segment; they reflect geographic concentration effects rather than fundamentally different customer profiles, reinforcing the decision to treat the hotspot as a geographic observation rather than a behavioural cluster.</p>
</div>
""", unsafe_allow_html=True)

    # Chart: Age distribution hotspot vs outside
    st.markdown("""
<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Age distribution: hotspot vs. rest of base</div>
</div>
""", unsafe_allow_html=True)
    customer_info_geo = load_csv_data("customer_info.csv")
    customer_info_geo = customer_info_geo.dropna(subset=["latitude", "longitude"]).copy()
    customer_info_geo["customer_birthdate"] = pd.to_datetime(customer_info_geo["customer_birthdate"], errors="coerce")
    customer_info_geo["customer_age"] = 2026 - customer_info_geo["customer_birthdate"].dt.year
    customer_info_geo["lat_bin"] = (customer_info_geo["latitude"] * 20).round(0) / 20
    customer_info_geo["lon_bin"] = (customer_info_geo["longitude"] * 20).round(0) / 20
    geo_grid = customer_info_geo.groupby(["lat_bin", "lon_bin"]).size().reset_index(name="count")
    geo_hotspot = geo_grid.sort_values("count", ascending=False).head(1).iloc[0]
    customer_info_geo["geo_group"] = "Rest of base"
    customer_info_geo.loc[
        (customer_info_geo.lat_bin == geo_hotspot.lat_bin) & (customer_info_geo.lon_bin == geo_hotspot.lon_bin),
        "geo_group"
    ] = "Hotspot"
    age_geo_clean = customer_info_geo.dropna(subset=["customer_age"])
    age_hotspot_chart = alt.Chart(age_geo_clean).mark_bar(opacity=0.6).encode(
        x=alt.X("customer_age:Q", bin=alt.Bin(maxbins=25), title="Age (years)"),
        y=alt.Y("count():Q", stack=None, title="Customers"),
        color=alt.Color("geo_group:N", scale=alt.Scale(domain=["Hotspot", "Rest of base"], range=[SEGMENT_COLORS[0], SEGMENT_COLORS[1]]), title="Group"),
        tooltip=["geo_group", alt.Tooltip("count():Q", title="Customers")]
    ).properties(height=320)
    st.altair_chart(age_hotspot_chart, use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The overlaid age histograms reveal a pronounced difference in age composition between the hotspot and the rest of the customer base. The hotspot skews clearly toward the 25 to 34 age band, while the broader base has a flatter distribution extending through the 40 to 55 range. This age gap is the most statistically robust finding of the geographic analysis: it corroborates the hypothesis that the dense area near Cidade Universitaria and Entrecampos attracts a younger urban population, consistent with proximity to university infrastructure and high-density residential areas. The finding is treated as an interpretive insight rather than an actionable segment boundary, because geography is deliberately excluded from the clustering distance. The age skew observed in the hotspot does not define a segment; rather, it adds spatial context to the behavioural profiles that the clustering model identifies independently.</p>
</div>
""", unsafe_allow_html=True)

    # Chart: Static scatter (NB2)
    st.markdown("""
<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Geographic distribution of all customers (raw)</div>
</div>
""", unsafe_allow_html=True)

    import matplotlib.pyplot as plt
    fig_geo, ax_geo = plt.subplots(figsize=(10, 6.5))
    ax_geo.scatter(
        customer_info['longitude'], 
        customer_info['latitude'], 
        s=3, 
        alpha=0.7, 
        color='#2563EB'
    )
    ax_geo.set_xlabel('Longitude', fontsize=10, color='#7a6454')
    ax_geo.set_ylabel('Latitude', fontsize=10, color='#7a6454')
    ax_geo.tick_params(colors='#7a6454', labelsize=9)
    ax_geo.grid(True, linestyle='--', alpha=0.3)
    
    # Set border color
    for spine in ax_geo.spines.values():
        spine.set_color((111/255, 79/255, 53/255, 0.16))
    
    fig_geo.patch.set_facecolor('none')
    ax_geo.set_facecolor('none')
    plt.tight_layout()
    
    st.pyplot(fig_geo)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>This static scatter plot represents every customer as a point at their recorded latitude and longitude coordinates, plotted without any segmentation labels. The visualisation reveals that the customer base is heavily concentrated in a narrow geographic band consistent with the Lisbon Metropolitan Area, with a pronounced density peak in the central zone. The clustering of points is not uniform: a high-density region emerges in the centre-north quadrant of the scatter, corresponding to the urban core identified in the interactive maps above. Surrounding this core, points become progressively sparser and more dispersed, consistent with suburban and commuter-belt customers. This raw spatial pattern is the starting point for the geographic analysis: the identification of the hotspot bounding box, the density map, and the behavioural comparisons are all derived from the concentration visible here. The static format makes the overall geographic footprint clear without the visual clutter of colour-coded segments, which are overlaid in the clustering-era scatter produced later in Notebook 4.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
        <div style='padding:24px; border-radius:16px; background:#f9fafb; border:1px solid #e5e7eb; margin-top:12px;'>
          <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:10px;'>Notebook 2: Conclusions</div>
          <p style='font-size:16px; color:#374151; line-height:1.9; margin:0 0 10px 0;'>The hotspot shows a distinct behavioural profile even before using any clustering labels. The strongest differences are age, product diversity, number of complaints, store visits, total spend and promotion usage. This suggests that the dense area is not only a map artefact; it also corresponds to a younger, more active customer profile.</p>
          <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Age is the strongest signal supporting the area near the university hypothesis. At this stage, it should not be treated as one of the final customer segments because geography is not part of the clustering distance. However, it is useful as a geographic profiling insight.</p>
        </div>
    """, unsafe_allow_html=True)
    render_footer()

elif selected_page == "NB3 Clustering":
    st.markdown('''
    <div style='margin-top: 0px; margin-bottom: 24px;'>
        <h2 style='font-size: 56px; font-weight: 800; color: #000000; margin: 0; letter-spacing: -0.03em;'>Clustering</h2>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
    <div style='margin-bottom:32px;'>
      <div>
        <div style='font-size:11px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#9ca3af; margin-bottom:10px;'>Notebook 3: Clustering</div>
        <p style='font-size:18px; color:#374151; line-height:1.9; margin:0 0 24px 0; text-align: justify;'>
          Notebook 3 is the core modelling stage. It systematically evaluates combinations of feature sets, scalers, and cluster counts (k=6 to 10) to determine the partition that best captures the underlying behavioural diversity in the customer base. The process is fully transparent: every diagnostic (silhouette, elbow, dendrogram, and PCA/UMAP projection) is documented and benchmarked.
        </p>
        <div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:14px; margin-bottom:28px;'>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:16px 18px; text-align:center;'>
            <div style='font-size:11px; font-weight:600; color:#7a6454; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;'>Algorithm</div>
            <div style='font-size:24px; font-weight:800; color:#c94f38;'>K-Means</div>
          </div>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:16px 18px; text-align:center;'>
            <div style='font-size:11px; font-weight:600; color:#7a6454; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;'>K selected</div>
            <div style='font-size:24px; font-weight:800; color:#c94f38;'>8</div>
          </div>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:16px 18px; text-align:center;'>
            <div style='font-size:11px; font-weight:600; color:#7a6454; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;'>Scaler</div>
            <div style='font-size:24px; font-weight:800; color:#c94f38;'>MinMax</div>
          </div>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:16px 18px; text-align:center;'>
            <div style='font-size:11px; font-weight:600; color:#7a6454; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;'>Feature set</div>
            <div style='font-size:16px; font-weight:800; color:#c94f38; line-height:1.2;'>spend + promo<br/>no groceries</div>
          </div>
        </div>
      </div>

      <div style='background:linear-gradient(145deg, #ffffff, #f9fafb); border:1px solid #e5e7eb; border-radius:12px; padding:20px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05); margin-top:20px;'>
        <div style='font-size:14px; font-weight:800; color:#111827; margin-bottom:16px; border-bottom:2px solid #f3eee6; padding-bottom:8px;'>Notebook 3 Index</div>
        <div style='display:flex; flex-wrap:wrap; gap:12px; align-items:center;'>
          <a href="#nb3-1" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>1) Imports & data loading</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb3-2" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>2) Outlier strategy check</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb3-3" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>3) Candidate feature sets & scalers</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb3-4" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>4) Diagnostics A - scalers & elbow</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb3-5" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>5) Diagnostics B - feature set & k grid</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb3-6" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>6) Model configuration</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb3-7" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>7) Model fitting & validation</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb3-8" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>8) Method benchmarks</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb3-9" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>9) Segment profiling</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb3-10" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>10) Reattach outliers & export</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb3-11" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>11) Final modelling conclusion</div></a>
        </div>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
      <div id="nb3-1" style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 20px; margin-bottom:28px;'>
        <h2 style='font-size:24px; font-weight:800; color:#111827; margin:0 0 8px 0;'>1) Imports & data loading</h2>
        <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Loading the core data libraries and the unscaled characterisation datasets prepared in Notebook 2.</div>
      </div>

      <div id="nb3-2" style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 20px; margin-bottom:28px;'>
        <h2 style='font-size:24px; font-weight:800; color:#111827; margin:0 0 8px 0;'>2) Outlier strategy check</h2>
        <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Verifying the impact of the consensus outlier separation from Notebook 1 on the cluster structures. Only customers flagged simultaneously by IQR, DBSCAN, and a third detection method were excluded (982 customers, 3.0%), ensuring extreme multivariate values do not disproportionately pull the centroids.</div>
      </div>

      <div id="nb3-3" style='margin-bottom:28px;'>
        <h2 style='font-size:24px; font-weight:800; color:#111827; margin:0 0 16px 0;'>3) Candidate feature sets & scalers</h2>
        <p style='font-size:16px; color:#374151; line-height:1.9; margin:0 0 24px 0;'>The dataset contains 40 engineered features. Not all of them are appropriate inputs to a distance-based clustering algorithm. The selection below is the result of a deliberate process: each feature was evaluated on whether it adds discriminative signal between customers or introduces noise, redundancy, or a dimension that confounds behavioural differences with demographic or geographic artefacts. The 13 candidate feature sets tested in the silhouette grid reflect systematic removal and addition of individual features and groups.</p>

        <div style='font-size:18px; font-weight:800; color:#111827; margin-bottom:12px;'>Features included in the final set</div>
        <div style='display:grid; grid-template-columns:repeat(2,1fr); gap:12px; margin-bottom:28px;'>
          <div style='background:#f0fdf4; border:1px solid #bbf7d0; border-radius:12px; padding:16px 18px;'>
            <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#15803d; margin-bottom:6px;'>Spend categories (excl. groceries)</div>
            <div style='font-size:15px; color:#374151; line-height:1.7;'><code>lifetime_spend_vegetables</code>, <code>lifetime_spend_meat</code>, <code>lifetime_spend_fish</code>, <code>lifetime_spend_hygiene</code>, <code>lifetime_spend_technology</code>, <code>lifetime_spend_petfood</code>, <code>lifetime_spend_nonalcohol_drinks</code>, <code>lifetime_spend_alcohol_drinks</code>. Each of these categories has at least one customer segment that substantially over-indexes on it relative to the global average. They carry genuine discriminative signal: high between-cluster variance and low within-cluster variance, exactly what K-Means optimises for.</div>
          </div>
          <div style='background:#f0fdf4; border:1px solid #bbf7d0; border-radius:12px; padding:16px 18px;'>
            <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#15803d; margin-bottom:6px;'>Promotional sensitivity</div>
            <div style='font-size:15px; color:#374151; line-height:1.7;'><code>percentage_of_products_bought_promotion</code> is orthogonal to absolute spend: a customer can spend heavily and never use a promotion, or spend modestly and always seek deals. Without this feature, Promoters and Regulars are nearly indistinguishable in spend-only space — both show moderate, broad-category purchasing. Including promo sensitivity adds a second axis that separates deal-seeking behaviour from spending intensity, recovering Promoters as a distinct segment. The silhouette grid confirms that removing this feature causes a measurable drop in score.</div>
          </div>
        </div>

        <div style='font-size:18px; font-weight:800; color:#111827; margin-bottom:12px;'>Features excluded and why</div>
        <div style='display:grid; grid-template-columns:repeat(2,1fr); gap:12px; margin-bottom:24px;'>
          <div style='background:#fff5f5; border:1px solid #fecaca; border-radius:12px; padding:16px 18px;'>
            <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#b91c1c; margin-bottom:6px;'>Groceries</div>
            <div style='font-size:15px; color:#374151; line-height:1.7;'><code>lifetime_spend_groceries</code> is near-universal: virtually every customer in the dataset purchases groceries regularly. Because variation across customers is small relative to the feature's magnitude, including it pulls K-Means centroids toward a groceries axis that separates customers by how long they have been shopping — not by what kind of shopper they are. The normalised heatmap in NB4 confirms this empirically: the groceries column shows nearly uniform shading across all 8 segments. It is retained for profiling but excluded from the distance computation.</div>
          </div>
          <div style='background:#fff5f5; border:1px solid #fecaca; border-radius:12px; padding:16px 18px;'>
            <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#b91c1c; margin-bottom:6px;'>Demographic features</div>
            <div style='font-size:15px; color:#374151; line-height:1.7;'><code>customer_age</code>, <code>is_male</code>, <code>education_level</code>, <code>kids_home</code>, <code>teens_home</code>, <code>total_children</code>. Including demographic variables would produce demographic segments — clusters defined by age bracket or household size — rather than behavioural segments defined by spending. The goal is to discover communities of similar buyers, then validate against demographics post-hoc. If demographics are cluster inputs, you cannot later use them as independent validation evidence.</div>
          </div>
          <div style='background:#fff5f5; border:1px solid #fecaca; border-radius:12px; padding:16px 18px;'>
            <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#b91c1c; margin-bottom:6px;'>Geographic features</div>
            <div style='font-size:15px; color:#374151; line-height:1.7;'><code>latitude</code>, <code>longitude</code>. NB2 showed that all spend behaviour types are distributed throughout the Lisbon Metropolitan Area — no geographic zone contains exclusively high- or low-spend customers. Including coordinates would produce geographic clusters (north vs south, urban vs suburban) that do not reflect purchasing behaviour. Geography is used post-hoc in NB4 to validate that segments are not spatially confounded.</div>
          </div>
          <div style='background:#fff5f5; border:1px solid #fecaca; border-radius:12px; padding:16px 18px;'>
            <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#b91c1c; margin-bottom:6px;'>Temporal features</div>
            <div style='font-size:15px; color:#374151; line-height:1.7;'><code>typical_hour_sin</code>, <code>typical_hour_cos</code>. Shopping hour captures when a customer visits, not what or how much they buy. In preliminary tests, including these features introduced a time-of-day axis that fragmented segments without adding interpretable business meaning. A segment defined by "shops at 08:00" rather than by what they spend is not actionable for product or promotional strategy.</div>
          </div>
          <div style='background:#fff5f5; border:1px solid #fecaca; border-radius:12px; padding:16px 18px;'>
            <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#b91c1c; margin-bottom:6px;'>Aggregate spend totals</div>
            <div style='font-size:15px; color:#374151; line-height:1.7;'><code>total_spend</code>, <code>log_total_spend</code>. These aggregate all category spend into a single dimension, collapsing the very heterogeneity the model is trying to capture. Including them alongside category-level features creates severe collinearity: total spend is a linear function of the category spends already in the feature set, giving high-spend customers disproportionate weight in the distance calculation regardless of their category mix.</div>
          </div>
          <div style='background:#fff5f5; border:1px solid #fecaca; border-radius:12px; padding:16px 18px;'>
            <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#b91c1c; margin-bottom:6px;'>Spend share features</div>
            <div style='font-size:15px; color:#374151; line-height:1.7;'><code>electronics_share</code>, <code>vegetables_share</code>, and all other category share variables. Shares normalise each category by total spend, which removes the information about whether a customer is a big or small spender. Two customers with identical spending profiles except one spends 10x more overall would appear identical in share space. The clustering objective requires both dimensions: what a customer buys and how much they spend on it.</div>
          </div>
          <div style='background:#fff5f5; border:1px solid #fecaca; border-radius:12px; padding:16px 18px;'>
            <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#b91c1c; margin-bottom:6px;'>Loyalty flag & complaints</div>
            <div style='font-size:15px; color:#374151; line-height:1.7;'><code>customer_loyalty_flag</code>, <code>number_complaints</code>. These are outcome variables rather than behavioural inputs. Loyalty flag has 38% missing values (retained as a binary class), making it unreliable as a distance feature. Complaints volume is low and relatively uniform across customers; including it would fragment clusters around complaint history rather than spending behaviour.</div>
          </div>
          <div style='background:#fff5f5; border:1px solid #fecaca; border-radius:12px; padding:16px 18px;'>
            <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#b91c1c; margin-bottom:6px;'>Tenure & distinct products</div>
            <div style='font-size:15px; color:#374151; line-height:1.7;'><code>tenure</code>, <code>lifetime_total_distinct_products</code>. Tenure is used to normalise lifetime spend into annual rates (spend ÷ tenure); it is an input to feature engineering, not a cluster feature itself. Distinct products is a breadth measure that is strongly collinear with total spend: customers who spend more tend to buy from more categories. Including it would amplify the weight of high-spend customers without adding new behavioural signal.</div>
          </div>
        </div>
      </div>
    ''', unsafe_allow_html=True)

    # 4) Diagnostics A
    st.markdown('''
<div id="nb3-4" style="margin-top:40px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">4) Diagnostics A - Scaler comparison & Elbow</h2></div>
''', unsafe_allow_html=True)
    
    _p = IMAGENS_DIR / "charts" / "scaler_comparison.png"
    if _p.exists():
        st.markdown("""<div style='margin-bottom:8px;'><div style='font-size:18px; font-weight:700; color:#111827;'>Scaler comparison</div></div>
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:12px 16px; margin-bottom:14px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:5px;'>Why this chart?</div>
  <div style='font-size:16px; color:#374151; line-height:1.7;'>Shows silhouette score vs. k for both MinMaxScaler and RobustScaler. A higher silhouette score means better-separated, more compact clusters. The comparison reveals whether the choice of scaler materially changes the solution quality, favouring MinMax.</div>
</div>""", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
    st.markdown('''
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Both MinMaxScaler and RobustScaler were tested. MinMaxScaler consistently produces higher silhouette scores at k=8 and shows a cleaner elbow in the curve.</p>
</div>
''', unsafe_allow_html=True)

    _p = IMAGENS_DIR / "charts" / "elbow_silhouette.png"
    if _p.exists():
        st.markdown("<div style='margin-bottom:12px;'><div style='font-size:18px; font-weight:700; color:#111827;'>Elbow method (Silhouette score)</div></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
    st.markdown('''
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The silhouette peaks locally near k=4 and k=8, with k=8 providing the best balance between geometric separation and the number of actionable segments.</p>
</div>
''', unsafe_allow_html=True)

    _p = IMAGENS_DIR / "charts" / "ward_dendrogram.png"
    if _p.exists():
        st.markdown("<div style='margin-bottom:12px;'><div style='font-size:18px; font-weight:700; color:#111827;'>Ward Dendrogram</div></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
    st.markdown('''
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The Ward linkage dendrogram is computed on a stratified 5,000-customer sample. It provides a hierarchical view of how clusters merge as the distance threshold increases. The dendrogram confirms that k=8 is a natural cut point: a horizontal cut at the appropriate height yields eight well-separated branches with no single branch dominating the structure. The vertical distance between successive merges (the height of each horizontal bar) encodes the within-cluster variance increase at that merge step. The large gap between the merge at k=8 and the next merge at k=7 is the main quantitative justification for choosing eight clusters: merging any two of the eight groups would incur a disproportionate loss of within-cluster homogeneity relative to the gain in parsimony. The Ward criterion also confirms that the chosen feature set produces cohesive, compact groups rather than elongated or chained structures.</p>
</div>
''', unsafe_allow_html=True)

    _p1 = IMAGENS_DIR / "charts" / "average_dendrogram__sample____spend___promo_no_groceries.png"
    _p2 = IMAGENS_DIR / "charts" / "complete_dendrogram__sample____spend___promo_no_groceries.png"
    _p3 = IMAGENS_DIR / "charts" / "single_dendrogram__sample____spend___promo_no_groceries.png"
    
    if _p1.exists() and _p2.exists() and _p3.exists():
        st.markdown("<div style='margin-bottom:12px;'><div style='font-size:18px; font-weight:700; color:#111827;'>Alternative linkages (Average, Complete, Single)</div></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1: st.image(str(_p1), use_container_width=True, caption="Average Linkage")
        with col2: st.image(str(_p2), use_container_width=True, caption="Complete Linkage")
        with col3: st.image(str(_p3), use_container_width=True, caption="Single Linkage")
        st.markdown('''
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Alternative linkage methods exhibit severe chaining and pathological imbalance, confirming that Ward linkage (which minimises within-cluster variance) is the only viable approach for this dataset.</p>
</div>
''', unsafe_allow_html=True)
    else:
        _p = IMAGENS_DIR / "charts" / "alt_dendrograms.png"
        if _p.exists():
            st.image(str(_p), use_container_width=True)

    # 5) Diagnostics B
    st.markdown('''
<div id="nb3-5" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">5) Diagnostics B - Silhouette score grid & Projections</h2></div>
''', unsafe_allow_html=True)
    
    _p = IMAGENS_DIR / "charts" / "silhouette_grid.png"
    if _p.exists():
        st.markdown("""<div style='margin-bottom:8px;'><div style='font-size:18px; font-weight:700; color:#111827;'>Silhouette score grid (K=6 to K=10)</div></div>
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:12px 16px; margin-bottom:14px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:5px;'>Why this chart?</div>
  <div style='font-size:16px; color:#374151; line-height:1.7;'>The silhouette grid plots average silhouette score for each combination of feature set and k value. A single best-performing row (feature set) and column (k) is the main selection criterion. The grid makes the selection transparent: K=8 with spend + promo, no groceries achieves the highest score, and the margin over K=7 and K=9 is visible and non-trivial.</div>
</div>""", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
    st.markdown('''
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The silhouette grid evaluates 13 feature sets across k values from 6 to 10. Each cell represents the average silhouette score for that feature set and cluster count combination. Higher values indicate better-separated and more cohesive clusters. The grid reveals that the feature set combining spend categories and promotional sensitivity without groceries consistently achieves the highest scores, and that k=8 is the optimal cluster count within this feature set: the score at k=8 exceeds those at k=7 and k=9 by a non-trivial margin, and the drop from k=8 to k=9 is steeper than the gain from k=7 to k=8, indicating diminishing returns beyond eight clusters. Feature sets that include groceries or exclude promotional sensitivity score systematically lower, providing quantitative support for those design choices.</p>
</div>
''', unsafe_allow_html=True)

    _p = IMAGENS_DIR / "charts" / "customers_per_cluster.png"
    if _p.exists():
        st.markdown("<div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;'>Customers per Cluster</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
    st.markdown('''
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The bar chart shows the number of customers assigned to each of the eight clusters after the outlier reattachment step. The distribution is reasonably balanced: no single cluster absorbs a dominant share of customers, and no cluster is too small to be statistically or commercially actionable. The largest segments (Regulars and Economizers) reflect the broad middle of the behavioural distribution, where spending patterns are moderate and undifferentiated. The smaller segments (Techies, Loyalists, Wellness) represent rarer but highly distinctive behavioural profiles. This balance is an important quality check: severe imbalance would suggest that the chosen k is either too low (merging distinct groups) or too high (fragmenting a homogeneous group). The near-uniform distribution here is consistent with the silhouette diagnostics and confirms that k=8 is a well-calibrated partition.</p>
</div>
''', unsafe_allow_html=True)

    _p = IMAGENS_DIR / "charts" / "silhouette_blades.png"
    if _p.exists():
        st.markdown("""<div style='margin-bottom:8px; margin-top:24px;'><div style='font-size:18px; font-weight:700; color:#111827;'>Silhouette blades (K=8)</div></div>
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:12px 16px; margin-bottom:14px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:5px;'>Why this chart?</div>
  <div style='font-size:16px; color:#374151; line-height:1.7;'>Blade width represents cluster size; blade length represents the silhouette score of each sample. Uniform blade widths confirm balanced cluster sizes. Mostly positive scores (right of the dashed average line) confirm that most customers are closer to their own cluster than to any neighbour. Negative scores (left of zero) flag potential misclassifications and are used to decide whether to keep or split a cluster.</div>
</div>""", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
        st.markdown('''
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The blades for k=8 show relatively uniform thickness (indicating balanced cluster sizes) and mostly positive scores, with few instances of negative silhouette scores (which would indicate misclassification).</p>
</div>
''', unsafe_allow_html=True)

    _p = IMAGENS_DIR / "charts" / "pca_projection.png"
    if _p.exists():
        st.markdown("<div style='margin-bottom:12px;'><div style='font-size:18px; font-weight:700; color:#111827;'>PCA Projection (2D)</div></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
        st.markdown('''
<div style='background:#f0f9ff; border:1px solid #bae6fd; border-radius:12px; padding:16px 20px; margin-top:8px; margin-bottom:8px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#0369a1; margin-bottom:6px;'>Why this chart?</div>
  <p style='font-size:15px; color:#374151; line-height:1.8; margin:0;'>Principal Component Analysis (PCA) projects the scaled feature space onto the two directions of greatest variance. As a linear technique, it is computationally efficient and deterministic, making it a reliable first-pass sanity check: if clusters are separable in PCA space, they are linearly well-defined. Because PCA cannot capture non-linear structure, partial overlap in this view does not invalidate the segmentation; it motivates the subsequent UMAP and t-SNE checks.</p>
</div>
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The PCA projection shows partial but meaningful separation across the two principal components. Several clusters form recognisable groupings along the principal axes, particularly Loyalists and Promoters, whose high spend and high promotional response respectively align with the dominant variance direction. The degree of overlap visible here is expected given PCA's linear constraint on a feature space with known non-linear interactions (e.g., promotion response scaling differently with income bracket). This partial separability in two linear dimensions already provides evidence that the clusters capture distinct behavioural profiles.</p>
</div>
''', unsafe_allow_html=True)

    _p = IMAGENS_DIR / "charts" / "umap_projection.png"
    if _p.exists():
        st.markdown("<div style='margin-bottom:12px;'><div style='font-size:18px; font-weight:700; color:#111827;'>UMAP Projection (2D)</div></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
        st.markdown('''
<div style='background:#f0f9ff; border:1px solid #bae6fd; border-radius:12px; padding:16px 20px; margin-top:8px; margin-bottom:8px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#0369a1; margin-bottom:6px;'>Why this chart?</div>
  <p style='font-size:15px; color:#374151; line-height:1.8; margin:0;'>Uniform Manifold Approximation and Projection (UMAP) is a non-linear dimensionality reduction technique that preserves both local and global topological structure. Unlike PCA, UMAP can reveal non-linear cluster boundaries and curved manifolds. It is used here as a secondary validation layer: strong visual separation in UMAP space provides robust evidence that the K-Means partitions correspond to genuinely distinct regions of the behavioural manifold, not artefacts of the Euclidean distance assumption.</p>
</div>
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The UMAP projection reveals considerably cleaner cluster separation than the linear PCA view, confirming that the segment boundaries are non-trivially structured in the original feature space. Several clusters form compact, well-bounded islands, most notably the high-spend segments (Loyalists, Techies) which appear as spatially isolated groups. The Promoters cluster, defined primarily by promotional responsiveness rather than absolute spend, occupies a distinct region with limited overlap. Segments with overlapping behavioural profiles (e.g., Regulars and Economizers, both characterised by moderate or low spend) show some proximity in UMAP space, which is consistent with the intuition that these groups lie on a continuum of spend intensity. The overall topology validates the k=8 partition: eight identifiable regions exist in the low-dimensional manifold.</p>
</div>
''', unsafe_allow_html=True)

    _p = IMAGENS_DIR / "charts" / "tsne_projection.png"
    if _p.exists():
        st.markdown("<div style='margin-bottom:12px;'><div style='font-size:18px; font-weight:700; color:#111827;'>t-SNE Projection (2D)</div></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
        st.markdown('''
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>While PCA shows some overlap due to its linear nature, the manifold learning techniques (UMAP and particularly t-SNE) demonstrate clear boundaries and distinct islands, visually validating the K-Means cluster assignments in a non-linear low-dimensional space.</p>
</div>
''', unsafe_allow_html=True)

    _p = IMAGENS_DIR / "charts" / "zscore_heatmap.png"
    if _p.exists():
        st.markdown("<div style='margin-bottom:12px;'><div style='font-size:18px; font-weight:700; color:#111827;'>Z-Score Heatmap (Features x Clusters)</div></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
        st.markdown('''
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>This heatmap visualises the z-score deviations of each cluster's centroid from the global mean across all features. Deep rust indicates values significantly above the average, while slate indicates values below the average. It confirms the distinct spending and behavioural profile of each segment: Techies dominate electronics and technology, Vegetarians dominate vegetables, and Families show strong deviations in meat and hygiene.</p>
</div>
''', unsafe_allow_html=True)

    # 6, 7, 8, 9
    st.markdown('''
      <div id="nb3-6" style='font-size:24px; font-weight:800; color:#111827; margin-bottom:16px; margin-top:40px;'>6) Model configuration & 7) Model fitting & validation</div>

      <div style='display:grid; grid-template-columns:repeat(2,1fr); gap:16px; margin-bottom:28px;'>

        <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:18px 20px;'>
          <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Algorithm choice</div>
          <div style='font-size:16px; font-weight:700; color:#111827; margin-bottom:6px;'>Why K-Means?</div>
          <div style='font-size:16px; color:#374151; line-height:1.7;'>K-Means produces hard, non-overlapping assignments that map cleanly to business segments. It is fast enough to iterate over a full grid of k values and feature sets, and its centroids are directly interpretable as average member profiles, making the naming rationale straightforward. Soft-assignment models (GMM) were considered but add complexity without a clear interpretability gain for this use case.</div>
        </div>

        <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:18px 20px;'>
          <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Number of clusters</div>
          <div style='font-size:16px; font-weight:700; color:#111827; margin-bottom:6px;'>Why K=8?</div>
          <div style='font-size:16px; color:#374151; line-height:1.7;'>k values from 6 to 10 were evaluated on three criteria: silhouette score (separation quality), elbow in the inertia curve (diminishing returns), and business interpretability. K=8 produced the best silhouette score in the spend + promo, no groceries feature set, showed a clear elbow, and yielded eight behaviourally distinct profiles each with a clear real-world label. K=6 merged groups that were meaningfully different; k=10 created near-duplicate clusters.</div>
        </div>

        <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:18px 20px;'>
          <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Scaler</div>
          <div style='font-size:16px; font-weight:700; color:#111827; margin-bottom:6px;'>Why MinMaxScaler over RobustScaler?</div>
          <div style='font-size:16px; color:#374151; line-height:1.7;'>Both scalers were tested across all feature-set × k combinations. MinMaxScaler consistently produced higher silhouette scores at k=8, and a cleaner elbow curve. RobustScaler clips the influence of outliers via the IQR; however, since outliers were already separated in NB1, the input to the model is already a clean distribution. MinMaxScaler compresses all features to [0, 1] while preserving their relative order, giving equal weight to each spend dimension without discarding distributional information.</div>
        </div>

        <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:18px 20px;'>
          <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Feature set</div>
          <div style='font-size:16px; font-weight:700; color:#111827; margin-bottom:6px;'>Why spend + promo, no groceries?</div>
          <div style='font-size:16px; color:#374151; line-height:1.7;'>Grocery spend is near-universal across all customers and contributes noise rather than signal to the clustering distance. Including it pulls centroids toward a groceries dimension that does not differentiate behaviour. Removing groceries from the feature set while retaining it for profiling gives a cleaner signal. Adding <code>percentage_of_products_bought_promotion</code> captures deal-seeking behaviour that is orthogonal to absolute spend, thus recovering the Promoters segment that would otherwise blend into Regulars.</div>
        </div>

      </div>

      <div id="nb3-8" style='font-size:24px; font-weight:800; color:#111827; margin-bottom:16px;'>8) Method benchmarks: Alternatives tested</div>
      <div style='display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-bottom:28px;'>
        <div style='background:#fff5f5; border:1px solid #fecaca; border-radius:12px; padding:18px 20px;'>
          <div style='font-size:16px; font-weight:700; color:#b91c1c; margin-bottom:6px;'>DBSCAN (rejected)</div>
          <div style='font-size:16px; color:#374151; line-height:1.7;'>DBSCAN requires a radius parameter (ε) that is sensitive to the high-dimensional feature space. With customer spend data it labelled 20–40% of customers as noise, making downstream business use impossible. Density-based boundaries also don't produce stable centroid profiles for profiling.</div>
        </div>
        <div style='background:#f0fdf4; border:1px solid #bbf7d0; border-radius:12px; padding:18px 20px;'>
          <div style='font-size:16px; font-weight:700; color:#15803d; margin-bottom:6px;'>Hierarchical Ward (validated)</div>
          <div style='font-size:16px; color:#374151; line-height:1.7;'>Ward linkage on a stratified 5,000-customer sample was run as a cross-check. The dendrogram confirmed that k=8 is a natural cut point. The resulting Ward labels showed high agreement with K-Means assignments (adjusted Rand Index > 0.70), validating the K-Means solution rather than replacing it.</div>
        </div>
        <div style='background:#f0fdf4; border:1px solid #bbf7d0; border-radius:12px; padding:18px 20px;'>
          <div style='font-size:16px; font-weight:700; color:#15803d; margin-bottom:6px;'>Self-Organising Map (validated)</div>
          <div style='font-size:16px; color:#374151; line-height:1.7;'>A SOM was tested as a topology-preserving alternative. It produced comparable cluster profiles but with higher run-time and harder-to-explain topology for a business audience. K-Means was preferred for its speed, reproducibility (fixed random_state), and direct centroid interpretability.</div>
        </div>
        <div style='background:#fff5f5; border:1px solid #fecaca; border-radius:12px; padding:18px 20px;'>
          <div style='font-size:16px; font-weight:700; color:#b91c1c; margin-bottom:6px;'>Mean Shift (rejected)</div>
          <div style='font-size:16px; color:#374151; line-height:1.7;'>Mean Shift was evaluated and discarded for three reasons. Its O(n²) time complexity makes it computationally prohibitive at 33,038 customers: K-Means runs in linear time. While it avoids specifying k, it requires a bandwidth parameter that is equally sensitive and lacks an objective selection criterion equivalent to elbow or silhouette analysis. Its kernel density estimation also degrades in higher-dimensional spaces, producing fragmented or heavily imbalanced clusters in mixed retail data distributions.</div>
        </div>
        <div style='background:#fff5f5; border:1px solid #fecaca; border-radius:12px; padding:18px 20px;'>
          <div style='font-size:16px; font-weight:700; color:#b91c1c; margin-bottom:6px;'>PCA (rejected)</div>
          <div style='font-size:16px; color:#374151; line-height:1.7;'>PCA was considered and rejected because it destroys the interpretability that is central to this project's value. Principal components are linear combinations of all features, meaning cluster centroids lose their business meaning: instead of this segment spends heavily on electronics, a result would read this segment has high loading on PC2. With only 15 engineered features there is no dimensionality problem that would justify this trade-off. MinMaxScaler already addresses the scale issue without sacrificing feature meaning. The entire value of this segmentation lies in named, explainable segments that a marketing team can act on directly.</div>
        </div>
      </div>
      
    ''', unsafe_allow_html=True)

    _p = IMAGENS_DIR / "charts" / "umap___separation__regular_only_.png"
    if _p.exists():
        st.markdown("<div style='font-size:16px; font-weight:700; color:#111827; margin-bottom:12px;'>K-Means vs Hierarchical Ward Labels (UMAP)</div>", unsafe_allow_html=True)
        st.image(str(_p), use_container_width=True)
    st.markdown('''
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>This UMAP projection overlays two independent sets of cluster labels: those produced by K-Means and those produced by hierarchical Ward linkage on the same feature space. Where the two labellings agree, customers appear in the same colour region under both schemes; where they disagree, colour divergence is visible. The high degree of visual overlap between the two solutions confirms that the eight-cluster structure is not an artefact of the K-Means algorithm but a genuine feature of the data. The adjusted Rand Index between the two solutions exceeds 0.70, indicating substantial label agreement well above chance. This cross-validation is important because K-Means and Ward linkage use fundamentally different optimisation criteria: K-Means minimises within-cluster sum of squares iteratively, while Ward linkage uses a bottom-up agglomerative procedure that minimises variance at each merge step. Agreement between these two approaches on the same partition provides strong evidence that eight distinct behavioural regions exist in the customer feature space.</p>
</div>
''', unsafe_allow_html=True)

    st.markdown('''
      <div id="nb3-9" style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:18px 20px; margin-bottom:20px;'>
        <div style='font-size:24px; font-weight:800; color:#111827; margin-bottom:8px;'>9) Segment profiling: Naming rationale</div>
        <div style='font-size:16px; color:#374151; line-height:1.8;'>Cluster names are assigned <em>after</em> the modelling stage is complete, using z-score deviation profiles from the centroid of each cluster. A name is only accepted if it captures the single most distinctive behavioural trait of that cluster relative to the global average. Names are never imposed before seeing the data; this prevents confirmation bias in centroid interpretation. The full per-cluster evidence table is in Notebook 4, Section 3.</div>
      </div>

      <div id="nb3-10" style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:18px 20px; margin-bottom:20px;'>
        <div style='font-size:24px; font-weight:800; color:#111827; margin-bottom:8px;'>10) Reattach outliers & export</div>
        <div style='font-size:16px; color:#374151; line-height:1.8;'>The 982 outlier customers separated in NB1 are reattached by computing their Euclidean distance (in the scaled feature space) to each of the 8 cluster centroids and assigning them to the nearest one. This ensures 100% of customers receive a segment label for downstream CRM and campaign use. Outliers are flagged in the export so that analysts can filter them out if they want to analyse only the regular base.</div>
      </div>

      <div id="nb3-11" style='padding:24px; border-radius:16px; background:#111827; border:1px solid #374151; margin-top:32px;'>
        <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:12px;'>Final modelling conclusion</div>
        <p style='font-size:16px; color:#d1d5db; line-height:1.9; margin:0;'>The final model is <strong style="color:#fff;">K-Means, k=8, MinMaxScaler, feature set: spend + promo, no groceries</strong>. This configuration was selected after a systematic grid search across 13 feature sets × 2 scalers × 5 k values. It produces the highest silhouette score, the cleanest elbow, and the most distinct and interpretable business profiles. All downstream characterisation, geographic analysis, and campaign recommendations are built on top of this solution.</p>
      </div>
    ''', unsafe_allow_html=True)
    render_footer()
elif selected_page == "NB4 Characterisation":
    cluster_id_map = {0: "Regulars", 1: "Families", 2: "Economizers", 3: "Vegetarians", 4: "Loyalists", 5: "Techies", 6: "Wellness", 7: "Promoters"}

    st.markdown("""
    <div style='margin-top: 0px; margin-bottom: 24px;'>
        <h2 style='font-size: 56px; font-weight: 800; color: #000000; margin: 0; letter-spacing: -0.03em;'>Cluster Characterisation</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-bottom:32px;'>
      <div>
        <div style='font-size:11px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#9ca3af; margin-bottom:10px;'>Notebook 4: Cluster Characterisation</div>
        <p style='font-size:18px; color:#374151; line-height:1.9; margin:0 0 24px 0; text-align: justify;'>
          Notebook 4 operationalises the clustering model by profiling each segment. It identifies the distinct behavioural and spending traits that define the eight communities, assigns data-grounded business names, and validates these labels against demographic and geographic metadata. The goal is to translate abstract cluster assignments into clear, distinct customer personas.
        </p>
        <div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:14px; margin-bottom:28px;'>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:16px 18px; text-align:center;'>
            <div style='font-size:11px; font-weight:600; color:#7a6454; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;'>Segments named</div>
            <div style='font-size:24px; font-weight:800; color:#c94f38;'>8</div>
          </div>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:16px 18px; text-align:center;'>
            <div style='font-size:11px; font-weight:600; color:#7a6454; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;'>Chart views used</div>
            <div style='font-size:24px; font-weight:800; color:#c94f38;'>7</div>
          </div>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:16px 18px; text-align:center;'>
            <div style='font-size:11px; font-weight:600; color:#7a6454; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;'>Customers profiled</div>
            <div style='font-size:24px; font-weight:800; color:#c94f38;'>33,038</div>
            <div style='font-size:11px; color:#7a6454; margin-top:4px; line-height:1.4;'>32,056 clustered + 982 outliers assigned to nearest cluster</div>
          </div>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:16px 18px; text-align:center;'>
            <div style='font-size:11px; font-weight:600; color:#7a6454; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;'>Min views to name</div>
            <div style='font-size:24px; font-weight:800; color:#c94f38;'>3</div>
          </div>
        </div>
      </div>

      <div style='background:linear-gradient(145deg, #ffffff, #f9fafb); border:1px solid #e5e7eb; border-radius:12px; padding:20px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05); margin-top:20px;'>
        <div style='font-size:14px; font-weight:800; color:#111827; margin-bottom:16px; border-bottom:2px solid #f3eee6; padding-bottom:8px;'>Notebook 4 Index</div>
        <div style='display:flex; flex-wrap:wrap; gap:12px; align-items:center;'>
          <a href="#nb4-1" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>1) Imports & data loading</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb4-2" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>2) Segment sizes</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb4-3" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>3) Spend profile</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb4-4" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>4) Behavioural & demographic profile</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb4-5" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>5) Loyalty & metadata checks</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb4-6" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>6) Normalised comparison</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb4-7" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>7) Feature plots</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb4-8" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>8) Cluster interpretation</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb4-9" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>9) Geographic check</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb4-10" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>10) Final segment names & export</div></a>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
      <div id="nb4-1"></div><div id="nb4-3"></div><div id="nb4-4"></div><div id="nb4-5"></div><div id="nb4-6"></div><div id="nb4-7"></div><div id="nb4-8"></div><div id="nb4-9"></div><div id="nb4-10"></div>
      <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 20px; margin-bottom:28px;'>
        <h2 style='font-size:24px; font-weight:800; color:#111827; margin:0 0 8px 0;'>1) Imports & data loading - Naming protocol</h2>
        <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Business names are assigned only after the modelling stage is complete. A name is only confirmed when the same pattern appears consistently across at least three views: the spend deviation table, the radar plot, the spend profile heatmap, and the demographic/behavioural profile. The final name of each segment is chosen only when the same pattern appears in more than one view. This prevents confirmation bias and ensures that names reflect stable, data-grounded patterns rather than single-chart impressions.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Cluster sizes
    st.markdown("""
<div id="nb4-2" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">2) Segment sizes - Customer count per community</h2></div>
""", unsafe_allow_html=True)
    id_cluster_df = load_csv_data("id_and_cluster.csv")
    cluster_counts = id_cluster_df.groupby("cluster_name").size().reset_index(name="customers")
    cluster_counts = cluster_counts.sort_values("customers", ascending=False)
    cluster_size_chart = alt.Chart(cluster_counts).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("cluster_name:N", sort="-y", title="Segment"),
        y=alt.Y("customers:Q", title="Number of customers"),
        color=alt.Color("cluster_name:N", scale=alt.Scale(domain=list(SEGMENT_NAME_COLORS.keys()), range=list(SEGMENT_NAME_COLORS.values())), legend=None),
        tooltip=["cluster_name", alt.Tooltip("customers:Q", format=",")]
    ).properties(height=320)
    st.altair_chart(cluster_size_chart, use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The distribution of customers across the eight communities is free from pathological size imbalance. No single community dominates the dataset and no community is too small to be actionable. This balance is a direct consequence of the outlier separation step in NB1: removing multivariate extremes before clustering produces a more homogeneous input space in which K-Means converges to more evenly populated centroids. The two largest segments (Regulars and Economizers) are also the most behaviourally moderate, which is consistent with a retail customer base where the majority of customers have unremarkable spending patterns. The three smallest segments (Techies, Loyalists, Wellness) are the most behaviourally distinctive; their smaller size reflects how rare those specific patterns are in the population, not a modelling failure. Segment sizes inform campaign prioritisation: larger segments offer higher absolute reach, while smaller but more homogeneous segments offer higher targeting precision.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style='margin-top:48px; margin-bottom:6px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#9ca3af; margin-bottom:6px;'>Notebook 4: Segment profiling charts</div>
  <div style='font-size:20px; font-weight:800; color:#111827; margin-bottom:4px;'>Deep-dive characterisation: all charts from NB4</div>
</div>
""", unsafe_allow_html=True)

    # Spend profile heatmap (NB4 version)
    st.markdown("""
<div id="nb4-3" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">3) Spend profile heatmap</h2></div>
<div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;'>Average lifetime spend per cluster (€)</div>
""", unsafe_allow_html=True)
    _p = IMAGENS_DIR / "charts" / "spend_heatmap.png"
    if _p.exists():
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The spend profile heatmap shows raw average lifetime spend in euros per cluster-category pair, with colour normalised across clusters per column so that the darkest cell identifies the highest-spending segment in each category. Cell annotations show the actual euro values. Techies stand out strongly in electronics, technology, and videogames, confirming they are the technology-oriented segment and making them the priority audience for any cross-sell campaign targeting premium devices. Vegetarians dominate vegetables and non-alcoholic drinks, consistent with a health- and diet-conscious profile. Families show elevated spend across groceries and hygiene, reflecting large household purchasing patterns. Loyalists rank highly across multiple categories simultaneously, consistent with a long-tenure, broad-basket profile. Economizers show consistently low raw spend values across all categories, reflecting a restrained purchasing style; importantly, this is not driven by promotion sensitivity (their promo usage is near the median), but by genuinely lower absolute spending levels. This raw-values version of the heatmap complements the normalised plotly version shown above by revealing the actual scale differences between clusters, which the normalised view compresses.</p>
</div>
""", unsafe_allow_html=True)

    # Behavioural + demographic heatmap (NB4)
    st.markdown("""
<div id="nb4-4" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">4) Behavioural & demographic profile</h2></div>
<div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;'>Z-scores by cluster</div>
""", unsafe_allow_html=True)
    _p = IMAGENS_DIR / "charts" / "behavioural_heatmap.png"
    if _p.exists():
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>This heatmap captures non-spend dimensions: customer age, tenure as a customer, number of children at home, number of teenagers at home, number of complaints, stores visited, and promotional sensitivity. Z-scores allow direct comparison across variables with different units and scales. Families show the strongest positive deviation on the children and teens dimensions, which is the primary naming driver for this segment. Loyalists score highest on tenure, consistent with their long-standing relationship with the retailer. Promoters score by far the strongest positive z-score on promotional sensitivity (percentage of products bought on promotion), confirming that this is their defining and differentiating characteristic. Regulars and Economizers have relatively flat profiles across behavioural dimensions, which contributes to their lower distinctiveness in the z-score space; their differentiation comes from the spend profile rather than demographic or behavioural attributes. Complaints vary modestly across segments; no cluster is systematically dissatisfied, reducing the risk that any identified community represents a cohort at high churn risk due to service quality alone.</p>
</div>
""", unsafe_allow_html=True)


    # 5) Loyalty & metadata checks
    st.markdown("""
<div id="nb4-5" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">5) Loyalty & metadata checks</h2></div>
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Loyalty program participation varies substantially by segment. Cluster 4 (Loyalists) and Cluster 1 (Families) show the highest share of loyalty card holders (77.6% and 68.4% respectively), suggesting strong brand attachment and habitual shopping patterns. Conversely, Cluster 2 (Economizers) and Cluster 5 (Techies) demonstrate lower participation rates (47.9% and 49.5%), reinforcing their more transactional or purpose-driven behavior. Gender distribution appears relatively balanced across all clusters, confirming that segmentation was driven more by behavioral and transactional variables than by demographic traits.</p>
</div>
""", unsafe_allow_html=True)
    # Interactive spend heatmap
    st.markdown("""
<div id="nb4-6" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">6) Normalised comparison</h2></div>
<div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;'>6.1) Normalised spend profile per cluster: interactive heatmap</div>
""", unsafe_allow_html=True)
    seg_spend_df = load_csv_data("segment_spend_profile.csv")
    spend_heat_cols = [c for c in seg_spend_df.columns if c.startswith("lifetime_spend_")]
    seg_spend_df["cluster"] = pd.to_numeric(seg_spend_df["cluster"], errors="coerce")
    seg_spend_df = seg_spend_df.dropna(subset=["cluster"])
    seg_spend_df = seg_spend_df.sort_values("cluster")
    seg_spend_df["segment_name"] = seg_spend_df["cluster"].astype(int).map(cluster_id_map)
    spend_matrix = seg_spend_df[spend_heat_cols].values.astype(float)
    col_min = spend_matrix.min(axis=0); col_max = spend_matrix.max(axis=0)
    col_range = col_max - col_min; col_range[col_range == 0] = 1
    spend_matrix_norm = (spend_matrix - col_min) / col_range
    spend_col_labels = [c.replace("lifetime_spend_", "").replace("_", " ").title() for c in spend_heat_cols]
    spend_heat_fig = px.imshow(spend_matrix_norm, x=spend_col_labels, y=seg_spend_df["segment_name"].tolist(),
        color_continuous_scale=list(SEGMENT_COLORS.values()), zmin=0, zmax=1, text_auto=".2f")
    spend_heat_fig.update_layout(margin=dict(l=80, r=20, t=60, b=80), height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(spend_heat_fig, use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Each cell is normalised to [0, 1] across segments per column, so the darkest cell identifies the highest-spending segment in that category. Techies concentrate spending in electronics, technology, and videogames. Vegetarians over-index in vegetables and non-alcoholic drinks. Families show elevated spend across groceries and hygiene. Groceries show similar shading across nearly all segments, confirming that its exclusion from the clustering distance was correct: it adds little discriminative power. Alcohol and petfood show very low values across all segments, confirming their niche status in the customer base. Hover over any cell to see the normalised score; compare columns to identify which category most cleanly separates one segment from the rest.</p>
</div>
""", unsafe_allow_html=True)

    # Interactive behavioural heatmap
    st.markdown("""
<div style="margin-top:36px; margin-bottom:12px; padding-top:24px;">
<div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;'>6.2) Normalised behavioural profile per cluster: interactive heatmap</div>
</div>
""", unsafe_allow_html=True)
    info_unscaled_comm = load_csv_data("info_clustering_unscaled.csv")
    customer_segments_comm = load_csv_data("customer_segments.csv")
    merged_comm = info_unscaled_comm.merge(customer_segments_comm, on="customer_id", how="inner")
    behav_features = ["percentage_of_products_bought_promotion", "tenure", "total_children", "number_complaints"]
    behav_by_cluster = merged_comm.groupby("cluster")[behav_features].mean().reset_index()
    behav_by_cluster["segment_name"] = behav_by_cluster["cluster"].map(cluster_id_map)
    behav_by_cluster = behav_by_cluster.sort_values("cluster")
    behav_matrix = behav_by_cluster[behav_features].values.astype(float)
    b_min = behav_matrix.min(axis=0); b_max = behav_matrix.max(axis=0)
    b_range = b_max - b_min; b_range[b_range == 0] = 1
    behav_matrix_norm = (behav_matrix - b_min) / b_range
    behav_heat_fig = px.imshow(behav_matrix_norm, x=["Promo sensitivity", "Tenure (years)", "Total children", "Avg complaints"],
        y=behav_by_cluster["segment_name"].tolist(), color_continuous_scale=list(SEGMENT_COLORS.values()), zmin=0, zmax=1,
        text_auto=".2f")
    behav_heat_fig.update_layout(margin=dict(l=80, r=20, t=60, b=80), height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(behav_heat_fig, use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Promotional sensitivity varies strongly across segments: Promoters register the maximum value on this dimension, confirming that their defining trait is price-driven purchasing. Tenure separates long-term customers (Loyalists, Families) from newer cohorts (Techies, Economizers), supporting differentiated retention versus acquisition strategies. Total children most strongly characterises the Families segment; the highest value on this axis was one of the primary naming criteria. Complaints vary modestly across segments; where elevated, they reflect higher transaction frequency rather than systematic dissatisfaction. Together, these four dimensions provide a multi-axis profile that is more actionable for campaign design than spend data alone.</p>
</div>
""", unsafe_allow_html=True)

    # Individual radar profiles
    st.markdown("""
<div id="nb4-7" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">7) Feature plots</h2></div>
<div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;'>7.1) Individual radar profiles: all 8 clusters (9-axis spider chart)</div>
""", unsafe_allow_html=True)
    _p = IMAGENS_DIR / "charts" / "radar_individual.png"
    if _p.exists():
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Each panel shows a single cluster's average profile across nine axes: electronics, vegetables, meat, fish, technology, petfood, videogames, hygiene, and promotional sensitivity. Values are normalised to [0, 1] relative to the dataset maximum for each axis, so the shape of each radar reflects relative spend intensity rather than absolute euros. Techies (C7) present a highly asymmetric shape with large extensions along electronics and technology and a very small promotional sensitivity arm, confirming they are full-price technology buyers. Vegetarians (C0) show a large extension along the vegetables axis with a near-zero promotional arm, consistent with quality-driven, full-price vegetable purchasers. Promoters (C3) show a large promotional sensitivity arm but a relatively flat spend profile across product categories, confirming that what defines this segment is how they buy rather than what they buy. Families (C5) show elevated hygiene and meat arms. Loyalists (C4) present a broadly extended shape across multiple arms, reflecting their high-basket, broad-category purchasing. The individual view makes segment-specific patterns clear without the visual complexity of the overlaid comparison.</p>
</div>
""", unsafe_allow_html=True)

    # Combined radar
    st.markdown("""
<div style='margin-top:36px; margin-bottom:12px; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>7.2) Combined radar: all 8 clusters overlaid</div>
</div>
""", unsafe_allow_html=True)
    _p = IMAGENS_DIR / "charts" / "radar_combined.png"
    if _p.exists():
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The combined radar overlays all eight cluster profiles in a single chart, enabling direct visual comparison of where each community stands relative to every other on the same axis. The chart reveals the concentration of most profiles near the centre for the majority of axes, confirming that most spending categories are at moderate or low levels for most customers, while a small number of clusters extend significantly outward on specific axes. This visual concentration pattern is the radar equivalent of the z-score heatmap's near-zero cells for moderate segments. The axes where clear separation occurs (electronics for Techies, vegetables for Vegetarians, promotional sensitivity for Promoters) are precisely the axes that carry the highest discriminative power in the clustering distance. The combined radar is particularly useful for campaign planning: any axis where the target segment extends furthest from the centre while others remain near the origin represents an opportunity for category-specific messaging with minimal audience overlap risk.</p>
</div>
""", unsafe_allow_html=True)

    # Feature barplots
    st.markdown("""
<div style='margin-top:36px; margin-bottom:12px; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>7.3) Average spend per category by cluster: grouped bar charts</div>
</div>
""", unsafe_allow_html=True)
    _p = IMAGENS_DIR / "charts" / "feature_barplots.png"
    if _p.exists():
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Each panel shows one product category, with bars coloured by cluster and the y-axis representing the average lifetime spend in euros. This view complements the heatmap by making absolute scale differences explicit: the electronics panel, for example, reveals that Techies spend roughly twice the next-highest cluster on electronics, while the vegetables panel shows a moderate but consistent advantage for Vegetarians. The fish panel shows that Families and Loyalists have the highest fish spend, while Vegetarians and Wellness sit near zero, consistent with plant-based and health-conscious profiles that avoid animal protein. The petfood panel confirms that the petfood feature, while excluded from the clustering distance, does differentiate one cluster (Families) in the profiling stage. The videogames panel shows that Techies dominate this category, while most other segments spend near zero, making videogames the most concentrated single-segment category in the dataset and therefore the most targeted cross-sell opportunity available from any campaign built on these segments.</p>
</div>
""", unsafe_allow_html=True)

    # Boxplot grid
    st.markdown("""
<div style='margin-top:36px; margin-bottom:12px; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>7.4) Key variable distributions by cluster: boxplot grid</div>
</div>
""", unsafe_allow_html=True)
    _p = IMAGENS_DIR / "charts" / "boxplot_grid.png"
    if _p.exists():
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The boxplot grid shows the within-cluster distribution of four key variables: total lifetime spend, promotional purchase ratio, customer age, and tenure. Unlike mean-based heatmaps, boxplots expose the spread and skew within each cluster, information that is essential for assessing how targeted a campaign can realistically be. The total spend panel reveals that Loyalists have the highest median spend and also the widest interquartile range, meaning that this segment contains both very high and moderately high spenders. Promoters show a very narrow promotional sensitivity distribution clustered near 1.0, confirming that their defining characteristic is consistent, not occasional, promotion usage. The age panel reveals that Economizers and Promoters skew notably younger (avg. ~48–50 years) while Loyalists and Vegetarians are older on average (~57–58 years). Tenure follows a complementary pattern: Loyalists and Families have the longest tenure, while Techies and Economizers are newer customers with wider tenure distributions, consistent with a more recently acquired and more heterogeneous cohort. These within-cluster distributions inform the confidence level with which each segment can be targeted: narrow distributions mean higher message precision; wide distributions mean a broader or tiered communication strategy is more appropriate.</p>
</div>
""", unsafe_allow_html=True)

    # Community cards
    try:
        seg_summary = load_csv_data("segment_summary.csv")
        seg_meta_grid = {
            0: {"name": "Regulars", "desc": "Active but newer, deal-aware shoppers. Strong targets for onboarding to the loyalty program.", "icon_idx": 0},
            1: {"name": "Families", "desc": "Large households (avg. 5.38 kids). Loyal without needing promotions. Target with bulk-buying bundles.", "icon_idx": 1},
            2: {"name": "Economizers", "desc": "Restrained, low-friction spenders who buy at baseline. NOT deal-chasers; value baseline pricing.", "icon_idx": 2},
            3: {"name": "Vegetarians", "desc": "Full-price, promotion-resistant shoppers. Lead with curation/quality framing rather than discounts.", "icon_idx": 3},
            4: {"name": "Loyalists", "desc": "Highest LTV, highest tenure (13.5 years), and highest loyalty flag adoption (77.6%). Reward and protect.", "icon_idx": 4},
            5: {"name": "Techies", "desc": "Small households buying high-value tech. Cleanest electronics and audio cross-sell campaign audience.", "icon_idx": 5},
            6: {"name": "Wellness", "desc": "Quiet, low-maintenance, and low-complaint shoppers. Exert low friction and buy full price.", "icon_idx": 6},
            7: {"name": "Promoters", "desc": "The ultimate deal-seekers (+149% promo share vs overall). Perfect for price-led campaign stacking.", "icon_idx": 7}
        }
        cluster_images = {0:REGULARS_URI,1:FAMILIES_URI,2:ECONOMIZERS_URI,3:VEGETARIANS_URI,4:LOYALISTS_URI,5:TECHIES_URI,6:WELLNESS_URI,7:PROMOTERS_URI}
        st.markdown('<div id="nb4-8" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">8) Cluster interpretation - Your 8 customer communities</h2></div>', unsafe_allow_html=True)
        cards_list_html = []
        for idx, row in seg_summary.iterrows():
            c_id = int(row['cluster']); share = row['share_%']; custs = int(row['customers'])
            meta = seg_meta_grid.get(c_id, {"name": f"Cluster {c_id}", "desc": "No description.", "icon_idx": 0})
            img_uri = cluster_images.get(c_id, SLICES_URIS[c_id % len(SLICES_URIS)])
            cards_list_html.append(f"<div class='community-card'><div class='community-card-icon-container'><img src='{img_uri}' class='community-card-img' /></div><div><h3 class='community-card-title'>{meta['name']}</h3><div class='community-card-value'>{share:.1f}%</div><div class='community-card-sub'>{custs:,} customers</div><div class='community-card-desc'>{meta['desc']}</div></div><div class='community-card-arrow'>→</div></div>")
        st.markdown(f"<div class='communities-grid'>{''.join(cards_list_html)}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading segment summary: {e}")

    # Geographic scatter by cluster (NB4)
    st.markdown("""
<div id="nb4-9" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">9) Geographic check</h2></div>
<div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;'>Geographic distribution by cluster: static scatter</div>
""", unsafe_allow_html=True)
    _p = IMAGENS_DIR / "charts" / "geo_scatter.png"
    if _p.exists():
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The geographic scatter overlays all eight cluster labels on the latitude-longitude coordinate space, using the same colour palette as the radar and heatmap charts. Because geography was deliberately excluded from the clustering distance, any spatial pattern visible here is an emergent property of the behavioural segmentation rather than a modelling artefact. The chart reveals that clusters are not randomly mixed across space: Techies and Loyalists show a higher concentration in the central urban zone, consistent with the younger and higher-income urban customer profile identified in the geographic analysis notebook. Families are more evenly distributed across the suburban periphery, consistent with lower population density in residential areas outside the city centre. Promoters appear throughout the map with no strong geographic concentration, suggesting that price-sensitivity is a behavioural trait not constrained to a particular residential area. This geographic overlay is used as a final profiling validation step: if a cluster appeared concentrated exclusively in a single neighbourhood, that would raise a flag that geography had leaked into the segmentation through a correlated variable. The relatively mixed spatial distribution across clusters confirms that the model is capturing behavioural rather than geographic patterns.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""<div id="nb4-10" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">10) Final segment names & export</h2></div><div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;'>Explore Clustered Data</div>""", unsafe_allow_html=True)

    try:
        seg_summary = load_csv_data("segment_summary.csv")
        seg_spend = load_csv_data("segment_spend_profile.csv")
        seg_complaints = load_csv_data("segment_complaints_profile.csv")
        
        cluster_options = {
            0: "Cluster 0: Regulars",
            1: "Cluster 1: Families",
            2: "Cluster 2: Economizers",
            3: "Cluster 3: Vegetarians",
            4: "Cluster 4: Loyalists",
            5: "Cluster 5: Techies",
            6: "Cluster 6: Wellness",
            7: "Cluster 7: Promoters"
        }
        
        selected_cluster = st.selectbox("Select cluster to inspect", options=list(cluster_options.keys()), format_func=lambda x: cluster_options[x])
        cluster_color = SEGMENT_COLORS.get(selected_cluster, "#ea580c")
        
        row_summary = seg_summary[pd.to_numeric(seg_summary['cluster'], errors='coerce') == selected_cluster]
        row_spend = seg_spend[pd.to_numeric(seg_spend['cluster'], errors='coerce') == selected_cluster]
        row_complaints = seg_complaints[pd.to_numeric(seg_complaints['cluster'], errors='coerce') == selected_cluster]
        
        if not row_summary.empty:
            cust_count = row_summary.iloc[0]['customers']
            cust_share = row_summary.iloc[0]['share_%']
            num_complaints = row_complaints.iloc[0]['avg_complaints'] if not row_complaints.empty else 0.0
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Customers in cluster", f"{cust_count:,}")
            c2.metric("Share of customer base", f"{cust_share:.1f}%")
            c3.metric("Average complaints", f"{num_complaints:.2f}")
            
            st.markdown("#### Average Spend per Product Category (€)")
            spend_cols = [c for c in row_spend.columns if c.startswith("lifetime_spend_")]
            spend_vals = [row_spend.iloc[0][col] for col in spend_cols]
            spend_labels = [col.replace("lifetime_spend_", "").replace("_", " ").title() for col in spend_cols]
            
            spend_df = pd.DataFrame({
                "Category": spend_labels,
                "Average Spend (€)": spend_vals
            })
            
            spend_chart_cluster = alt.Chart(spend_df).mark_bar(color=cluster_color, cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                x=alt.X("Category:N", sort="-y", title="Spend Category"),
                y=alt.Y("Average Spend (€):Q", title="Average Spend (€)"),
                tooltip=["Category", alt.Tooltip("Average Spend (€):Q", format=",.2f")]
            ).properties(height=300)
            
            st.altair_chart(spend_chart_cluster, use_container_width=True)
            st.markdown(f'''
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>This bar chart displays the average lifetime spend per product category for the selected cluster. The categories are sorted in descending order of spend, making the dominant consumption driver immediately visible. Comparing this breakdown across clusters reveals the behavioural specialisation that distinguishes each segment: Loyalists and Families show elevated grocery spend, Techies exhibit disproportionate electronics expenditure, and Vegetarians record near-zero meat and fish categories. This per-cluster decomposition is the primary evidence base for the campaign targeting logic in Notebook 5, confirming that the product associations discovered by the Apriori algorithm are consistent with the underlying spend structure of each segment.</p>
</div>
''', unsafe_allow_html=True)
    except Exception as e:
        st.info("Dynamic cluster files not fully loaded. Displaying static mockup.")

    st.markdown("""
<div style='padding:24px; border-radius:16px; background:#111827; border:1px solid #374151; margin-top:48px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:12px;'>Notebook 4: Conclusions</div>
  <p style='font-size:16px; color:#d1d5db; line-height:1.9; margin:0 0 12px 0;'>The eight segments produced by the K-Means model are behaviourally distinct and individually interpretable. Each cluster has a clearly identifiable spending driver: Families and Loyalists are high-value, full-price purchasers; Vegetarians and Wellness customers are produce and hygiene specialists with low promotion sensitivity; Techies dominate electronics; Promoters are price-driven with the highest promotion rate; Economizers are budget-conscious across all categories; and Regulars represent the broad baseline without a dominant specialisation.</p>
  <p style='font-size:16px; color:#d1d5db; line-height:1.9; margin:0 0 12px 0;'>Demographic and geographic validation confirms that the segments reflect genuine behavioural patterns rather than artefacts of sampling. Gender is balanced across all clusters. Families are identified not by the label but by statistical confirmation of household size. The Loyalists segment is validated by the 13.5-year average tenure and the 77.6% loyalty card rate. Geographic distribution shows that all eight segments exist throughout the Lisbon Metropolitan Area, supporting the decision to exclude location from the clustering feature set.</p>
  <p style='font-size:16px; color:#d1d5db; line-height:1.9; margin:0;'>Segment names are assigned post-hoc, grounded in the z-score evidence from Section 3 and the normalised spend profiles from Section 6. No name was assigned before examining the data. These named communities are the direct input to Notebook 5, where association rule mining is run per segment to generate targeted campaign recommendations.</p>
</div>
""", unsafe_allow_html=True)
    render_footer()

elif selected_page == "Targeter Promotion":
    _basket_uri = load_image_as_base64(IMAGENS_DIR / "basket.png")
    st.markdown(f"""
    <div style='display:flex; align-items:flex-start; justify-content:space-between; margin-top:0; margin-bottom:24px; gap:32px;'>
        <div style='flex:1;'>
          <h2 style='font-size:56px; font-weight:800; color:#000000; margin:0 0 16px 0; letter-spacing:-0.03em;'>Targeter Promotion</h2>
          <div style='font-size:11px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#9ca3af; margin-bottom:10px;'>Notebook 5: Association Rules</div>
          <p style='font-size:18px; color:#374151; line-height:1.9; margin:0;'>The final notebook operationalises the cluster definitions by mining product associations specific to each community. Using the Apriori algorithm on the transaction-level <code>customer_basket</code> dataset, it discovers what products are frequently bought together by each segment. These patterns are then translated into concrete, data-driven cross-selling campaigns.</p>
        </div>
        <img src='{_basket_uri}' style='height:420px; width:auto; object-fit:contain; border-radius:16px; flex-shrink:0;' />
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-bottom:32px;'>
      <div>
        <div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:14px; margin-bottom:28px;'>
          <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 18px; text-align:center;'>
            <div style='font-size:11px; font-weight:600; color:#9ca3af; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;'>Min Support</div>
            <div style='font-size:26px; font-weight:800; color:#111827;'>1%</div>
            <div style='font-size:12px; color:#6b7280; margin-top:2px;'>intentionally low</div>
          </div>
          <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 18px; text-align:center;'>
            <div style='font-size:11px; font-weight:600; color:#9ca3af; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;'>Min Confidence</div>
            <div style='font-size:26px; font-weight:800; color:#111827;'>20%</div>
          </div>
          <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 18px; text-align:center;'>
            <div style='font-size:11px; font-weight:600; color:#9ca3af; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;'>Min Lift</div>
            <div style='font-size:26px; font-weight:800; color:#111827;'>1.2</div>
          </div>
          <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 18px; text-align:center;'>
            <div style='font-size:11px; font-weight:600; color:#9ca3af; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;'>Robustness split</div>
            <div style='font-size:26px; font-weight:800; color:#111827;'>80/20</div>
            <div style='font-size:12px; color:#6b7280; margin-top:2px;'>train/test</div>
          </div>
        </div>
      </div>

      <div style='background:linear-gradient(145deg, #ffffff, #f9fafb); border:1px solid #e5e7eb; border-radius:12px; padding:20px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05); margin-top:20px;'>
        <div style='font-size:14px; font-weight:800; color:#111827; margin-bottom:16px; border-bottom:2px solid #f3eee6; padding-bottom:8px;'>Notebook 5 Index</div>
        <div style='display:flex; flex-wrap:wrap; gap:12px; align-items:center;'>
          <a href="#nb5-1" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>1) Imports & data loading</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb5-2" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>2) Transaction preparation per segment</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb5-3" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>3) Apriori parameters</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb5-4" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>4) Association rules per segment</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb5-5" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>5) Top rules per segment</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb5-6" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>6) Rule robustness check</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb5-7" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>7) Campaign suggestions & Creative campaign texts</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb5-8" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>8) Final interpretation</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb5-9" style="text-decoration:none;"><div style='font-size:18px; color:#374151; font-weight:500;'>9) Cupons</div></a>
        </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


    st.markdown('''
<div id="nb5-1" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">1) Imports & data loading</h2></div>
<div style='font-size:16px; color:#6b7280; line-height:1.8; margin-bottom:24px;'>Initialisation of the environment, loading the customer transactions and cluster assignments.</div>

<div id="nb5-2" style="margin-top:32px; margin-bottom:24px; border-top:1px solid #e5e7eb; padding-top:24px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">2) Transaction preparation per segment</h2></div>
<div style='font-size:16px; color:#6b7280; line-height:1.8; margin-bottom:24px;'>Separating the global transactions into segment-specific baskets to run Apriori per community.</div>
''', unsafe_allow_html=True)

    st.markdown('''
<div id="nb5-3" style="margin-top:32px; margin-bottom:24px; border-top:1px solid #e5e7eb; padding-top:24px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">3) Apriori parameters</h2></div>
<div style='display:grid; grid-template-columns:repeat(2,1fr); gap:20px; margin-bottom:24px;'>
    <div style='border-left:3px solid #111827; padding-left:20px;'>
        <div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>Why support is set at 1%</div>
        <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Rules are mined per segment: each sub-population has far fewer transactions than the full dataset. A 1% support threshold ensures enough rules are discovered while still requiring meaningful co-occurrence frequency within each community.</div>
    </div>
    <div style='border-left:3px solid #111827; padding-left:20px;'>
        <div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>Lift-derived campaign discounts</div>
        <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Suggested campaign discounts are not fixed; they are derived from the lift value of each rule. A higher lift means a stronger-than-random co-purchase signal, which justifies a larger promotional incentive. This ties the commercial decision directly to statistical evidence.</div>
    </div>
</div>
''', unsafe_allow_html=True)
    # Chart 1: Top rules by lift per segment (grouped horizontal bar)
    st.markdown("""
<div id="nb5-4" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">4) Association rules per segment - Association rules by lift per segment</h2></div>
""", unsafe_allow_html=True)
    rules_df = load_csv_data("segment_campaign_rules.csv")
    rules_df["rule_label"] = rules_df["if_buys"] + " -> " + rules_df["promote"]
    lift_chart = alt.Chart(rules_df).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        y=alt.Y("rule_label:N", sort="-x", title="Rule (if buys -> promote)"),
        x=alt.X("lift:Q", title="Lift"),
        color=alt.Color("segment:N", title="Segment", scale=alt.Scale(domain=list(SEGMENT_NAME_COLORS.keys()), range=list(SEGMENT_NAME_COLORS.values()))),
        tooltip=["segment", "if_buys", "promote", alt.Tooltip("confidence:Q", format=".2f", title="Confidence"), alt.Tooltip("lift:Q", format=".2f", title="Lift")]
    ).properties(height=520)
    st.altair_chart(lift_chart, use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Lift measures how much more likely two products are to be purchased together compared to what would be expected if they were purchased independently. A lift of 1.0 indicates no association beyond chance; a lift of 1.2 indicates that the joint purchase is 20% more likely than random co-occurrence; values above 2.0 represent strong non-random co-purchase patterns. The chart reveals that the strongest lift values are concentrated in a small number of rules: vegetable-combination rules for produce-focused segments (Regulars, Promoters) and technology cross-sell rules (Techies, Economizers). These high-lift rules are the primary candidates for campaign deployment because they represent the strongest statistical evidence that promoting item B to a customer who bought item A will generate a genuine incremental purchase rather than recovering a purchase that would have occurred anyway. Lower-lift rules remain valid but justify smaller promotional incentives.</p>
</div>
""", unsafe_allow_html=True)

    # Chart 2: Confidence vs lift scatter
    st.markdown("""
<div id="nb5-5" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">5) Top rules per segment - Confidence vs. lift across all segments</h2></div>
""", unsafe_allow_html=True)
    scatter_fig = px.scatter(
        rules_df,
        x="confidence",
        y="lift",
        color="segment",
        color_discrete_map=SEGMENT_NAME_COLORS,
        hover_data={"if_buys": True, "promote": True, "segment": True, "confidence": ":.2f", "lift": ":.2f"},
        labels={"confidence": "Confidence", "lift": "Lift", "segment": "Segment"}
    )
    scatter_fig.update_traces(marker=dict(size=10))
    scatter_fig.update_layout(margin=dict(l=40, r=20, t=60, b=60), height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(scatter_fig, use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Confidence and lift are complementary metrics that together characterise the commercial quality of an association rule. Confidence measures the conditional probability that a customer who buys the antecedent will also buy the consequent; lift adjusts this probability for the base rate of the consequent across all transactions. Rules positioned in the upper-right quadrant of this scatter plot (high confidence and high lift) represent the strongest candidates for campaign deployment: they are both reliable (customers who trigger the rule frequently also complete the recommended purchase) and non-trivial (the co-purchase is substantially more likely than random). Rules with high confidence but modest lift may reflect a consequent that is purchased frequently regardless of the antecedent, diminishing the causal interpretation of the rule. Rules with high lift but low confidence identify genuine but infrequent co-purchase patterns that may be better suited to targeted micro-campaigns than broad promotional rollouts.</p>
</div>
""", unsafe_allow_html=True)


    st.markdown('''
<div id="nb5-6" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">6) Rule robustness check</h2></div>
<div style='border-left:3px solid #111827; padding-left:20px; margin-bottom:24px;'>
    <div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>Robustness validation</div>
    <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Each segment's rules are validated on an 80/20 train/test split. Segments with many matched rules and low mean lift difference between train and test have stable co-purchase patterns. Segments with few matched rules should be interpreted with caution.</div>
</div>
''', unsafe_allow_html=True)

    st.markdown('''
<div id="nb5-7" style="margin-top:32px; margin-bottom:24px; border-top:1px solid #e5e7eb; padding-top:24px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">7) Campaign suggestions & Creative campaign texts</h2></div>
<div style='border-left:3px solid #111827; padding-left:20px; margin-bottom:24px;'>
    <div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>Excluded recommendations: Vegetarians (cluster 3)</div>
    <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Chicken, meat, and fish are excluded from recommendations for cluster 3 (Vegetarians). The Apriori rules initially suggested these items, but they contradict the segment's defining behavioural trait. Notebook 4 confirms that this segment's identity is plant-based, and the exclusion ensures campaign coherence.</div>
</div>
''', unsafe_allow_html=True)
    try:
        campaign_rules = load_csv_data("segment_campaign_rules.csv")
        unique_segments = campaign_rules['segment'].unique()
        
        # Segment label map to color rules by matching SEGMENT_COLORS keys
        segment_label_map = {
            "Regulars": 0, "Families": 1, "Economizers": 2, "Vegetarians": 3,
            "Loyalists": 4, "Techies": 5, "Wellness": 6, "Promoters": 7
        }
        
        selected_segment = st.selectbox("Select segment for campaigns", options=unique_segments)
        segment_color_idx = segment_label_map.get(selected_segment, 0)
        promo_color = SEGMENT_COLORS.get(segment_color_idx, "#ea580c")
        
        segment_rules = campaign_rules[campaign_rules['segment'] == selected_segment]
        
        st.markdown(f"<div id='nb5-8' style='margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;'><h2 style='font-size:24px; font-weight:800; color:#111827; margin:0;'>8) Final interpretation</h2></div>\n\n#### Top Association Rules for {selected_segment}", unsafe_allow_html=True)
        
        for idx, rule in segment_rules.iterrows():
            if_buys = rule['if_buys']
            promote = rule['promote']
            conf = rule['confidence'] * 100
            lift = rule['lift']
            
            st.markdown(f"""
            <div style='background: #ffffff; border: 1px solid rgba(0,0,0,0.05); border-radius: 12px; padding: 18px; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between;'>
              <div style='flex: 2;'>
                <div style='font-size: 11px; text-transform: uppercase; color: #8c8c8c; font-weight: 700; margin-bottom: 4px;'>Trigger purchase</div>
                <div style='font-size: 16px; font-weight: 700; color: #1a1a1a;'>If customer buys: <code style='font-size: 13px; background: rgba(0,0,0,0.04); padding: 3px 6px; border-radius: 6px;'>{if_buys}</code></div>
              </div>
              <div style='flex: 1.5; border-left: 1px solid rgba(0,0,0,0.08); padding-left: 20px;'>
                <div style='font-size: 11px; text-transform: uppercase; color: {promo_color}; font-weight: 700; margin-bottom: 4px;'>Targeted promotion</div>
                <div style='font-size: 16px; font-weight: 700; color: {promo_color};'>Promote: <strong>{promote.upper()}</strong></div>
              </div>
              <div style='flex: 1; border-left: 1px solid rgba(0,0,0,0.08); padding-left: 20px; text-align: center;'>
                <div style='font-size: 11px; text-transform: uppercase; color: #8c8c8c; font-weight: 700; margin-bottom: 4px;'>Confidence</div>
                <div style='font-size: 18px; font-weight: 800; color: #000000;'>{conf:.0f}%</div>
              </div>
              <div style='flex: 1; border-left: 1px solid rgba(0,0,0,0.08); padding-left: 20px; text-align: center;'>
                <div style='font-size: 11px; text-transform: uppercase; color: #8c8c8c; font-weight: 700; margin-bottom: 4px;'>Lift ratio</div>
                <div style='font-size: 18px; font-weight: 800; color: #10b981;'>{lift:.2f}x</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.info("No campaign rules CSV found. Add datasets/segment_campaign_rules.csv to enable targeter simulation.")

    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The per-segment rule cards above show the top association rules discovered by the Apriori algorithm for the selected community. Each rule states a trigger purchase (the antecedent) and a recommended promotional target (the consequent), accompanied by confidence and lift metrics. Confidence quantifies how reliably a customer who buys the trigger item also completes the recommended purchase; lift measures how much stronger that co-purchase is compared to what would be expected by chance. A high-confidence, high-lift rule indicates a commercially robust promotional signal: the intervention is both predictable and non-trivial. Campaign discounts are derived directly from lift: a lift of 1.2 justifies a modest incentive, while a lift above 2.0 supports a more aggressive promotional investment. Rules with lift below 1.2 were filtered out during parameterisation as they do not represent actionable co-purchase patterns above baseline. The creative coupon designs in Section 9 operationalise the top rule for each segment, translating the statistical output into ready-to-deploy campaign assets.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown('''
<div id="nb5-9" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">9) Cupons (Campaign Creative)</h2></div>
<div class="coupons-grid-wrapper">
''', unsafe_allow_html=True)
    import os, base64
    cupoes_dir = IMAGENS_DIR / "cupoes"

    # Filename -> (segment label, short description)
    COUPON_META = {
        "1.png": (
            "Promoters",
            "Bundle salad, tomatoes, asparagus and carrots together for 29% off. Targets high-frequency produce buyers who respond strongly to multi-item bundle promotions."
        ),
        "2.png": (
            "Families",
            "The Family Breakfast Combo: cereals, chocolate bread, eggs and fresh bread bundled for 10% off the whole basket. Designed for large-household shoppers who prioritise everyday staples."
        ),
        "eco.png": (
            "Economizers",
            "Stock up on asparagus and spinach and bring avocado home for 23% less. Targets budget-conscious shoppers who seek value on healthy everyday greens."
        ),
        "Adobe Express - file.png": (
            "Regulars",
            "Your weekly staples for less: avocado, carrots, asparagus and salad at 25% off. Rewards loyal, routine shoppers who consistently fill their basket with fresh produce."
        ),
        "Untitled - 09 de junho de 2026 às 00.41.01 (1).png": (
            "Vegetarians",
            "Add cooking oil and tomato sauce to your basket and babies food joins at 10% off. Designed for plant-based shoppers who regularly purchase cooking essentials."
        ),
        "Untitled - 09 de junho de 2026 às 00.41.01 (1) (1).png": (
            "Wellness",
            "Powered up on an energy bar? Treat your ears too: AirPods at 10% off. Targets health-oriented customers who also invest in personal wellbeing and lifestyle products."
        ),
        "Untitled - 09 de junho de 2026 às 00.41.01 (1) (4).png": (
            "Loyalists",
            "Members who buy an iPhone 10 unlock AirPods at 10% off, exclusively for them. A VIP reward for high-value, brand-loyal customers who purchase premium electronics."
        ),
        "Untitled - 09 de junho de 2026 às 00.41.01 (1) (5).png": (
            "Techies",
            "Complete the setup: buy bluetooth headphones and an energy drink to unlock AirPods at 24% off. Targets tech-savvy customers who pair digital accessories with high-energy lifestyle purchases."
        ),
    }

    if cupoes_dir.exists():
        images = [f for f in os.listdir(cupoes_dir) if f.endswith(".png")]
        html_imgs = ""
        for f in images:
            img_path = cupoes_dir / f
            with open(img_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            segment, description = COUPON_META.get(f, ("", ""))
            caption_html = f'<div class="coupon-caption"><div class="coupon-segment">{segment}</div><div class="coupon-desc">{description}</div></div>' if (segment or description) else ""
            html_imgs += f'<div class="coupon-item"><img src="data:image/png;base64,{encoded_string}" style="width:100%; border-radius:12px; mix-blend-mode: multiply;">{caption_html}</div>'

        grid_html = f'''
        <style>
        .coupons-scroller {{
            display: grid;
            grid-template-columns: repeat(4, 300px);
            grid-template-rows: repeat(2, auto);
            gap: 24px;
            overflow-x: auto;
            padding-bottom: 16px;
        }}
        .coupon-item {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .coupon-caption {{
            padding: 0 4px;
        }}
        .coupon-segment {{
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #6b7280;
            margin-bottom: 4px;
        }}
        .coupon-desc {{
            font-size: 13px;
            color: #374151;
            line-height: 1.6;
        }}
        @media (max-width: 800px) {{
            .coupons-scroller {{
                grid-template-columns: repeat(4, 250px);
            }}
        }}
        </style>
        <div class="coupons-scroller">
            {html_imgs}
        </div>
        '''
        st.markdown(grid_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
<div style='padding:24px; border-radius:16px; background:#111827; border:1px solid #374151; margin-top:48px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:12px;'>Notebook 5: Conclusions</div>
  <p style='font-size:16px; color:#d1d5db; line-height:1.9; margin:0 0 12px 0;'>The Apriori algorithm, applied independently to each of the eight customer segments, identifies co-purchase patterns that are both statistically robust and commercially actionable. The choice to mine rules per segment rather than on the full transaction pool is deliberate: global rules would reflect the aggregate behaviour of all customers and mask the segment-specific associations that make targeted campaigns possible. Segment-level mining recovers patterns that would be invisible at the population level.</p>
  <p style='font-size:16px; color:#d1d5db; line-height:1.9; margin:0 0 12px 0;'>The parameter choices (minimum support 1%, minimum confidence 20%, minimum lift 1.2) reflect the reduced transaction volumes in per-segment populations. A higher support threshold would suppress too many rules for smaller segments; a lower lift floor would include associations that are statistically marginal and commercially unconvincing. The 80/20 robustness validation confirms that the rules are not artefacts of the training sample.</p>
  <p style='font-size:16px; color:#d1d5db; line-height:1.9; margin:0;'>The coupon designs in Section 9 translate the top rule per segment into ready-to-deploy campaign assets. The creative direction for each coupon is grounded in the segment's defining behavioural trait: vegetable bundles for produce-focused segments, technology cross-sells for Techies, volume deals for Families, and so on. This alignment between statistical rule and campaign creative ensures that the promotional message reinforces rather than contradicts the segment identity established in Notebook 4.</p>
</div>
""", unsafe_allow_html=True)
    render_footer()
elif selected_page == "Conclusion & Recommendations":
    st.markdown("""
    <div style='margin-top: 0px; margin-bottom: 32px;'>
        <h2 style='font-size: 56px; font-weight: 800; color: #000000; margin: 0; letter-spacing: -0.03em;'>Conclusion & Recommendations</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <!-- Executive Summary -->
    <div style='background:#111827; border-radius:16px; padding:28px 32px; margin-bottom:32px;'>
      <div style='font-size:11px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#9ca3af; margin-bottom:12px;'>Executive Summary</div>
      <p style='font-size:16px; color:#d1d5db; line-height:1.9; margin:0;'>
        This project segmented <strong style="color:#fff;">33,038 supermarket customers</strong> into <strong style="color:#fff;">8 behavioural communities</strong> using K-Means clustering on annual category spend and promotion sensitivity. The pipeline spans five notebooks: data exploration and quality diagnosis (NB0), preprocessing and feature engineering (NB1), geographic analysis (NB2), systematic model selection and validation (NB3), cluster characterisation and naming (NB4), and campaign design via association rule mining (NB5). Every modelling decision (algorithm, k, scaler, feature set, outlier strategy) was made empirically, benchmarked against alternatives, and is documented in this app.
      </p>
    </div>

    <!-- Key decisions -->
    <div style='font-size:24px; font-weight:800; color:#111827; margin-bottom:16px;'>Key methodological decisions</div>
    <div style='display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:32px;'>

      <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 18px;'>
        <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:6px;'>Outlier strategy</div>
        <div style='font-size:16px; font-weight:700; color:#111827; margin-bottom:4px;'>Consensus separation</div>
        <div style='font-size:16px; color:#374151; line-height:1.6;'>Only customers flagged simultaneously by IQR, DBSCAN, and a third detection method are separated (982 customers, 3.0%). This avoids aggressive exclusion while still protecting the K-Means centroid computation from multivariate extremes. Outliers are reattached after fitting.</div>
      </div>

      <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 18px;'>
        <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:6px;'>Feature engineering</div>
        <div style='font-size:16px; font-weight:700; color:#111827; margin-bottom:4px;'>Tenure, cyclic hour, annual rates</div>
        <div style='font-size:16px; color:#374151; line-height:1.6;'>Raw dates and hours are not interpretable as distances. Tenure normalises spend for relationship length. Sin/cos encoding makes shopping-hour distances circular. Annual spend rates remove the confound between high spend and long tenure.</div>
      </div>

      <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 18px;'>
        <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:6px;'>Imputation</div>
        <div style='font-size:16px; font-weight:700; color:#111827; margin-bottom:4px;'>KNN imputation (k=5)</div>
        <div style='font-size:16px; color:#374151; line-height:1.6;'>Missing values are inferred from the five most behaviourally similar customers, preserving local correlation structure. Imputation is run after outlier separation to prevent extreme customers from biasing neighbour lookups.</div>
      </div>

      <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 18px;'>
        <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:6px;'>Algorithm</div>
        <div style='font-size:16px; font-weight:700; color:#111827; margin-bottom:4px;'>K-Means over DBSCAN / GMM</div>
        <div style='font-size:16px; color:#374151; line-height:1.6;'>K-Means produces hard, non-overlapping assignments with directly interpretable centroids. DBSCAN labelled 20–40% of customers as noise. GMM adds complexity without a clear interpretability gain. Ward linkage and SOM were tested as cross-checks and confirmed the K-Means solution.</div>
      </div>

      <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 18px;'>
        <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:6px;'>Scaler</div>
        <div style='font-size:16px; font-weight:700; color:#111827; margin-bottom:4px;'>MinMaxScaler over RobustScaler</div>
        <div style='font-size:16px; color:#374151; line-height:1.6;'>After outlier separation the input distribution is already clean. RobustScaler's IQR clipping is redundant and compresses spend differences that are real and informative. MinMaxScaler preserves magnitude ordering within each category while normalising all features to [0, 1].</div>
      </div>

      <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 18px;'>
        <div style='font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:6px;'>Feature set & k</div>
        <div style='font-size:16px; font-weight:700; color:#111827; margin-bottom:4px;'>K=8, spend + promo, no groceries</div>
        <div style='font-size:16px; color:#374151; line-height:1.6;'>Groceries are universal and add noise. Including promotion sensitivity recovers the deal-seeker segment. K=8 was selected from a 13 × 2 × 5 grid (feature sets × scalers × k values) by highest silhouette score, cleanest elbow, and maximum business interpretability.</div>
      </div>

    </div>

    <!-- Key findings per segment -->
    <div style='font-size:24px; font-weight:800; color:#111827; margin-bottom:16px;'>Key findings per segment</div>
    <div style='display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:32px;'>
      <div style='border-left:4px solid #b76563; background:#fdf8f8; border-radius:8px; padding:14px 16px;'>
        <div style='font-size:16px; font-weight:700; color:#b76563; margin-bottom:4px;'>Regulars (0)</div>
        <div style='font-size:16px; color:#374151; line-height:1.6;'>Broad shoppers with no dominant category. Newer customer base (lower tenure). Best entry point for loyalty programme onboarding and first-purchase incentives.</div>
      </div>
      <div style='border-left:4px solid #bc7933; background:#fdf9f4; border-radius:8px; padding:14px 16px;'>
        <div style='font-size:16px; font-weight:700; color:#bc7933; margin-bottom:4px;'>Families (1)</div>
        <div style='font-size:16px; color:#374151; line-height:1.6;'>Highest household size (+169% children), highest total spend, highest meat and fish. Loyal without needing promotions. Respond to volume deals and family-pack formats.</div>
      </div>
      <div style='border-left:4px solid #687643; background:#f7f9f3; border-radius:8px; padding:14px 16px;'>
        <div style='font-size:16px; font-weight:700; color:#687643; margin-bottom:4px;'>Economizers (2)</div>
        <div style='font-size:16px; color:#374151; line-height:1.6;'>Low total spend, moderate promo sensitivity, newer tenure. Budget-conscious. Respond to everyday low prices and clear value signalling.</div>
      </div>
      <div style='border-left:4px solid #b64828; background:#fdf5f2; border-radius:8px; padding:14px 16px;'>
        <div style='font-size:16px; font-weight:700; color:#b64828; margin-bottom:4px;'>Vegetarians (3)</div>
        <div style='font-size:16px; color:#374151; line-height:1.6;'>Highest vegetable spend (+176%), near-zero meat/fish, lowest promotion usage (-69%). Full-price, quality-driven. Lead with curation, provenance, and seasonal produce messaging.</div>
      </div>
      <div style='border-left:4px solid #c88d40; background:#fdf9f2; border-radius:8px; padding:14px 16px;'>
        <div style='font-size:16px; font-weight:700; color:#c88d40; margin-bottom:4px;'>Loyalists (4)</div>
        <div style='font-size:16px; color:#374151; line-height:1.6;'>Highest LTV. 77.6% loyalty flag, 13.5-year average tenure, highest grocery spend (+80%). Protect with VIP recognition and early access; avoid discount dependence.</div>
      </div>
      <div style='border-left:4px solid #368689; background:#f2f9f9; border-radius:8px; padding:14px 16px;'>
        <div style='font-size:16px; font-weight:700; color:#368689; margin-bottom:4px;'>Techies (5)</div>
        <div style='font-size:16px; color:#374151; line-height:1.6;'>Highest electronics (+284%) and videogames (+289%). Small households. Best cross-sell audience for tech accessories at checkout and subscription services.</div>
      </div>
      <div style='border-left:4px solid #36668d; background:#f2f6fa; border-radius:8px; padding:14px 16px;'>
        <div style='font-size:16px; font-weight:700; color:#36668d; margin-bottom:4px;'>Wellness (6)</div>
        <div style='font-size:16px; color:#374151; line-height:1.6;'>Highest hygiene spend (+146%), high vegetables (+107%), low meat/fish/alcohol. Quiet, low-friction, full-price shoppers. Respond to health-focused product ranges and clean-label messaging.</div>
      </div>
      <div style='border-left:4px solid #6c4d36; background:#f7f4f1; border-radius:8px; padding:14px 16px;'>
        <div style='font-size:16px; font-weight:700; color:#6c4d36; margin-bottom:4px;'>Promoters (7)</div>
        <div style='font-size:16px; color:#374151; line-height:1.6;'>Highest promotion rate (+149%), lowest absolute spend. Youngest average age. Ideal for stacked discount campaigns and loss-leader cross-sells that convert promo visits into fuller baskets.</div>
      </div>
    </div>

    <!-- Strategic recommendations -->
    <div style='font-size:24px; font-weight:800; color:#111827; margin-bottom:16px;'>Strategic recommendations</div>
    <div style='display:grid; grid-template-columns:repeat(2,1fr); gap:16px; margin-bottom:32px;'>

      <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:18px 20px;'>
        <div style='font-size:16px; font-weight:700; color:#111827; margin-bottom:6px;'>1. Protect and reward high-LTV segments first</div>
        <div style='font-size:14px; color:#374151; line-height:1.7;'>Loyalists (13.5-year tenure, 77.6% loyalty flag) and Families (highest total spend) are the most commercially valuable segments. Any churn risk from these groups disproportionately impacts revenue. Implement a VIP tier with early product access, dedicated checkout lanes, and personalised communication before investing in acquisition.</div>
      </div>

      <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:18px 20px;'>
        <div style='font-size:16px; font-weight:700; color:#111827; margin-bottom:6px;'>2. Use association rules for cross-sell at the point of sale</div>
        <div style='font-size:14px; color:#374151; line-height:1.7;'>NB5 mined high-lift product pairs per segment. These should be deployed as dynamic checkout suggestions: show segment-specific cross-sell prompts based on basket contents at the time of purchase. High-lift rules (>2.0) generate incremental purchases rather than just recovering ones that would happen anyway.</div>
      </div>

      <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:18px 20px;'>
        <div style='font-size:16px; font-weight:700; color:#111827; margin-bottom:6px;'>3. Convert Promoters without training discount dependency</div>
        <div style='font-size:14px; color:#374151; line-height:1.7;'>Promoters respond strongly to discounts but have the lowest total spend. The campaign objective is to use price-led entry offers to drive basket expansion, not to sustain a permanently discounted relationship. Use time-limited bundles and auto-upgrade mechanisms that surface full-price products once a promotional relationship is established.</div>
      </div>

      <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:18px 20px;'>
        <div style='font-size:16px; font-weight:700; color:#111827; margin-bottom:6px;'>4. Differentiate Vegetarians and Wellness with curation, not price</div>
        <div style='font-size:14px; color:#374151; line-height:1.7;'>Both segments are promotion-resistant and buy at full price. Discounting these segments signals commodity positioning, which conflicts with their quality orientation. Lead with curated shelves, provenance labelling, seasonal produce features, and health-claim product ranges.</div>
      </div>

      <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:18px 20px;'>
        <div style='font-size:16px; font-weight:700; color:#111827; margin-bottom:6px;'>5. Target Techies with electronics cross-sell at checkout</div>
        <div style='font-size:14px; color:#374151; line-height:1.7;'>Techies have the strongest electronics affinity in the dataset (+284%) and are also concentrated in the urban core. A dedicated electronics endcap or app-based tech accessory bundle, shown when their basket contains a tech item, is a high-precision intervention with minimal spend per conversion.</div>
      </div>

      <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:18px 20px;'>
        <div style='font-size:16px; font-weight:700; color:#111827; margin-bottom:6px;'>6. Use geography to allocate resources, not to define segments</div>
        <div style='font-size:14px; color:#374151; line-height:1.7;'>Geography was deliberately excluded from the clustering feature set. The geographic overlay in NB4 shows that all 8 segments are distributed across the Lisbon Metropolitan Area, confirming that behaviour, not location, drives segmentation. Geography should inform store-level assortment and staff allocation, but CRM communications should be driven by segment membership.</div>
      </div>

    </div>


    """, unsafe_allow_html=True)
    render_footer()

elif selected_page == "Customer Simulator":
    st.markdown("""
    <div style='margin-top: 0px; margin-bottom: 24px;'>
        <h2 style='font-size: 56px; font-weight: 800; color: #000000; margin: 0; letter-spacing: -0.03em;'>Customer Simulator</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='page-shell'>
      <div class='page-text'>
        <p>Simulate a customer's spending and complaints behavior to classify them into their most likely K-Means segment. The simulator uses the overall averages from the dataset to compute a normalized Euclidean distance to each segment centroid, assigning the simulated customer to the nearest community in real-time.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-bottom:28px;'>
      <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>How to Play</div>
      <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>
        A virtual customer arrives with a pre-loaded basket. Add items you think they would buy, then checkout.
        After checkout, guess which of the eight customer segments they belong to based on their purchases.
        Score a point for every correct identification. Can you beat the algorithm?
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Data ─────────────────────────────────────────────────────────────
    _SHOP_EMOJI = {
        "airpods":"🎧","almonds":"🥜","antioxydant juice":"🧃","asparagus":"🌿",
        "avocado":"🥑","babies food":"👶","bacon":"🥓","barbecue sauce":"🍖",
        "beer":"🍺","black beer":"🍺","black tea":"🫖","blueberries":"🫐",
        "bluetooth headphones":"🎧","body spray":"✨","bramble":"🌿","brownies":"🍫",
        "burger sauce":"🍔","burgers":"🍔","butter":"🧈","cake":"🎂",
        "candy bars":"🍬","canned_tuna":"🐟","carrots":"🥕","cat food":"🐈",
        "catfish":"🐟","cauliflower":"🥦","cereals":"🥣","champagne":"🥂",
        "chicken":"🍗","chili":"🌶","chocolate":"🍫","chocolate bread":"🍫",
        "chutney":"🫙","cider":"🍺","cologne":"🧴","cookies":"🍪",
        "cooking oil":"🫙","corn":"🌽","cottage cheese":"🧀","cotton buds":"🪥",
        "cream":"🥛","deodorant":"💨","dessert wine":"🍷","dog food":"🐕",
        "eggplant":"🍆","eggs":"🥚","energy bar":"⚡","energy drink":"⚡",
        "escalope":"🥩","extra dark chocolate":"🍫","final fantasy XIX":"🎮",
        "final fantasy XX":"🎮","final fantasy XXII":"🎮","flax seed":"🌾",
        "french fries":"🍟","french wine":"🍷","fresh bread":"🍞","fresh tuna":"🐟",
        "fromage blanc":"🧀","frozen smoothie":"🥤","frozen vegetables":"🥦",
        "gadget for tiktok streaming":"📱","gluten free bar":"💪","grated cheese":"🧀",
        "green beans":"🫘","green grapes":"🍇","green tea":"🍵","ground beef":"🥩",
        "gums":"🫧","half-life 2":"🎮","half-life: alyx":"🎮","ham":"🥩",
        "hand protein bar":"💪","herb & pepper":"🌿","honey":"🍯","hot dogs":"🌭",
        "iMac":"🖥","iPad":"📱","iphone 10":"📱","ketchup":"🍅","laptop":"💻",
        "light cream":"🥛","light mayo":"🫙","low fat yogurt":"🍦",
        "mashed potato":"🥔","mayonnaise":"🫙","meatballs":"🍖",
        "megaman zero":"🤖","megaman zero 2":"🤖","megaman zero 3":"🤖",
        "megaman zero 4":"🤖","melons":"🍈","metroid fusion":"🎮","metroid prime":"🎮",
        "milk":"🥛","minecraft":"⛏","mineral water":"💧","mint":"🌿",
        "mint green tea":"🍵","muffins":"🧁","mushroom cream sauce":"🍄",
        "napkins":"🧻","nonfat milk":"🥛","oatmeal":"🌾","oil":"🫙","olive oil":"🫙",
        "pancakes":"🥞","parmesan cheese":"🧀","pasta":"🍝","pepper":"🫑",
        "pet food":"🐾","phone car charger":"🔌","phone charger":"🔌","pickles":"🥒",
        "pokemon scarlet":"🎮","pokemon shield":"🎮","pokemon sword":"🎮",
        "pokemon violet":"🎮","portal":"🎮","portal 2":"🎮","protein bar":"💪",
        "ratchet & clank":"🎮","ratchet & clank 2":"🎮","ratchet & clank 3":"🎮",
        "razor":"🪒","red wine":"🍷","rice":"🍚","ring light":"💡","salad":"🥗",
        "salmon":"🐟","salt":"🧂","samsung galaxy 10":"📱","sandwich":"🥪",
        "seabass":"🐟","shallot":"🧅","shampoo":"🧴","shower gel":"🚿",
        "shrimp":"🦐","soda":"🥤","soup":"🍲","spaghetti":"🍝",
        "sparkling water":"💧","spinach":"🥬","strawberries":"🍓",
        "strong cheese":"🧀","tea":"☕","toilet paper":"🧻","tomato juice":"🧃",
        "tomato sauce":"🍅","tomatoes":"🍅","tooth brush":"🪥","toothpaste":"🦷",
        "trout":"🐟","turkey":"🦃","vacuum cleaner":"🧹","vegetables mix":"🥦",
        "water spray":"💦","white wine":"🍾","whole weat flour":"🌾",
        "whole wheat pasta":"🍝","whole wheat rice":"🍚","yams":"🍠",
        "yogurt cake":"🍰","zucchini":"🥒",
    }
    _SHOP_PRICES = {
        "airpods":49.99,"almonds":3.99,"antioxydant juice":2.99,"asparagus":2.49,
        "avocado":1.99,"babies food":3.99,"bacon":4.99,"barbecue sauce":2.49,
        "beer":1.99,"black beer":2.49,"black tea":2.29,"blueberries":2.99,
        "bluetooth headphones":39.99,"body spray":6.99,"bramble":1.79,"brownies":3.49,
        "burger sauce":1.99,"burgers":5.99,"butter":2.79,"cake":4.99,
        "candy bars":1.99,"canned_tuna":2.49,"carrots":1.29,"cat food":4.99,
        "catfish":7.99,"cauliflower":1.99,"cereals":3.49,"champagne":24.99,
        "chicken":6.99,"chili":2.29,"chocolate":2.49,"chocolate bread":2.99,
        "chutney":2.99,"cider":2.49,"cologne":19.99,"cookies":2.49,
        "cooking oil":3.99,"corn":1.29,"cottage cheese":2.99,"cotton buds":1.99,
        "cream":1.99,"deodorant":4.49,"dessert wine":12.99,"dog food":5.99,
        "eggplant":1.49,"eggs":2.99,"energy bar":1.99,"energy drink":1.99,
        "escalope":8.99,"extra dark chocolate":3.49,"final fantasy XIX":59.99,
        "final fantasy XX":59.99,"final fantasy XXII":59.99,"flax seed":3.49,
        "french fries":2.99,"french wine":14.99,"fresh bread":2.29,"fresh tuna":9.99,
        "fromage blanc":2.29,"frozen smoothie":3.49,"frozen vegetables":2.99,
        "gadget for tiktok streaming":29.99,"gluten free bar":2.49,"grated cheese":3.49,
        "green beans":1.99,"green grapes":2.49,"green tea":2.49,"ground beef":7.99,
        "gums":1.49,"half-life 2":14.99,"half-life: alyx":39.99,"ham":4.99,
        "hand protein bar":2.99,"herb & pepper":1.99,"honey":4.49,"hot dogs":3.49,
        "iMac":1299.99,"iPad":399.99,"iphone 10":799.99,"ketchup":1.99,
        "laptop":899.99,"light cream":1.79,"light mayo":1.99,"low fat yogurt":1.99,
        "mashed potato":1.99,"mayonnaise":2.29,"meatballs":5.99,
        "megaman zero":19.99,"megaman zero 2":19.99,"megaman zero 3":19.99,
        "megaman zero 4":19.99,"melons":2.99,"metroid fusion":19.99,"metroid prime":19.99,
        "milk":1.49,"minecraft":19.99,"mineral water":0.99,"mint":0.99,
        "mint green tea":2.79,"muffins":2.99,"mushroom cream sauce":3.49,
        "napkins":1.99,"nonfat milk":1.49,"oatmeal":2.49,"oil":3.49,"olive oil":5.99,
        "pancakes":2.99,"parmesan cheese":4.49,"pasta":1.99,"pepper":1.29,
        "pet food":5.49,"phone car charger":19.99,"phone charger":19.99,"pickles":2.29,
        "pokemon scarlet":59.99,"pokemon shield":59.99,"pokemon sword":59.99,
        "pokemon violet":59.99,"portal":9.99,"portal 2":9.99,"protein bar":2.99,
        "ratchet & clank":19.99,"ratchet & clank 2":19.99,"ratchet & clank 3":19.99,
        "razor":4.99,"red wine":8.99,"rice":1.99,"ring light":24.99,"salad":1.99,
        "salmon":8.99,"salt":0.99,"samsung galaxy 10":399.99,"sandwich":3.49,
        "seabass":9.99,"shallot":0.99,"shampoo":5.99,"shower gel":3.99,"shrimp":9.99,
        "soda":1.29,"soup":3.49,"spaghetti":1.99,"sparkling water":1.29,
        "spinach":1.79,"strawberries":2.99,"strong cheese":4.49,"tea":2.29,
        "toilet paper":3.99,"tomato juice":1.99,"tomato sauce":1.99,"tomatoes":1.99,
        "tooth brush":2.99,"toothpaste":3.49,"trout":7.99,"turkey":7.99,
        "vacuum cleaner":89.99,"vegetables mix":2.49,"water spray":2.99,
        "white wine":9.99,"whole weat flour":2.49,"whole wheat pasta":2.29,
        "whole wheat rice":2.49,"yams":2.49,"yogurt cake":3.49,"zucchini":1.29,
    }
    _SHOP_CATEGORIES = {
        "🥦 Produce":["asparagus","avocado","blueberries","bramble","carrots","cauliflower",
            "corn","eggplant","frozen vegetables","green beans","green grapes",
            "herb & pepper","melons","mint","pepper","salad","shallot","spinach",
            "strawberries","tomatoes","vegetables mix","yams","zucchini"],
        "🥛 Dairy":["butter","cottage cheese","cream","fromage blanc","grated cheese",
            "light cream","low fat yogurt","milk","nonfat milk","parmesan cheese","strong cheese"],
        "🍞 Bakery":["cake","cereals","chocolate bread","cookies","fresh bread","muffins",
            "oatmeal","pancakes","whole weat flour","whole wheat pasta","whole wheat rice","yogurt cake"],
        "🍗 Meat & Fish":["bacon","burgers","catfish","chicken","escalope","fresh tuna",
            "ground beef","ham","hot dogs","meatballs","salmon","sandwich","seabass",
            "shrimp","trout","turkey","canned_tuna"],
        "🍷 Drinks":["antioxydant juice","beer","black beer","black tea","champagne","cider",
            "dessert wine","energy drink","french wine","frozen smoothie","green tea",
            "mineral water","mint green tea","red wine","soda","sparkling water","tea",
            "tomato juice","white wine"],
        "🥫 Pantry":["almonds","barbecue sauce","brownies","candy bars","chili","chocolate",
            "chutney","cooking oil","energy bar","extra dark chocolate","flax seed",
            "french fries","gluten free bar","hand protein bar","honey","ketchup",
            "light mayo","mashed potato","mayonnaise","mushroom cream sauce","oil","olive oil",
            "pasta","pickles","protein bar","rice","salt","soup","spaghetti","tomato sauce"],
        "🧴 Hygiene":["body spray","cologne","cotton buds","deodorant","gums","napkins",
            "razor","shampoo","shower gel","toilet paper","tooth brush","toothpaste","water spray"],
        "🐾 Pet & Baby":["babies food","cat food","dog food","pet food"],
        "📱 Electronics":["airpods","bluetooth headphones","gadget for tiktok streaming",
            "iMac","iPad","iphone 10","laptop","phone car charger","phone charger",
            "ring light","samsung galaxy 10","vacuum cleaner"],
        "🎮 Games":["final fantasy XIX","final fantasy XX","final fantasy XXII",
            "half-life 2","half-life: alyx","megaman zero","megaman zero 2",
            "megaman zero 3","megaman zero 4","metroid fusion","metroid prime","minecraft",
            "pokemon scarlet","pokemon shield","pokemon sword","pokemon violet",
            "portal","portal 2","ratchet & clank","ratchet & clank 2","ratchet & clank 3"],
    }
    _SHOP_SEGMENTS = {
        "👨‍👩‍👧 Families": {
            "desc": "Bread, eggs, cereals, butter, tea",
            "top": ["eggs","cereals","fresh bread","butter","bacon","tea","honey",
                    "sandwich","oatmeal","milk","black tea","chocolate bread","salt",
                    "oil","whole weat flour","cooking oil"],
        },
        "🧴 Techies": {
            "desc": "Shower gel, deodorant, shampoo, protein bars",
            "top": ["shower gel","tooth brush","deodorant","shampoo","toothpaste",
                    "razor","cotton buds","body spray","antioxydant juice","energy bar",
                    "protein bar","toilet paper","almonds","gluten free bar",
                    "nonfat milk","green tea","low fat yogurt"],
        },
        "🥦 Vegetarians": {
            "desc": "Dog food, babies food, pasta, chicken",
            "top": ["napkins","dog food","babies food","cooking oil","pet food","chicken",
                    "cat food","rice","spaghetti","pasta","meatballs","milk",
                    "toilet paper","fresh bread","cereals","soup","tomato sauce",
                    "mayonnaise","eggs","ketchup"],
        },
        "🎮 Regulars": {
            "desc": "Airpods, headphones, games, ring light",
            "top": ["final fantasy XX","ratchet & clank 3","metroid fusion",
                    "final fantasy XIX","ring light","bluetooth headphones",
                    "final fantasy XXII","phone car charger","airpods","pokemon sword",
                    "gadget for tiktok streaming","vacuum cleaner","pokemon shield",
                    "energy drink","iPad","ratchet & clank","portal 2","minecraft","portal"],
        },
        "🏷 Promoters": {
            "desc": "Tomatoes, carrots, spinach, fruits, laptop",
            "top": ["tomatoes","carrots","spinach","eggplant","laptop","strawberries",
                    "salad","corn","green beans","green grapes","asparagus","avocado",
                    "blueberries","cauliflower","frozen vegetables"],
        },
        "💰 Economizers": {
            "desc": "Spinach, tomatoes, carrots, avocado, berries",
            "top": ["spinach","tomatoes","carrots","avocado","strawberries","green beans",
                    "salad","green grapes","corn","zucchini","asparagus","blueberries",
                    "mineral water","cauliflower","mashed potato","eggplant"],
        },
        "⭐ Loyalists": {
            "desc": "Champagne, beer, french wine, iMac, pasta",
            "top": ["phone car charger","iMac","whole wheat pasta","flax seed",
                    "megaman zero 4","champagne","beer","mushroom cream sauce",
                    "half-life 2","final fantasy XIX","french wine","french fries",
                    "pickles","green beans","light mayo","melons","gums","turkey"],
        },
        "🧘 Wellness": {
            "desc": "Red wine, salmon, gaming, cologne, yams",
            "top": ["red wine","bramble","yams","black beer","portal","seabass",
                    "ratchet & clank 3","pokemon sword","megaman zero","cologne",
                    "ratchet & clank 2","ratchet & clank","cottage cheese","half-life 2",
                    "french fries","portal 2","metroid prime","salmon","samsung galaxy 10"],
        },
    }
    _SHOP_NAMES = ["Alice","Bruno","Carla","David","Eva","Fábio","Gina","Hugo",
                   "Inês","João","Lara","Miguel","Nuno","Olga","Pedro","Rita"]
    _SHOP_ALL_PRODUCTS = sorted(_SHOP_PRICES.keys())

    # ── Session state helpers ─────────────────────────────────────────────
    def _shop_init():
        defaults = {
            "shop_cart": {},
            "shop_true_seg": "",
            "shop_customer_name": "",
            "shop_checked_out": False,
            "shop_guessed": False,
            "shop_guess_result": None,
            "shop_score": 0,
            "shop_rounds": 0,
            "shop_round_started": False,
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    def _shop_new_round():
        import random as _rnd
        seg = _rnd.choice(list(_SHOP_SEGMENTS.keys()))
        pool = list(_SHOP_SEGMENTS[seg]["top"])
        _rnd.shuffle(pool)
        n = _rnd.randint(5, 9)
        basket = pool[:min(n, len(pool))]
        cart = {}
        for item in basket:
            cart[item] = cart.get(item, 0) + 1
        st.session_state.shop_cart = cart
        st.session_state.shop_true_seg = seg
        st.session_state.shop_customer_name = _rnd.choice(_SHOP_NAMES)
        st.session_state.shop_checked_out = False
        st.session_state.shop_guessed = False
        st.session_state.shop_guess_result = None
        st.session_state.shop_round_started = True

    def _shop_add(product):
        st.session_state.shop_cart[product] = st.session_state.shop_cart.get(product, 0) + 1

    def _shop_remove(product):
        if product in st.session_state.shop_cart:
            if st.session_state.shop_cart[product] > 1:
                st.session_state.shop_cart[product] -= 1
            else:
                del st.session_state.shop_cart[product]

    def _shop_total():
        return sum(_SHOP_PRICES.get(p, 0) * q for p, q in st.session_state.shop_cart.items())

    _shop_init()

    # ── Scoreboard ───────────────────────────────────────────────────────
    _sc1, _sc2, _sc3 = st.columns(3)
    _sc1.metric("Score", st.session_state.shop_score)
    _sc2.metric("Rounds", st.session_state.shop_rounds)
    _shop_acc = (
        f"{round(st.session_state.shop_score / st.session_state.shop_rounds * 100)}%"
        if st.session_state.shop_rounds else "—"
    )
    _sc3.metric("Accuracy", _shop_acc)
    st.divider()

    # ── First start ──────────────────────────────────────────────────────
    if not st.session_state.shop_round_started:
        if st.button("▶ Start — first customer", use_container_width=True, type="primary", key="shop_start_btn"):
            _shop_new_round()
            st.rerun()
        st.stop()

    # ── Customer banner ──────────────────────────────────────────────────
    if not st.session_state.shop_checked_out:
        st.info(
            f"**Customer: {st.session_state.shop_customer_name}** — basket loaded. "
            "Add more items if you like, then checkout."
        )
    else:
        st.info(
            f"**Customer: {st.session_state.shop_customer_name}** checked out — "
            "which segment do they belong to?"
        )

    # ── Layout: shop | cart ──────────────────────────────────────────────
    _shop_col, _cart_col = st.columns([2, 1])

    with _shop_col:
        st.subheader("🏪 Products")
        st.markdown(f"""
<div style='display:flex; gap:16px; margin-bottom:16px;'>
  <img src='{PRODUTOS_TECH_URI}' style='width:50%; border-radius:12px; object-fit:cover;' />
  <img src='{PRODUTOS_MEAT_URI}' style='width:50%; border-radius:12px; object-fit:cover;' />
</div>
""", unsafe_allow_html=True)
        _cs, _cc = st.columns([2, 1])
        _search = _cs.text_input("Search", placeholder="Search…", label_visibility="collapsed", key="shop_search")
        _cat_filter = _cc.selectbox(
            "Category", ["All"] + list(_SHOP_CATEGORIES.keys()),
            label_visibility="collapsed", key="shop_cat_filter"
        )
        _filtered = _SHOP_ALL_PRODUCTS
        if _cat_filter != "All":
            _filtered = [p for p in _filtered if p in _SHOP_CATEGORIES[_cat_filter]]
        if _search:
            _filtered = [p for p in _filtered if _search.lower() in p.lower()]

        _grid_cols = st.columns(4)
        for _i, _product in enumerate(_filtered):
            with _grid_cols[_i % 4]:
                _qty = st.session_state.shop_cart.get(_product, 0)
                _badge = f" ×{_qty}" if _qty else ""
                _label = f"{_SHOP_EMOJI.get(_product,'🛒')} {_product.title()}{_badge}\n€{_SHOP_PRICES[_product]:.2f}"
                    
                if st.button(_label, key=f"shop_add_{_product}",
                             disabled=st.session_state.shop_checked_out,
                             use_container_width=True):
                    _shop_add(_product)
                    st.rerun()

    with _cart_col:
        st.subheader("🧺 Cart")
        if not st.session_state.shop_cart:
            st.caption("Cart is empty")
        else:
            for _product, _qty in list(st.session_state.shop_cart.items()):
                _ca, _cb, _cc2 = st.columns([3, 2, 1])
                _ca.write(f"{_SHOP_EMOJI.get(_product,'🛒')} {_product.title()}")
                _cb.write(f"×{_qty}  **€{_SHOP_PRICES[_product]*_qty:.2f}**")
                if not st.session_state.shop_checked_out:
                    if _cc2.button("✕", key=f"shop_rm_{_product}"):
                        _shop_remove(_product)
                        st.rerun()
            st.write(f"**Total: €{_shop_total():.2f}**")

        st.divider()

        if not st.session_state.shop_checked_out:
            if st.session_state.shop_cart:
                if st.button("💳 Checkout", use_container_width=True, type="primary", key="shop_checkout_btn"):
                    st.session_state.shop_checked_out = True
                    st.rerun()
        else:
            if not st.session_state.shop_guessed:
                st.write("**Which segment?**")
                for _seg_name, _seg_data in _SHOP_SEGMENTS.items():
                    if st.button(
                        f"{_seg_name}\n*{_seg_data['desc']}*",
                        key=f"shop_guess_{_seg_name}",
                        use_container_width=True
                    ):
                        _correct = _seg_name == st.session_state.shop_true_seg
                        st.session_state.shop_guessed = True
                        st.session_state.shop_guess_result = _correct
                        st.session_state.shop_rounds += 1
                        if _correct:
                            st.session_state.shop_score += 1
                        st.rerun()
            else:
                _true = st.session_state.shop_true_seg
                _info = _SHOP_SEGMENTS[_true]
                if st.session_state.shop_guess_result:
                    st.success(f"✅ **{_true}**\n\n*{_info['desc']}*")
                else:
                    st.error(f"❌ It was **{_true}**\n\n*{_info['desc']}*")
                if st.button("▶ Next customer", use_container_width=True, type="primary", key="shop_next_btn"):
                    _shop_new_round()
                    st.rerun()

    render_footer()
