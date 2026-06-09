import streamlit as st

st.set_page_config(page_title="Sidebar Button Style Test", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "introduction"

# Custom CSS for styling buttons in sidebar to look like links
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ea580c 0%, #f97316 100%) !important;
    }
    
    /* Reset and style all sidebar buttons */
    [data-testid="stSidebar"] div.stButton > button {
        background-color: transparent !important;
        color: rgba(255, 255, 255, 0.8) !important;
        border: none !important;
        padding: 10px 16px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        text-align: left !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        margin: 0 !important;
    }
    
    [data-testid="stSidebar"] div.stButton > button:hover {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Style active state via parent wrapper */
    div.sidebar-btn-active > div.stButton > button {
        color: #ea580c !important;
        background-color: #ffffff !important;
        font-weight: 700 !important;
    }
    div.sidebar-btn-active > div.stButton > button:hover {
        color: #ea580c !important;
        background-color: #ffffff !important;
    }
    
    /* Add SVGs before button text */
    #btn-intro button::before {
        content: "";
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 12px;
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.8)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>');
        background-size: contain;
        background-repeat: no-repeat;
    }
    
    div.sidebar-btn-active#btn-intro button::before {
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23ea580c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>');
    }
    
    #btn-analysis button::before {
        content: "";
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 12px;
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.8)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>');
        background-size: contain;
        background-repeat: no-repeat;
    }
    
    div.sidebar-btn-active#btn-analysis button::before {
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23ea580c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>');
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.write("### Navigation")

# Sidebar button 1
p = st.session_state.page
intro_class = "sidebar-btn-active" if p == "introduction" else "sidebar-btn-inactive"
st.sidebar.markdown(f'<div class="{intro_class}" id="btn-intro">', unsafe_allow_html=True)
if st.sidebar.button("Overview", key="btn_intro_val"):
    st.session_state.page = "introduction"
    st.rerun()
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Sidebar button 2
analysis_class = "sidebar-btn-active" if p == "data-analysis" else "sidebar-btn-inactive"
st.sidebar.markdown(f'<div class="{analysis_class}" id="btn-analysis">', unsafe_allow_html=True)
if st.sidebar.button("Data Analysis", key="btn_analysis_val"):
    st.session_state.page = "data-analysis"
    st.rerun()
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.title(f"Page: {st.session_state.page.capitalize()}")
st.write("Some text here...")
st.slider("Test Slider", 0, 100, 50)
