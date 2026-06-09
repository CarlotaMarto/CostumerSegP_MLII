import streamlit as st

st.set_page_config(page_title="Radio Style Navigation Test", layout="wide")

# Custom CSS for styling st.sidebar.radio
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ea580c 0%, #f97316 100%) !important;
    }
    
    /* Style the radio container to look like our sidebar item index */
    [data-testid="stSidebar"] div[role="radiogroup"] {
        background-color: transparent !important;
        padding: 0 !important;
        gap: 0 !important;
    }
    
    /* Hide the default radio circle and its container */
    [data-testid="stSidebar"] div[role="radiogroup"] label [role="presentation"],
    [data-testid="stSidebar"] div[role="radiogroup"] label div[dir="ltr"] {
        display: none !important;
    }
    
    /* Style all radio labels to look like sidebar menu links */
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: transparent !important;
        color: rgba(255, 255, 255, 0.8) !important;
        border: none !important;
        padding: 10px 16px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        border-radius: 12px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        margin-bottom: 6px !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        box-shadow: none !important;
    }
    
    /* Hover state */
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Selected/Active state using CSS :has selector */
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input[type="radio"]:checked) {
        color: #ea580c !important;
        background-color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Add SVGs or styling for active state */
    /* (we can prefix labels with emojis or map them) */
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.write("### Section Index")

page = st.sidebar.radio(
    label="Index",
    options=[
        "Overview",
        "Data Analysis",
        "Data Preprocessing",
        "Data in Geography",
        "Customer Communities",
        "Targeter Promotion",
        "Conclusion & Recommendations",
        "Customer Simulator"
    ],
    label_visibility="collapsed"
)

st.title(f"Page: {page}")
st.write("This is a demo page. Moving sliders or changing widgets will run natively without page reloads!")
st.slider("Slider State Test", 0, 100, 50)
