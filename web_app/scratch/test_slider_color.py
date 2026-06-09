import streamlit as st

st.set_page_config(page_title="Slider Custom Color Test")

color = st.text_input("Enter color (hex):", value="#ea580c")

# Inject custom CSS dynamically based on the color input
st.markdown(
    f"""
    <style>
    /* Thumb styling */
    div[data-testid="stSlider"] div[role="slider"] {{
        background-color: {color} !important;
    }}
    
    /* Active progress bar styling in baseui slider */
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div > div {{
        background-color: {color} !important;
    }}
    
    /* Hover and focus ring style for thumb */
    div[data-testid="stSlider"] div[role="slider"]:focus, 
    div[data-testid="stSlider"] div[role="slider"]:hover {{
        box-shadow: 0px 0px 0px 8px {color}33 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Slider Color Override Test")
st.slider("Adjust me!", 0, 100, 50)
