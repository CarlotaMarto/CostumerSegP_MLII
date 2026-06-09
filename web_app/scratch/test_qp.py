import streamlit as st

st.set_page_config(page_title="Query Parameter Navigation Test")

# Get query parameters
page = st.query_params.get("page", "home")

# Render custom sidebar
st.sidebar.markdown(
    f"""
    <div style="padding: 10px;">
        <h3>Navigation</h3>
        <a href="?page=home" target="_self" style="display:block; padding: 10px; color: {"red" if page == "home" else "black"}; font-weight: {"bold" if page == "home" else "normal"};">Home</a>
        <a href="?page=about" target="_self" style="display:block; padding: 10px; color: {"red" if page == "about" else "black"}; font-weight: {"bold" if page == "about" else "normal"};">About</a>
        <a href="?page=contact" target="_self" style="display:block; padding: 10px; color: {"red" if page == "contact" else "black"}; font-weight: {"bold" if page == "contact" else "normal"};">Contact</a>
    </div>
    """,
    unsafe_allow_html=True
)

st.title(f"Currently on: {page.capitalize()}")

if page == "home":
    st.write("Welcome to the home page!")
    st.slider("Home Slider", 0, 100, 50)
elif page == "about":
    st.write("This is the about page.")
elif page == "contact":
    st.write("Get in touch with us.")
