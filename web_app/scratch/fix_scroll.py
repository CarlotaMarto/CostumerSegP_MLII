import sys

app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

target = """selected_page = st.sidebar.radio(
    label="Index",
    options=page_list,
    index=current_index,
    format_func=lambda x: _page_labels[x],
    label_visibility="collapsed"
)
st.session_state.current_page = selected_page"""

replacement = """selected_page = st.sidebar.radio(
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
    import streamlit.components.v1 as components
    components.html(
        '''
        <script>
            var body = window.parent.document.querySelector(".main");
            if (body) {
                body.scrollTo(0, 0);
            }
            window.parent.scrollTo(0, 0);
        </script>
        ''',
        height=0
    )"""

if target in content:
    content = content.replace(target, replacement)
    with open(app_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Scroll fix applied successfully.")
else:
    print("Target not found.")
    sys.exit(1)
