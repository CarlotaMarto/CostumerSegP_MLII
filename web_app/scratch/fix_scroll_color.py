import sys
import re

app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update the bottom bar colors
old_bar = "background-color: rgba(0, 0, 0, 0.04); color: #9ca3af;"
new_bar = "background-color: #f7e6e1; color: #a36154;"
content = content.replace(old_bar, new_bar)

# 2. Fix the scroll logic
# Remove the old scroll block
old_scroll_regex = r"if st\.session_state\.previous_page != selected_page:\n\s*st\.session_state\.previous_page = selected_page\n\s*import streamlit\.components\.v1 as components\n\s*components\.html\(\n\s*'''\n\s*<script>.*?</script>\n\s*''',\n\s*height=0\n\s*\)"

# We'll just manually replace the exact string to be safe
start_str = "if st.session_state.previous_page != selected_page:"
end_str = "        height=0\n    )"
start_idx = content.find(start_str)
end_idx = content.find(end_str, start_idx) + len(end_str)

if start_idx != -1 and end_idx != -1:
    old_scroll_block = content[start_idx:end_idx]
    new_scroll_trigger = """if st.session_state.previous_page != selected_page:
    st.session_state.previous_page = selected_page
    st.session_state.scroll_trigger = True"""
    content = content.replace(old_scroll_block, new_scroll_trigger)

# Now inject the scroll execution at the end of render_footer
footer_end_str = '        """,\n        unsafe_allow_html=True\n    )'
footer_end_idx = content.find(footer_end_str) + len(footer_end_str)

scroll_injection = """
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
"""

if footer_end_idx != -1:
    content = content[:footer_end_idx] + "\n" + scroll_injection + content[footer_end_idx:]

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Applied footer color and scroll fix.")
