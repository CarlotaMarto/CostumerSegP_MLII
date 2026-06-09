import re
import os

app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update the Coupons grid in NB5
old_grid = """<div style="display:flex; flex-wrap:wrap; gap:16px; justify-content:center;">
''', unsafe_allow_html=True)
    import os
    cupoes_dir = IMAGENS_DIR / "cupoes"
    if cupoes_dir.exists():
        for f in os.listdir(cupoes_dir):
            if f.endswith(".png"):
                st.image(str(cupoes_dir / f), width=400)
    st.markdown("</div>", unsafe_allow_html=True)"""

new_grid = """<div class="coupons-grid-wrapper">
''', unsafe_allow_html=True)
    import os
    cupoes_dir = IMAGENS_DIR / "cupoes"
    if cupoes_dir.exists():
        # Display 8 images
        st.markdown('<div class="coupons-scroller">', unsafe_allow_html=True)
        cols = st.columns(4)
        images = [f for f in os.listdir(cupoes_dir) if f.endswith(".png")]
        for idx, f in enumerate(images):
            # We want 2 rows, so we can just let them wrap in CSS or use columns
            pass
        # Actually it's easier to use raw HTML for the grid to ensure perfect horizontal scrolling and mix-blend-mode
        import base64
        html_imgs = ""
        for f in images:
            img_path = cupoes_dir / f
            with open(img_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            html_imgs += f'<div class="coupon-item"><img src="data:image/png;base64,{encoded_string}" style="width:100%; border-radius:12px; mix-blend-mode: multiply;"></div>'
        
        grid_html = f'''
        <style>
        .coupons-scroller {{
            display: grid;
            grid-template-columns: repeat(4, 300px);
            grid-template-rows: repeat(2, auto);
            gap: 16px;
            overflow-x: auto;
            padding-bottom: 16px;
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
        
    st.markdown("</div>", unsafe_allow_html=True)"""

content = content.replace(old_grid, new_grid)

# 2. Fix the simulator products rendering (remove images, keep emojis)
old_simulator_logic = """                _img_name = _product.title() + ".png"
                _img_path = BASE_DIR / "imagens" / "guardadas" / _img_name
                
                # Check if we have an image for this product
                if _img_path.exists():
                    st.image(str(_img_path), use_container_width=True)
                    _label = f"{_product.title()}{_badge}\\n€{_SHOP_PRICES[_product]:.2f}"
                else:
                    _label = f"{_SHOP_EMOJI.get(_product,'🛒')} {_product.title()}{_badge}\\n€{_SHOP_PRICES[_product]:.2f}"
                    
                if st.button(_label, key=f"shop_add_{_product}",
                             disabled=st.session_state.shop_checked_out,
                             use_container_width=True):"""

new_simulator_logic = """                _label = f"{_SHOP_EMOJI.get(_product,'🛒')} {_product.title()}{_badge}\\n€{_SHOP_PRICES[_product]:.2f}"
                    
                if st.button(_label, key=f"shop_add_{_product}",
                             disabled=st.session_state.shop_checked_out,
                             use_container_width=True):"""

content = content.replace(old_simulator_logic, new_simulator_logic)

# 3. Change button background from #000000 to #c94f38 in CSS
# I will use regex to be safe, replacing background-color: #000000 !important; to #c94f38 inside stButton
css_old = "background-color: #000000 !important;"
css_new = "background-color: #c94f38 !important;"
# We will do a generic replacement for the stButton style block if possible
import re
content = re.sub(r'(\[data-testid="stButton"\] button \{[^\}]*?background-color:\s*)#000000(\s*!important;)', r'\1#c94f38\2', content)

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)
