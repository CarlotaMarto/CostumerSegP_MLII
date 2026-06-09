import sys

app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

target = """        _grid_cols = st.columns(4)
        for _i, _product in enumerate(_filtered):
            with _grid_cols[_i % 4]:
                _qty = st.session_state.shop_cart.get(_product, 0)
                _badge = f" ×{_qty}" if _qty else ""
                _label = f"{_SHOP_EMOJI.get(_product,'🛒')} {_product.title()}{_badge}\\n€{_SHOP_PRICES[_product]:.2f}"
                if st.button(_label, key=f"shop_add_{_product}",
                             disabled=st.session_state.shop_checked_out,
                             use_container_width=True):
                    _shop_add(_product)
                    st.rerun()"""

replacement = """        _grid_cols = st.columns(4)
        for _i, _product in enumerate(_filtered):
            with _grid_cols[_i % 4]:
                _qty = st.session_state.shop_cart.get(_product, 0)
                _badge = f" ×{_qty}" if _qty else ""
                _img_name = _product.title() + ".png"
                _img_path = BASE_DIR / "imagens" / "guardadas" / _img_name
                
                # Check if we have an image for this product
                if _img_path.exists():
                    st.image(str(_img_path), use_container_width=True)
                    _label = f"{_product.title()}{_badge}\\n€{_SHOP_PRICES[_product]:.2f}"
                else:
                    _label = f"{_SHOP_EMOJI.get(_product,'🛒')} {_product.title()}{_badge}\\n€{_SHOP_PRICES[_product]:.2f}"
                    
                if st.button(_label, key=f"shop_add_{_product}",
                             disabled=st.session_state.shop_checked_out,
                             use_container_width=True):
                    _shop_add(_product)
                    st.rerun()"""

if target in content:
    content = content.replace(target, replacement)
    with open(app_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Replaced successfully.")
else:
    print("Target block not found.")
    sys.exit(1)
