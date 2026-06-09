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
