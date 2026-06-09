                y=alt.Y("Average Spend (€):Q", title="Average Spend (€)"),
                tooltip=["Category", alt.Tooltip("Average Spend (€):Q", format=",.2f")]
            ).properties(height=300)
            
            st.altair_chart(spend_chart_cluster, use_container_width=True)
    except Exception as e:
        st.info("Dynamic cluster files not fully loaded. Displaying static mockup.")
    render_footer()

elif selected_page == "Targeter Promotion":
    st.markdown("""
    <div style='margin-top: 0px; margin-bottom: 24px;'>
        <h2 style='font-size: 56px; font-weight: 800; color: #000000; margin: 0; letter-spacing: -0.03em;'>Targeter Promotion</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-bottom:32px;'>
      <div>
        <div style='font-size:11px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#9ca3af; margin-bottom:10px;'>Notebook 5 — Association Rules</div>
        <p style='font-size:18px; color:#374151; line-height:1.9; margin:0 0 24px 0; text-align: justify;'>
          The final notebook operationalises the cluster definitions by mining product associations specific to each community. Using the Apriori algorithm on the transaction-level <code>customer_basket</code> dataset, it discovers what products are frequently bought together by each segment. These patterns are then translated into concrete, data-driven cross-selling campaigns.
        </p>
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
          <a href="#nb5-1" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>1) Imports & data loading</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb5-2" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>2) Transaction preparation per segment</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb5-3" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>3) Apriori parameters</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb5-4" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>4) Association rules per segment</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb5-5" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>5) Top rules per segment</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb5-6" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>6) Rule robustness check</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb5-7" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>7) Campaign suggestions & Creative campaign texts</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb5-8" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>8) Final interpretation</div></a>
        </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
      <div id="nb5-1"></div><div id="nb5-2"></div>
      <div style='display:grid; grid-template-columns:repeat(2,1fr); gap:20px; margin-bottom:24px;'>
        <div style='border-left:3px solid #111827; padding-left:20px;'>
          <div id="nb5-3" style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>3) Apriori parameters - Why support is set at 1%</div>
          <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Rules are mined per segment — each sub-population has far fewer transactions than the full dataset. A 1% support threshold ensures enough rules are discovered while still requiring meaningful co-occurrence frequency within each community.</div>
        </div>
        <div style='border-left:3px solid #111827; padding-left:20px;'>
          <div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>Lift-derived campaign discounts</div>
          <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Suggested campaign discounts are not fixed — they are derived from the lift value of each rule. A higher lift means a stronger-than-random co-purchase signal, which justifies a larger promotional incentive. This ties the commercial decision directly to statistical evidence.</div>
        </div>
        <div style='border-left:3px solid #111827; padding-left:20px;'>
          <div id="nb5-7" style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>7) Campaign suggestions - Excluded recommendations: Vegetarians (cluster 0)</div>
          <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Chicken, meat, and fish are excluded from recommendations for cluster 0 (Vegetarians). The Apriori rules initially suggested these items, but they contradict the segment's defining behavioural trait. Notebook 4 confirms that this segment's identity is plant-based — the exclusion ensures campaign coherence.</div>
        </div>
        <div style='border-left:3px solid #111827; padding-left:20px;'>
          <div id="nb5-6" style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>6) Rule robustness check - Robustness validation</div>
          <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Each segment's rules are validated on an 80/20 train/test split. Segments with many matched rules and low mean lift difference between train and test have stable co-purchase patterns. Segments with few matched rules should be interpreted with caution.</div>
        </div>
      </div>
    </div>
    <div style='height:1px; background:#e5e7eb; margin-bottom:24px;'></div>
    """, unsafe_allow_html=True)

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

    st.markdown("<div style='height:1px; background:#e5e7eb; margin-bottom:24px;'></div>", unsafe_allow_html=True)

    try:
        campaign_rules = load_csv_data("segment_campaign_rules.csv")
        unique_segments = campaign_rules['segment'].unique()
        
        # Segment label map to color rules by matching SEGMENT_COLORS keys
        segment_label_map = {
            "Vegetarians": 0, "Regulars": 1, "Wellness": 2, "Promoters": 3,
            "Loyalists": 4, "Families": 5, "Economizers": 6, "Techies": 7
        }
        
        selected_segment = st.selectbox("Select segment for campaigns", options=unique_segments)
        segment_color_idx = segment_label_map.get(selected_segment, 0)
        promo_color = SEGMENT_COLORS.get(segment_color_idx, "#ea580c")
        
        segment_rules = campaign_rules[campaign_rules['segment'] == selected_segment]
        
        st.markdown(f"<div id='nb5-8'></div>\n\n#### Top Association Rules for {selected_segment}", unsafe_allow_html=True)
        
        for idx, rule in segment_rules.iterrows():
            if_buys = rule['if_buys']
            promote = rule['promote']
            conf = rule['confidence'] * 100
            lift = rule['lift']
            
            st.markdown(f"""
            <div style='background: #ffffff; border: 1px solid rgba(0,0,0,0.05); border-radius: 12px; padding: 18px; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between;'>
              <div style='flex: 2;'>
                <div style='font-size: 11px; text-transform: uppercase; color: #8c8c8c; font-weight: 700; margin-bottom: 4px;'>Trigger purchase</div>
                <div style='font-size: 15px; font-weight: 700; color: #1a1a1a;'>If customer buys: <code style='font-size: 13px; background: rgba(0,0,0,0.04); padding: 3px 6px; border-radius: 6px;'>{if_buys}</code></div>
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
    render_footer()

elif selected_page == "Conclusion & Recommendations":
    st.markdown("""
    <div style='margin-top: 0px; margin-bottom: 24px;'>
        <h2 style='font-size: 56px; font-weight: 800; color: #000000; margin: 0; letter-spacing: -0.03em;'>Conclusion & Recommendations</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='page-shell'>
      <div class='page-text'>
        <p>The Customer Intelligence dashboard successfully maps raw transactional behavior and customer attributes into actionable insights. By cleaning negative entries, correcting future transaction records, and mapping spatial density, we have built a stable dataset for business decisions.</p>
        <h3 style='margin-top: 24px; margin-bottom: 16px; font-family:"Playfair Display", serif;'>Strategic Actions</h3>
        <ul style='padding-left: 20px;'>
          <li style='margin-bottom: 12px;'><strong>Targeted Campaign Lift</strong>: Apply the association rules (e.g., salad cross-sell for Promo Shoppers, airpods for Technologists) directly in checkout systems to drive average order value.</li>
          <li style='margin-bottom: 12px;'><strong>Geographic Expansion</strong>: The university and urban corridor hotspots present immediate physical expansion opportunities for premium and student convenience stores.</li>
          <li style='margin-bottom: 12px;'><strong>Retention Programs</strong>: Build tailored communication lines focusing on vegetable-focused mature segments and premium wellness buyers, which show higher loyalty flag consistency.</li>
        </ul>
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
