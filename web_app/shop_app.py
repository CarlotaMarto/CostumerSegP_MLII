import streamlit as st
import random

st.set_page_config(page_title="Market Shop – Guess the Segment", layout="wide")

# ── Data ─────────────────────────────────────────────────────────────────────

EMOJI = {
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

PRICES = {
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

CATEGORIES = {
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

SEGMENTS = {
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
    "⭐ Loyalists": {
        "desc": "Champagne, beer, french wine, iMac, pasta",
        "top": ["phone car charger","iMac","whole wheat pasta","flax seed",
                "megaman zero 4","champagne","beer","mushroom cream sauce",
                "half-life 2","final fantasy XIX","french wine","french fries",
                "pickles","green beans","light mayo","melons","gums","turkey"],
    },
    "🧘 Wellness": {
        "desc": "Red wine, salmon, gaming, cologne, yams",
        "top": ["red wine","bramble","yams","black beer","portal","seabass",
                "ratchet & clank 3","pokemon sword","megaman zero","cologne",
                "ratchet & clank 2","ratchet & clank","cottage cheese","half-life 2",
                "french fries","portal 2","metroid prime","salmon","samsung galaxy 10"],
    },
}

NAMES = ["Alice","Bruno","Carla","David","Eva","Fábio","Gina","Hugo",
         "Inês","João","Lara","Miguel","Nuno","Olga","Pedro","Rita"]

ALL_PRODUCTS = sorted(PRICES.keys())

# ── Session state ─────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "cart": {},
        "true_seg": "",
        "customer_name": "",
        "checked_out": False,
        "guessed": False,
        "guess_result": None,
        "score": 0,
        "rounds": 0,
        "round_started": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def new_round():
    seg = random.choice(list(SEGMENTS.keys()))
    pool = list(SEGMENTS[seg]["top"])
    random.shuffle(pool)
    n = random.randint(5, 9)
    basket = pool[:min(n, len(pool))]
    cart = {}
    for item in basket:
        cart[item] = cart.get(item, 0) + 1
    st.session_state.cart = cart
    st.session_state.true_seg = seg
    st.session_state.customer_name = random.choice(NAMES)
    st.session_state.checked_out = False
    st.session_state.guessed = False
    st.session_state.guess_result = None
    st.session_state.round_started = True

def add_item(product):
    st.session_state.cart[product] = st.session_state.cart.get(product, 0) + 1

def remove_item(product):
    if product in st.session_state.cart:
        if st.session_state.cart[product] > 1:
            st.session_state.cart[product] -= 1
        else:
            del st.session_state.cart[product]

def cart_total():
    return sum(PRICES.get(p, 0) * q for p, q in st.session_state.cart.items())

# ── App ───────────────────────────────────────────────────────────────────────

init_state()

st.title("🛒 Market Shop — Guess the Segment")

# Scoreboard
c1, c2, c3 = st.columns(3)
c1.metric("Score", st.session_state.score)
c2.metric("Rounds", st.session_state.rounds)
acc = (f"{round(st.session_state.score / st.session_state.rounds * 100)}%"
       if st.session_state.rounds else "—")
c3.metric("Accuracy", acc)

st.divider()

# First start
if not st.session_state.round_started:
    if st.button("▶ Start — first customer", use_container_width=True, type="primary"):
        new_round()
        st.rerun()
    st.stop()

# Customer banner
if not st.session_state.checked_out:
    st.info(f"**Customer: {st.session_state.customer_name}** — basket loaded. "
            f"Add more items if you like, then checkout.")
else:
    st.info(f"**Customer: {st.session_state.customer_name}** checked out — "
            f"which segment do they belong to?")

# Layout: shop | cart
shop_col, cart_col = st.columns([2, 1])

# ── Shop grid ──
with shop_col:
    st.subheader("🏪 Products")
    col_s, col_c = st.columns([2, 1])
    search = col_s.text_input("Search", placeholder="Search…", label_visibility="collapsed")
    cat_filter = col_c.selectbox("Category", ["All"] + list(CATEGORIES.keys()),
                                 label_visibility="collapsed")

    filtered = ALL_PRODUCTS
    if cat_filter != "All":
        filtered = [p for p in filtered if p in CATEGORIES[cat_filter]]
    if search:
        filtered = [p for p in filtered if search.lower() in p.lower()]

    grid_cols = st.columns(4)
    for i, product in enumerate(filtered):
        with grid_cols[i % 4]:
            qty = st.session_state.cart.get(product, 0)
            badge = f" ×{qty}" if qty else ""
            label = f"{EMOJI.get(product,'🛒')} {product.title()}{badge}\n€{PRICES[product]:.2f}"
            if st.button(label, key=f"add_{product}",
                         disabled=st.session_state.checked_out,
                         use_container_width=True):
                add_item(product)
                st.rerun()

# ── Cart ──
with cart_col:
    st.subheader("🧺 Cart")

    if not st.session_state.cart:
        st.caption("Cart is empty")
    else:
        for product, qty in list(st.session_state.cart.items()):
            a, b, c = st.columns([3, 2, 1])
            a.write(f"{EMOJI.get(product,'🛒')} {product.title()}")
            b.write(f"×{qty}  **€{PRICES[product]*qty:.2f}**")
            if not st.session_state.checked_out:
                if c.button("✕", key=f"rm_{product}"):
                    remove_item(product)
                    st.rerun()
        st.write(f"**Total: €{cart_total():.2f}**")

    st.divider()

    # Checkout button
    if not st.session_state.checked_out:
        if st.session_state.cart:
            if st.button("💳 Checkout", use_container_width=True, type="primary"):
                st.session_state.checked_out = True
                st.rerun()
    else:
        # Segment guess
        if not st.session_state.guessed:
            st.write("**Which segment?**")
            for seg_name, seg_data in SEGMENTS.items():
                if st.button(f"{seg_name}\n*{seg_data['desc']}*",
                             key=f"guess_{seg_name}", use_container_width=True):
                    correct = seg_name == st.session_state.true_seg
                    st.session_state.guessed = True
                    st.session_state.guess_result = correct
                    st.session_state.rounds += 1
                    if correct:
                        st.session_state.score += 1
                    st.rerun()
        else:
            true = st.session_state.true_seg
            info = SEGMENTS[true]
            if st.session_state.guess_result:
                st.success(f"✅ **{true}**\n\n*{info['desc']}*")
            else:
                st.error(f"❌ It was **{true}**\n\n*{info['desc']}*")

            if st.button("▶ Next customer", use_container_width=True, type="primary"):
                new_round()
                st.rerun()
