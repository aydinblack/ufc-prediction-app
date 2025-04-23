# --- GEREKLİ KÜTÜPHANELER ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import base64

# --- MODEL VE VERİ SETLERİ YÜKLE ---
model = joblib.load("best_catboost_model.pkl")
scaler = joblib.load("robust_scaler.pkl")
model_columns = joblib.load("model_columns.pkl")
full_df = pd.read_csv("Datasets/large_dataset.csv")
athlete_df = pd.read_csv("Datasets/athlete_data.csv")


# --- UFC TEMALI CSS STİLLERI ---
def load_ufc_styles():
    return """
    <style>
        /* MODERN UFC TEMA RENKLERİ VE STILLER */
        .stApp {
            background-image: linear-gradient(to bottom, #0f0f1a, #1a1a2a);
            color: #f0f0f0;
            font-family: 'Segoe UI', 'Roboto', sans-serif;
        }
        h1, h2 {
            color: #ff1e27 !important;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            font-weight: 800;
            letter-spacing: 1px;
        }
        h3 {
            color: #f0f0f0 !important;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            font-weight: 800;
            letter-spacing: 1px;
        }
        /* Gradient button styling */
        .stButton>button {
            background: linear-gradient(90deg, #ff1e27, #1e3cff) !important;
            color: white !important;
            font-weight: 700 !important;
            font-size: 18px !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 14px 20px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.5) !important;
            letter-spacing: 1.5px !important;
            cursor: pointer !important;
            text-align: center !important;
            font-family: 'Segoe UI', 'Roboto', sans-serif !important;
        }
        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5), 
                        0 0 10px rgba(255, 30, 39, 0.5), 
                        0 0 15px rgba(30, 60, 255, 0.5) !important;
            background: linear-gradient(90deg, #ff1e27, #1e3cff) !important;
        }
        .stButton>button:active {
            transform: translateY(1px) !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
        }
        .fight-card {
            border-radius: 8px;
            overflow: hidden;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }
        .fight-button-container {
            margin: 15px 0;
            text-align: center;
        }
        .fight-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        }
        .selectbox {
            background-color: rgba(255, 255, 255, 0.08) !important;
            color: white !important;
            border-radius: 4px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        .stSelectbox>div>div {
            background-color: #1a1a2a !important;
            color: white !important;
        }
        /* Animasyon Sınıfları */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideInFromLeft {
            0% { transform: translateX(-100%); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideInFromRight {
            0% { transform: translateX(100%); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
        }
        @keyframes bounceIn {
            0% { transform: scale(0.3); opacity: 0; }
            50% { transform: scale(1.05); opacity: 0.9; }
            70% { transform: scale(0.9); opacity: 1; }
            100% { transform: scale(1); opacity: 1; }
        }
        @keyframes glowingText {
            0% { text-shadow: 0 0 5px #ff1e27, 0 0 10px #ff1e27; }
            50% { text-shadow: 0 0 15px #ff1e27, 0 0 20px #ff1e27; }
            100% { text-shadow: 0 0 5px #ff1e27, 0 0 10px #ff1e27; }
        }
        @keyframes glowingTextBlue {
            0% { text-shadow: 0 0 5px #1e3cff, 0 0 10px #1e3cff; }
            50% { text-shadow: 0 0 15px #1e3cff, 0 0 20px #1e3cff; }
            100% { text-shadow: 0 0 5px #1e3cff, 0 0 10px #1e3cff; }
        }
        .animate-slide-left {
            animation: slideInFromLeft 0.6s ease-out forwards;
        }
        .animate-slide-right {
            animation: slideInFromRight 0.6s ease-out forwards;
        }
        .animate-bounce {
            animation: bounceIn 0.7s ease-out forwards;
        }
        .animate-pulse {
            animation: pulse 1.5s infinite;
        }
        .animate-glow {
            animation: glowingText 1.5s infinite;
        }
        .animate-glow-blue {
            animation: glowingTextBlue 1.5s infinite;
        }
        .stat-card {
            background-color: rgba(30, 30, 50, 0.2);
            backdrop-filter: blur(10px);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 8px;
            transition: all 0.2s ease;
            border-left: 3px solid;
        }
        .stat-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        .red-stats {
            border-left-color: #ff1e27;
        }
        .blue-stats {
            border-left-color: #1e3cff;
        }
        /* Vs İkonunun Animasyonu */
        .vs-animation {
            animation: pulse 1.5s infinite;
            font-size: 50px;
            color: #ffffff;
            text-shadow: 0 0 10px rgba(255, 30, 39, 0.6), 0 0 20px rgba(30, 60, 255, 0.6);
        }
        /* Octagon Arka Plan */
        .octagon-bg {
            position: relative;
        }
        .octagon-bg::before {
            content: "";
            background-image: url('https://www.transparentpng.com/thumb/ufc/ufc-logo-transparent-background-9.png');
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            opacity: 0.05;
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: -1;
        }
        /* Modern glassmorphism effect */
        .glass-card {
            background: rgba(30, 30, 50, 0.2);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }
        /* Modern progress bar */
        .modern-progress {
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            overflow: hidden;
            margin: 10px 0;
        }
        .modern-progress-bar {
            height: 100%;
            border-radius: 2px;
        }
        /* Kazanan animasyonu */
        @keyframes winnerGlow {
            0% { box-shadow: 0 0 5px rgba(255, 215, 0, 0.7), 0 0 10px rgba(255, 215, 0, 0.5); }
            50% { box-shadow: 0 0 20px rgba(255, 215, 0, 0.9), 0 0 30px rgba(255, 215, 0, 0.7); }
            100% { box-shadow: 0 0 5px rgba(255, 215, 0, 0.7), 0 0 10px rgba(255, 215, 0, 0.5); }
        }

        @keyframes championRibbonAnimation {
            0% { transform: translateY(-5px); opacity: 0.8; }
            50% { transform: translateY(0px); opacity: 1; }
            100% { transform: translateY(-5px); opacity: 0.8; }
        }

        .winner-card {
            position: relative;
            animation: winnerGlow 2s infinite;
            border: 2px solid gold !important;
        }

        .champion-ribbon {
            position: absolute;
            top: -10px;
            left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(90deg, #FFD700, #FFA500);
            color: #000;
            padding: 5px 15px;
            border-radius: 15px;
            font-weight: bold;
            letter-spacing: 1px;
            text-transform: uppercase;
            font-size: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            animation: championRibbonAnimation 2s infinite ease-in-out;
            z-index: 10;
        }

        /* Dövüşçü kartları için yeni stil */
        .fighter-card-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 20px;
        }

        .fighter-card {
            display: inline-block;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
            background-color: #1a1a2a;
            width: 160px;
            transition: transform 0.3s ease;
        }

        .fighter-card:hover {
            transform: translateY(-5px);
        }

        .fighter-card img {
            width: 100%;
            height: 160px;
            object-fit: cover;
            display: block;
        }

        .fighter-card .fighter-name {
            padding: 8px;
            font-weight: 500;
            font-size: 16px;
            letter-spacing: 0.5px;
            color: white;
            text-align: center;
        }

        /* VS bölümü için stil */
        .vs-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0 20px;
        }
    </style>
    """


# --- GÖRSEL GETİRME ---
fighter_image_map = dict(zip(athlete_df["Athlete Name"], athlete_df["Image URL"]))


def get_fighter_image(fighter_name):
    url = fighter_image_map.get(fighter_name)
    if isinstance(url, str) and url.startswith("http"):
        return url
    else:
        return "https://cdn.pixabay.com/photo/2024/03/22/15/32/ai-generated-8649918_1280.png"


# --- DÖVÜŞÇÜ KARTI GÖSTERME ---
def show_fighter_card(fighter_name, border_color, is_winner=False):
    """En basit yapıda kart gösterimi"""
    img_url = get_fighter_image(fighter_name)

    # Basit HTML yapısı
    card_html = f"""
    <div style="width: 160px; margin: 0 auto; text-align: center;">
        <img src="{img_url}" style="width: 160px; height: 160px; object-fit: cover; border-radius: 8px 8px 0 0; display: block;" />
        <div style="background-color: {border_color}; padding: 8px; border-radius: 0 0 8px 8px;">
            <span style="color: white; font-weight: 500;">{fighter_name}</span>
        </div>
    </div>
    """

    return card_html


# --- KAZANAN İŞARETİ GÖSTERME ---
def show_winner_badge(fighter_name):
    """Kazanan rozeti gösterimi - ayrı bir fonksiyon olarak"""
    return f"""
    <div style="width: 160px; margin: 10px auto; text-align: center;">
        <div style="background: linear-gradient(90deg, #FFD700, #FFA500); 
                  color: black; padding: 5px 10px; border-radius: 15px; 
                  font-weight: bold; display: inline-block; font-size: 12px;">
            KAZANAN
        </div>
    </div>
    """


# --- PARLAYAN ÇERÇEVE GÖSTERME ---
def show_glowing_frame(fighter_name, border_color, is_winner=False):
    """Kazanan için parlayan çerçeve gösterimi"""
    img_url = get_fighter_image(fighter_name)

    # Kazanan için animasyonlu çerçeve, kaybeden için düz çerçeve
    if is_winner:
        border_style = "border: 2px solid gold;"
        glow_style = "box-shadow: 0 0 15px rgba(255, 215, 0, 0.7);"
        animation = "animation: pulse 1.5s infinite;"
    else:
        border_style = "border: 1px solid rgba(255, 255, 255, 0.1);"
        glow_style = "box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);"
        animation = ""

    return f"""
    <div style="width: 160px; margin: 0 auto; {border_style} {glow_style} {animation} border-radius: 8px; overflow: hidden;">
        <img src="{img_url}" style="width: 100%; height: 160px; object-fit: cover; display: block;" />
        <div style="background-color: {border_color}; padding: 8px; text-align: center;">
            <span style="color: white; font-weight: 500;">{fighter_name}</span>
        </div>
    </div>
    """


# --- DÖVÜŞÇÜ BİLGİ GETİRME ---
def get_fighter_info(df, fighter, prefix):
    # Dövüşçü adındaki fazla boşlukları temizle
    fighter = fighter.strip()

    # Dövüşçünün veri setinde olup olmadığını kontrol et
    if fighter not in df['r_fighter'].values and fighter not in df['b_fighter'].values:
        st.error(f"Dövüşçü {fighter} veri setinde bulunamadı.")
        return {}

    # Dövüşçüye ait satırları seç
    row = df[(df["r_fighter"] == fighter) | (df["b_fighter"] == fighter)]

    if row.empty:
        st.warning(f"Dövüşçü {fighter} için satır bulunamadı.")
        return {}

    # Bilgileri topla
    info = {}
    try:
        # "r_" veya "b_" prefixlerine göre uygun kolonu seç
        if prefix == "r":
            fighter_prefix = "r_"
        else:
            fighter_prefix = "b_"

        # Galibiyetler, mağlubiyetler ve toplam maçlar
        if f"{fighter_prefix}wins_total" in row.columns:
            # NaN kontrolü yap
            wins_mean = row[f"{fighter_prefix}wins_total"].mean()
            if pd.isna(wins_mean):
                wins = 0
            else:
                wins = int(wins_mean)
            info["Galibiyet"] = wins
        else:
            st.warning(f"{fighter_prefix}wins_total kolonu bulunamadı")
            wins = 0

        if f"{fighter_prefix}losses_total" in row.columns:
            # NaN kontrolü yap
            losses_mean = row[f"{fighter_prefix}losses_total"].mean()
            if pd.isna(losses_mean):
                losses = 0
            else:
                losses = int(losses_mean)
            info["Mağlubiyet"] = losses
        else:
            st.warning(f"{fighter_prefix}losses_total kolonu bulunamadı")
            losses = 0

        total_matches = wins + losses
        info["Toplam Maç"] = total_matches

        # Galibiyet oranı
        win_rate = (wins / total_matches * 100) if total_matches > 0 else 0
        info["Galibiyet Oranı (%)"] = f"%{win_rate:.2f}"

        # Yaş
        if f"{fighter_prefix}age" in row.columns:
            age_mean = row[f"{fighter_prefix}age"].mean()
            if not pd.isna(age_mean):
                info["Yaş"] = int(age_mean)

        # Boy
        if f"{fighter_prefix}height" in row.columns:
            height_mean = row[f"{fighter_prefix}height"].mean()
            if not pd.isna(height_mean):
                info["Boy (cm)"] = f"{height_mean:.0f} cm"

        # Kilo
        if f"{fighter_prefix}weight" in row.columns:
            weight_mean = row[f"{fighter_prefix}weight"].mean()
            if not pd.isna(weight_mean):
                info["Kilo (kg)"] = f"{weight_mean:.0f} kg"

        # Dövüş stili
        if f"{fighter_prefix}stance" in row.columns and not row[f"{fighter_prefix}stance"].empty:
            # NaN değerlerini filtrele
            valid_stances = row[f"{fighter_prefix}stance"].dropna()
            if not valid_stances.empty:
                stance_mode = valid_stances.mode()
                info["Stil"] = stance_mode.iloc[0] if not stance_mode.empty else "-"
            else:
                info["Stil"] = "-"
        else:
            info["Stil"] = "-"

    except Exception as e:
        st.error(f"Dövüşçü bilgileri alınırken hata oluştu: {e}")

    return info


# --- YÜKLEME ANİMASYONU ---
def loading_animation():
    with st.spinner(""):
        # Daha minimal, modern yükleme ekranı
        progress_placeholder = st.empty()
        message_placeholder = st.empty()

        for percent_complete in range(0, 101, 4):
            progress_bar_html = f"""
            <div style="width: 100%; background-color: rgba(255,255,255,0.1); height: 4px; border-radius: 2px; margin: 10px 0;">
                <div style="width: {percent_complete}%; background-color: {'#ff1e27' if percent_complete < 50 else '#1e3cff'}; 
                      height: 100%; border-radius: 2px; transition: width 0.2s ease;"></div>
            </div>
            """
            progress_placeholder.markdown(progress_bar_html, unsafe_allow_html=True)

            if percent_complete == 25:
                message_placeholder.markdown(
                    "<div style='text-align:center; font-size:14px; color:#f0f0f0; opacity:0.7;'>Veri analiz ediliyor</div>",
                    unsafe_allow_html=True
                )
            elif percent_complete == 50:
                message_placeholder.markdown(
                    "<div style='text-align:center; font-size:14px; color:#f0f0f0; opacity:0.7;'>Model çalıştırılıyor</div>",
                    unsafe_allow_html=True
                )
            elif percent_complete == 75:
                message_placeholder.markdown(
                    "<div style='text-align:center; font-size:14px; color:#f0f0f0; opacity:0.7;'>Tahmin hesaplanıyor</div>",
                    unsafe_allow_html=True
                )

            time.sleep(0.03)

        # Animasyonu temizle
        progress_placeholder.empty()
        message_placeholder.empty()


# --- UFC Temalı Sayfa Başlığı ---
def show_ufc_header():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 25px;">
        <img src="https://pngimg.com/d/ufc_PNG61.png" style="max-width: 200px; margin-bottom: 10px;">
        <div style="font-size: 18px; color: #f0f0f0; margin-top: 5px; opacity: 0.9; font-weight: 300;">DÖVÜŞ TAHMİN UYGULAMASI</div>
    </div>
    <div style="text-align: center; margin-bottom: 25px; padding: 8px; border-radius: 4px; 
          background: linear-gradient(90deg, #ff1e27, rgba(30, 30, 50, 0.5), #1e3cff);">
        <span style="color: white; font-weight: 500; letter-spacing: 1px; font-size: 14px;">GERÇEK ZAMANDA TAHMİN GÜCÜ</span>
    </div>
    """, unsafe_allow_html=True)


# --- COUNTDOWNA ANİMASYONU ---
def show_fight_countdown():
    # Daha minimalist sayaç animasyonu
    countdown_placeholder = st.empty()

    for i in range(3, 0, -1):
        countdown_placeholder.markdown(f"""
        <div style="text-align: center; animation: fadeIn 0.3s;">
            <div style="font-size: 40px; font-weight: 500; color: {'#ff1e27' if i == 3 else '#f0f0f0'}; opacity: 0.9;">{i}</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.4)

    countdown_placeholder.markdown("""
    <div style="text-align: center; animation: fadeIn 0.3s;">
        <div style="font-size: 40px; font-weight: 600; color: #1e3cff;">FIGHT!</div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(0.4)
    countdown_placeholder.empty()


# --- KAZANAN DUYURU EFEKTİ ---
def show_winner_announcement(winner_name, winning_proba, winner_color):
    # Kazanan rengi belirleme (kırmızı veya mavi)
    solid_color = "#ff1e27" if winner_color == "#ff1e27" or winner_color == "#FF0000" else "#1e3cff"

    st.markdown(f"""
    <div style="text-align: center; animation: fadeIn 0.5s;">
        <div style="background-color: {solid_color};
             padding: 15px; border-radius: 8px; margin: 10px 0;">
            <div style="font-size: 14px; color: white; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px;">KAZANAN</div>
            <div style="font-size: 32px; font-weight: 700; color: white; margin: 5px 0;">
                {winner_name}
            </div>
            <div style="font-size: 16px; color: white; opacity: 0.9;">
                Kazanma Olasılığı: <span style="font-weight: 600;">%{winning_proba * 100:.1f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- UYGULAMA BAŞLANGICI ---
def main():
    st.set_page_config(
        page_title="🏆 UFC Octagon Fight Predictor",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # UFC temayı uygula
    st.markdown(load_ufc_styles(), unsafe_allow_html=True)

    # UFC temali başlık
    show_ufc_header()

    # Octagon Arka Plan
    st.markdown('<div class="octagon-bg">', unsafe_allow_html=True)

    # Cinsiyet ve dövüşçü filtreleme
    fighter_gender_map = full_df[['r_fighter', 'gender']].drop_duplicates().set_index('r_fighter').to_dict()['gender']
    fighter_gender_map.update(
        full_df[['b_fighter', 'gender']].drop_duplicates().set_index('b_fighter').to_dict()['gender'])
    fighters = sorted(set(full_df['r_fighter'].unique()).union(set(full_df['b_fighter'].unique())))

    genders = sorted(full_df["gender"].dropna().unique())
    selected_gender = st.selectbox("👤 Cinsiyet Seçin", genders)
    filtered_fighters = sorted([f for f in fighters if fighter_gender_map.get(f) == selected_gender])

    # Dövüşçü Seçimi
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="text-align: center; background-color: rgba(255, 30, 39, 0.3); padding: 10px; border-radius: 4px; 
                   border-left: 4px solid #ff1e27; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
            <div style="font-size: 16px; color: #ffffff; font-weight: 700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); letter-spacing: 1px;">RED KÖŞE</div>
        </div>
        """, unsafe_allow_html=True)
        red_fighter = st.selectbox("", filtered_fighters, label_visibility="collapsed")

    with col2:
        st.markdown("""
        <div style="text-align: center; background-color: rgba(30, 60, 255, 0.3); padding: 10px; border-radius: 4px;
                   border-left: 4px solid #1e3cff; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
            <div style="font-size: 16px; color: #ffffff; font-weight: 700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); letter-spacing: 1px;">BLUE KÖŞE</div>
        </div>
        """, unsafe_allow_html=True)
        valid_blue_fighters = [f for f in filtered_fighters if f != red_fighter]
        blue_fighter = st.selectbox("", valid_blue_fighters, label_visibility="collapsed")

    # Tahmin butonu
    colA, colB, colC = st.columns([1, 2, 1])
    with colB:
        # Normal button with custom CSS applied
        tahmin = st.button("DÖVÜŞÜ BAŞLAT", use_container_width=True)

    if tahmin:
        # Yükleme animasyonu göster
        loading_animation()

        # Dövüş geri sayımı
        show_fight_countdown()

        # --- Hızlı Çözüm: Model inputunda alfabetik sıralama zorunluluğu ---
        fighter_a, fighter_b = sorted([red_fighter, blue_fighter])

        red_stats = full_df[(full_df["r_fighter"] == fighter_a) | (full_df["b_fighter"] == fighter_a)]
        blue_stats = full_df[(full_df["r_fighter"] == fighter_b) | (full_df["b_fighter"] == fighter_b)]

        numeric_cols = red_stats.select_dtypes(include=[np.number]).columns

        red_row = {col: red_stats[red_stats["r_fighter"] == fighter_a][col].mean() if col.startswith("r_") else
        red_stats[red_stats["b_fighter"] == fighter_a][col].mean() for col in numeric_cols}
        blue_row = {col: blue_stats[blue_stats["r_fighter"] == fighter_b][col].mean() if col.startswith("r_") else
        blue_stats[blue_stats["b_fighter"] == fighter_b][col].mean() for col in numeric_cols}

        red_row = pd.Series(red_row)
        blue_row = pd.Series(blue_row)

        sample = pd.DataFrame()
        for col in red_row.index:
            if col.startswith("r_"):
                sample[col] = [red_row[col]]
            elif col.startswith("b_"):
                sample[col] = [blue_row[col]]

        shared_cols = ["weight_class", "gender", "is_title_bout", "total_rounds"]
        for col in shared_cols:
            if col in full_df.columns:
                try:
                    filtered = full_df[((full_df['r_fighter'] == fighter_a) & (full_df['b_fighter'] == fighter_b)) | (
                            (full_df['r_fighter'] == fighter_b) & (full_df['b_fighter'] == fighter_a))]
                    mode_val = filtered[col].mode()
                    sample[col] = [mode_val.iloc[0] if not mode_val.empty else np.nan]
                except:
                    sample[col] = [np.nan]
            else:
                sample[col] = [np.nan]

        sample["wins_total_diff"] = red_row.get("r_wins_total", 0) - blue_row.get("b_wins_total", 0)
        sample["losses_total_diff"] = red_row.get("r_losses_total", 0) - blue_row.get("b_losses_total", 0)
        sample["age_diff"] = red_row.get("r_age", 0) - blue_row.get("b_age", 0)
        sample["td_avg_diff"] = red_row.get("r_td_avg", 0) - blue_row.get("b_td_avg", 0)
        sample["kd_diff"] = red_row.get("r_kd", 0) - blue_row.get("b_kd", 0)

        for col in model_columns:
            if col not in sample.columns:
                sample[col] = 0
        sample = sample[model_columns]

        sample_scaled = scaler.transform(sample)
        proba = model.predict_proba(sample_scaled)[0][1]

        # Modelde fighter_a vs fighter_b input verildiği için, red/blue gösterimi kullanıcı tarafında tutuluyor
        winner_name = red_fighter if ((red_fighter == fighter_a and proba >= 0.5) or (
                red_fighter == fighter_b and proba < 0.5)) else blue_fighter
        winning_proba = proba if (winner_name == fighter_a) else 1 - proba

        # Kazananın rengini ayarla
        winner_color = "#FF0000" if winner_name == red_fighter else "#3399FF"

        # Kazananı duyur
        show_winner_announcement(winner_name, winning_proba, winner_color)

        # --- Seçilen Dövüşçüler ve VS İkonu ---
        st.markdown(
            '<hr style="border:none; height:1px; background: linear-gradient(to right, transparent, rgba(255,255,255,0.1), transparent); margin:20px 0;">',
            unsafe_allow_html=True)

        # 3-kolonlu düzen
        cols = st.columns([40, 20, 40])

        # Kırmızı dövüşçü
        with cols[0]:
            # 1. Parlayan çerçeve
            st.markdown(
                show_glowing_frame(red_fighter, "#ff1e27", is_winner=(red_fighter == winner_name)),
                unsafe_allow_html=True
            )

            # 2. Sadece kazanan için rozet göster
            if red_fighter == winner_name:
                st.markdown(show_winner_badge(red_fighter), unsafe_allow_html=True)

        # VS yazısı
        with cols[1]:
            st.markdown(
                '<div style="height: 160px; display: flex; align-items: center; justify-content: center;">'
                '<span style="font-size: 40px; color: white; font-weight: bold; '
                'text-shadow: 0 0 10px rgba(255, 30, 39, 0.6), 0 0 20px rgba(30, 60, 255, 0.6); '
                'animation: pulse 1.5s infinite;">'
                'VS'
                '</span>'
                '</div>',
                unsafe_allow_html=True
            )

        # Mavi dövüşçü
        with cols[2]:
            # 1. Parlayan çerçeve
            st.markdown(
                show_glowing_frame(blue_fighter, "#1e3cff", is_winner=(blue_fighter == winner_name)),
                unsafe_allow_html=True
            )

            # 2. Sadece kazanan için rozet göster
            if blue_fighter == winner_name:
                st.markdown(show_winner_badge(blue_fighter), unsafe_allow_html=True)

        # --- Dövüşçü Özeti ---
        st.markdown("""
        <div style="text-align: center; margin: 25px 0 15px;">
            <div style="font-size: 18px; color: #f0f0f0; text-transform: uppercase; letter-spacing: 1px; font-weight: 300;">
                DÖVÜŞÇÜ İSTATİSTİKLERİ
            </div>
            <div style="width: 50px; height: 2px; background: linear-gradient(to right, #ff1e27, #1e3cff); margin: 8px auto;"></div>
        </div>
        """, unsafe_allow_html=True)

        # Dövüşçü bilgilerini al
        red_info = get_fighter_info(full_df, red_fighter, "r")
        blue_info = get_fighter_info(full_df, blue_fighter, "b")

        # İstatistikleri görselleştir
        plot_stats_bar_chart(red_info, blue_info, red_fighter, blue_fighter)

    st.markdown('</div>', unsafe_allow_html=True)  # Octagon bg kapanış


# --- İSTATİSTİKLERİ BAR GRAFİKLE GÖSTERME ---
def plot_stats_bar_chart(fighter1_info, fighter2_info, red_fighter, blue_fighter):
    """
    İki dövüşçünün istatistiklerini bar grafik olarak göster.
    Her iki dövüşçü için barlar alt alta gösterilir.
    Her istatistik için ayrı bir başlık.
    """
    if not fighter1_info or not fighter2_info:
        st.error("Dövüşçü bilgileri alınamadı. Bar grafikler gösterilemiyor.")
        return

    # İstatistik kategorilerini çıkar
    common_stats = []
    for stat in fighter1_info:
        if stat in fighter2_info:
            # "Stil" istatistiğini atlayalım çünkü sayısal bir değer değil
            if stat != "Stil":
                common_stats.append(stat)

    if not common_stats:
        st.warning("Karşılaştırılabilir istatistik bulunamadı.")
        return

    # Dövüşçülerin değerlerini al ve float'a çevir
    red_values = []
    blue_values = []
    valid_stats = []

    for stat in common_stats:
        try:
            # Yüzde, cm, kg değerlerini işle
            red_val = fighter1_info[stat]
            blue_val = fighter2_info[stat]

            # Null veya NaN değerleri kontrol et
            if red_val is None or (isinstance(red_val, float) and pd.isna(red_val)) or \
                    blue_val is None or (isinstance(blue_val, float) and pd.isna(blue_val)):
                continue

            # String değerleri float'a çevir
            if isinstance(red_val, str):
                # % işaretini, cm veya kg birimlerini kaldır
                clean_red = red_val.replace("%", "").replace(" cm", "").replace("cm", "").replace(" kg", "").replace(
                    "kg", "")
                try:
                    red_val = float(clean_red)
                except ValueError:
                    # Sayıya çevrilemiyorsa atla
                    continue
            elif isinstance(red_val, (int, float)):
                red_val = float(red_val)
            else:
                # Sayısal olmayan değerleri atla
                continue

            if isinstance(blue_val, str):
                clean_blue = blue_val.replace("%", "").replace(" cm", "").replace("cm", "").replace(" kg", "").replace(
                    "kg", "")
                try:
                    blue_val = float(clean_blue)
                except ValueError:
                    # Sayıya çevrilemiyorsa atla
                    continue
            elif isinstance(blue_val, (int, float)):
                blue_val = float(blue_val)
            else:
                # Sayısal olmayan değerleri atla
                continue

            # NaN kontrolleri
            if pd.isna(red_val) or pd.isna(blue_val):
                continue

            # Listelere ekle
            red_values.append(red_val)
            blue_values.append(blue_val)
            valid_stats.append(stat)
        except (ValueError, TypeError) as e:
            # Stil değeri hata verdiğinde gizli uyarı - konsola yazdırılacak
            if stat != "Stil":
                st.warning(f"{stat} değeri hesaplanamadı: {e}")
            continue

    if not valid_stats:
        st.warning("Gösterilebilecek sayısal istatistik bulunamadı.")
        return

    # Değerleri formatlama fonksiyonu
    def format_value(val):
        # Float değer ise ve tam sayı ise, int olarak göster
        if isinstance(val, float) and val.is_integer():
            return str(int(val))
        # Integer ise direkt string'e çevir
        elif isinstance(val, int):
            return str(val)
        # Float ve tam sayı değilse iki ondalık basamakla göster
        else:
            return f"{val:.2f}"

    # Her istatistik için bar grafiklerini oluştur
    for i, stat in enumerate(valid_stats):
        r_val = red_values[i]
        b_val = blue_values[i]

        # Son bir kez daha NaN kontrolü
        if pd.isna(r_val) or pd.isna(b_val):
            continue

        # Maksimum değerleri hesapla
        max_val = max(r_val, b_val) * 1.2
        if max_val == 0:  # Sıfıra bölünmeyi önle
            max_val = 1

        # Yüzde hesapla - NaN kontrolleri ekle
        try:
            red_percentage = int((r_val / max_val) * 100) if max_val > 0 and not pd.isna(r_val) else 0
            blue_percentage = int((b_val / max_val) * 100) if max_val > 0 and not pd.isna(b_val) else 0
        except (TypeError, ValueError):
            # Herhangi bir hesaplama hatası olursa bu istatistiği atla
            continue

        # İstatistik başlığı
        st.markdown(f"""
        <div style="text-align: center; margin-top: 20px; margin-bottom: 10px;">
            <h3 style="color: #f0f0f0; font-size: 18px; margin: 0; padding: 0;">{stat}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Kırmızı dövüşçü barı
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="width: 40px; text-align: right; color: white; font-weight: bold; margin-right: 10px;">
                {format_value(r_val)}
            </div>
            <div style="flex-grow: 1; height: 30px; background-color: rgba(50, 50, 50, 0.3); position: relative; border-radius: 4px; overflow: hidden;">
                <div style="position: absolute; left: 0; top: 0; width: {red_percentage}%; height: 100%; background-color: #ff1e27; border-radius: 4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Mavi dövüşçü barı
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="width: 40px; text-align: right; color: white; font-weight: bold; margin-right: 10px;">
                {format_value(b_val)}
            </div>
            <div style="flex-grow: 1; height: 30px; background-color: rgba(50, 50, 50, 0.3); position: relative; border-radius: 4px; overflow: hidden;">
                <div style="position: absolute; left: 0; top: 0; width: {blue_percentage}%; height: 100%; background-color: #1e3cff; border-radius: 4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()