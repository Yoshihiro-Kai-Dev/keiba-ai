import streamlit as st
import pandas as pd
import joblib
import requests
import re
import datetime
import json
import textwrap
import time
import random
import base64
import os
import numpy as np
import concurrent.futures
import threading 
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
import queue
import streamlit.components.v1 as components
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
from sqlalchemy import text
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import google.generativeai as genai

# ★ secretsからキーを読み込むようにする
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    # secretsがない場合（ここにはキーを書かない！）
    GEMINI_API_KEY = None 

def generate_gemini_comment(row):
    """
    特徴量データを受け取り、Geminiに寸評を書かせる関数
    戻り値: (寸評テキスト, 使用したモデル名) のタプル
    """
    api_key = None
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        api_key = GEMINI_API_KEY 

    if not api_key:
        return "⚠️ APIキー設定が必要です (.streamlit/secrets.toml)", "No Key"

    # フォールバック順序
    candidate_models = [
        'gemini-2.0-flash-exp',
        'gemini-2.0-flash',
        'gemini-2.0-flash-001',
        'gemini-2.0-flash-exp-image-generation',
        'gemini-2.0-flash-lite-001',
        'gemini-2.0-flash-lite',
        'gemini-2.0-flash-lite-preview-02-05',
        'gemini-2.0-flash-lite-preview',
        'gemini-exp-1206',
        # 'gemini-2.5-flash-preview-tts',
        # 'gemini-2.5-pro-preview-tts',
        'gemini-2.5-flash-image-preview',
        'gemini-2.5-flash-image',
        'gemini-2.5-flash-preview-09-2025',
        'gemini-2.5-flash-lite-preview-09-2025',
        'gemini-3-pro-preview',
        'gemini-3-flash-preview',
        'gemini-3-pro-image-preview',
        'gemini-robotics-er-1.5-preview',
        'gemini-2.5-computer-use-preview-10-2025',
        # 'gemini-embedding-exp-03-07',
        # 'gemini-embedding-exp',
        # 'gemini-embedding-001',
        # 'gemini-2.5-flash-native-audio-latest',
        # 'gemini-2.5-flash-native-audio-preview-09-2025',
        # 'gemini-2.5-flash-native-audio-preview-12-2025',
        'gemini-2.5-pro',
        'gemini-2.5-flash',
        'gemini-2.5-flash-lite',
        'gemini-flash-latest',
        'gemini-flash-lite-latest',
        'gemini-pro-latest'
    ]

    # プロンプト（前回と同じ熱血版）
    prompt = f"""
    あなたは日本一の競馬予想AIです。以下のデータに基づき、この馬が「なぜ買いなのか」を
    競馬新聞のベテラン記者が書くような、読み手の心を揺さぶる「熱い推奨コメント」で書いてください。

    【馬データ】
    ・馬名: {row['馬名']}
    ・騎手: {row['騎手']} (勝率: {row.get('jockey_win_rate', 0)*100:.1f}%)
    ・調教師: {row['調教師']} (勝率: {row.get('trainer_win_rate', 0)*100:.1f}%)
    ・AI信頼度: {row['AIスコア']*100:.1f}% (高い！)
    ・近走3走平均着順: {row.get('recent_rank_avg', '不明')}位
    ・脚質傾向: {"先行" if row.get('run_style_ratio', 0) > 0.5 else "差し・追込"}
    
    【執筆ルール（絶対遵守）】
    1. **250文字程度**でわかりやすくまとめること。
    2. 「～です」「～ます」は禁止。「～だ！」「～に違いない！」と断定口調にする。
    3. 数値を並べるのではなく、「驚異の勝率」「安定感抜群」といった**感情的な言葉**に変換する。
    4. 最後に必ず、「迷わず買え！」「本命はこの馬だ！」といった力強い一言で締める。
    5. 競馬ファンが好む「専門用語（脚質、展開、手綱捌きなど）」を自然に混ぜる。
    """

    genai.configure(api_key=api_key)

    # 修正版ループ処理
    errors = []
    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text, model_name 

        except Exception as e:
            error_msg = str(e)
            errors.append(f"{model_name}: {error_msg}") # エラーを記録
            
            # フォールバック処理（次へ）
            if "429" in error_msg or "Quota" in error_msg or "404" in error_msg or "not found" in error_msg:
                continue
            else:
                return f"致命的エラー: {error_msg}", model_name

    return f"🚫 本日のAI予測利用枠（全モデル合計）を使い切りました。明日またお試しください。"

# ---------------------------------------------------------
# 1. 設定 & ページ初期化
# ---------------------------------------------------------
st.set_page_config(
    page_title="KaiのゆるっとAI",
    page_icon="🦄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BACKTEST_PERIOD = "2024/01 ～ Present"

# ---------------------------------------------------------
# 2. UI/UX 定義 (CSS & JS)
# ---------------------------------------------------------
def load_custom_css():
    if 'theme_color' not in st.session_state:
        themes = [
            {'p': '#3b82f6', 'g': 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)', 'h': 'linear-gradient(-45deg, #1e1b4b, #312e81, #4338ca)'},
            {'p': '#f59e0b', 'g': 'linear-gradient(135deg, #f59e0b 0%, #ef4444 100%)', 'h': 'linear-gradient(-45deg, #451a03, #78350f, #b45309)'},
            {'p': '#10b981', 'g': 'linear-gradient(135deg, #10b981 0%, #3b82f6 100%)', 'h': 'linear-gradient(-45deg, #064e3b, #065f46, #059669)'},
            {'p': '#8b5cf6', 'g': 'linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%)', 'h': 'linear-gradient(-45deg, #4c1d95, #5b21b6, #7c3aed)'},
        ]
        st.session_state.theme_color = random.choice(themes)
    
    theme = st.session_state.theme_color

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;900&family=Noto+Sans+JP:wght@400;500;700;900&family=Roboto+Mono:wght@500&display=swap');
        
        :root {{ 
            --primary-color: {theme['p']};
            --primary-gradient: {theme['g']};
            --header-gradient: {theme['h']};
            --bg-color: #f8fafc;
            --card-bg: rgba(255, 255, 255, 0.95);
            --glass-border: 1px solid rgba(255, 255, 255, 0.6);
            --text-main: #1e293b;
            --sub-text: #64748b;
            --shadow-lg: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        }}

        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg-color: #0f172a;
                --card-bg: rgba(30, 41, 59, 0.95);
                --glass-border: 1px solid rgba(255, 255, 255, 0.1);
                --text-main: #f1f5f9;
                --sub-text: #94a3b8;
                --shadow-lg: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
            }}
            .stApp {{ background-color: var(--bg-color) !important; color: var(--text-main) !important; }}
            .stCard {{ background-color: var(--card-bg) !important; border: var(--glass-border) !important; }}
            .hero-rating-box {{ background: #1e293b !important; border: 1px solid #334155; }}
            .pedigree-rate {{ background: #334155 !important; color: #e2e8f0; }}
            .ai-list-card {{ background: #1e293b !important; border-color: #334155 !important; }}
            .report-card-dual {{ background: #1e293b !important; border-color: #334155 !important; }}
            .report-header {{ color: #cbd5e1 !important; }}
            .report-val-sm {{ color: #f1f5f9 !important; }}
            .ev-legend-box {{ background: rgba(30, 41, 59, 0.6) !important; border-color: #475569 !important; }}
            .badge-legend {{ background: #1e293b !important; border-color: #334155 !important; }}
            .hero-stat-val {{ color: #f1f5f9 !important; }}
            .pedigree-name {{ color: #cbd5e1 !important; }}
            .ai-list-card.fire {{ background-color: #3f1818 !important; border-color: #7f1d1d !important; }}
            
            /* 文字色強制 (スマホDark Mode対策) */
            .ai-list-card div {{ color: #f1f5f9 !important; }}
            .ai-list-card span {{ color: #cbd5e1 !important; }}
            .ai-list-card .strategy-badge {{ color: #fff !important; }} 
            .ai-list-card .boost-badge {{ color: #fff !important; }}
            
            /* テーブル内の文字色 */
            div[data-testid="stDataFrame"] {{ color: #f1f5f9 !important; }}
        }}
        
        html {{ scroll-behavior: smooth; }}
        .stApp {{ background-color: var(--bg-color) !important; font-family: 'Inter', 'Noto Sans JP', sans-serif; color: var(--text-main) !important; }}
        
        /* Header */
        .header-container {{ 
            position: relative; padding: 100px 20px; border-radius: 0 0 50px 50px; 
            margin: -6rem -4rem 3rem -4rem; text-align: center; box-shadow: var(--shadow-lg); 
            overflow: hidden; background: #000; 
        }}
        .video-background {{
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            z-index: 0; pointer-events: none; opacity: 0.5; overflow: hidden;
        }}
        /* videoタグ用のCSS (object-fit:coverで全画面対応) */
        .video-background video {{
            width: 100%; height: 100%; 
            position: absolute; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            object-fit: cover;
        }}
        .header-overlay {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: var(--header-gradient); mix-blend-mode: multiply; z-index: 1; opacity: 0.7; }}
        .header-content {{ position: relative; z-index: 2; animation: slideUpFade 1s ease-out; }}
        
        .header-title {{ 
            font-size: 3.5rem; font-weight: 900; margin: 0; color: #ffffff !important; 
            text-shadow: 0 4px 20px rgba(0,0,0,0.5); line-height: 1.1; letter-spacing: -0.02em;
        }}
        .beta-badge {{ 
            font-size: 0.4em; background: rgba(255,255,255,0.15); padding: 6px 18px; 
            border-radius: 30px; border: 1px solid rgba(255,255,255,0.3); 
            backdrop-filter: blur(5px); text-transform: uppercase; letter-spacing: 0.05em; vertical-align: middle;
        }}
        .header-subtitle {{ 
            font-size: 1.3rem; color: rgba(255, 255, 255, 0.95) !important; margin-top: 15px; 
            font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.5); 
        }}
        
        /* Cards */
        .stCard {{ 
            background-color: var(--card-bg) !important; backdrop-filter: blur(12px); 
            border-radius: 24px; padding: 24px; box-shadow: var(--shadow-lg); 
            border: var(--glass-border); margin-bottom: 24px; transition: transform 0.2s;
        }}
        
        /* Hero Card */
        .hero-card {{ border-left: 6px solid var(--primary-color); background: linear-gradient(to right, #fff, #f8fafc); position: relative; overflow: hidden; }}
        .hero-card.fire {{ border-color: #ef4444; background: linear-gradient(135deg, #fffafa 0%, #fff1f2 100%); box-shadow: 0 10px 30px -10px rgba(239, 68, 68, 0.3); }}
        .hero-label {{ font-size: 0.85rem; font-weight: 900; text-transform: uppercase; color: var(--primary-color); letter-spacing: 0.05em; margin-bottom: 4px; }}
        .hero-card.fire .hero-label {{ color: #ef4444; }}
        .hero-horse {{ font-size: 2.4rem; font-weight: 900; color: var(--text-main) !important; margin: 5px 0 20px 0; line-height: 1.1; letter-spacing: -0.02em; }}
        
        .hero-stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; background: rgba(255,255,255,0.1); padding: 20px; border-radius: 16px; border: 1px solid rgba(0,0,0,0.03); }}
        .hero-stat-item {{ display: flex; flex-direction: column; padding: 4px; }}
        .hero-stat-label {{ font-size: 0.75rem; color: var(--sub-text); font-weight: 700; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.03em; }}
        .hero-stat-val {{ font-size: 1.05rem; font-weight: 600; color: var(--text-main); }}
        
        .pedigree-box {{ display: flex; flex-direction: column; gap: 6px; font-size: 0.95rem; }}
        .pedigree-row {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px dashed #cbd5e1; padding-bottom: 4px; }}
        .pedigree-row:last-child {{ border-bottom: none; }}
        .pedigree-name {{ font-weight: 700; color: var(--text-main); }}
        .pedigree-rate {{ font-size: 0.85em; font-weight: 700; background: #f1f5f9; padding: 2px 6px; border-radius: 4px; color: #334155; }}
        .rate-high {{ color: #fff !important; background: #ef4444 !important; }}
        
        .hero-rating-box {{ position: absolute; top: 24px; right: 24px; text-align: center; background: #fff; padding: 12px 24px; border-radius: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
        .hero-rating-val {{ font-size: 2.2rem; font-weight: 900; background: var(--primary-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1; }}
        .hero-card.fire .hero-rating-val {{ background: linear-gradient(135deg, #ef4444 0%, #f59e0b 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .hero-rating-label {{ font-size: 0.7rem; color: #94a3b8; font-weight: 700; letter-spacing: 0.05em; }}

        /* Strategy Lists */
        .ai-list-card {{ background: #fff; border-radius: 16px; padding: 18px; margin-bottom: 16px; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); position: relative; overflow: hidden; transition: all 0.2s; }}
        .ai-list-card:hover {{ transform: translateY(-2px); box-shadow: 0 10px 20px -5px rgba(0,0,0,0.05); border-color: var(--primary-color); }}
        .ai-list-card::before {{ content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; background: var(--primary-gradient); }}
        .ai-list-card.fire::before {{ background: linear-gradient(135deg, #ef4444 0%, #f59e0b 100%); }}
        .ai-list-card.fire {{ background-color: #fffbfc; }}
        .ai-list-card.fire::after {{ content: '🔥'; position: absolute; top: 10px; right: 80px; font-size: 2rem; opacity: 0.1; transform: rotate(15deg); }}

        .ai-card-badges {{ position: absolute; top: 16px; right: 16px; display: flex; gap: 6px; }}
        
        .grade-badge {{ display: inline-block; width: 60px; text-align: center; padding: 5px 0; border-radius: 8px; font-size: 0.75rem; font-weight: 800; color: #fff !important; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .bg-G1 {{ background: linear-gradient(135deg, #3b82f6, #1d4ed8); }} 
        .bg-G2 {{ background: linear-gradient(135deg, #ef4444, #b91c1c); }} 
        .bg-G3 {{ background: linear-gradient(135deg, #22c55e, #15803d); }} 
        .bg-LOP {{ background: linear-gradient(135deg, #f97316, #c2410c); }} 
        .bg-GEN {{ background: #94a3b8; }}

        .strategy-badge {{ display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 900; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
        .boost-badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 0.7rem; font-weight: bold; margin-right: 6px; margin-bottom: 6px; color: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        
        .boost-jockey {{ background: linear-gradient(135deg, #ef4444, #b91c1c); }} 
        .boost-blood {{ background: linear-gradient(135deg, #8b5cf6, #6d28d9); }} 
        .boost-leader {{ background: linear-gradient(135deg, #f59e0b, #d97706); }} 
        .boost-course {{ background: linear-gradient(135deg, #10b981, #047857); }} 
        .boost-speed {{ background: linear-gradient(135deg, #3b82f6, #1d4ed8); }} 
        .boost-dist {{ background: linear-gradient(135deg, #06b6d4, #0891b2); }} 
        .boost-pace {{ background: linear-gradient(135deg, #ec4899, #be185d); }} 
        .boost-gokyak {{ background: linear-gradient(135deg, #6366f1, #4f46e5); }}
        .boost-horse-course {{ background: linear-gradient(135deg, #ea580c, #c2410c); }} 

        .report-card-dual {{ background: #fff; border: 1px solid #e2e8f0; padding: 20px; border-radius: 24px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.03); transition: all 0.3s; position: relative; overflow: hidden; }}
        .report-card-dual:hover {{ transform: translateY(-5px); border-color: var(--primary-color); box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1); }}
        .report-header {{ font-size: 1.0rem; color: var(--sub-text); font-weight: 800; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.05em; display: flex; justify-content: space-between; align-items: center; }}
        .report-grid {{ display: flex; justify-content: space-between; align-items: center; margin-top: 10px; }}
        .report-item {{ text-align: center; width: 48%; }}
        .report-sublabel {{ font-size: 0.7rem; color: #94a3b8; font-weight: 700; margin-bottom: 4px; text-transform: uppercase; }}
        .report-val-sm {{ font-size: 1.6rem; font-weight: 900; line-height: 1.1; color: var(--text-main); }}
        .val-win {{ color: #ef4444; }} 
        .val-neutral {{ color: #3b82f6; }}
        .val-empty {{ color: #94a3b8; font-weight: 400; }}

        .ev-legend-box {{ background: rgba(128,128,128,0.05); padding: 15px; border-radius: 12px; border: 1px dashed var(--border-color); margin-bottom: 20px; color: var(--sub-text); }}
        .badge-legend {{ margin-top: 15px; background: var(--card-bg); padding: 15px; border-radius: 12px; font-size: 0.8rem; color: var(--sub-text); border: 1px solid var(--border-color); }}

        @media (max-width: 768px) {{
            .header-container {{ margin: -4rem -1rem 2rem -1rem; padding: 60px 10px; border-radius: 0 0 30px 30px; }}
            .header-title {{ font-size: 2.2rem; }}
            .hero-stats-grid {{ grid-template-columns: repeat(1, 1fr); }}
            .hero-horse {{ margin-top: 50px; font-size: 1.8rem; }}
            .report-val-sm {{ font-size: 1.3rem; }}
        }}
        
        .to-top-btn {{
            position: fixed; bottom: 30px; right: 30px; width: 50px; height: 50px;
            background: var(--primary-gradient); color: white; border-radius: 50%;
            display: flex; justify-content: center; align-items: center;
            font-size: 24px; box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            z-index: 9999; cursor: pointer; transition: transform 0.2s;
            text-decoration: none; border: 2px solid rgba(255,255,255,0.2);
        }}
        .to-top-btn:hover {{ transform: scale(1.1); }}
    </style>
    <div id="top-anchor"></div>
    <a href="#top-anchor" class="to-top-btn" title="Go to Top">⬆️</a>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. リソース & 定数設定
# ---------------------------------------------------------
MODEL_PATH = 'models/lgbm_pace_tuned.pkl'
ENCODER_PATH = 'models/pace_encoders.pkl'

# クラウドDB接続先 (ローカル実行時のエラー回避対応)
try:
    if "DATABASE_URL" in st.secrets:
        DATABASE_URL = st.secrets["DATABASE_URL"]
    else:
        # secretsファイルはあるがキーがない場合
        DATABASE_URL = 'postgresql://neondb_owner:npg_4HTcfQoa0Suq@ep-empty-fog-a1m9gve8-pooler.ap-southeast-1.aws.neon.tech/keiba_db?sslmode=require&channel_binding=require'
except:
    # ローカルで secrets.toml 自体がない場合（今回のエラーはここで吸収）
    DATABASE_URL = 'postgresql://neondb_owner:npg_4HTcfQoa0Suq@ep-empty-fog-a1m9gve8-pooler.ap-southeast-1.aws.neon.tech/keiba_db?sslmode=require&channel_binding=require'

COURSE_START_TO_CORNER = {
    ('東京', '芝', 1400): 350, ('東京', '芝', 1600): 550, ('東京', '芝', 1800): 150, 
    ('東京', '芝', 2000): 130, ('東京', '芝', 2400): 350, ('東京', '芝', 2500): 450,
    ('東京', 'ダ', 1300): 340, ('東京', 'ダ', 1400): 440, ('東京', 'ダ', 1600): 150, ('東京', 'ダ', 2100): 240,
    ('中山', '芝', 1200): 275, ('中山', '芝', 1600): 240, ('中山', '芝', 1800): 205, 
    ('中山', '芝', 2000): 405, ('中山', '芝', 2200): 432, ('中山', '芝', 2500): 192,
    ('中山', 'ダ', 1200): 502, ('中山', 'ダ', 1800): 375,
    ('京都', '芝', 1200): 300, ('京都', '芝', 1400): 500, ('京都', '芝', 1600): 700, 
    ('京都', '芝', 1800): 900, ('京都', '芝', 2000): 300, ('京都', '芝', 2200): 400, ('京都', '芝', 3000): 200,
    ('京都', 'ダ', 1200): 400, ('京都', 'ダ', 1400): 600, ('京都', 'ダ', 1800): 280, ('京都', 'ダ', 1900): 380,
    ('阪神', '芝', 1200): 250, ('阪神', '芝', 1400): 450, ('阪神', '芝', 1600): 444, 
    ('阪神', '芝', 1800): 644, ('阪神', '芝', 2000): 325, ('阪神', '芝', 2200): 525, ('阪神', '芝', 2400): 300,
    ('阪神', 'ダ', 1200): 350, ('阪神', 'ダ', 1400): 550, ('阪神', 'ダ', 1800): 300, ('阪神', 'ダ', 2000): 500,
    ('中京', '芝', 1200): 312, ('中京', '芝', 1400): 512, ('中京', '芝', 1600): 590, 
    ('中京', '芝', 2000): 314, ('中京', '芝', 2200): 514,
    ('中京', 'ダ', 1200): 400, ('中京', 'ダ', 1400): 600, ('中京', 'ダ', 1800): 290, ('中京', 'ダ', 1900): 390,
    ('札幌', '芝', 1200): 412, ('札幌', '芝', 1500): 180, ('札幌', '芝', 1800): 185, ('札幌', '芝', 2000): 380,
    ('札幌', 'ダ', 1000): 130, ('札幌', 'ダ', 1700): 240,
    ('函館', '芝', 1200): 480, ('函館', '芝', 1800): 275, ('函館', '芝', 2000): 476,
    ('函館', 'ダ', 1000): 150, ('函館', 'ダ', 1700): 330,
    ('福島', '芝', 1200): 412, ('福島', '芝', 1800): 305, ('福島', '芝', 2000): 505,
    ('福島', 'ダ', 1150): 150, ('福島', 'ダ', 1700): 338,
    ('新潟', '芝', 1000): 1000, ('新潟', '芝', 1200): 550, ('新潟', '芝', 1400): 650, ('新潟', '芝', 1600): 550, 
    ('新潟', '芝', 1800): 750, ('新潟', '芝', 2000): 950,
    ('新潟', 'ダ', 1200): 400, ('新潟', 'ダ', 1800): 390,
    ('小倉', '芝', 1200): 480, ('小倉', '芝', 1800): 272, ('小倉', '芝', 2000): 472,
    ('小倉', 'ダ', 1000): 150, ('小倉', 'ダ', 1700): 340,
}

PLACE_MAP = {'01':'札幌', '02':'函館', '03':'福島', '04':'新潟', '05':'東京', '06':'中山', '07':'中京', '08':'京都', '09':'阪神', '10':'小倉'}
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36', 'Referer': 'https://race.netkeiba.com/'}

MANUAL_TRAINER_MAP = {
    "高橋文": "[東] 高橋文雅", "和田郎": "[東] 和田正一", "武井": "[東] 武井亮",
    "武英": "[西] 武英智", "堀": "[東] 堀宣行", "手塚久": "[東] 手塚貴久",
    "高橋亮": "[西] 高橋亮", "清水久": "[西] 清水久詞", "高橋忠": "[西] 高橋義忠",
    "飯田": "[西] 飯田祐史", # 候補: ['[地] 飯田弘道', '[西] 飯田祐史', '[西] 飯田雄三']
    "林": "[東] 林徹", # 候補: []
    "千葉": "[東] 千葉直人", # 候補: ['[地] 千葉幸喜', '[東] 千葉直人']
    "柄崎": "[東] 柄崎将寿", # 候補: ['[東] 柄崎孝', '[東] 柄崎将寿']
    "矢野": "[東] 矢野英一", # 候補: ['[地] 矢野久美', '[地] 矢野義幸', '[東] 矢野英一']
    "石坂": "[西] 石坂公一", # 候補: ['[西] 石坂公一', '[西] 石坂正']
    "福永": "[西] 福永祐一", # 候補: ['[地] 福永敏', '[西] 福永祐一']
    "古賀": "[東] 古賀慎明", # 候補: ['[地] 古賀光範', '[東] 古賀史生', '[東] 古賀慎明']
    "昆": "[西] 昆貢", # 候補: []
    "池添": "[西] 池添学", # 候補: ['[西] 池添兼雄', '[西] 池添学']
    "辻": "[東] 辻哲英", # 候補: []
    "藤原": "[西] 藤原英昭", # 候補: ['[地] 藤原智行', '[東] 藤原辰雄', '[西] 藤原英昭']
    "宮": "[西] 宮徹", # 候補: []
    "谷": "[西] 谷潔", # 候補: []
    "高野": "[西] 高野友和", # 候補: ['[地] 高野毅', '[西] 高野友和']
    "ユースタス": "[外] ユースタ", # 候補: []
    "牧": "[東] 牧光二", # 候補: []
    "小野": "[東] 小野次郎", # 候補: ['[地] 小野望', '[東] 小野次郎']
    "石橋": "[西] 石橋守", # 候補: ['[地] 石橋満', '[西]  石橋守', '[西] 石橋守']
    "長谷川": "[西] 長谷川浩", # 候補: ['[地] 長谷川忍', '[西] 長谷川浩']
    "田中勝": "[東] 田中勝春", # 候補: []
    "角田": "[西] 角田晃一", # 候補: ['[地] 角田輝也', '[西] 角田晃一']
    "安田": "[西] 安田翔伍", # 候補: ['[地] 安田武広', '[西]  安田翔伍', '[西] 安田翔伍', '[西] 安田隆行']
    "グラファー": "[外] グラファ"
}
MANUAL_JOCKEY_MAP = {
    "鮫島駿": "鮫島克駿", "秋山稔": "秋山稔樹", "プーシャン": "プーシャ",
    "マーカンド": "マーカン", "Ｃデムーロ": "Ｃ．デム", "Ｍデムーロ": "Ｍ．デム",
    "石神道": "石神深道", "角田和": "角田大和", "シュタルケ": "シュタル", "吉村": "吉村誠之",
    "団野": "団野大成", "丹内": "丹内祐次", "西村淳": "西村淳也", "佐々木": "佐々木大輔",
    "岩田望": "岩田望来", "菅原明": "菅原明良", "横山典": "横山典弘", "横山武": "横山武史", 
    "横山和": "横山和生", "吉田豊": "吉田豊", "吉田隼": "吉田隼人", "木幡初": "木幡初也",
    "木幡巧": "木幡巧也", "大野": "大野拓弥", "北村宏": "北村宏司", "北村友": "北村友一",
    "津村": "津村明秀", "三浦": "三浦皇成", "田辺": "田辺裕信", "戸崎圭": "戸崎圭太",
    "川田": "川田将雅", "ルメール": "ルメール", "松山": "松山弘平", "坂井": "坂井瑠星",
    "ザーラ": "ザーラ", "ジェルー": "ジェルー","バルザロー": "バルザロ",
    "川原正": "川原正一",
    "松戸政": "松戸政也",
    "石川裕紀人": "石川裕紀",
    "吉村誠之助": "吉村誠之"
}

# ---------------------------------------------------------
# 4. ヘルパー関数
# ---------------------------------------------------------
def js_scroll_to(element_id):
    components.html(f"<script>var element = window.parent.document.getElementById('{element_id}'); if (element) {{ element.scrollIntoView({{behavior: 'smooth', block: 'start'}}); }}</script>", height=0)

def convert_raw_margin(x):
    if pd.isna(x) or x == '': return 0.0
    s = str(x).strip()
    if s in ['ハナ', 'アタマ']: return 0.05
    if s == 'クビ': return 0.1
    if s == '大差': return 2.5
    if '/' in s: return 0.2
    try: return float(s)
    except: return 0.0

def get_grade(title):
    if 'G1' in title or 'JpnI' in title: return 'G1'
    if 'GII' in title or 'G2' in title or 'JpnII' in title: return 'G2'
    if 'GIII' in title or 'G3' in title or 'JpnIII' in title: return 'G3'
    if '(L)' in title or '[L]' in title: return 'L' 
    if '(OP)' in title: return 'OP'
    return '一般'

def get_grade_class_name(grade):
    if 'G1' in grade: return 'bg-G1'
    if 'G2' in grade: return 'bg-G2'
    if 'G3' in grade: return 'bg-G3'
    if grade in ['L', 'OP', '特別']: return 'bg-LOP'
    return 'bg-GEN'

def detect_grade_from_icon(element):
    if not element: return None
    icon = element.find('span', class_=re.compile(r'Icon_GradeType\d+'))
    if not icon: return None
    c = icon.get('class')
    if 'Icon_GradeType1' in c: return 'G1'
    if 'Icon_GradeType2' in c: return 'G2'
    if 'Icon_GradeType3' in c: return 'G3'
    if 'Icon_GradeType15' in c: return 'L'
    if 'Icon_GradeType5' in c: return 'OP'
    return None

# ---------------------------------------------------------
# 【完成版】ハイブリッド取得関数
# requestsで高速取得し、ダメなら自動でSelenium(ブラウザ)に切り替える
# ---------------------------------------------------------
def get_html_content(url, driver=None):
    # 中身がちゃんとあるか判定する関数
    def is_valid_html(html):
        if not html: return False
        # レース一覧系 or 出馬表系のキーワードが含まれているか
        keywords = ["RaceList", "RaceTop", "HorseList", "Umaban", "Kaisai", "RaceTable"]
        return any(k in html for k in keywords)

    # 1. まずは高速な requests でトライ
    try:
        res = requests.get(url, headers=HEADERS, timeout=5)
        if res.status_code == 200:
            for enc in ['euc-jp', 'utf-8', 'shift_jis', 'cp932']:
                try: 
                    decoded = res.content.decode(enc)
                    # 中身が空っぽ(ダミー)じゃないか確認
                    if is_valid_html(decoded):
                        return decoded
                except: continue
    except: pass # requests失敗時は何もしないで次へ

    # 2. ダメなら Selenium (Chrome) を起動して確実に取る
    try:
        # ドライバが渡されていない場合のみ、ここで新規作成・破棄を行う（単発利用）
        local_driver = False
        if driver is None:
            local_driver = True
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            from selenium.webdriver.chrome.service import Service
            service = Service()
            driver = webdriver.Chrome(options=options, service=service)
        
        try:
            driver.get(url)
            time.sleep(2) # 読み込み待ち
            html = driver.page_source
            
            # Seleniumでも一応中身チェック
            if is_valid_html(html):
                return html
            else:
                return None
        finally:
            # 自分で作ったドライバなら閉じる。渡されたものなら閉じない。
            if local_driver and driver:
                driver.quit()
    except Exception:
        return None

def render_grade_badge_html(grade):
    cls = get_grade_class_name(grade)
    return f'<span class="grade-badge {cls}">{grade}</span>'

def render_hero_card(row):
    rating = min(99, int(row.get('AI Rating', 50)))
    sire = row.get('sire_name', '-')
    sire_win = row.get('sire_win_rate', 0)
    bms = row.get('bms_name', '-')
    bms_win = row.get('bms_win_rate', 0)
    jockey_win = row.get('jockey_win_rate', 0)
    trainer_name = row.get('調教師', '-')
    trainer_win = row.get('trainer_win_rate', 0)
    
    dist_change_val = row.get('dist_change', 0)
    if pd.isna(dist_change_val): dist_change_val = 0
    dist_txt = "同距離"
    if dist_change_val < 0: dist_txt = f"{int(dist_change_val)}m (短縮)"
    elif dist_change_val > 0: dist_txt = f"+{int(dist_change_val)}m (延長)"
    
    fire_class = "fire" if rating >= 80 else ""
    def fmt_stat(val):
        return f"<span style='font-weight:bold;'>{val*100:.1f}%</span>"

    html = f"""
    <div class="hero-card stCard {fire_class}">
        <div class="hero-rating-box">
            <div class="hero-rating-label">AI RATING</div>
            <div class="hero-rating-val">{rating}</div>
        </div>
        <div class="hero-label">🏆 AI推奨 No.1</div>
        <div class="hero-horse"><span style="font-size:0.6em; opacity:0.6;">#{int(row['馬番'])}</span> {row['馬名']}</div>
        <div class="hero-stats-grid">
            <div class="hero-stat-item">
                <div class="hero-stat-label">血統 (父 / 母父)</div>
                <div class="hero-stat-val pedigree-box">
                    <div class="pedigree-row">
                        <span class="pedigree-name">父: {sire}</span>
                        <span class="pedigree-rate { 'rate-high' if sire_win >= 0.1 else '' }">勝 {fmt_stat(sire_win)}</span>
                    </div>
                    <div class="pedigree-row">
                        <span class="pedigree-name">母父: {bms}</span>
                        <span class="pedigree-rate { 'rate-high' if bms_win >= 0.1 else '' }">勝 {fmt_stat(bms_win)}</span>
                    </div>
                </div>
            </div>
            <div class="hero-stat-item">
                <div class="hero-stat-label">ローテーション</div>
                <div class="hero-stat-val">{dist_txt}</div>
            </div>
            <div class="hero-stat-item">
                <div class="hero-stat-label">人間力 (勝率)</div>
                <div class="hero-stat-val pedigree-box">
                    <div class="pedigree-row">
                        <span class="pedigree-name">{row['騎手']}</span>
                        <span class="pedigree-rate { 'rate-high' if jockey_win >= 0.15 else '' }">{fmt_stat(jockey_win)}</span>
                    </div>
                    <div class="pedigree-row">
                        <span class="pedigree-name">{trainer_name}</span>
                        <span class="pedigree-rate { 'rate-high' if trainer_win >= 0.15 else '' }">{fmt_stat(trainer_win)}</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    return html.replace('\n', '')

def render_ai_list_item(row, overlap_badges):
    rating = min(99, int(row.get('AI Rating', 50)))
    
    # ★追加: 購入対象(Bet Target)なら強調表示するためのフラグ確認
    is_bet = row.get('is_bet_target', False)
    
    badges_html = ""
    # ★追加: 購入対象なら「BUY」バッジを先頭に追加
    if is_bet:
        badges_html += '<span class="strategy-badge" style="background:#ffd700; color:#000; border:2px solid #000; font-weight:900; font-size:0.85rem;">🎯 BUY</span> '

    if "pace" in overlap_badges: badges_html += '<span class="strategy-badge" style="background:#fce7f3; color:#be185d;">🚀 展開神</span>'
    if "ai" in overlap_badges: badges_html += '<span class="strategy-badge" style="background:#fffbeb; color:#d97706;">🦄 鉄板</span>'
    if "hole" in overlap_badges: badges_html += '<span class="strategy-badge" style="background:#fee2e2; color:#b91c1c;">💣 穴馬</span>'
    
    # ★変更: 購入対象なら枠線を赤く太くし、背景色を微調整
    if is_bet:
        fire_class = "fire" # 強制的に炎エフェクト有効
        card_style = "border: 3px solid #ef4444 !important; background-color: #fffaf0 !important; transform: scale(1.01); box-shadow: 0 8px 16px rgba(239, 68, 68, 0.15) !important;"
    else:
        fire_class = "fire" if rating >= 80 else ""
        card_style = ""

    bar_color = "linear-gradient(135deg, #ef4444 0%, #f59e0b 100%)" if rating >= 80 else "var(--primary-gradient)"
    boost_html = ""
    reason_str = row.get('BoostReason', '')
    if reason_str:
        reasons = reason_str.split(' ')
        for r in reasons:
            if not r: continue
            cls = "boost-speed"
            if "騎手" in r: cls = "boost-jockey"
            elif "良血" in r: cls = "boost-blood" 
            elif "先行" in r: cls = "boost-leader"
            elif "コース" in r: cls = "boost-course"
            elif "巧者" in r: cls = "boost-horse-course"
            elif "短縮" in r: cls = "boost-dist"
            elif "展開" in r: cls = "boost-pace"
            elif "豪脚" in r: cls = "boost-gokyak"
            
            boost_html += f'<span class="boost-badge {cls}">{r}</span>'

    # ★変更: style属性を追加して強調デザインを適用
    html = f"""
    <div class="ai-list-card {fire_class}" style="{card_style}">
        <div class="ai-card-badges">{badges_html}</div>
        <div style="font-size:1.1rem; font-weight:bold; color:var(--text-main); margin-bottom:4px;">
            <span style="opacity:0.6; font-size:0.8em;">#{int(row['馬番'])}</span> {row['馬名']}
        </div>
        <div style="display:flex; align-items:center; gap:10px; font-size:0.85rem; color:var(--sub-text); margin-bottom:8px;">
            <span><i class="icon">🏇</i> {row['騎手']}</span>
            <span>単勝 {row['オッズ']}倍</span>
        </div>
        <div style="margin-bottom:8px;">{boost_html}</div>
        <div style="background:rgba(128,128,128,0.1); border-radius:8px; padding:8px; font-size:0.8rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                <span style="color:var(--sub-text);">AI Rating</span>
                <span style="font-weight:bold; color:var(--primary-color);">{rating}</span>
            </div>
            <div style="background:rgba(128,128,128,0.2); height:6px; border-radius:3px; overflow:hidden;">
                <div style="background:{bar_color}; width:{rating}%; height:100%;"></div>
            </div>
            <div style="margin-top:6px; font-size:0.75rem; color:var(--sub-text);">父: {row.get('sire_name','-')} / 母父: {row.get('bms_name','-')}</div>
        </div>
    </div>
    """
    return html.replace('\n', '')

def render_ev_legend():
    html = f"""
    <div class="ev-legend-box">
        <span style="font-weight:bold; font-size:0.85rem;">📊 勝利の方程式 (The Holy Grail)</span>
        <span style="font-size:0.85rem; margin-left:10px;"><span style="color:#be185d; font-weight:900;">🚀 展開の神</span> : 展開利あり & 勝率10%↑ & オッズ10~100倍 (ROI: 108%🏆)</span>
        <span style="font-size:0.85rem; margin-left:10px;"><span style="color:#d97706; font-weight:900;">🦄 鉄板の軸</span> : AI自信度No.1 (ROI: 80% / 的中重視)</span>
        <span style="font-size:0.85rem; margin-left:10px;"><span style="color:#b91c1c; font-weight:900;">💣 穴馬の極意</span> : 勝率8%↑ & オッズ50~100倍 (ROI: 87%)</span>
        <br>
        <span style="font-size:0.8rem; margin-left:120px;">
            ※ <strong>展開利</strong>: スローの先行、ハイペースの差しなど、展開が味方する馬。<br>
            ※ <strong>ROI集計期間</strong>: {BACKTEST_PERIOD}
        </span>
    </div>
    """
    return html.replace('\n', '')

def render_badge_legend():
    html = """
    <div class="badge-legend">
        <div style="font-weight:bold; margin-bottom:5px;">✨ 注目ポイント（バッジ）ガイド</div>
        <div class="badge-legend-item"><span class="boost-badge boost-pace">🌀展開利</span> : ペース予測と脚質が完全に噛み合う</div>
        <div class="badge-legend-item"><span class="boost-badge boost-horse-course">🐴コース巧者(馬)</span> : 馬がこのコースを得意としている</div>
        <div class="badge-legend-item"><span class="boost-badge boost-course">🏰コース巧者(人)</span> : 騎手がこのコースを得意としている</div>
        <div class="badge-legend-item"><span class="boost-badge boost-jockey">🔥高勝率騎手</span> : 騎手の通算勝率が15%以上</div>
        <div class="badge-legend-item"><span class="boost-badge boost-leader">🚀先行型</span> : 過去5走で「4角4番手以内」のレースが半数以上</div>
        <div class="badge-legend-item"><span class="boost-badge boost-gokyak">⚡豪脚</span> : 後方待機から鋭い末脚を使うタイプ</div>
    </div>
    """
    return html.replace('\n', '')

def render_report_card_dual(label, hit_count, bets, win_roi, place_roi):
    if bets > 0:
        hit_rate_str = f"{(hit_count / bets * 100):.1f}%"
        win_roi_str = f"{win_roi:.0f}%"
        place_roi_str = f"{place_roi:.0f}%"
        win_class = "val-win" if win_roi >= 100 else "val-neutral"
        place_class = "val-win" if place_roi >= 100 else "val-neutral"
        hit_sub = f"({hit_count}/{bets})"
    else:
        hit_rate_str = "-"
        win_roi_str = "-"
        place_roi_str = "-"
        win_class = "val-empty"
        place_class = "val-empty"
        hit_sub = "(対象なし)"
    
    html = f"""
    <div class="report-card-dual">
        <div class="report-header">{label}</div>
        <div class="report-grid">
            <div class="report-item">
                <div class="report-sublabel">的中率 (単/複)</div>
                <div class="report-val-sm">{hit_rate_str}</div>
                <div style="font-size:0.7em; color:var(--sub-text);">{hit_sub}</div>
            </div>
            <div class="report-item">
                <div class="report-sublabel">回収率 (単/複)</div>
                <div style="line-height:1.1;">
                    <span class="{win_class}" style="font-weight:900;">{win_roi_str}</span> / 
                    <span class="{place_class}" style="font-weight:900;">{place_roi_str}</span>
                </div>
            </div>
        </div>
    </div>
    """
    return html.replace('\n', '')

def get_base64_video(file_paths):
    valid_paths = [p for p in file_paths if os.path.exists(p)]
    if not valid_paths: return None
    selected = random.choice(valid_paths)
    try:
        with open(selected, "rb") as f: return base64.b64encode(f.read()).decode()
    except: return None

@st.cache_resource
def load_resources(mtime):
    logs = {}
    try:
        if os.path.exists(MODEL_PATH):
            pack = joblib.load(MODEL_PATH)
            model = pack['model']
            calibrator = pack['calibrator']
            feature_cols = pack['features']
            return {'model': model, 'calibrator': calibrator, 'features': feature_cols}, joblib.load(ENCODER_PATH), create_engine(DATABASE_URL), logs
        else: return None, None, None, {}
    except Exception as e: return None, None, None, {'error': str(e)}

@st.cache_data(ttl=600)
def get_race_list_by_date(target_date):
    date_str = target_date.strftime('%Y%m%d')
    race_list = []
    seen_ids = set()
    target_urls = [f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}", f"https://db.netkeiba.com/race/list/{date_str}/"]
    for url in target_urls:
        content = get_html_content(url)
        if not content: continue
        found_ids = re.findall(r'(20\d{10})', content)
        unique_ids = sorted(list(set(found_ids)))
        if unique_ids:
            try: soup = BeautifulSoup(content, 'lxml')
            except: soup = BeautifulSoup(content, 'html.parser')
            for race_id in unique_ids:
                if race_id in seen_ids: continue
                p_code = race_id[4:6]
                if p_code not in PLACE_MAP: continue
                place_name = PLACE_MAP.get(p_code, '開催')
                try: r_no_str = f"{int(race_id[10:12])}R"
                except: r_no_str = "R"
                title_str = "レース詳細"
                icon_grade = None
                link_tag = soup.find('a', href=re.compile(race_id))
                if link_tag:
                    item_title = link_tag.find(class_='ItemTitle')
                    if item_title: title_str = item_title.get_text(strip=True)
                    else: title_str = link_tag.get('title') or link_tag.get_text(strip=True)
                    title_str = re.sub(r'^\d+R|\d{2}:\d{2}|[\[\(].*?[\]\)]', '', title_str).strip()
                    icon_grade = detect_grade_from_icon(link_tag)
                    if not icon_grade:
                        parent = link_tag.find_parent('li')
                        if parent: icon_grade = detect_grade_from_icon(parent)
                grade = icon_grade if icon_grade else get_grade(title_str)
                race_list.append({'label': f"【{place_name} {r_no_str}】 {title_str}", 'id': race_id, 'url': f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}", 'grade': grade})
                seen_ids.add(race_id)
            if race_list: return race_list
    return race_list

def scrape_race_result(race_id):
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        content = get_html_content(url)
        if not content: return None, None, None, []
        soup = BeautifulSoup(content, 'lxml')
        table = soup.find('table', class_='RaceTable01')
        rank_map = {}
        if table:
            for row in table.find_all('tr'):
                tds = row.find_all('td')
                if len(tds) > 3:
                    try:
                        r = tds[0].get_text(strip=True)
                        u = tds[2].get_text(strip=True)
                        if r.isdigit() and u.isdigit(): rank_map[int(u)] = int(r)
                    except: continue
        win_map = {}
        fukusho_map = {}
        all_tables = soup.find_all('table')
        for t in all_tables:
            if "単勝" in t.get_text():
                for row in t.find_all('tr'):
                    if not row.find('th') or "単勝" not in row.find('th').get_text(): continue
                    tds = row.find_all('td')
                    if len(tds) >= 2:
                        us = tds[0].get_text(strip=True, separator='|').split('|')
                        ps = tds[1].get_text(strip=True, separator='|').split('|')
                        for u, p in zip(us, ps):
                            try: win_map[int(u.strip())] = int(p.strip().replace(',', '').replace('円', ''))
                            except: continue
            if "複勝" in t.get_text():
                for row in t.find_all('tr'):
                    if not row.find('th') or "複勝" not in row.find('th').get_text(): continue
                    tds = row.find_all('td')
                    if len(tds) >= 2:
                        us = tds[0].get_text(strip=True, separator='|').split('|')
                        ps = tds[1].get_text(strip=True, separator='|').split('|')
                        for u, p in zip(us, ps):
                            try: fukusho_map[int(u.strip())] = int(p.strip().replace(',', '').replace('円', ''))
                            except: continue
        return (rank_map if rank_map else None), (win_map if win_map else None), (fukusho_map if fukusho_map else None), []
    except: return None, None, None, []

def scrape_race_data(url, driver=None):
    try:
        content = get_html_content(url, driver=driver)
        if not content: return None
        soup = BeautifulSoup(content, 'lxml')
        api_odds_map = {}
        race_id_match = re.search(r'race_id=(\d+)', url)
        rid = race_id_match.group(1) if race_id_match else None
        if rid:
            try:
                r_api = requests.get(f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={rid}&type=1&action=init", headers=HEADERS, timeout=5)
                if r_api.status_code == 200:
                    raw_odds = r_api.json().get('data', {}).get('odds', {}).get('1', {})
                    for h, i in raw_odds.items(): api_odds_map[int(h)] = i[0]
            except: pass
        
        intro = soup.find('div', class_='RaceData01')
        intro_text = intro.get_text().replace('\n', '').strip() if intro else ""
        dist_match = re.search(r'(芝|ダ|障)(\d+)m', intro_text)
        course_type = dist_match.group(1) if dist_match else 'Unknown'
        distance = int(dist_match.group(2)) if dist_match else 0
        if course_type == '障': course_type = '障害'
        elif course_type == 'ダ': course_type = 'ダート'
        direction = '左' if '左' in intro_text else ('直線' if '直線' in intro_text else '右')
        
        race_date = None
        title = soup.find('title')
        if title:
            m = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', title.get_text())
            if m: race_date = f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
        if not race_date and intro_text:
             m = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', intro_text)
             if m: race_date = f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
        
        div_race_name = soup.find('div', class_='RaceName')
        if div_race_name:
             race_title = div_race_name.get_text(strip=True)
        else:
             page_title = soup.title.get_text() if soup.title else ""
             race_title = page_title.split('|')[0].strip() if '|' in page_title else page_title

        title_norm = race_title.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
        title_norm = re.sub(r'(２歳|３歳|４歳|以上|牝|混合|限定)', '', title_norm)
        
        race_class = "OP"
        if "新馬" in title_norm: race_class = "新馬"
        elif "未勝利" in title_norm: race_class = "未勝利"
        elif "G1" in title_norm or "Jpn1" in title_norm: race_class = "G1"
        elif "G2" in title_norm or "Jpn2" in title_norm: race_class = "G2"
        elif "G3" in title_norm or "Jpn3" in title_norm: race_class = "G3"
        elif "L" in title_norm: race_class = "L"
        elif "3勝" in title_norm or "1600万" in title_norm: race_class = "3勝クラス"
        elif "2勝" in title_norm or "1000万" in title_norm: race_class = "2勝クラス"
        elif "1勝" in title_norm or "500万" in title_norm: race_class = "1勝クラス"
        
        rows = soup.find_all('tr', class_='HorseList')
        if not rows:
            t = soup.find('table', class_='Shutuba_Table') or soup.find('table', class_='race_table_01')
            if t: rows = [r for r in t.find_all('tr') if r.find('td', class_=re.compile(r'Umaban|Waku'))]
            
        data_list = []
        for idx, row in enumerate(rows):
            try:
                tds = row.find_all('td')
                if len(tds) < 7: continue
                w = tds[0].get_text(strip=True)
                waku = int(w) if w.isdigit() else 0
                u = tds[1].get_text(strip=True)
                umaban = int(u) if u.isdigit() else (idx+1)
                h_tag = tds[3].find('span', class_='HorseName')
                horse = h_tag.get_text(strip=True) if h_tag else ""
                if not horse: continue
                
                sex_age = tds[4].get_text(strip=True)
                sex = sex_age[0] if sex_age else "Unknown"
                age = 3 
                age_match = re.search(r'\d+', sex_age)
                if age_match: age = int(age_match.group())
                
                kin = tds[5].get_text(strip=True)
                joc = re.sub(r'[▲△☆★◇\s]+', '', tds[6].get_text(strip=True))
                tra = ""
                try:
                    t_td = row.find('td', class_='Trainer')
                    if t_td: tra = t_td.find('a').get('title') if t_td.find('a') else t_td.get_text(strip=True)
                    elif len(tds)>7: tra = tds[7].get_text(strip=True)
                    tra = re.sub(r'\(.*?\)|\[.*?\]|^(栗東|美浦|外)', '', tra).strip()
                except: pass
                
                odd = "-"
                if umaban in api_odds_map: odd = api_odds_map[umaban]
                else:
                    o_span = row.find('span', id=re.compile(r'^odds-\d+_\d+'))
                    if o_span: odd = o_span.get_text(strip=True)
                    elif len(tds) > 9: odd = tds[9].get_text(strip=True)
                
                place_code = re.search(r'race_id=\d{4}(\d{2})', url).group(1)
                data_list.append({
                    '枠番': waku, '馬番': umaban, '馬名': horse, '性別': sex, '年齢': age, '斤量': kin, 
                    '騎手': joc, '調教師': tra, 'オッズ': odd, '開催場所': PLACE_MAP.get(place_code, '中山'), 
                    'コース区分': course_type, '距離': distance, '回り': direction, 'クラス': race_class, 
                    'race_id': rid, 'date': race_date, 'race_title_raw': race_title 
                })
            except: continue
        return pd.DataFrame(data_list) if data_list else None
    except: return None

@st.cache_data(ttl=3600)
def resolve_jockey_names(_engine, target_jockeys):
    try: 
        if 'db_jockeys' not in st.session_state:
            st.session_state.db_jockeys = pd.read_sql('SELECT DISTINCT "騎手" FROM raw_race_results', _engine)['騎手'].dropna().unique().tolist()
        db_jockeys = st.session_state.db_jockeys
    except: return {}, []
    mapping = {}
    missing = []
    for target in target_jockeys:
        if not target: continue
        if target in MANUAL_JOCKEY_MAP: mapping[target] = MANUAL_JOCKEY_MAP[target]; continue
        if target in db_jockeys: mapping[target] = target; continue
        found_prefix = next((d for d in db_jockeys if d.startswith(target)), None)
        if found_prefix: mapping[target] = found_prefix; continue
        missing.append(target); mapping[target] = target
    return mapping, missing

@st.cache_data(ttl=3600)
def resolve_trainer_names(_engine, target_trainers):
    try: 
        if 'db_trainers' not in st.session_state:
            st.session_state.db_trainers = pd.read_sql('SELECT DISTINCT "調教師" FROM raw_race_results', _engine)['調教師'].dropna().unique().tolist()
        db_trainers = st.session_state.db_trainers
        
        db_map_clean = {}
        for db_name in db_trainers:
            clean_name = re.sub(r'\[.*?\]|（.*?）|\(.*?\)|[ \u3000]+', '', db_name)
            if clean_name not in db_map_clean:
                db_map_clean[clean_name] = []
            db_map_clean[clean_name].append(db_name)
    except: return {}, [], pd.DataFrame()
    
    mapping = {}
    missing = []
    debug_rows = []

    for target in target_trainers:
        if not target: continue
        
        if target in MANUAL_TRAINER_MAP: 
            mapping[target] = MANUAL_TRAINER_MAP[target]
            continue
            
        clean_target = re.sub(r'[ \u3000]+', '', target) 
        
        found = None
        candidates = []
        
        # A. 完全一致
        if clean_target in db_map_clean:
            candidates = db_map_clean[clean_target]
        else:
            # B. 部分一致
            if len(clean_target) >= 2:
                for db_clean_key, original_names in db_map_clean.items():
                    if clean_target in db_clean_key:
                         candidates.extend(original_names)
        
        # 候補の正規化 (スペース違い等の実質重複を排除)
        core_candidates = set()
        for cand in candidates:
            core = re.sub(r'\[.*?\]|[ \u3000]+', '', cand)
            core_candidates.add(core)
        
        unique_core_count = len(core_candidates)
        
        # 候補が実質1人なら自動採用
        if unique_core_count == 1:
            found = candidates[0]
        
        debug_rows.append({
            "Web取得名": target,
            "正規化名": clean_target,
            "マッチ結果": found if found else "❌",
            "候補数": unique_core_count,
            "候補リスト": str(candidates)
        })

        if found:
            mapping[target] = found
        else:
            missing.append(target)
            mapping[target] = target 

    return mapping, missing, pd.DataFrame(debug_rows)

def process_passage_rank(passage):
    if not isinstance(passage, str): return np.nan
    try: return int(passage.split('-')[0])
    except: return np.nan

@st.cache_data(ttl=600)
def calc_horse_history(_engine, horse_names, target_date):
    clean_names = [n.replace(" ", "").replace("　", "") for n in horse_names]
    names_str = "', '".join([n.replace("'", "''") for n in clean_names])
    
    query = f"""
    SELECT "date", "馬名", "着順", "上り", "着差", "通過", "賞金(万円)", "距離", "コース区分", "騎手"
    FROM raw_race_results
    WHERE REPLACE(REPLACE("馬名", ' ', ''), '　', '') IN ('{names_str}') 
      AND "date" < '{target_date}'
    ORDER BY "date" ASC
    """
    
    debug_info = {"sql": query}
    
    try:
        hist_df = pd.read_sql(query, _engine)
        hist_df['date'] = pd.to_datetime(hist_df['date'])
        hist_df['着順'] = pd.to_numeric(hist_df['着順'], errors='coerce')
        hist_df['上り'] = pd.to_numeric(hist_df['上り'], errors='coerce')
        hist_df['着差'] = hist_df['着差'].apply(convert_raw_margin)
        hist_df['money'] = pd.to_numeric(hist_df['賞金(万円)'], errors='coerce').fillna(0)
        hist_df['is_win'] = (hist_df['着順'] == 1).astype(int)
        hist_df['馬名_clean'] = hist_df['馬名'].astype(str).str.replace(" ", "").str.replace("　", "")
        hist_df['騎手_clean'] = hist_df['騎手'].astype(str).str.replace(" ", "").str.replace("　", "")
        
        hist_df['first_pos'] = hist_df['通過'].apply(process_passage_rank)
        hist_df['pos_rate'] = hist_df['first_pos'] / 14.0 
        hist_df['is_nige'] = (hist_df['first_pos'] == 1).astype(int)
        hist_df['is_senko'] = (hist_df['pos_rate'] <= 0.3).astype(int)

        stats = []
        for horse in horse_names:
            h_clean = horse.replace(" ", "").replace("　", "")
            h_data = hist_df[hist_df['馬名_clean'] == h_clean]
            
            if h_data.empty:
                stats.append({
                    '馬名': horse, 'interval_weeks': 0, 'prev_rank': 0, 'prev_3f': 36.0, 'prev_margin': 0.5, 
                    'recent_3f_avg': 36.0, 'recent_rank_avg': 8.0, 'run_style_ratio': 0, 
                    'total_wins': 0, 'total_money': 0, 'win_ratio': 0,
                    'prev_distance': 0, 'prev_course_type': 'Unknown', 'prev_jockey': 'Unknown',
                    'nige_rate': 0, 'senko_rate': 0, 'avg_pos_rate': 0.5
                })
                continue
            
            last = h_data.iloc[-1]
            interval = (pd.to_datetime(target_date) - last['date']).days / 7
            recent = h_data.tail(3)
            rec_3f = recent['上り'].mean()
            rec_rank = recent['着順'].mean()
            recent5 = h_data.tail(5)
            
            nige_rate = recent5['is_nige'].mean()
            senko_rate = recent5['is_senko'].mean()
            avg_pos_rate = recent5['pos_rate'].mean()
            
            run_style = senko_rate
            wins = h_data['is_win'].sum()
            total_money = h_data['money'].sum()
            cnt = len(h_data)
            
            stats.append({
                '馬名': horse,
                'interval_weeks': interval,
                'prev_rank': last['着順'],
                'prev_3f': last['上り'],
                'prev_margin': last['着差'],
                'prev_distance': last['距離'],        
                'prev_course_type': last['コース区分'], 
                'prev_jockey': last['騎手_clean'],    
                'recent_3f_avg': rec_3f,
                'recent_rank_avg': rec_rank,
                'run_style_ratio': run_style,
                'total_wins': wins,
                'total_money': total_money,
                'win_ratio': wins/cnt if cnt>0 else 0,
                'nige_rate': nige_rate,
                'senko_rate': senko_rate,
                'avg_pos_rate': avg_pos_rate
            })
        return pd.DataFrame(stats), debug_info
        
    except Exception as e:
        return pd.DataFrame(), {'error': str(e)}

def predict_race(df, model_pack, encoders, _engine):
    model = model_pack['model']
    calibrator = model_pack['calibrator']
    feature_cols = model_pack['features']

    j_map, missing_j = resolve_jockey_names(_engine, df['騎手'].unique().tolist())
    df['騎手_db'] = df['騎手'].map(j_map)
    # 修正: 戻り値変更に対応
    t_map, missing_t, t_debug = resolve_trainer_names(_engine, df['調教師'].unique().tolist())
    df['調教師_db'] = df['調教師'].map(t_map)
    
    missing_info = {'jockey': missing_j, 'trainer': missing_t, 'trainer_debug': t_debug}
    
    diag_data = {
        'pedigree': pd.DataFrame(), 
        'jockey': pd.DataFrame(), 
        'trainer': pd.DataFrame(), 
        'prev_trace': pd.DataFrame(), 
        'sql_debug': {}, 
        'jockey_check_df': pd.DataFrame(),
        'missing_advanced': [],
        'class_debug': {} 
    }
    
    try:
        j_stats = pd.read_sql('SELECT "騎手", AVG(CASE WHEN "着順"=\'1\' THEN 1.0 ELSE 0.0 END) as jockey_win_rate, AVG(CASE WHEN "着順"<=\'2\' THEN 1.0 ELSE 0.0 END) as jockey_rentai_rate, AVG(CASE WHEN "着順"<=\'2\' THEN 1.0 ELSE 0.0 END) as jockey_course_rentai_rate FROM raw_race_results GROUP BY "騎手"', _engine)
        t_stats = pd.read_sql('SELECT "調教師", AVG(CASE WHEN "着順"=\'1\' THEN 1.0 ELSE 0.0 END) as trainer_win_rate FROM raw_race_results GROUP BY "調教師"', _engine)
        
        df = df.merge(j_stats, left_on='騎手_db', right_on='騎手', how='left', suffixes=('', '_j'))
        df = df.merge(t_stats, left_on='調教師_db', right_on='調教師', how='left', suffixes=('', '_t'))
        
        diag_data['jockey'] = j_stats[j_stats['騎手'].isin(df['騎手_db'])].copy() if not j_stats.empty else pd.DataFrame()
        diag_data['trainer'] = t_stats[t_stats['調教師'].isin(df['調教師_db'])].copy() if not t_stats.empty else pd.DataFrame()
    except: pass

    target_date = df['date'].iloc[0]
    horse_stats, hist_debug = calc_horse_history(_engine, df['馬名'].tolist(), target_date)
    diag_data['sql_debug'] = hist_debug
    
    if not horse_stats.empty:
        df = df.merge(horse_stats, on='馬名', how='left')
    
    # Advanced Features Implementation (SQL)
    place_name = df['開催場所'].iloc[0]
    c_type = df['コース区分'].iloc[0]
    names_tuple = tuple(df['馬名'].unique())
    names_str_sql = str(names_tuple) if len(names_tuple) > 1 else f"('{names_tuple[0]}')"
    
    # 1. crs_rate
    try:
        crs_query = f"""
        SELECT "馬名", count(*) as runs, SUM(CASE WHEN "着順" IN ('1','2','3') THEN 1 ELSE 0 END) as top3 
        FROM raw_race_results 
        WHERE "馬名" IN {names_str_sql}
          AND "開催場所" LIKE '%%{place_name}%%'
          AND "コース区分" = '{c_type}'
        GROUP BY "馬名"
        """
        crs_df = pd.read_sql(crs_query, _engine)
        if not crs_df.empty:
            crs_df['crs_rate'] = crs_df['top3'] / crs_df['runs']
            df = df.merge(crs_df[['馬名', 'crs_rate']], on='馬名', how='left')
        else:
            df['crs_rate'] = 0.0
    except Exception as e:
        df['crs_rate'] = 0.0
        diag_data['crs_rate_error'] = str(e)

    # 2. jockey_course_win_rate
    try:
        jockeys = tuple(df['騎手_db'].dropna().unique())
        if jockeys:
            j_str = str(jockeys) if len(jockeys) > 1 else f"('{jockeys[0]}')"
            jc_query = f"""
            SELECT "騎手", AVG(CASE WHEN "着順"='1' THEN 1.0 ELSE 0.0 END) as jockey_course_win_rate
            FROM raw_race_results
            WHERE "騎手" IN {j_str}
              AND "開催場所" LIKE '%%{place_name}%%'
              AND "コース区分" = '{c_type}'
            GROUP BY "騎手"
            """
            jc_df = pd.read_sql(jc_query, _engine)
            if not jc_df.empty:
                df = df.merge(jc_df, left_on='騎手_db', right_on='騎手', how='left', suffixes=('', '_jc'))
    except: pass

    # 3. tag_win_rate
    try:
        if jockeys:
            tag_query = f"""
            SELECT "騎手", "調教師", AVG(CASE WHEN "着順"='1' THEN 1.0 ELSE 0.0 END) as tag_win_rate
            FROM raw_race_results
            WHERE "騎手" IN {j_str}
            GROUP BY "騎手", "調教師"
            """
            tag_df = pd.read_sql(tag_query, _engine)
            if not tag_df.empty:
                df = df.merge(tag_df, left_on=['騎手_db', '調教師_db'], right_on=['騎手', '調教師'], how='left', suffixes=('', '_tag'))
    except: pass

    # 4. course_waku_win_rate
    try:
        cw_query = f"""
        SELECT "枠番", AVG(CASE WHEN "着順"='1' THEN 1.0 ELSE 0.0 END) as course_waku_win_rate
        FROM raw_race_results
        WHERE "開催場所" LIKE '%%{place_name}%%'
          AND "コース区分" = '{c_type}'
          AND "距離" = '{int(df['距離'].iloc[0])}'
        GROUP BY "枠番"
        """
        cw_df = pd.read_sql(cw_query, _engine)
        if not cw_df.empty:
            cw_df['枠番'] = pd.to_numeric(cw_df['枠番'], errors='coerce')
            df['枠番'] = pd.to_numeric(df['枠番'], errors='coerce')
            df = df.merge(cw_df, on='枠番', how='left')
    except Exception as e:
        diag_data['cw_error'] = str(e)

    for col in ['jockey_course_win_rate', 'tag_win_rate', 'course_waku_win_rate', 'crs_rate']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = 0.0

    try:
        names_str = "', '".join([n.replace("'", "''") for n in df['馬名'].tolist()])
        ped_info = pd.read_sql(f"SELECT horse_name, sire_name, bms_name FROM horses WHERE horse_name IN ('{names_str}')", _engine)
        if not ped_info.empty:
            ped_info = ped_info.drop_duplicates('horse_name')
            df = df.merge(ped_info, left_on='馬名', right_on='horse_name', how='left')
            
            s_stats = pd.read_sql('SELECT sire_name, AVG(CASE WHEN "着順"=\'1\' THEN 1.0 ELSE 0.0 END) as sire_win_rate, AVG(CASE WHEN "着順"<=\'2\' THEN 1.0 ELSE 0.0 END) as sire_rentai_rate FROM raw_race_results r JOIN horses h ON r."馬名"=h.horse_name GROUP BY sire_name', _engine)
            b_stats = pd.read_sql('SELECT bms_name, AVG(CASE WHEN "着順"=\'1\' THEN 1.0 ELSE 0.0 END) as bms_win_rate, AVG(CASE WHEN "着順"<=\'2\' THEN 1.0 ELSE 0.0 END) as bms_rentai_rate FROM raw_race_results r JOIN horses h ON r."馬名"=h.horse_name GROUP BY bms_name', _engine)
            df = df.merge(s_stats, on='sire_name', how='left')
            df = df.merge(b_stats, on='bms_name', how='left')
            
            sires = tuple(df['sire_name'].dropna().unique())
            if sires:
                s_str = str(sires) if len(sires) > 1 else f"('{sires[0]}')"
                surf_query = f"""
                SELECT h.sire_name, AVG(CASE WHEN r."着順"='1' THEN 1.0 ELSE 0.0 END) as sire_surface_win_rate
                FROM raw_race_results r JOIN horses h ON r."馬名"=h.horse_name 
                WHERE h.sire_name IN {s_str} AND r."コース区分" = '{c_type}'
                GROUP BY h.sire_name
                """
                surf_df = pd.read_sql(surf_query, _engine)
                if not surf_df.empty:
                    df = df.merge(surf_df, on='sire_name', how='left')
            diag_data['pedigree'] = ped_info
    except: pass

    if 'sire_name' not in df.columns: df['sire_name'] = '-'
    if 'bms_name' not in df.columns: df['bms_name'] = '-'

    df['距離'] = pd.to_numeric(df['距離'], errors='coerce').fillna(1600)
    df['prev_distance'] = pd.to_numeric(df['prev_distance'], errors='coerce').fillna(df['距離'])
    df['dist_change'] = df['距離'] - df['prev_distance']
    df['is_dist_shorten'] = (df['dist_change'] < 0).astype(int)
    df['is_dist_extend'] = (df['dist_change'] > 0).astype(int)
    
    df['prev_course_type'] = df['prev_course_type'].fillna(df['コース区分'])
    df['course_change'] = (df['コース区分'] != df['prev_course_type']).astype(int)
    
    df['騎手_clean'] = df['騎手_db'].fillna(df['騎手']).astype(str).str.replace(" ", "").str.replace("　", "")
    df['prev_jockey'] = df['prev_jockey'].fillna(df['騎手_clean'])
    df['is_same_jockey'] = (df['騎手_clean'] == df['prev_jockey']).astype(int)
    diag_data['jockey_check_df'] = df[['馬名', '騎手', '騎手_db', 'prev_jockey', 'is_same_jockey']].copy()

    if 'nige_rate' not in df.columns: df['nige_rate'] = 0
    if 'senko_rate' not in df.columns: df['senko_rate'] = 0
    
    nige_count = (df['nige_rate'] > 0.5).sum()
    senko_count = (df['senko_rate'] > 0.5).sum()
    horse_count = len(df)
    senko_ratio = senko_count / horse_count if horse_count > 0 else 0
    
    df['nige_count'] = nige_count
    df['senko_count'] = senko_count
    df['senko_ratio_in_race'] = senko_ratio
    df['horse_count'] = horse_count
    
    is_high = 1 if senko_ratio >= 0.4 else 0
    is_slow = 1 if senko_ratio <= 0.2 else 0
    df['is_high_pace_forecast'] = is_high
    df['is_slow_pace_forecast'] = is_slow
    
    def get_dist(row):
        key = (row['開催場所'], row['コース区分'], row['距離'])
        return COURSE_START_TO_CORNER.get(key, 300)
    df['dist_to_first_corner'] = df.apply(get_dist, axis=1)
    
    df['枠番'] = pd.to_numeric(df['枠番'], errors='coerce').fillna(0)
    df['dist_to_corner_x_waku'] = df['dist_to_first_corner'] * df['枠番']
    
    df['is_pace_advantage'] = 0
    mask_slow = (is_slow == 1) & (df['avg_pos_rate'] <= 0.3)
    df.loc[mask_slow, 'is_pace_advantage'] = 1
    mask_high = (is_high == 1) & (df['avg_pos_rate'] > 0.6)
    df.loc[mask_high, 'is_pace_advantage'] = 1

    adv_cols = ['jockey_course_rentai_rate', 'sire_surface_win_rate', 'tag_win_rate']
    for c in adv_cols:
        if c not in df.columns: df[c] = 0.0

    cols_fill_0 = ['jockey_win_rate', 'jockey_rentai_rate', 'trainer_win_rate', 
                   'sire_win_rate', 'sire_rentai_rate', 'bms_win_rate', 'bms_rentai_rate',
                   'interval_weeks', 'prev_rank', 'prev_3f', 'prev_margin', 
                   'recent_3f_avg', 'recent_rank_avg', 'run_style_ratio', 'total_wins', 'total_money', 'win_ratio',
                   'dist_change', 'is_dist_shorten', 'is_dist_extend', 'course_change', 'is_same_jockey', 'crs_rate']
    for c in cols_fill_0:
        if c not in df.columns: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    def calc_std(x):
        if x.std() == 0: return 0
        return (x - x.mean()) / x.std()
        
    df['std_recent_3f'] = calc_std(df['recent_3f_avg'])
    df['std_recent_rank'] = calc_std(df['recent_rank_avg'])
    df['std_jockey_win'] = calc_std(df['jockey_win_rate'])
    df['std_trainer_win'] = calc_std(df['trainer_win_rate'])
    df['std_sire_win'] = calc_std(df['sire_win_rate'])

    if 'クラス' in df.columns and 'race_title_raw' in df.columns:
        diag_data['class_debug'] = df[['race_title_raw', 'クラス']].drop_duplicates().to_dict(orient='records')
        diag_data['raw_class_values'] = df[['race_title_raw', 'クラス']].copy()

    cat_cols = ['開催場所', 'コース区分', '回り', 'クラス']
    for c in cat_cols:
        if c in encoders:
            le = encoders[c]
            df[c] = df[c].astype(str).map(lambda x: x if x in le.classes_ else 'Unknown')
            df[c] = df[c].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
        else:
            df[c] = 0

    X = df[feature_cols]
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    try:
        raw_preds = model.predict(X)
        df['raw_preds'] = raw_preds # Keep raw for tie-break
        probs = calibrator.transform(raw_preds)
        df['AIスコア'] = probs
    except Exception as e:
        df['AIスコア'] = 0
        df['raw_preds'] = 0
        
    df['AI Rating'] = (df['AIスコア'] * 500).clip(0, 99).astype(int)
    
    df['騎手'] = df['騎手_db'].fillna(df['騎手'])
    df['調教師'] = df['調教師_db'].fillna(df['調教師'])
    
    def get_reason(row):
        r = []
        if row['is_pace_advantage'] == 1: r.append("🌀展開利")
        if row.get('crs_rate', 0) >= 0.5: r.append("🐴コース巧者")
        if row.get('jockey_course_rentai_rate', 0) > 0.3: r.append("🏰コース巧者")
        if row['jockey_win_rate'] > 0.15: r.append("🔥高勝率騎手")
        if row['run_style_ratio'] > 0.5: r.append("🚀先行型")
        if row.get('avg_pos_rate', 0) > 0.7 and row.get('std_recent_3f', 0) < -0.5: r.append("⚡豪脚")
        if row['recent_rank_avg'] < 3.0 and row['recent_rank_avg'] > 0: r.append("📈近走好調")
        if row['sire_win_rate'] > 0.1: r.append("🩸良血")
        if row.get('is_dist_shorten', 0) == 1: r.append("📏距離短縮")
        return " ".join(r)
    df['BoostReason'] = df.apply(get_reason, axis=1)
    
    def get_rec(row):
        p = row['AIスコア']
        try:
            o_str = str(row['オッズ']).replace('-','0')
            if o_str == '0': o = 0.0
            else: o = float(o_str)
        except: o = 0.0
        
        if row['is_pace_advantage'] == 1 and p >= 0.10 and 10.0 <= o <= 100.0:
            return "🚀 展開の神", "-"
        if p >= 0.08 and 50.0 <= o <= 100.0:
            return "-", "💣 穴馬の極意"
        if p >= 0.20 and 5.0 <= o <= 30.0:
            return "💎 黄金法則", "-"
        return "-", "-"

    df[['判定', '判定_穴']] = df.apply(lambda x: pd.Series(get_rec(x)), axis=1)
    
    trace_cols = ['馬名', 'date', 'interval_weeks', 'prev_3f', 'prev_margin', 'recent_3f_avg', 'jockey_win_rate', 'dist_change', 'nige_rate', 'senko_rate', 'is_pace_advantage']
    for c in trace_cols:
        if c not in df.columns: df[c] = np.nan
    trace_df = df[trace_cols].copy()
    
    return df.sort_values(['AIスコア', 'raw_preds'], ascending=[False, False]), df, X, diag_data, missing_info, trace_df

def process_one_race(race, model, encoders, engine, driver=None):
    """並列処理用の単一レース処理関数"""
    try:
        df = scrape_race_data(race['url'], driver=driver)
        if df is not None and not df.empty:
            res, _, _, _, missing_info, _ = predict_race(df, model, encoders, engine)
            
            # 結果サマリー
            top_ai = res.iloc[0]
            
            # ★変更: 抽出された馬リストの中で、最もAIスコアが高い馬を「購入対象」としてマークする
            
            # 1. 展開の神 (スコア順にソートして先頭1頭をBUY対象にする)
            pace_hits = res[res['判定'] == "🚀 展開の神"].copy()
            if not pace_hits.empty:
                pace_hits = pace_hits.sort_values(['AIスコア', 'raw_preds'], ascending=[False, False])
                pace_hits['is_bet_target'] = False
                # 先頭行(最高スコア)をTrueに
                pace_hits.iat[0, pace_hits.columns.get_loc('is_bet_target')] = True
            
            # 2. 穴馬 (同様にソートして先頭1頭をBUY対象にする)
            hole_hits = res[res['判定_穴'] == "💣 穴馬の極意"].copy()
            if not hole_hits.empty:
                hole_hits = hole_hits.sort_values(['AIスコア', 'raw_preds'], ascending=[False, False])
                hole_hits['is_bet_target'] = False
                hole_hits.iat[0, hole_hits.columns.get_loc('is_bet_target')] = True
            
            try: top_odds = float(str(top_ai['オッズ']).replace('-','0'))
            except: top_odds = 0
            is_ai_target = (3.0 <= top_odds <= 30.0)
            
            # 3. 鉄板 (条件を満たせばTrue)
            ai_hit_df = res.iloc[[0]].copy()
            ai_hit_df['is_bet_target'] = is_ai_target

            # バッジ処理 (元のresに対して行う)
            res['overlap_badges'] = [[] for _ in range(len(res))]
            if is_ai_target: res.at[res.index[0], 'overlap_badges'].append("ai")
            # 注意: ここでのループは元のresに対するものなので、マーク済みDFとは別管理
            for idx in pace_hits.index: res.at[idx, 'overlap_badges'].append("pace")
            for idx in hole_hits.index: res.at[idx, 'overlap_badges'].append("hole")

            # 成績集計用データ
            race_id = df.iloc[0]['race_id']
            ranks, win_p, place_p, _ = scrape_race_result(race_id)
            
            return {
                'status': 'success',
                'race': race,
                'df': res,
                'pace_hits': pace_hits, # マーク付きDFを返す
                'hole_hits': hole_hits, # マーク付きDFを返す
                'ai_hit_df': ai_hit_df, # マーク付きDFを返す
                'is_ai_target': is_ai_target,
                'top_ai': top_ai,
                'ranks': ranks,
                'win_p': win_p,
                'place_p': place_p,
                'missing_info': missing_info
            }
        return {'status': 'empty', 'race': race}
    except Exception as e:
        return {'status': 'error', 'race': race, 'error': str(e)}

# ---------------------------------------------------------
# 新規追加: レースの塊(チャンク)を1つのブラウザで連続処理する関数
# ---------------------------------------------------------
# ---------------------------------------------------------
# 新規追加: レースの塊(チャンク)を1つのブラウザで連続処理する関数
# ---------------------------------------------------------
# ---------------------------------------------------------
# 新規追加: レースの塊(チャンク)を1つのブラウザで連続処理する関数
# ---------------------------------------------------------
def process_race_chunk(chunk, model, encoders, engine, ctx, result_queue): # ★変更: result_queue を追加
    # 別スレッドでもStreamlitの機能が使えるように設定
    if ctx: add_script_run_ctx(threading.current_thread(), ctx)

    # resultsリストは廃止し、随時 queue に入れる
    # スレッドごとに専用のブラウザを起動
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = None
    try:
        from selenium.webdriver.chrome.service import Service
        service = Service()
        driver = webdriver.Chrome(options=options, service=service)
        
        # 起動したブラウザを使い回して、担当分のレースを全て処理
        for race in chunk:
            res = process_one_race(race, model, encoders, engine, driver=driver)
            result_queue.put(res) # ★変更: 処理が終わったら即座にキューへ入れる
            
    except Exception as e:
        # エラー時もキューに入れてカウントを進める
        for race in chunk:
            result_queue.put({'status': 'error', 'race': race, 'error': str(e)}) # ★変更
    finally:
        if driver:
            driver.quit()
    
    return True # 戻り値は使わないので適当に

def scan_races(target_date, race_list, model, encoders, engine):
    if 'report_stats' in st.session_state and st.session_state.report_stats:
        stats = st.session_state.report_stats
    else:
        stats = {k: {'bets':0, 'hit_count':0, 'win_ret':0, 'place_ret':0} for k in ['pace', 'hole', 'ai']}
        
    results = {'pace': [], 'hole': [], 'ai': []}
    st.session_state.hits_details = []
    
    all_missing = {'jockey': set(), 'trainer': set()}
    trainer_debug_list = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("🚀 Initializing parallel workers...")

    # 新馬・障害を除外したリストを作成
    target_races = [r for r in race_list if "新馬" not in r['label'] and "障害" not in r['label']]
    total_races = len(target_races)
    
    if total_races == 0:
        status_text.text("No target races found.")
        return results

    # リストを2分割する (クラウドのメモリ制限を考慮し、並列数は2とする)
    num_workers = 2
    chunks = [target_races[i::num_workers] for i in range(num_workers)]
    
    # メインスレッドのコンテキストを取得
    try:
        ctx = get_script_run_ctx()
    except:
        ctx = None

    # 結果受け取り用のキューを作成
    result_queue = queue.Queue() # ★追加

    # 並列実行
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 各チャンクを処理開始 (queueも渡す)
        # ★変更: process_race_chunk の引数に result_queue を追加
        futures = [executor.submit(process_race_chunk, chunk, model, encoders, engine, ctx, result_queue) for chunk in chunks]
        
        completed_races = 0
        
        # ★変更: レース数分だけループして、キューから結果を1つずつ取り出す
        while completed_races < total_races:
            try:
                # キューから結果を取得 (タイムアウト付きで無限待ち回避)
                data = result_queue.get(timeout=180) 
                
                completed_races += 1
                status_text.text(f"Processing... ({completed_races}/{total_races} completed)")
                
                if data['status'] == 'success':
                    res = data['df']
                    race = data['race']
                    
                    # Missing Info集計
                    m_info = data['missing_info']
                    for j in m_info['jockey']: all_missing['jockey'].add(j)
                    for t in m_info['trainer']: all_missing['trainer'].add(t)
                    if 'trainer_debug' in m_info and not m_info['trainer_debug'].empty:
                        trainer_debug_list.append(m_info['trainer_debug'])
                    
                    # 結果リストへの追加
                    # ★変更: マーク付きのDF (pace_hits, hole_hits, ai_hit_df) を使用する
                    if not data['pace_hits'].empty: 
                        results['pace'].append({'race': race['label'], 'url': race['url'], 'hits': data['pace_hits'], 'grade': race['grade']})
                    
                    if not data['hole_hits'].empty: 
                        results['hole'].append({'race': race['label'], 'url': race['url'], 'hits': data['hole_hits'], 'grade': race['grade']})
                    
                    if data['is_ai_target']: 
                        # ここも ai_hit_df を使う
                        results['ai'].append({'race': race['label'], 'url': race['url'], 'hits': data['ai_hit_df'], 'grade': race['grade']})
                    
                    # 成績集計
                    ranks = data['ranks']
                    win_p = data['win_p']
                    place_p = data['place_p']
                    
                    if ranks:
                        def update(cat, horse):
                            stats[cat]['bets'] += 1
                            r = ranks.get(horse['馬番'], 99)
                            if r == 1: stats[cat]['win_ret'] += win_p.get(horse['馬番'], 0)
                            if r <= 3:
                                stats[cat]['hit_count'] += 1
                                stats[cat]['place_ret'] += place_p.get(horse['馬番'], 0)
                                st.session_state.hits_details.append({"戦略": cat, "レース": race['label'], "馬名": horse['馬名'], "着順": r, "単勝": win_p.get(horse['馬番'], 0), "複勝": place_p.get(horse['馬番'], 0)})

                        if data['is_ai_target']: update('ai', data['top_ai'])
                        if not data['pace_hits'].empty: update('pace', data['pace_hits'].iloc[0])
                        if not data['hole_hits'].empty: 
                            hole_sorted = data['hole_hits'].sort_values(['AIスコア', 'raw_preds'], ascending=[False, False])
                            update('hole', hole_sorted.iloc[0])
            
                # プログレスバー更新 (1件ごとに進む)
                progress_bar.progress(min(1.0, completed_races / total_races))
            
            except queue.Empty:
                # 万が一タイムアウトしたらループを抜ける
                break
    
    status_text.empty()
    progress_bar.empty()
    st.session_state.report_stats = stats
    st.session_state.missing_data = all_missing
    if trainer_debug_list:
        st.session_state.trainer_debug_all = pd.concat(trainer_debug_list, ignore_index=True)
    return results

def toggle_expander(key):
    if key in st.session_state.expander_states:
        st.session_state.expander_states[key] = not st.session_state.expander_states[key]

def main():
    load_custom_css()
    
    video_files = ["resource/競馬シーン_サイト埋め込み用動画.mp4", "resource/競馬シーン_サイト埋め込み用動画_2.mp4", "resource/競馬シーン_サイト埋め込み用動画_3.mp4"]
    b64 = get_base64_video(video_files)
    video_src = f"data:video/mp4;base64,{b64}" if b64 else "https://videos.pexels.com/video-files/5230349/5230349-uhd_2560_1440_25fps.mp4"
    
    # YouTube random
    yt_ids = ["TZXtiFh3AM8", "7omyMKRLIrc", "tMHitvVB4lU"]
    yt_id = random.choice(yt_ids)

    st.markdown(f"""
        <div class="header-container">
            <div class="video-background">
                <video src="{video_src}" autoplay loop muted playsinline></video>
            </div>
            <div class="header-overlay"></div>
            <div class="header-content">
                <h1 class="header-title">KaiのゆるっとAI<br><span class="beta-badge">Ver.1.0.0</span></h1>
                <div class="header-subtitle">展開利を見抜いて回収率108%!?</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("System Status")
        model, encoders, engine, logs = load_resources(0)
        if model: st.success(f"✅ Model Loaded: Pace Aware (Feats: {len(model['features'])})")
        else: st.error("Model Load Failed")

    if 'race_list' not in st.session_state: st.session_state.race_list = []
    if 'selected_race_url' not in st.session_state: st.session_state.selected_race_url = ""
    if 'selected_race_name' not in st.session_state: st.session_state.selected_race_name = ""
    if 'view_mode' not in st.session_state: st.session_state.view_mode = 'list'
    if 'auto_predict' not in st.session_state: st.session_state.auto_predict = False
    if 'scan_results' not in st.session_state: st.session_state.scan_results = None
    if 'report_stats' not in st.session_state: st.session_state.report_stats = None
    if 'hits_details' not in st.session_state: st.session_state.hits_details = []
    if 'expander_states' not in st.session_state: st.session_state.expander_states = {'pace': True, 'hole': False, 'ai': False}
    if 'missing_data' not in st.session_state: st.session_state.missing_data = {'jockey': set(), 'trainer': set()}
    if 'trainer_debug_all' not in st.session_state: st.session_state.trainer_debug_all = pd.DataFrame()
    
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)
    st.markdown("### 📅 Race Selection")
    
    col_a, col_b = st.columns([1, 2])
    with col_a:
        target_date = st.date_input("開催日を選択", datetime.date.today())
        if st.button("レース一覧を取得", type="primary", use_container_width=True):
            with st.spinner("取得中..."):
                st.session_state.race_list = get_race_list_by_date(target_date)
                # Reset view but keep nothing until scan
                st.session_state.scan_results = None 
                st.session_state.report_stats = None
                st.session_state.view_mode = 'list'
                if not st.session_state.race_list: st.toast("レース情報なし", icon="⚠️")
                else: st.toast(f"{len(st.session_state.race_list)} 件取得", icon="✅")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        scan_label = "🚀 全レース一括スキャン"
        if st.button(scan_label, use_container_width=True):
            with st.spinner("開催情報を取得中..."):
                current_race_list = get_race_list_by_date(target_date)
            
            if current_race_list:
                # ★追加: 表示用に、スキャン対象(新馬・障害以外)の数をあらかじめ計算する
                scan_targets = [r for r in current_race_list if "新馬" not in r['label'] and "障害" not in r['label']]
                
                st.session_state.race_list = current_race_list
                st.session_state.report_stats = None 
                st.session_state.scan_results = None
                
                # ★変更: len(current_race_list) ではなく len(scan_targets) を表示
                with st.spinner(f"AIが全集中で予想中... (対象: {len(scan_targets)}レース)"):
                    results = scan_races(target_date, current_race_list, model, encoders, engine)
                    st.session_state.scan_results = results
                    st.session_state.view_mode = 'list'
                    st.rerun()
            else:
                st.error("レース情報が見つかりませんでした。開催日を確認してください。")

    # --- View Mode Control ---
    if st.session_state.view_mode == 'list' and st.session_state.scan_results:
        # Show Results
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔽 リストを閉じる (クリア)", key="clear_scan"):
            st.session_state.scan_results = None
            st.session_state.report_stats = None
            st.rerun()
        
        missing = st.session_state.missing_data
        if missing['jockey'] or missing['trainer']:
            with st.expander("⚠️ 未マッピング情報 (診断用)", expanded=False):
                if 'trainer_debug_all' in st.session_state and not st.session_state.trainer_debug_all.empty:
                    df_debug = st.session_state.trainer_debug_all
                    df_fail = df_debug[df_debug['マッチ結果'] == '❌']
                    if not df_fail.empty:
                        st.dataframe(df_fail)
                
                st.warning("以下のリストを `MANUAL_MAP` に追加してください。")
                code_str = ""
                if missing['jockey']:
                    code_str += "# --- 未登録騎手 ---\n"
                    for m in missing['jockey']: code_str += f'"{m}": "???",\n'
                if missing['trainer']:
                    code_str += "\n# --- 未登録調教師 ---\n"
                    for m in missing['trainer']: 
                        cands = "[]"
                        if 'trainer_debug_all' in st.session_state and not st.session_state.trainer_debug_all.empty:
                            row = st.session_state.trainer_debug_all[st.session_state.trainer_debug_all['Web取得名'] == m]
                            if not row.empty:
                                cands = row.iloc[0]['候補リスト']
                        code_str += f'"{m}": "???", # 候補: {cands}\n'
                st.code(code_str, language='python')

        stats = st.session_state.report_stats
        if stats:
            st.markdown(f"### 📈 回収率シミュレーション")
            r1, r2, r3 = st.columns(3)
            def get_rois(d):
                b = d['bets']; 
                if b == 0: return 0, 0, 0.0, 0.0
                return d['hit_count'], b, (d['win_ret']/(b*100))*100, (d['place_ret']/(b*100))*100

            w1, b1, roi_w1, roi_p1 = get_rois(stats['pace'])
            with r1:
                st.markdown(render_report_card_dual("🚀 展開の神", w1, b1, roi_w1, roi_p1), unsafe_allow_html=True)
                if st.button("🔽 リストを開く", key="jump_pace", on_click=toggle_expander, args=('pace',), use_container_width=True): pass
                if st.session_state.expander_states['pace']: js_scroll_to('section_pace')

            w3, b3, roi_w3, roi_p3 = get_rois(stats['ai'])
            with r2:
                st.markdown(render_report_card_dual("🦄 鉄板の軸", w3, b3, roi_w3, roi_p3), unsafe_allow_html=True)
                if st.button("🔽 リストを開く", key="jump_ai", on_click=toggle_expander, args=('ai',), use_container_width=True): pass
                if st.session_state.expander_states['ai']: js_scroll_to('section_ai')

            w2, b2, roi_w2, roi_p2 = get_rois(stats['hole'])
            with r3:
                st.markdown(render_report_card_dual("💣 穴馬の極意", w2, b2, roi_w2, roi_p2), unsafe_allow_html=True)
                if st.button("🔽 リストを開く", key="jump_hole", on_click=toggle_expander, args=('hole',), use_container_width=True): pass
                if st.session_state.expander_states['hole']: js_scroll_to('section_hole')
            
            with st.expander("🏆 的中実績の詳細", expanded=False):
                if st.session_state.hits_details:
                    df_hits = pd.DataFrame(st.session_state.hits_details)
                    t1, t2, t3 = st.tabs(["🚀 展開の神", "🦄 鉄板", "💣 穴馬"])
                    with t1:
                        sub = df_hits[df_hits['戦略'] == 'pace']
                        if not sub.empty: st.dataframe(sub, hide_index=True)
                        else: st.info("該当なし")
                    with t2:
                        sub = df_hits[df_hits['戦略'] == 'ai']
                        if not sub.empty: st.dataframe(sub, hide_index=True)
                        else: st.info("該当なし")
                    with t3:
                        sub = df_hits[df_hits['戦略'] == 'hole']
                        if not sub.empty: st.dataframe(sub, hide_index=True)
                        else: st.info("該当なし")
                else: st.info("的中なし")
            st.divider()

        results = st.session_state.scan_results
        st.markdown(render_ev_legend(), unsafe_allow_html=True)
        st.markdown(render_badge_legend(), unsafe_allow_html=True)
        
        def render_scan_list(hits_list, mode):
            if not hits_list: st.info("該当なし"); return
            for item in hits_list:
                hits_df = item['hits'].copy()
                hits_df = hits_df.sort_values('馬番')
                
                with st.container():
                    c1, c2 = st.columns([0.15, 0.85])
                    with c1: st.markdown(render_grade_badge_html(item['grade']), unsafe_allow_html=True)
                    with c2: st.markdown(f"#### {item['race']}")
                    
                    for idx, row in hits_df.iterrows():
                        st.markdown(render_ai_list_item(row, row.get('overlap_badges', [])), unsafe_allow_html=True)
                            
                    if st.button(f"詳細を見る ➡️", key=f"btn_{mode}_{item['race']}"):
                        st.session_state.selected_race_url = item['url']
                        st.session_state.selected_race_name = item['race']
                        st.session_state.view_mode = 'detail' # Switch to detail mode
                        st.session_state.auto_predict = True
                        st.rerun()
                st.divider()
            
            if st.button(f"🔼 リストを閉じる", key=f"close_{mode}", on_click=toggle_expander, args=(mode,), use_container_width=True): pass

        st.markdown('<div id="section_pace"></div>', unsafe_allow_html=True)
        with st.expander("🚀 展開の神 (ROI 108%〜)", expanded=st.session_state.expander_states['pace']):
            render_scan_list(results['pace'], 'pace')
        st.markdown('<div id="section_ai"></div>', unsafe_allow_html=True)
        with st.expander("🦄 鉄板の軸 (ROI 80%)", expanded=st.session_state.expander_states['ai']):
            render_scan_list(results['ai'], 'ai')
        st.markdown('<div id="section_hole"></div>', unsafe_allow_html=True)
        with st.expander("💣 穴馬の極意 (ROI 87%)", expanded=st.session_state.expander_states['hole']):
            render_scan_list(results['hole'], 'hole')

    with col_b:
        if st.session_state.race_list:
             if st.session_state.view_mode == 'list':
                st.caption("👇 レース名を選択してAI予想を実行")
                if st.button("🔼 リストを閉じる", key="close_list_btn"):
                     st.session_state.race_list = [] # Clear list
                     st.rerun()
                with st.container(height=400):
                    for r in st.session_state.race_list:
                        c1, c2 = st.columns([0.15, 0.85])
                        with c1: st.markdown(render_grade_badge_html(r['grade']), unsafe_allow_html=True)
                        with c2:
                            if st.button(r['label'], key=r['id'], use_container_width=True):
                                st.session_state.selected_race_url = r['url']
                                st.session_state.selected_race_name = r['label']
                                st.session_state.view_mode = 'detail'
                                st.session_state.auto_predict = True 
                                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("🔗 URL直接入力"):
        input_url = st.text_input("レースURL", key="manual_url")
        if st.button("URLで予想", key="btn_manual"):
            st.session_state.selected_race_url = input_url
            st.session_state.selected_race_name = "URL指定レース"
            st.session_state.view_mode = 'detail'
            st.session_state.auto_predict = True
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- Detail View Logic ---
    should_predict = st.session_state.auto_predict
    
    if st.session_state.view_mode == 'detail' and st.session_state.selected_race_url:
        st.session_state.auto_predict = False # Reset flag
        target = st.session_state.selected_race_url
        
        st.divider()
        if st.session_state.scan_results:
            if st.button("🔙 スキャン結果に戻る", use_container_width=True):
                st.session_state.view_mode = 'list'
                st.rerun()
        
        st.markdown(f"### 🎯 AI Forecast: {st.session_state.selected_race_name or '指定レース'}")
        
        with st.spinner("🦄 AIが全集中で予想中..."):
            df_in = scrape_race_data(target)
            if df_in is not None and not df_in.empty:
                try:
                    res, debug, X_renamed, diag_data, missing_info, trace_df = predict_race(df_in, model, encoders, engine)
                    res = res.drop_duplicates(subset=['馬名'])
                    
                    if not res.empty:
                        top = res.iloc[0]
                        
                        # ★ ここでGeminiを呼び出す！
                        # ★ 変更点: 戻り値を2つ受け取る
                        with st.spinner("🦄 Geminiが寸評を執筆中..."):
                            ai_comment, used_model = generate_gemini_comment(top)

                        # Hero Cardの下にコメントを表示するエリアを追加
                        st.markdown(render_hero_card(top), unsafe_allow_html=True)
                        
                        # 寸評表示用のおしゃれなボックス（モデル名を追加）
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #fdfbf7 0%, #fff 100%); 
                            border: 2px solid #d4af37; 
                            border-radius: 12px; 
                            padding: 15px; 
                            margin-bottom: 20px; 
                            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                            position: relative;
                        ">
                            <div style="
                                position: absolute; 
                                top: -12px; 
                                left: 20px; 
                                background: #d4af37; 
                                color: white; 
                                padding: 2px 10px; 
                                border-radius: 4px; 
                                font-weight: bold; 
                                font-size: 0.8rem;
                                display: flex;
                                align-items: center;
                                gap: 5px;
                            ">
                                <span>✨ Gemini's Eye</span>
                                <span style="
                                    background: rgba(255,255,255,0.2); 
                                    padding: 0px 6px; 
                                    border-radius: 3px; 
                                    font-size: 0.7em; 
                                    font-weight: normal;
                                ">
                                    by {used_model}
                                </span>
                            </div>
                            <div style="
                                font-family: 'Hiragino Mincho ProN', serif; 
                                font-size: 1.1rem; 
                                color: #333; 
                                line-height: 1.6;
                                margin-top: 5px;
                            ">
                                {ai_comment}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        pace_hits = res[res['判定'] == "🚀 展開の神"]
                        if not pace_hits.empty:
                            st.markdown("### 🚀 展開の神 (ROI 108%🏆)")
                            for _, row in pace_hits.iterrows():
                                ev_val = row['AIスコア'] * float(str(row['オッズ']).replace('-','0'))
                                st.success(f"**#{row['馬番']} {row['馬名']}** ({row['騎手']}) - 単勝{row['オッズ']}倍 (EV: {ev_val:.2f}) - {row['BoostReason']}")

                        hole_hits = res[res['判定_穴'] == "💣 穴馬の極意"]
                        if not hole_hits.empty:
                            st.markdown("### 💣 穴馬の極意")
                            for _, row in hole_hits.iterrows():
                                ev_val = row['AIスコア'] * float(str(row['オッズ']).replace('-','0'))
                                st.error(f"**#{row['馬番']} {row['馬名']}** ({row['騎手']}) - 単勝{row['オッズ']}倍 (EV: {ev_val:.2f})")

                        st.markdown("##### 📊 全頭診断リスト")
                        st.markdown(render_ev_legend(), unsafe_allow_html=True)
                        st.markdown(render_badge_legend(), unsafe_allow_html=True)
                        
                        disp = res.copy()
                        disp['AIスコア'] *= 100
                        disp['sire_win_rate'] *= 100 
                        disp['bms_win_rate'] *= 100 
                        disp['枠'] = pd.to_numeric(disp['枠番'], errors='coerce').fillna(0).astype(int)
                        disp['番'] = pd.to_numeric(disp['馬番'], errors='coerce').fillna(0).astype(int)
                        disp['trainer_win_rate'] *= 100
                        disp['jockey_win_rate'] *= 100
                        disp = disp.rename(columns={'sire_name':'父', 'bms_name':'母父', 'sire_win_rate':'父勝率', 'bms_win_rate':'母父勝率', 'trainer_win_rate':'厩舎勝率', 'jockey_win_rate': '騎手勝率', 'オッズ':'単勝'})
                        
                        st.dataframe(
                            disp[['枠', '番', '馬名', '騎手', '騎手勝率', '調教師', '単勝', '判定', '判定_穴', 'AI Rating', 'AIスコア', 'BoostReason', '父', '父勝率', '母父', '母父勝率', '厩舎勝率']],
                            column_config={
                                "AI Rating": st.column_config.ProgressColumn("AI Rating (偏差値)", min_value=0, max_value=100),
                                "AIスコア": st.column_config.NumberColumn("AI自信度(%)", format="%.1f%%"),
                                "父勝率": st.column_config.NumberColumn("父勝率", format="%.1f%%"),
                                "母父勝率": st.column_config.NumberColumn("母父勝率", format="%.1f%%"),
                                "厩舎勝率": st.column_config.NumberColumn("厩舎勝率", format="%.1f%%"),
                                "騎手勝率": st.column_config.NumberColumn("騎手勝率", format="%.1f%%"),
                                "BoostReason": "注目点"
                            },
                            hide_index=True, use_container_width=True
                        )
                        
                        csv = disp.to_csv(index=False).encode('utf-8_sig')
                        st.download_button(label="📥 予想結果をCSVでダウンロード", data=csv, file_name=f"prediction_{datetime.date.today()}.csv", mime="text/csv")
                        
                        with st.expander("🕵️‍♂️ AI内部データ診断", expanded=False):
                            tabs_diag = st.tabs(["Hero Card Source", "Calculation Trace", "Prev Race Lookup", "血統", "騎手", "調教師", "Full Data", "Advanced Debug"])
                            with tabs_diag[0]:
                                st.code(render_hero_card(top), language='html')
                            with tabs_diag[1]:
                                st.write("変数の計算過程追跡ログ (Raw -> Feature)")
                                st.dataframe(trace_df)
                            with tabs_diag[2]:
                                st.write("前走レース選択の正当性チェック (Target Date vs Prev Date)")
                                st.dataframe(diag_data['prev_trace'])
                            with tabs_diag[3]:
                                if not diag_data['pedigree'].empty: st.dataframe(diag_data['pedigree'])
                            with tabs_diag[4]:
                                if not diag_data['jockey'].empty: st.dataframe(diag_data['jockey'])
                            with tabs_diag[5]:
                                if not diag_data['trainer'].empty: st.dataframe(diag_data['trainer'])
                            with tabs_diag[6]:
                                st.subheader("データマッチング確認")
                                st.dataframe(debug)
                                debug_csv = debug.to_csv(index=False).encode('utf-8_sig')
                                st.download_button("📥 診断用データをCSVでダウンロード", debug_csv, "debug_data.csv", "text/csv")
                            with tabs_diag[7]:
                                st.subheader("Advanced Features Status")
                                if diag_data['missing_advanced']:
                                    st.warning(f"⚠️ 以下のAdvanced Featuresは現在ダミー値(0.0)で代用されています: {diag_data['missing_advanced']}")
                                else:
                                    st.success("All Advanced Features loaded.")
                                
                                if 'crs_rate_error' in diag_data:
                                    st.error(f"CRS Rate Calculation Error: {diag_data['crs_rate_error']}")
                                    
                                if 'cw_error' in diag_data:
                                    st.error(f"Course Waku Win Rate Error: {diag_data['cw_error']}")
                                    
                                if 'class_debug' in diag_data:
                                    st.write("Class Value Check:")
                                    st.write(diag_data['class_debug'])
                                    
                                if 'trainer_debug' in missing_info:
                                     st.write("Trainer Matching Logic Trace:")
                                     st.dataframe(missing_info['trainer_debug'])

                    else: st.warning("有効なデータがありません")
                except Exception as e: st.error(f"エラー: {e}")
            else: st.error("レース情報の取得に失敗しました")

if __name__ == '__main__':
    main()