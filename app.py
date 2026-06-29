import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="FIFA 2026 Predictor", page_icon="⚽", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { background-color: #050505 !important; color: #ffffff !important; }
.stApp { background-color: #050505; }

.stTabs [data-baseweb="tab-list"] { background-color: #050505; border-bottom: 1px solid #1a1a1a; gap: 0; }
.stTabs [data-baseweb="tab"] { font-family: 'Inter', sans-serif; font-size: 11px; font-weight: 600; letter-spacing: 3px; text-transform: uppercase; color: #444444 !important; padding: 14px 28px; }
.stTabs [aria-selected="true"] { color: #c9a84c !important; border-bottom: 2px solid #c9a84c !important; background: transparent !important; }

.hero-container { background: linear-gradient(135deg, #0c0c0c 0%, #050505 100%); padding: 40px; border-left: 4px solid #c9a84c; margin-bottom: 30px; }
.hero-eyebrow { font-family: 'Inter', sans-serif; font-size: 11px; font-weight: 600; letter-spacing: 6px; color: #c9a84c; text-transform: uppercase; margin-bottom: 12px; }
.hero-title { font-family: 'Bebas Neue', sans-serif; font-size: 92px; letter-spacing: 4px; line-height: 0.90; color: #ffffff; margin: 0; text-transform: uppercase; }
.hero-title span { color: #c9a84c; }

.stat-card { background: #0c0c0c; border: 1px solid #1a1a1a; border-top: 2px solid #c9a84c; padding: 26px 20px 22px; border-radius: 2px; }
.stat-number { font-family: 'Bebas Neue', sans-serif; font-size: 56px; color: #ffffff; line-height: 1; }
.stat-label { font-family: 'Inter', sans-serif; font-size: 10px; font-weight: 600; letter-spacing: 3px; color: #666666; text-transform: uppercase; margin-top: 8px; }

.section-eyebrow { font-family: 'Inter', sans-serif; font-size: 11px; font-weight: 600; letter-spacing: 4px; color: #c9a84c; text-transform: uppercase; margin-bottom: 24px; padding-bottom: 12px; border-bottom: 1px solid #1a1a1a; }

.match-card { background: #0c0c0c; border: 1px solid #1a1a1a; padding: 24px 32px; margin-bottom: 12px; border-radius: 2px; }
.team-name { font-family: 'Bebas Neue', sans-serif; font-size: 32px; letter-spacing: 2.5px; color: #ffffff; line-height: 1; }
.elo-label { font-family: 'Inter', sans-serif; font-size: 10px; letter-spacing: 2px; color: #555555; text-transform: uppercase; margin-top: 4px; }
.vs-badge { font-family: 'Inter', sans-serif; font-size: 10px; font-weight: 700; letter-spacing: 3px; color: #c9a84c; border: 1px solid #c9a84c; padding: 5px 12px; text-align: center; background: #050505; }
.date-text { font-family: 'Inter', sans-serif; font-size: 11px; letter-spacing: 2px; color: #666666; text-transform: uppercase; margin-bottom: 16px; font-weight: 500; }

div[data-testid="stSelectbox"] label { font-family: 'Inter', sans-serif; font-size: 10px; letter-spacing: 3px; color: #c9a84c; text-transform: uppercase; }
div[data-baseweb="select"] { background-color: #0c0c0c !important; border: 1px solid #1a1a1a !important; border-radius: 2px !important; }
.stSlider label { font-family: 'Inter', sans-serif; font-size: 10px; letter-spacing: 3px; color: #c9a84c; text-transform: uppercase; }

footer { display: none !important; }
#MainMenu { display: none !important; }
header { display: none !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    predictions = pd.read_csv("data/predictions.csv").dropna(subset=["home_team", "away_team"])
    predictions["date"] = pd.to_datetime(predictions["date"], utc=True)
    predictions["date_fmt"] = predictions["date"].dt.strftime("%b %d · %H:%M UTC")
    completed = pd.read_csv("data/wc2026_completed.csv").dropna(subset=["home_team", "away_team"])
    elo = pd.read_csv("data/elo_ratings.csv")
    return predictions, completed, elo


predictions, completed, elo = load_data()

# HERO
st.markdown("""
<div class="hero-container">
    <p class="hero-eyebrow">FIFA World Cup 2026 · AI Match Predictor</p>
    <h1 class="hero-title">FIFA 2026<br><span>Official</span><br>Dashboard.</h1>
</div>
""", unsafe_allow_html=True)

# STATS
c1, c2, c3, c4 = st.columns(4)
home_wins = len(predictions[predictions["prediction"] == "Home Win"])
draws     = len(predictions[predictions["prediction"] == "Draw"])
avg_conf  = predictions[["home_win_prob","draw_prob","away_win_prob"]].max(axis=1).mean()

c1.markdown(f'<div class="stat-card"><div class="stat-number">{len(predictions)}</div><div class="stat-label">Matches to Predict</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="stat-card"><div class="stat-number">{home_wins}</div><div class="stat-label">Home Win Predictions</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="stat-card"><div class="stat-number">{draws}</div><div class="stat-label">Draw Predictions</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#c9a84c;">{avg_conf:.0f}%</div><div class="stat-label">Avg Confidence</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# TABS
tab1, tab2, tab3 = st.tabs(["🔮   Predictions", "📊   ELO Rankings", "✅   Completed"])

# TAB 1
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-eyebrow">Upcoming Schedule · Match Outcome Models</p>', unsafe_allow_html=True)

    filter_col, _ = st.columns([2, 4])
    with filter_col:
        filter_opt = st.selectbox("Filter by model prediction", ["All", "Home Win", "Draw", "Away Win"])

    filtered = predictions if filter_opt == "All" else predictions[predictions["prediction"] == filter_opt]

    for _, row in filtered.iterrows():
        pred = row['prediction']
        left_border = '#c9a84c'
        badge_bg = '#161616'
        badge_color = '#ffffff' if pred == 'Draw' else '#c9a84c'

        hw = row['home_win_prob']
        dp = row['draw_prob']
        aw = row['away_win_prob']
        home = row['home_team']
        away = row['away_team']
        home_elo = f"{row['home_elo']:.0f}"
        away_elo = f"{row['away_elo']:.0f}"
        date_fmt = row['date_fmt']

        html = f"""<div class="match-card" style="border-left:2px solid {left_border};">
<div class="date-text">{date_fmt}</div>
<div style="display:flex;align-items:center;justify-content:space-between;">
<div style="flex:1;">
<div style="display:flex;align-items:center;gap:20px;">
<div style="text-align:right;flex:1;"><div class="team-name">{home}</div><div class="elo-label">ELO {home_elo}</div></div>
<div class="vs-badge">VS</div>
<div style="text-align:left;flex:1;"><div class="team-name">{away}</div><div class="elo-label">ELO {away_elo}</div></div>
</div>
<div style="display:flex;gap:2px;margin:18px 0 6px 0;height:4px;background:#111;">
<div style="flex:{hw};background:#ffffff;"></div>
<div style="flex:{dp};background:#c9a84c;"></div>
<div style="flex:{aw};background:#444444;"></div>
</div>
<div style="display:flex;justify-content:space-between;font-family:Inter,sans-serif;font-size:10px;letter-spacing:1px;font-weight:500;">
<span style="color:#ffffff;">{hw}%</span>
<span style="color:#c9a84c;">{dp}%</span>
<span style="color:#888888;">{aw}%</span>
</div>
</div>
<div style="margin-left:40px;">
<span style="font-family:Inter,sans-serif;font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;padding:8px 18px;background:{badge_bg};color:{badge_color};border:1px solid #222;">{pred}</span>
</div>
</div>
</div>"""
        st.markdown(html, unsafe_allow_html=True)

# TAB 2
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-eyebrow">Current Global ELO Ratings Matrix</p>', unsafe_allow_html=True)

    top_n = st.slider("Show top N tournament teams", 10, 50, 20)
    top_elo = elo.head(top_n).sort_values("elo_rating")

    fig = go.Figure(go.Bar(
        x=top_elo["elo_rating"],
        y=top_elo["team"],
        orientation="h",
        marker=dict(
            color=top_elo["elo_rating"],
            colorscale=[[0, "#161616"], [0.7, "#a68632"], [1, "#c9a84c"]],
            showscale=False
        ),
        text=top_elo["elo_rating"].round(0).astype(int),
        textposition="outside",
        textfont=dict(color="#888888", size=11, family="Inter")
    ))

    fig.update_layout(
        paper_bgcolor="#050505",
        plot_bgcolor="#050505",
        font=dict(color="#ffffff", family="Inter"),
        height=max(400, top_n * 28),
        margin=dict(l=20, r=60, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor="#111111", tickfont=dict(color="#444444"),
                   range=[elo["elo_rating"].min() - 50, elo["elo_rating"].max() + 100]),
        yaxis=dict(tickfont=dict(color="#ffffff", size=12))
    )

    st.plotly_chart(fig, use_container_width=True)

# TAB 3
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-eyebrow">Match History Ledger & Realised Scores</p>', unsafe_allow_html=True)

    completed["date"] = pd.to_datetime(completed["date"], utc=True)
    completed_sorted = completed.sort_values("date", ascending=False)

    for _, row in completed_sorted.iterrows():
        hg = int(row["home_goals"]) if pd.notna(row["home_goals"]) else 0
        ag = int(row["away_goals"]) if pd.notna(row["away_goals"]) else 0

        if hg > ag:
            h_style, a_style = "color:#ffffff;", "color:#333333;"
        elif ag > hg:
            h_style, a_style = "color:#333333;", "color:#ffffff;"
        else:
            h_style = a_style = "color:#c9a84c;"

        date_str = pd.to_datetime(row["date"]).strftime("%b %d")
        home = row['home_team']
        away = row['away_team']

        html = f"""<div class="match-card" style="padding:16px 32px;">
<div style="display:flex;align-items:center;gap:16px;">
<div class="date-text" style="width:50px;margin:0;">{date_str}</div>
<div style="flex:1;text-align:right;font-family:'Bebas Neue',sans-serif;font-size:24px;letter-spacing:2px;{h_style}">{home}</div>
<div style="font-family:'Bebas Neue',sans-serif;font-size:26px;color:#ffffff;padding:4px 24px;background:#161616;min-width:90px;text-align:center;border:1px solid #222;border-radius:2px;">{hg} – {ag}</div>
<div style="flex:1;text-align:left;font-family:'Bebas Neue',sans-serif;font-size:24px;letter-spacing:2px;{a_style}">{away}</div>
</div>
</div>"""
        st.markdown(html, unsafe_allow_html=True)

# FOOTER
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div style="text-align:center;font-family:Inter,sans-serif;font-size:10px;letter-spacing:4px;color:#333333;text-transform:uppercase;">Powered by XGBoost · ELO Ratings · football-data.org</div>', unsafe_allow_html=True)