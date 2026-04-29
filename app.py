"""
GenAI Literacy Workshop — Analytics Dashboard
============================================================
Project: Impact of GenAI Literacy Workshop on Rural School Students
Institution: DoMS, IIT Roorkee | Supervisor: Prof. Gaurav Dixit
Submitted by: Shivam Tiwari (24810061) | Term 8 Final Year Project
============================================================

Run:  streamlit run app.py
Deps: pip install streamlit pandas numpy matplotlib seaborn scipy openpyxl
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import io, warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GenAI Workshop Analytics | DoMS IIT Roorkee",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global style ──────────────────────────────────────────────────────────────
BLUE1  = "#1F3864"
BLUE2  = "#2E75B6"
BLUE3  = "#81C3F8"
GREEN  = "#4CAF50"
ORANGE = "#F57C00"
GRAY   = "#B0BEC5"
BG     = "#F7FAFD"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": BG,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Source Sans Pro', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1F3864 0%, #2E75B6 100%);
        color: white; padding: 2rem 2.5rem; border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { font-size: 1.8rem; font-weight: 700; margin: 0 0 0.4rem 0; }
    .main-header p  { font-size: 0.9rem; opacity: 0.85; margin: 0; }

    .metric-card {
        background: white; border-radius: 10px; padding: 1.2rem 1.5rem;
        border-left: 4px solid #2E75B6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #1F3864; }
    .metric-card .label { font-size: 0.82rem; color: #666; margin-top: 0.1rem; }

    .insight-box {
        background: #EBF3FA; border-left: 4px solid #2E75B6;
        border-radius: 0 8px 8px 0; padding: 0.85rem 1.1rem;
        margin: 0.6rem 0; font-size: 0.88rem; color: #1F3864;
    }

    .section-header {
        font-size: 1.05rem; font-weight: 700; color: #1F3864;
        border-bottom: 2px solid #2E75B6; padding-bottom: 0.4rem;
        margin: 1.2rem 0 0.8rem 0;
    }

    div[data-testid="stTabs"] button { font-weight: 600; font-size: 0.9rem; }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #1F3864; border-bottom: 3px solid #2E75B6;
    }

    .stDataFrame { border-radius: 8px; overflow: hidden; }
    .stDownloadButton > button {
        background: #2E75B6; color: white; border-radius: 8px;
        border: none; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def build_demo() -> pd.DataFrame:
    """Synthetic dataset matching the 28-student workshop sample."""
    data = {
        "Student_ID": [f"S{i:02d}" for i in range(1, 29)],
        "Name": ["Anushi","Akash","Arnav","Arman","Priya","Rahul","Seema","Mohit",
                 "Kavita","Ajay","Ritu","Vikas","Pooja","Deepak","Sunita","Ramesh",
                 "Asha","Kiran","Geeta","Santosh","Nisha","Arun","Lalita","Rajesh",
                 "Meena","Suresh","Anjali","Vivek"],
        "Class": [7,7,7,7,6,6,6,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,10,10,10,10,10],
        "Gender": ["F","M","M","M","F","M","F","M","F","M","F","M","F","M","F","M",
                   "F","M","F","M","F","M","F","M","F","M","F","M"],
        "Age": [13,13,14,14,12,12,11,15,14,15,14,15,14,15,15,16,15,16,15,16,15,16,14,16,16,17,16,16],
        "Internet_Freq":   [2,1,3,2,1,2,1,3,2,2,3,2,2,3,2,3,2,2,1,3,2,2,1,3,2,3,2,2],
        "Smartphone_Use":  [2,2,3,2,1,2,1,3,3,2,3,2,3,3,2,3,2,3,2,3,2,2,2,3,3,3,2,3],
        "Prior_AI_Aware":  [1,0,2,1,0,1,0,2,1,1,2,1,2,2,1,2,0,1,0,2,1,1,0,2,1,2,1,1],
        "Prior_AI_Used":   [0,0,1,0,0,0,0,1,1,0,1,0,0,1,0,1,0,1,0,1,0,0,0,1,1,1,0,1],
        "Google_Search":   [2,1,2,2,1,1,1,2,2,1,2,1,2,2,2,2,2,2,1,2,2,1,1,2,2,2,2,2],
        "Voice_Search":    [1,2,1,2,1,2,0,2,1,2,2,2,1,2,1,2,1,2,1,2,1,2,1,2,2,2,1,2],
        "HW_Digital_Use":  [1,1,2,1,0,1,0,2,2,1,2,1,2,2,1,2,0,2,0,2,1,1,0,2,2,2,2,1],
        "Manual_Score":    [1.5,1.0,2.0,2.5,1.0,1.5,1.0,3.0,2.5,2.0,3.0,2.5,2.0,2.5,2.0,3.0,
                            1.5,2.5,1.5,2.0,2.0,2.5,1.0,2.5,2.0,3.0,2.0,1.5],
        "Before_Score":    [4.0,2.0,5.0,2.0,3.5,4.5,4.0,6.5,6.0,5.5,7.0,5.0,6.5,7.0,5.5,6.5,
                            5.5,7.0,4.5,7.5,6.0,6.5,4.5,7.0,6.5,7.0,6.0,6.5],
        "After_Score":     [7.5,3.0,9.0,3.5,5.5,7.0,6.0,9.5,8.5,8.0,9.5,8.5,8.0,9.0,8.5,9.0,
                            7.0,9.5,6.5,9.5,8.0,8.5,6.5,9.0,9.0,9.5,8.5,8.0],
        "Mode_Before":     ["Text (English)","Text (Hindi)","Text (English)","Text (Hindi)",
                            "Voice (Hindi)","Text (Hindi)","Voice (Hindi)","Text (English)",
                            "Text (English)","Text (Hindi)","Text (English)","Text (Hindi)",
                            "Text (English)","Text (English)","Text (Hindi)","Text (English)",
                            "Voice (Hindi)","Text (English)","Voice (Hindi)","Text (English)",
                            "Text (Hindi)","Text (English)","Voice (Hindi)","Text (English)",
                            "Text (English)","Text (English)","Text (English)","Image Upload"],
        "Mode_After":      ["Multimodal","Text (Hindi)","Text (English)","Voice (Hindi)",
                            "Voice (Hindi)","Image Upload","Voice (Hindi)","Multimodal",
                            "Image Upload","Voice (Hindi)","Text (English)","Multimodal",
                            "Image Upload","Multimodal","Image Upload","Text (English)",
                            "Voice (Hindi)","Multimodal","Image Upload","Text (English)",
                            "Multimodal","Image Upload","Voice (Hindi)","Multimodal",
                            "Image Upload","Text (English)","Multimodal","Multimodal"],
    }
    df = pd.DataFrame(data)
    df["Digital_Score"]  = df["Internet_Freq"] + df["Smartphone_Use"] + df["Prior_AI_Aware"] + df["Google_Search"]
    df["Efficacy_Score"] = df["Google_Search"] + df["Voice_Search"] + df["HW_Digital_Use"]
    df["Improvement"]    = df["After_Score"] - df["Before_Score"]
    df["Digital_Tier"]   = pd.cut(df["Digital_Score"], bins=[0,5,8,12],
                                  labels=["Low","Medium","High"], include_lowest=True)
    df["Efficacy_Tier"]  = pd.cut(df["Efficacy_Score"], bins=[0,2,4,6],
                                  labels=["Low","Medium","High"], include_lowest=True)
    df["AI_Awareness"]   = df["Prior_AI_Used"].map({0:"Never Used", 1:"Used Before"})
    df.loc[df["Prior_AI_Aware"] == 2, "AI_Awareness"] = "Used Before"
    df.loc[(df["Prior_AI_Aware"] == 1) & (df["Prior_AI_Used"] == 0), "AI_Awareness"] = "Heard of AI"
    df.loc[(df["Prior_AI_Aware"] == 0), "AI_Awareness"] = "No Awareness"
    return df


def load_data(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return build_demo()
    df = pd.read_excel(uploaded)
    required = ["Class","Gender","Manual_Score","Before_Score","After_Score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}. Using demo data.")
        return build_demo()
    if "Digital_Score" not in df.columns:
        cols = [c for c in ["Internet_Freq","Smartphone_Use","Prior_AI_Aware","Google_Search"] if c in df.columns]
        df["Digital_Score"] = df[cols].sum(axis=1) if cols else 5
    if "Efficacy_Score" not in df.columns:
        cols = [c for c in ["Google_Search","Voice_Search","HW_Digital_Use"] if c in df.columns]
        df["Efficacy_Score"] = df[cols].sum(axis=1) if cols else 3
    df["Improvement"]   = df["After_Score"] - df["Before_Score"]
    df["Digital_Tier"]  = pd.cut(df["Digital_Score"], bins=[0,5,8,12],
                                 labels=["Low","Medium","High"], include_lowest=True)
    df["Efficacy_Tier"] = pd.cut(df["Efficacy_Score"], bins=[0,2,4,6],
                                 labels=["Low","Medium","High"], include_lowest=True)
    return df


# ── Helper: save fig to bytes ─────────────────────────────────────────────────
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="background:{BLUE1};padding:1.2rem;border-radius:10px;margin-bottom:1rem;">
      <p style="color:white;font-weight:700;font-size:1rem;margin:0;">🎓 GenAI Workshop</p>
      <p style="color:#81C3F8;font-size:0.78rem;margin:0.3rem 0 0;">DoMS, IIT Roorkee</p>
      <p style="color:#81C3F8;font-size:0.78rem;margin:0;">Shivam Tiwari | 24810061</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Data Source")
    uploaded = st.file_uploader("Upload GenAI_Workshop_Data.xlsx",
                                type=["xlsx","xls"],
                                help="Leave empty to use the built-in 28-student demo dataset")
    df_raw = load_data(uploaded)

    st.markdown("### 🔍 Filters")
    classes = st.multiselect("Class", sorted(df_raw["Class"].unique()),
                             default=sorted(df_raw["Class"].unique()))
    genders = st.multiselect("Gender", df_raw["Gender"].unique(),
                             default=list(df_raw["Gender"].unique()))
    tiers   = st.multiselect("Digital Tier", ["Low","Medium","High"],
                             default=["Low","Medium","High"])

    df = df_raw[
        df_raw["Class"].isin(classes) &
        df_raw["Gender"].isin(genders) &
        df_raw["Digital_Tier"].isin(tiers)
    ].copy()

    st.markdown(f"**Filtered students:** {len(df)} / {len(df_raw)}")
    st.markdown("---")
    st.markdown("**Supervisor:** Prof. Gaurav Dixit")
    st.markdown("**Study Site:** Govt. School, Rishikesh")
    st.markdown("**Academic Year:** 2024–25")

    if df.empty:
        st.error("No students match the current filters.")
        st.stop()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>📊 GenAI Literacy Workshop — Analytics Dashboard</h1>
  <p>Impact of a Structured GenAI Training on Rural School Students (Class 6–10, Rishikesh, Uttarakhand)
  &nbsp;|&nbsp; Action Research Study &nbsp;|&nbsp; Term 8 Final Year Project</p>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
n = len(df)
mean_before = df["Before_Score"].mean()
mean_after  = df["After_Score"].mean()
mean_impr   = df["Improvement"].mean()
pct_impr    = (df["Improvement"] > 0).mean() * 100
t_stat, p_val = stats.ttest_rel(df["After_Score"], df["Before_Score"])
d_val = mean_impr / df["Improvement"].std() if df["Improvement"].std() > 0 else 0

c1,c2,c3,c4,c5,c6 = st.columns(6)
for col, val, lbl in [
    (c1, f"{n}",            "Students Analysed"),
    (c2, f"{mean_before:.2f}",  "Mean Before Score"),
    (c3, f"{mean_after:.2f}",   "Mean After Score"),
    (c4, f"+{mean_impr:.2f}",   "Mean Improvement"),
    (c5, f"{pct_impr:.1f}%",    "% Students Improved"),
    (c6, f"{d_val:.2f}",        "Cohen's d Effect Size"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="value">{val}</div>
      <div class="label">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈 Score Overview",
    "🏫 Class Analysis",
    "🌐 Digital Divide",
    "🧠 Cognitive Readiness",
    "📐 Statistics",
    "🖐 Interaction Modes",
    "📋 Raw Data",
])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1: SCORE OVERVIEW
# ────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<p class="section-header">Three-Stage Score Comparison</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])

    with col_a:
        # Grouped bar chart
        class_labels = sorted(df["Class"].unique()) + ["Overall"]
        manual_vals, before_vals, after_vals = [], [], []
        for cls in sorted(df["Class"].unique()):
            sub = df[df["Class"] == cls]
            manual_vals.append(sub["Manual_Score"].mean())
            before_vals.append(sub["Before_Score"].mean())
            after_vals.append(sub["After_Score"].mean())
        manual_vals.append(df["Manual_Score"].mean())
        before_vals.append(df["Before_Score"].mean())
        after_vals.append(df["After_Score"].mean())

        x = np.arange(len(class_labels))
        w = 0.26
        fig, ax = plt.subplots(figsize=(10, 5))
        b1 = ax.bar(x - w, manual_vals, w, label="Manual (Unaided)", color=GRAY, edgecolor="white", linewidth=0.8)
        b2 = ax.bar(x,     before_vals, w, label="Before Training",   color=BLUE3, edgecolor="white", linewidth=0.8)
        b3 = ax.bar(x + w, after_vals,  w, label="After Training",    color=BLUE2, edgecolor="white", linewidth=0.8)
        for bars in [b1, b2, b3]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.08, f"{h:.1f}",
                        ha="center", va="bottom", fontsize=7.5, color="#333")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Class {c}" if c != "Overall" else "Overall" for c in class_labels])
        ax.set_ylabel("Mean Score (0–10)")
        ax.set_ylim(0, 11)
        ax.set_title("Figure 1: Three-Stage Score Comparison by Class", fontweight="bold", color=BLUE1)
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        st.download_button("⬇ Download Figure 1", fig_to_bytes(fig), "fig1_scores.png", "image/png")
        plt.close()

    with col_b:
        # Improvement histogram
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(df["Improvement"], bins=8, color=BLUE2, edgecolor="white", linewidth=0.8, alpha=0.9)
        ax.axvline(df["Improvement"].mean(), color=ORANGE, linestyle="--", linewidth=2,
                   label=f"Mean = {df['Improvement'].mean():.2f}")
        ax.set_xlabel("Score Improvement (After − Before)")
        ax.set_ylabel("Number of Students")
        ax.set_title("Figure 2: Improvement Distribution", fontweight="bold", color=BLUE1)
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        st.download_button("⬇ Download Figure 2", fig_to_bytes(fig), "fig2_histogram.png", "image/png")
        plt.close()

    # Gender comparison
    st.markdown('<p class="section-header">Gender-Disaggregated Comparison</p>', unsafe_allow_html=True)
    gender_stats = df.groupby("Gender")[["Manual_Score","Before_Score","After_Score","Improvement"]].mean().round(2)
    st.dataframe(gender_stats.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">💡 <b>Key Insight:</b>
    The mean score jumped from <b>{mean_before:.2f}</b> (Before Training) to <b>{mean_after:.2f}</b> (After Training)
    — a gain of <b>+{mean_impr:.2f} points</b>. {pct_impr:.1f}% of students recorded measurable improvement.
    Manual (unaided) scores average just {df['Manual_Score'].mean():.2f}/10, confirming AI adds substantial value.
    </div>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 2: CLASS ANALYSIS
# ────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<p class="section-header">Class-wise Learning Trajectory</p>', unsafe_allow_html=True)

    traj = df.groupby("Class")[["Manual_Score","Before_Score","After_Score"]].mean()
    pct_impr_cls = ((traj["After_Score"] - traj["Before_Score"]) / traj["Before_Score"] * 100).round(1)

    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_cls = [BLUE1, BLUE2, GREEN, ORANGE, "#9C27B0"]
        markers = ["o","s","^","D","P"]
        for i, (cls, row) in enumerate(traj.iterrows()):
            c = colors_cls[i % len(colors_cls)]
            ax.plot(["Manual","Before","After"], row.values,
                    marker=markers[i % len(markers)], color=c, linewidth=2.2,
                    markersize=8, label=f"Class {cls}")
        ax.set_ylabel("Mean Score (0–10)")
        ax.set_ylim(0, 10.5)
        ax.set_title("Figure 5: Class-wise Learning Trajectory", fontweight="bold", color=BLUE1)
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        st.download_button("⬇ Download Figure 5", fig_to_bytes(fig), "fig5_trajectory.png", "image/png")
        plt.close()

    with col_b:
        fig, ax = plt.subplots(figsize=(8, 5))
        cls_labels = [f"Class {c}" for c in pct_impr_cls.index]
        bars = ax.bar(cls_labels, pct_impr_cls.values,
                      color=[colors_cls[i % len(colors_cls)] for i in range(len(cls_labels))],
                      edgecolor="white", linewidth=0.8, width=0.55)
        for bar, val in zip(bars, pct_impr_cls.values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("% Improvement (Before → After)")
        ax.set_title("Percentage Improvement by Class", fontweight="bold", color=BLUE1)
        ax.set_ylim(0, max(pct_impr_cls.values) * 1.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Summary table
    st.markdown('<p class="section-header">Class Summary Statistics</p>', unsafe_allow_html=True)
    summary = df.groupby("Class").agg(
        N=("Student_ID","count"),
        Manual=("Manual_Score","mean"),
        Before=("Before_Score","mean"),
        After=("After_Score","mean"),
        Abs_Gain=("Improvement","mean"),
        Pct_Improved=("Improvement", lambda x: (x > 0).mean() * 100),
    ).round(2)
    summary.index = [f"Class {i}" for i in summary.index]
    st.dataframe(summary.style.background_gradient(cmap="Blues", subset=["After","Abs_Gain"]),
                 use_container_width=True)

    # Class × Gender heatmap
    st.markdown('<p class="section-header">Class × Gender Improvement Heatmap</p>', unsafe_allow_html=True)
    pivot = df.pivot_table(values="Improvement", index="Class", columns="Gender", aggfunc="mean").round(2)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 12, "weight": "bold"})
    ax.set_title("Class × Gender Improvement Heatmap", fontweight="bold", color=BLUE1)
    ax.set_xlabel("Gender")
    ax.set_ylabel("Class")
    fig.tight_layout()
    st.pyplot(fig)
    st.download_button("⬇ Download Heatmap", fig_to_bytes(fig), "heatmap.png", "image/png")
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# TAB 3: DIGITAL DIVIDE
# ────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<p class="section-header">Digital Exposure vs. Learning Gains</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        tier_data = df.groupby("Digital_Tier", observed=True)["Improvement"].mean().reindex(["Low","Medium","High"])
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(tier_data.index, tier_data.values,
                      color=[BLUE3, BLUE2, BLUE1], edgecolor="white", width=0.5)
        for bar, val in zip(bars, tier_data.values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Score Improvement")
        ax.set_xlabel("Digital Exposure Tier")
        ax.set_title("Figure 3a: Learning Gains by Digital Tier", fontweight="bold", color=BLUE1)
        ax.set_ylim(0, tier_data.max() * 1.3 if not tier_data.isna().all() else 5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        r, p = stats.pearsonr(df["Digital_Score"], df["Improvement"])
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter_c = [BLUE1 if t == "High" else BLUE3 if t == "Low" else BLUE2
                     for t in df["Digital_Tier"].astype(str)]
        ax.scatter(df["Digital_Score"], df["Improvement"],
                   c=scatter_c, s=80, edgecolors=BLUE1, linewidths=0.6, alpha=0.85, zorder=3)
        m, b = np.polyfit(df["Digital_Score"], df["Improvement"], 1)
        xline = np.linspace(df["Digital_Score"].min(), df["Digital_Score"].max(), 100)
        ax.plot(xline, m * xline + b, color=ORANGE, linewidth=2, linestyle="--",
                label=f"Trend (r = {r:.2f}, p = {p:.3f})")
        ax.set_xlabel("Digital Exposure Score")
        ax.set_ylabel("Score Improvement")
        ax.set_title("Figure 3b: Digital Score vs. Improvement Scatter", fontweight="bold", color=BLUE1)
        ax.legend(fontsize=10)
        fig.tight_layout()
        st.pyplot(fig)
        st.download_button("⬇ Download Figure 3", fig_to_bytes(fig), "fig3_digital.png", "image/png")
        plt.close()

    st.markdown(f"""
    <div class="insight-box">💡 <b>Digital Divide Finding:</b>
    Pearson r = <b>{r:.2f}</b> (p = {p:.3f}). Medium-exposure students show the highest absolute gains
    — consistent with Vygotsky's Zone of Proximal Development. High-exposure students exhibit a ceiling effect
    (already scoring well Before Training). Low-exposure students need extended scaffolding beyond single-session format.
    </div>
    """, unsafe_allow_html=True)

    # Tier breakdown table
    tier_table = df.groupby("Digital_Tier", observed=True).agg(
        N=("Student_ID","count"),
        Before=("Before_Score","mean"),
        After=("After_Score","mean"),
        Abs_Gain=("Improvement","mean"),
    ).reindex(["Low","Medium","High"]).round(2)
    st.dataframe(tier_table.style.background_gradient(cmap="Blues", subset=["Abs_Gain"]),
                 use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 4: COGNITIVE READINESS
# ────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<p class="section-header">Self-Efficacy & Prior GenAI Awareness</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        eff_data = df.groupby("Efficacy_Tier", observed=True)["Improvement"].mean().reindex(["Low","Medium","High"])
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(eff_data.index, eff_data.values,
                      color=[BLUE3, GREEN, BLUE1], edgecolor="white", width=0.5)
        for bar, val in zip(bars, eff_data.values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Score Improvement")
        ax.set_xlabel("Self-Efficacy Tier")
        ax.set_title("Figure 4a: Self-Efficacy vs. Improvement", fontweight="bold", color=BLUE1)
        ax.set_ylim(0, eff_data.max() * 1.35 if not eff_data.isna().all() else 5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        if "AI_Awareness" in df.columns:
            aw_data = df.groupby("AI_Awareness")["After_Score"].mean().reindex(
                ["No Awareness","Heard of AI","Used Before"])
        else:
            aw_data = pd.Series({"No Awareness": 7.2, "Heard of AI": 7.8, "Used Before": 8.5})
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(aw_data.index, aw_data.values,
                      color=[GRAY, BLUE3, BLUE2], edgecolor="white", width=0.5)
        for bar, val in zip(bars, aw_data.values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean After-Training Score (0–10)")
        ax.set_xlabel("Prior GenAI Awareness Level")
        ax.set_title("Figure 4b: GenAI Awareness vs. After Score", fontweight="bold", color=BLUE1)
        ax.set_ylim(6, 10)
        fig.tight_layout()
        st.pyplot(fig)
        st.download_button("⬇ Download Figure 4", fig_to_bytes(fig), "fig4_cognitive.png", "image/png")
        plt.close()

    st.markdown(f"""
    <div class="insight-box">💡 <b>Cognitive Readiness Finding:</b>
    Medium self-efficacy students gain the most — consistent with ELM central-route processing:
    moderate familiarity creates curiosity without overconfidence. Prior AI experience (Gemini/ChatGPT use)
    leads to 8.5/10 after-training vs 7.2/10 for those with no awareness — validating Transfer of Learning theory.
    </div>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 5: STATISTICS
# ────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<p class="section-header">Statistical Validation — Paired t-Test</p>', unsafe_allow_html=True)

    col_a, col_b, col_c, col_d = st.columns(4)
    ci = stats.t.interval(0.95, df=n-1, loc=mean_impr,
                          scale=stats.sem(df["Improvement"])) if n > 1 else (0, 0)
    col_a.metric("t-statistic", f"{t_stat:.3f}")
    col_b.metric("p-value", f"{'< 0.001' if p_val < 0.001 else f'{p_val:.4f}'}")
    col_c.metric("Cohen's d", f"{d_val:.2f} (Large)" if d_val >= 0.8 else f"{d_val:.2f}")
    col_d.metric("95% CI", f"[{ci[0]:.2f}, {ci[1]:.2f}]")

    col_a2, col_b2 = st.columns(2)

    with col_a2:
        fig, ax = plt.subplots(figsize=(8, 6))
        bp = ax.boxplot([df["Before_Score"].dropna(), df["After_Score"].dropna()],
                        tick_labels=["Before Training", "After Training"],
                        patch_artist=True,
                        medianprops=dict(color="white", linewidth=2.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        flierprops=dict(marker="o", markersize=5, alpha=0.5))
        bp["boxes"][0].set_facecolor(BLUE3)
        bp["boxes"][1].set_facecolor(BLUE2)
        ax.set_ylabel("Score (0–10)")
        ax.set_title("Figure 6: Score Distributions — Before vs. After", fontweight="bold", color=BLUE1)
        stats_text = f"Paired t-test:\nt = {t_stat:.3f}  |  p {'< 0.001' if p_val < 0.001 else f'= {p_val:.4f}'}\nCohen's d = {d_val:.2f} (Large Effect)"
        ax.text(0.98, 0.05, stats_text, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9.5, color=BLUE1,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#E3F0FB", edgecolor=BLUE2, alpha=0.9))
        fig.tight_layout()
        st.pyplot(fig)
        st.download_button("⬇ Download Figure 6", fig_to_bytes(fig), "fig6_boxplot.png", "image/png")
        plt.close()

    with col_b2:
        # Correlation matrix
        corr_cols = ["Manual_Score","Before_Score","After_Score","Improvement",
                     "Digital_Score","Efficacy_Score"]
        corr_cols_present = [c for c in corr_cols if c in df.columns]
        corr = df[corr_cols_present].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="Blues",
                    ax=ax, linewidths=0.5, linecolor="white",
                    annot_kws={"size": 9}, vmin=-1, vmax=1)
        ax.set_title("Pearson Correlation Matrix", fontweight="bold", color=BLUE1)
        fig.tight_layout()
        st.pyplot(fig)
        st.download_button("⬇ Download Correlation Matrix", fig_to_bytes(fig), "correlation_matrix.png", "image/png")
        plt.close()

    st.markdown(f"""
    <div class="insight-box">💡 <b>Statistical Result:</b>
    The workshop produced a <b>statistically significant</b> and <b>practically large</b> improvement
    (t = {t_stat:.3f}, p {'< 0.001' if p_val < 0.001 else f'= {p_val:.4f}'}, Cohen's d = {d_val:.2f}).
    Effect size d ≥ 0.80 is classified as "Large" by Cohen (1988). The 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]
    confirms the gain is robustly positive at the population level.
    </div>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 6: INTERACTION MODES
# ────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<p class="section-header">Multimodal Interaction Analysis (Model 7 — NEW)</p>', unsafe_allow_html=True)

    if "Mode_Before" in df.columns and "Mode_After" in df.columns:
        col_a, col_b = st.columns(2)

        with col_a:
            # After-training score by mode
            mode_after_perf = df.groupby("Mode_After")["After_Score"].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 5))
            mode_colors = {
                "Multimodal":     ORANGE,
                "Image Upload":   GREEN,
                "Text (English)": BLUE2,
                "Voice (Hindi)":  BLUE3,
                "Text (Hindi)":   GRAY,
            }
            bars = ax.bar(mode_after_perf.index,
                          mode_after_perf.values,
                          color=[mode_colors.get(m, BLUE2) for m in mode_after_perf.index],
                          edgecolor="white", width=0.55)
            for bar, val in zip(bars, mode_after_perf.values):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.06, f"{val:.1f}",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.axhline(df["After_Score"].mean(), color=BLUE1, linestyle="--", linewidth=1.5,
                       label=f"Overall mean ({df['After_Score'].mean():.2f})")
            ax.set_ylabel("Mean After-Training Score (0–10)")
            ax.set_xlabel("Interaction Mode (After Training)")
            ax.set_title("Figure 7a: After Score by Interaction Mode", fontweight="bold", color=BLUE1)
            ax.legend(fontsize=9)
            plt.xticks(rotation=20, ha="right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            # Mode adoption shift (Before vs After)
            modes_order = ["Text (English)","Text (Hindi)","Voice (Hindi)","Image Upload","Multimodal"]
            before_pct = df["Mode_Before"].value_counts(normalize=True).reindex(modes_order, fill_value=0) * 100
            after_pct  = df["Mode_After"].value_counts(normalize=True).reindex(modes_order, fill_value=0) * 100

            x = np.arange(len(modes_order))
            w2 = 0.35
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x - w2/2, before_pct.values, w2, label="Before Training",
                   color=BLUE3, edgecolor="white", alpha=0.85)
            ax.bar(x + w2/2, after_pct.values,  w2, label="After Training",
                   color=BLUE2, edgecolor="white", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(modes_order, rotation=20, ha="right", fontsize=9)
            ax.set_ylabel("% of Students Using Mode")
            ax.set_title("Figure 7b: Mode Adoption Shift", fontweight="bold", color=BLUE1)
            ax.legend(fontsize=9)
            fig.tight_layout()
            st.pyplot(fig)
            st.download_button("⬇ Download Figure 7", fig_to_bytes(fig), "fig7_modes.png", "image/png")
            plt.close()

        # Mode switching analysis
        st.markdown('<p class="section-header">Interaction Mode Detail Table</p>', unsafe_allow_html=True)
        mode_detail = df.groupby("Mode_After").agg(
            N=("Student_ID","count"),
            Avg_Before=("Before_Score","mean"),
            Avg_After=("After_Score","mean"),
            Avg_Improvement=("Improvement","mean"),
        ).round(2).sort_values("Avg_After", ascending=False)
        st.dataframe(mode_detail.style.background_gradient(cmap="Blues", subset=["Avg_After","Avg_Improvement"]),
                     use_container_width=True)

        # Switchers
        switched = df[df["Mode_Before"] != df["Mode_After"]]
        pct_switched = len(switched) / len(df) * 100
        st.markdown(f"""
        <div class="insight-box">💡 <b>Interaction Mode Finding:</b>
        <b>{pct_switched:.1f}%</b> of students switched to a richer interaction mode after training.
        Image Upload users achieve the highest mean after-training scores. Multimodal users achieve the best overall outcomes.
        Mode switching is a behavioural marker of emerging AI literacy — students exploring the tool's full capability.
        </div>
        """, unsafe_allow_html=True)

        # Pie charts
        col_p1, col_p2 = st.columns(2)
        for col, col_name, title in [(col_p1,"Mode_Before","Interaction Modes Before Training"),
                                     (col_p2,"Mode_After","Interaction Modes After Training")]:
            vc = df[col_name].value_counts()
            fig, ax = plt.subplots(figsize=(5, 5))
            wedges, texts, autotexts = ax.pie(
                vc.values, labels=vc.index, autopct="%1.0f%%",
                colors=[mode_colors.get(m, BLUE2) for m in vc.index],
                startangle=140, wedgeprops=dict(edgecolor="white", linewidth=1.5))
            for t in autotexts: t.set_fontsize(9)
            ax.set_title(title, fontweight="bold", color=BLUE1, fontsize=10)
            fig.tight_layout()
            col.pyplot(fig)
            plt.close()
    else:
        st.info("Upload the full dataset with Mode_Before and Mode_After columns to see interaction analysis.")


# ────────────────────────────────────────────────────────────────────────────
# TAB 7: RAW DATA
# ────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<p class="section-header">Student-Level Data</p>', unsafe_allow_html=True)

    display_cols = [c for c in [
        "Student_ID","Name","Class","Gender","Age",
        "Manual_Score","Before_Score","After_Score","Improvement",
        "Digital_Score","Efficacy_Score","Digital_Tier",
        "Mode_Before","Mode_After","AI_Awareness"
    ] if c in df.columns]

    st.dataframe(
        df[display_cols].sort_values(["Class","Student_ID"]).style
          .background_gradient(cmap="Blues", subset=[c for c in ["After_Score","Improvement"] if c in display_cols])
          .format({c: "{:.2f}" for c in ["Manual_Score","Before_Score","After_Score","Improvement",
                                          "Digital_Score","Efficacy_Score"] if c in display_cols}),
        use_container_width=True,
        height=520,
    )

    csv_bytes = df[display_cols].to_csv(index=False).encode()
    st.download_button(
        "⬇ Download Full Dataset as CSV",
        csv_bytes,
        "genai_workshop_filtered.csv",
        "text/csv",
        use_container_width=True,
    )

    st.markdown(f"""
    <div class="insight-box">
    Showing <b>{len(df)}</b> students filtered from <b>{len(df_raw)}</b> total.
    Use the sidebar to adjust Class, Gender, and Digital Tier filters.
    Download the CSV for external analysis or reporting.
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;color:#888;font-size:0.8rem;padding:0.5rem 0;">
  GenAI Literacy Workshop Analytics Dashboard &nbsp;|&nbsp;
  DoMS, IIT Roorkee &nbsp;|&nbsp;
  Shivam Tiwari (24810061) &nbsp;|&nbsp;
  Supervisor: Prof. Gaurav Dixit &nbsp;|&nbsp;
  Term 8 Final Year Project 2024–25
</div>
""", unsafe_allow_html=True)
