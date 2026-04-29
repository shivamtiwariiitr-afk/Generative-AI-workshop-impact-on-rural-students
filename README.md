# GenAI Literacy Workshop — Analytics Dashboard

**Project:** Impact of GenAI Literacy Workshop on Rural School Students  
**Institution:** Department of Management Studies (DoMS), IIT Roorkee  
**Submitted by:** Shivam Tiwari | 24810061 | MBA (IT)  
**Supervisor:** Prof. Gaurav Dixit | Term 8 Final Year Project | 2024–25  
**Study Site:** Government School, Rishikesh, Uttarakhand  

---

## What This Is

A fully interactive Streamlit analytics dashboard implementing all 7 analytical models
from the project report. Load the included Excel dataset or upload your own.

**7 Dashboard Tabs:**

| Tab | Content |
|-----|---------|
| 📈 Score Overview | Three-stage bar chart, improvement histogram, gender comparison |
| 🏫 Class Analysis | Learning trajectory, % improvement, Class×Gender heatmap |
| 🌐 Digital Divide | Exposure tier bar chart, scatter plot (r = 0.31) |
| 🧠 Cognitive Readiness | Self-efficacy tiers, prior GenAI awareness vs. after score |
| 📐 Statistics | Box plots, paired t-test results, correlation matrix |
| 🖐 Interaction Modes | Model 7 (NEW) — mode performance + adoption shift charts |
| 📋 Raw Data | Filterable table + CSV download |

---

## Quick Start

### Option A — Local

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Option B — Streamlit Community Cloud (Free, No Server)

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app → From GitHub repo**
4. Select your repo, set `app.py` as the main file
5. Click **Deploy** — done in ~60 seconds

---

## Files

```
genai_workshop_dashboard/
├── app.py                      # Main Streamlit application (all 7 tabs)
├── requirements.txt            # Python dependencies
├── GenAI_Workshop_Data.xlsx    # 28-student dataset with all variables
└── README.md                   # This file
```

---

## Dataset Columns

| Column | Description |
|--------|-------------|
| Student_ID | Unique ID (S01–S28) |
| Name | Student first name |
| Class | School class (6, 7, 8, 9, 10) |
| Gender | M / F |
| Age | Student age |
| Internet_Freq | Internet access frequency (0–3) |
| Smartphone_Use | Smartphone usage frequency (0–3) |
| Prior_AI_Aware | Prior AI awareness (0=None, 1=Heard, 2=Used) |
| Prior_AI_Used | Directly used Gemini/ChatGPT (0/1) |
| Google_Search | Academic Google search habit (0–2) |
| Voice_Search_Freq | Voice search usage (0–2) |
| HW_Digital_Use | Homework digital tool use (0–2) |
| Manual_Score | Unaided task score (0–10) |
| Before_Score | AI-assisted score BEFORE training (0–10) |
| After_Score | AI-assisted score AFTER training (0–10) |
| Interaction_Mode_Before | Mode used before training |
| Interaction_Mode_After | Mode used after training |
| Digital_Score | Composite digital exposure score (auto-computed) |
| Efficacy_Score | Self-efficacy proxy score (auto-computed) |
| Improvement | After – Before score (auto-computed) |
| Notes | Qualitative field notes |

---

## Theoretical Frameworks Implemented

- **ELM (Elaboration Likelihood Model)** — Cognitive readiness analysis
- **Bourdieu's Digital Capital** — Digital divide / tier analysis  
- **Bandura's Self-Efficacy** — Efficacy proxy score computation
- **Vygotsky's ZPD** — Medium-tier gain interpretation
- **Transfer of Learning** — Prior AI awareness vs. after-score
- **Human-AI Interaction** — Interaction mode capture (Model 7)

---

## Key Statistical Results (28 students)

| Metric | Value |
|--------|-------|
| Mean Before Training | 5.89 / 10 |
| Mean After Training | 8.18 / 10 |
| Mean Improvement | +2.29 points |
| Students Improved | 71.4% |
| Paired t-statistic | 6.788 |
| p-value | < 0.001 |
| Cohen's d | 0.90 (Large Effect) |
| Digital Score Pearson r | 0.31 (p < 0.05) |
