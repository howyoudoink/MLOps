import requests
import plotly.graph_objects as go
import streamlit as st

API_URL = "http://localhost:8000/predict"
API_KEY = "mysecretkey"

st.set_page_config(
    page_title="Automotive Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    :root {
        --text: #0f172a;
        --muted: #64748b;
        --blue: #2d6cdf;
        --card: rgba(255, 255, 255, 0.96);
        --border: rgba(15, 23, 42, 0.08);
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
        color: var(--text);
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(45,108,223,0.07), transparent 30%),
            radial-gradient(circle at 95% 10%, rgba(15,23,42,0.05), transparent 28%),
            linear-gradient(180deg, #ffffff 0%, #f5f7fb 100%);
    }

    .block-container {
        max-width: 1380px;
        padding-top: 1.4rem;
        padding-bottom: 1.5rem;
    }

    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] label,
    [data-testid="stSlider"] label,
    [data-testid="stSelectSlider"] label,
    [data-testid="stForm"] label,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stSlider"] p,
    [data-testid="stSelectSlider"] p {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    .hero {
        text-align: center;
        margin-bottom: 0.7rem;
        animation: rise 550ms ease-out both;
    }

    .hero h1 {
        margin: 0;
        font-size: clamp(2.4rem, 4.6vw, 4.2rem);
        line-height: 1.06;
        letter-spacing: -0.04em;
        color: #0f172a;
    }

    .hero p {
        margin: 0.7rem auto 0;
        max-width: 760px;
        color: #64748b;
        font-size: 1rem;
    }

    .divider {
        margin: 1rem auto 0;
        width: min(900px, 92vw);
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(15,23,42,0.12), transparent);
    }

    .top-band {
        border: 1px solid var(--border);
        background: var(--card);
        border-radius: 20px;
        padding: 0.85rem;
        box-shadow: 0 16px 34px rgba(15, 23, 42, 0.06);
        animation: rise 680ms ease-out both;
        margin-bottom: 1rem;
    }

    .top-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.7rem;
    }

    .mini-card {
        border-radius: 14px;
        border: 1px solid rgba(15,23,42,0.08);
        background: linear-gradient(180deg, #ffffff, #f8fafc);
        padding: 0.75rem 0.8rem;
    }

    .mini-card .kicker {
        display: block;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        color: #64748b;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .mini-card .datum {
        display: block;
        color: #0f172a;
        font-size: 0.92rem;
        font-weight: 700;
    }

    .section-title {
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.82rem;
        margin-bottom: 0.35rem;
    }

    .glass-card {
        border: 1px solid var(--border);
        background: var(--card);
        border-radius: 22px;
        padding: 1rem;
        box-shadow: 0 16px 36px rgba(15, 23, 42, 0.06);
        animation: rise 700ms ease-out both;
    }

    .big-score {
        color: #0f172a;
        font-size: clamp(2.5rem, 5.5vw, 5rem);
        line-height: 0.95;
        letter-spacing: -0.05em;
        font-weight: 800;
        margin: 0.15rem 0 0;
    }

    .risk-pill {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.45rem 0.8rem;
        font-size: 0.84rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }

    .risk-low { background: rgba(34,197,94,0.12); color: #15803d; }
    .risk-medium { background: rgba(249,115,22,0.12); color: #c2410c; }
    .risk-high { background: rgba(239,68,68,0.12); color: #b91c1c; }

    .stMetric {
        border: 1px solid var(--border);
        border-radius: 16px;
        background: linear-gradient(180deg, #ffffff, #f8fafc);
        padding: 0.65rem 0.8rem;
    }

    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricValue"] {
        color: #0f172a !important;
    }

    div[data-testid="stProgress"] > div {
        height: 10px;
        border-radius: 999px;
        background: rgba(45,108,223,0.12);
    }

    div[data-testid="stProgress"] > div > div {
        border-radius: 999px;
        background: linear-gradient(90deg, #2d6cdf, #5d8ff0);
    }

    .timeline {
        margin-top: 0.7rem;
        display: grid;
        gap: 0.55rem;
    }

    .timeline-row {
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 14px;
        background: rgba(255,255,255,0.9);
        padding: 0.65rem 0.75rem;
        display: grid;
        grid-template-columns: 4.3rem 1fr;
        gap: 0.65rem;
        align-items: center;
    }

    .timeline-time {
        color: #64748b;
        font-size: 0.76rem;
        letter-spacing: 0.08em;
        font-weight: 800;
    }

    .timeline-text {
        color: #0f172a;
        font-size: 0.87rem;
        font-weight: 600;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.86rem;
        margin-top: 1rem;
    }

    @keyframes rise {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
"""


def build_gauge(probability: float) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 34, "color": "#0f172a"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#cbd5e1"},
                "bar": {"color": "#2d6cdf", "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#e8f6ee"},
                    {"range": [30, 70], "color": "#fff4e6"},
                    {"range": [70, 100], "color": "#fdecec"},
                ],
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    figure.update_layout(
        margin={"l": 12, "r": 12, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=250,
        font={"family": "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"},
    )
    return figure


def risk_style(risk_level: str) -> tuple[str, str]:
    mapping = {
        "Low": ("risk-pill risk-low", "Low risk - stable operating conditions"),
        "Medium": ("risk-pill risk-medium", "Medium risk - review operating trends"),
        "High": ("risk-pill risk-high", "High risk - immediate action recommended"),
    }
    return mapping.get(risk_level, mapping["Medium"])


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <h1>Automotive Predictive Maintenance</h1>
        <p>Real-time AI-driven failure analysis for manufacturing equipment, presented as a polished product experience.</p>
        <div class="divider"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="top-band">
      <div class="top-grid">
        <div class="mini-card"><span class="kicker">Signal profile</span><span class="datum">Heat, torque, wear, speed</span></div>
        <div class="mini-card"><span class="kicker">Engine mode</span><span class="datum">Real-time API analysis</span></div>
        <div class="mini-card"><span class="kicker">Interface</span><span class="datum">Interactive analysis</span></div>
        <div class="mini-card"><span class="kicker">Motion</span><span class="datum">Smooth reveal animation</span></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.03, 0.97], gap="large")

with left_col:
    st.markdown('<div class="section-title">Sensor Inputs</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    with st.form("prediction_form", border=False):
        col_a, col_b = st.columns(2, gap="medium")

        with col_a:
            air_temp = st.slider("Air Temperature (K)", min_value=250.0, max_value=400.0, value=300.0, step=0.1)
            process_temp = st.slider("Process Temperature (K)", min_value=250.0, max_value=450.0, value=310.0, step=0.1)
            rpm = st.slider("Rotational Speed (RPM)", min_value=500, max_value=3500, value=1500, step=10)

        with col_b:
            torque = st.slider("Torque (Nm)", min_value=0.0, max_value=120.0, value=40.0, step=0.1)
            tool_wear = st.slider("Tool Wear (min)", min_value=0, max_value=300, value=120, step=1)
            machine_type_label = st.select_slider("Machine Type", options=["Low", "Medium", "High"], value="Medium")

        btn_l, btn_c, btn_r = st.columns([1, 1.2, 1])
        with btn_c:
            submit = st.form_submit_button("Analyze")

    stat_cols = st.columns(4, gap="small")
    stats = [
        ("Temp drift", f"{process_temp - air_temp:.1f} K"),
        ("Load proxy", f"{(rpm * torque) / 1000:.1f}"),
        ("Wear ratio", f"{tool_wear / 300:.2f}"),
        ("Machine", machine_type_label),
    ]
    for col, (title, value) in zip(stat_cols, stats):
        with col:
            st.markdown(
                f'<div class="mini-card"><span class="kicker">{title}</span><span class="datum">{value}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-title">Model Output</div>', unsafe_allow_html=True)

if submit:
    payload = {
        "air_temperature": float(air_temp),
        "process_temperature": float(process_temp),
        "rotational_speed": float(rpm),
        "torque": float(torque),
        "tool_wear": float(tool_wear),
        "machine_type": ["Low", "Medium", "High"].index(machine_type_label),
    }

    headers = {"X-API-Key": API_KEY}

    try:
        with st.spinner("Analyzing machine condition..."):
            response = requests.post(API_URL, json=payload, headers=headers, timeout=15)

        with right_col:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)

            if response.status_code == 200:
                result = response.json()
                prob = float(result["failure_probability"])
                risk = result["risk_level"]
                rec = result["recommendation"]
                risk_class, risk_message = risk_style(risk)
                percent = prob * 100

                st.markdown(f'<div class="big-score">{percent:.1f}%</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="{risk_class}">{risk} risk</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="color:#64748b; margin-top:0.35rem;">{risk_message}</div>', unsafe_allow_html=True)

                g_col, s_col = st.columns([1.03, 0.97], gap="medium")
                with g_col:
                    st.plotly_chart(build_gauge(prob), use_container_width=True, config={"displayModeBar": False})
                with s_col:
                    st.metric("Failure Probability", f"{percent:.2f}%")
                    st.progress(min(max(prob, 0), 1))
                    st.markdown(
                        f"""
                        <div class="timeline">
                            <div class="timeline-row"><span class="timeline-time">NOW</span><span class="timeline-text">{rec}</span></div>
                            <div class="timeline-row"><span class="timeline-time">RISK</span><span class="timeline-text">Current risk band: {risk}</span></div>
                            <div class="timeline-row"><span class="timeline-time">NEXT</span><span class="timeline-text">Re-run with updated sensor values</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                info_cols = st.columns(4, gap="small")
                info = [
                    ("Air", f"{air_temp:.1f} K"),
                    ("Process", f"{process_temp:.1f} K"),
                    ("RPM", f"{rpm:.0f}"),
                    ("Torque", f"{torque:.1f} Nm"),
                ]
                for col, (k, v) in zip(info_cols, info):
                    with col:
                        st.markdown(
                            f'<div class="mini-card"><span class="kicker">{k}</span><span class="datum">{v}</span></div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.error(f"API Error: {response.text}")

            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        with right_col:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.error(f"Connection Error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
else:
    with right_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="big-score" style="font-size:2.4rem;">Ready</div>', unsafe_allow_html=True)
        st.metric("Failure Probability", "0.00%")
        st.progress(0)
        st.markdown(
            """
            <div class="timeline">
                <div class="timeline-row"><span class="timeline-time">01</span><span class="timeline-text">Set sensor values on the left panel</span></div>
                <div class="timeline-row"><span class="timeline-time">02</span><span class="timeline-text">Click Analyze to trigger model inference</span></div>
                <div class="timeline-row"><span class="timeline-time">03</span><span class="timeline-text">Review probability, risk band, and recommendation</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer"> Predictive Maintenance System</div>', unsafe_allow_html=True)
