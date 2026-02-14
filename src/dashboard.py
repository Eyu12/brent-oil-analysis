"""
Interactive Streamlit dashboard for Brent oil analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add src to path for module imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config import AppConfig
from src.data_loader import DataLoader
from src.change_point import ChangePointDetector, ChangePointMethod
from src.risk_metrics import RiskAnalyzer


# -------------------------
# Page configuration
# -------------------------
st.set_page_config(
    page_title="Brent Oil Risk Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; font-weight: bold; margin-bottom: 1rem; }
    .metric-card { background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .risk-high { color: #d62728; font-weight: bold; }
    .risk-medium { color: #ff7f0e; font-weight: bold; }
    .risk-low { color: #2ca02c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# -------------------------
# Data and config loaders
# -------------------------
@st.cache_resource
def load_config():
    """Load app configuration once per session."""
    return AppConfig()


@st.cache_data
def load_data():
    """Load and cache Brent oil data."""
    config = load_config()
    loader = DataLoader(config.data)
    data = loader.load_data()
    summary = loader.get_summary_statistics()
    return data, summary


@st.cache_data
def detect_change_points(data, method: str, penalty: float = 1.0):
    """Detect change points with caching."""
    config = load_config()
    config.model.pelt_penalty = penalty
    detector = ChangePointDetector(config.model)

    method_map = {
        "PELT": ChangePointMethod.PELT,
        "Binary Segmentation": ChangePointMethod.BINARY_SEG,
        "Window Sliding": ChangePointMethod.WINDOW_SLIDING
    }

    change_points = detector.detect(data, method=method_map.get(method, ChangePointMethod.PELT))
    return change_points


@st.cache_data
def calculate_risk_metrics(data):
    """Calculate risk metrics with caching."""
    config = load_config()
    analyzer = RiskAnalyzer(config.risk)
    metrics = analyzer.calculate_metrics(data['Price'])
    summary = analyzer.get_risk_summary()
    return metrics, summary


# -------------------------
# Main dashboard
# -------------------------
def main():
    st.markdown('<p class="main-header">📈 Brent Oil Risk Analytics Dashboard</p>',
                unsafe_allow_html=True)

    # Sidebar navigation & settings
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/oil-industry.png", width=80)
        st.title("Navigation & Settings")

        # Load data
        data, summary = load_data()

        # Change point detection settings
        st.subheader("Change Point Detection")
        method = st.selectbox("Detection Method", ["PELT", "Binary Segmentation", "Window Sliding"])
        sensitivity = st.slider("Detection Sensitivity", min_value=0.5, max_value=3.0, value=1.5, step=0.1)

        # Risk analysis settings
        st.subheader("Risk Analysis")
        confidence_level = st.select_slider("VaR Confidence Level", options=[0.90, 0.95, 0.99], value=0.95)

        # Date filter
        st.subheader("Date Range")
        min_date = data.index.min().date()
        max_date = data.index.max().date()

        start_date = st.date_input("Start Date", value=max_date - timedelta(days=5*365),
                                   min_value=min_date, max_value=max_date)
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

        filtered_data = data[(data.index.date >= start_date) & (data.index.date <= end_date)]

        # Navigation pages
        st.subheader("Sections")
        page = st.radio("Go to", ["Executive Summary", "Change Point Analysis", "Risk Metrics", 
                                 "Stress Testing", "Reports"])

    # Display selected page
    if page == "Executive Summary":
        show_executive_summary(filtered_data, summary, sensitivity, confidence_level)
    elif page == "Change Point Analysis":
        show_change_point_analysis(filtered_data, method, sensitivity)
    elif page == "Risk Metrics":
        show_risk_metrics(filtered_data, confidence_level)
    elif page == "Stress Testing":
        show_stress_testing(filtered_data)
    else:
        show_reports()


# -------------------------
# Page rendering functions
# -------------------------
def show_executive_summary(data, summary, sensitivity, confidence_level):
    st.header("📊 Executive Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_price = data['Price'].iloc[-1]
        total_change = ((current_price / data['Price'].iloc[0]) - 1) * 100
        st.metric("Current Price", f"${current_price:.2f}", f"{total_change:.1f}% overall")

    with col2:
        volatility = data['Price'].pct_change().std() * np.sqrt(252)
        st.metric("Annual Volatility", f"{volatility:.2%}",
                  "High" if volatility > 0.3 else "Moderate" if volatility > 0.2 else "Low")

    with col3:
        change_points = detect_change_points(data, "PELT", penalty=sensitivity)
        st.metric("Change Points", len(change_points),
                  f"{sum(1 for cp in change_points if cp.significance == 'High')} significant")

    with col4:
        returns = data['Price'].pct_change().dropna()
        var_95 = np.percentile(returns, 5) * 100
        st.metric("95% VaR (Daily)", f"{var_95:.2f}%", "Monitor" if var_95 < -2 else "Normal")

    # Price chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Price'], mode='lines', name='Brent Oil Price',
                             line=dict(color='#1f77b4', width=2)))

    for cp in change_points:
        color = '#d62728' if cp.direction == 'Decrease' else '#2ca02c'
        symbol = 'triangle-down' if cp.direction == 'Decrease' else 'triangle-up'
        fig.add_trace(go.Scatter(x=[cp.date], y=[cp.price_after], mode='markers+text',
                                 marker=dict(size=12, color=color, symbol=symbol),
                                 text=[f"{cp.change_pct:.1f}%"],
                                 textposition="top center" if cp.direction == 'Increase' else "bottom center",
                                 showlegend=False))
    fig.update_layout(height=500, hovermode='x unified', xaxis_title="Date",
                      yaxis_title="Price (USD/barrel)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


def show_change_point_analysis(data, method, sensitivity):
    st.header("🔍 Change Point Analysis")
    change_points = detect_change_points(data, method, penalty=sensitivity)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Change Points", len(change_points))
    with col2:
        st.metric("High Significance", sum(1 for cp in change_points if cp.significance == 'High'))
    with col3:
        st.metric("Price Increases", sum(1 for cp in change_points if cp.direction == 'Increase'))
    with col4:
        avg_change = np.mean([abs(cp.change_pct) for cp in change_points]) if change_points else 0
        st.metric("Avg Change", f"{avg_change:.1f}%")

    fig = make_subplots(rows=3, cols=1, subplot_titles=("Price with Change Points", "Change Magnitude", "Confidence Scores"),
                        vertical_spacing=0.1, row_heights=[0.5, 0.25, 0.25])

    # Price line
    fig.add_trace(go.Scatter(x=data.index, y=data['Price'], mode='lines', name='Price', line=dict(color='#1f77b4')), row=1, col=1)
    if change_points:
        dates = [cp.date for cp in change_points]
        changes = [abs(cp.change_pct) for cp in change_points]
        colors = ['#d62728' if cp.direction == 'Decrease' else '#2ca02c' for cp in change_points]
        confidences = [cp.confidence for cp in change_points]

        for cp in change_points:
            color = '#d62728' if cp.direction == 'Decrease' else '#2ca02c'
            fig.add_trace(go.Scatter(x=[cp.date], y=[cp.price_after], mode='markers', marker=dict(size=10, color=color),
                                     showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=dates, y=changes, marker_color=colors, name='Change Magnitude'), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=confidences, mode='lines+markers', line=dict(color='#ff7f0e'), name='Confidence'), row=3, col=1)

    fig.update_layout(height=800, showlegend=False, template="plotly_white")
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Change (%)", row=2, col=1)
    fig.update_yaxes(title_text="Confidence", row=3, col=1, range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    if change_points:
        df = pd.DataFrame([{'Date': cp.date.strftime('%Y-%m-%d'),
                            'Price Before': f"${cp.price_before:.2f}",
                            'Price After': f"${cp.price_after:.2f}",
                            'Change (%)': f"{cp.change_pct:.2f}%",
                            'Direction': cp.direction,
                            'Significance': cp.significance,
                            'Confidence': f"{cp.confidence:.2%}"} for cp in change_points])
        st.subheader("Detected Change Points")
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.download_button("📥 Download Change Points CSV", df.to_csv(index=False), "change_points.csv", "text/csv")


def show_risk_metrics(data, confidence_level):
    st.header("📊 Risk Metrics Analysis")
    metrics, summary = calculate_risk_metrics(data)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card'><h4>Volatility Analysis</h4>"
                    f"<p>Historical Vol: {metrics.historical_volatility:.2%}</p>"
                    f"<p>Current Vol: {metrics.rolling_volatility.iloc[-1]:.2%}</p>"
                    f"<p>Vol Regime: {summary['volatility_level']}</p></div>", unsafe_allow_html=True)
    with col2:
        var_value = metrics.var_historical.get(confidence_level, 0) * 100
        cvar_value = metrics.cvar.get(confidence_level, 0) * 100
        st.markdown(f"<div class='metric-card'><h4>Value at Risk ({int(confidence_level*100)}%)</h4>"
                    f"<p>Historical VaR: {var_value:.2f}%</p>"
                    f"<p>Conditional VaR: {cvar_value:.2f}%</p>"
                    f"<p>Risk Level: {summary['risk_return_profile']}</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h4>Drawdown Analysis</h4>"
                    f"<p>Max Drawdown: {metrics.max_drawdown*100:.2f}%</p>"
                    f"<p>Duration: {metrics.max_drawdown_duration} days</p>"
                    f"<p>Current: {metrics.current_drawdown*100:.2f}%</p></div>", unsafe_allow_html=True)

    # Plotting risk visualization
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Rolling Volatility", "Drawdown History", "Returns Distribution", "VaR Comparison"))
    fig.add_trace(go.Scatter(x=metrics.rolling_volatility.index, y=metrics.rolling_volatility,
                             mode='lines', line=dict(color='#ff7f0e'), name='Rolling Vol'), row=1, col=1)
    fig.add_hline(y=metrics.rolling_volatility.mean(), line_dash="dash", line_color="red", row=1, col=1)
    fig.add_trace(go.Scatter(x=metrics.drawdown_series.index, y=metrics.drawdown_series * 100,
                             mode='lines', fill='tozeroy', line=dict(color='#d62728'), name='Drawdown'), row=1, col=2)
    returns = data['Price'].pct_change().dropna() * 100
    fig.add_trace(go.Histogram(x=returns, nbinsx=50, marker_color='#2ca02c', name='Returns'), row=2, col=1)

    conf_levels = list(metrics.var_historical.keys())
    hist_var = [metrics.var_historical[c] * 100 for c in conf_levels]
    param_var = [metrics.var_parametric[c] * 100 for c in conf_levels]
    fig.add_trace(go.Bar(name='Historical', x=[f"{int(c*100)}%" for c in conf_levels], y=hist_var, marker_color='#1f77b4'), row=2, col=2)
    fig.add_trace(go.Bar(name='Parametric', x=[f"{int(c*100)}%" for c in conf_levels], y=param_var, marker_color='#ff7f0e'), row=2, col=2)

    fig.update_layout(height=800, showlegend=True, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


def show_stress_testing(data):
    st.header("⚠️ Stress Testing & Scenario Analysis")
    returns = data['Price'].pct_change().dropna()
    current_price = data['Price'].iloc[-1]

    scenarios = {"Financial Crisis 2008": -0.50, "COVID-19 Crash 2020": -0.30, "Gulf War 1991": -0.25,
                 "Moderate Correction": -0.15, "Minor Pullback": -0.05}
    results = []
    for scenario, shock in scenarios.items():
        shocked_price = current_price * (1 + shock)
        loss_pct = shock * 100
        results.append({'Scenario': scenario, 'Price Impact': f"{loss_pct:.1f}%", 'New Price': f"${shocked_price:.2f}"})
    df = pd.DataFrame(results)
    st.dataframe(df, hide_index=True, use_container_width=True)


def show_reports():
    st.header("📑 Reports & Export")
    st.markdown("### Generate Professional Reports")
    st.write("Download analysis reports in PDF/HTML formats (simulated for demo).")


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    main()
