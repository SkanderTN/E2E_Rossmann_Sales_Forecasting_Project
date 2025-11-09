"""
Streamlit dashboard for interactive sales forecasting visualization.

Displays historical sales and forecast overlays with ability to generate
forecasts via API and export results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import httpx
from pathlib import Path
from datetime import timedelta
import os

# Page configuration
st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

# Get base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
FEATURES_FILE = "rossmann_features.parquet"


def load_historical_data():
    """Load historical data from processed features file."""
    features_path = DATA_PROCESSED_DIR / FEATURES_FILE
    if not features_path.exists():
        return None
    try:
        df = pd.read_parquet(features_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def call_api_predict(api_url, store_id, start_date, horizon):
    """Call prediction API."""
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{api_url}/predict",
                json={
                    "store_id": store_id,
                    "start_date": start_date,
                    "horizon": horizon
                }
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API error: {e.response.status_code} - {e.response.text}")
        return None
    except httpx.ConnectError:
        st.error(f"Cannot connect to API at {api_url}. Ensure API is running.")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def test_api_connection(api_url):
    """Test API health endpoint."""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{api_url}/health")
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "ok" and data.get("model_loaded"):
                return True, "API is healthy and model is loaded"
            else:
                return False, f"API returned: {data}"
    except Exception as e:
        return False, f"Connection failed: {e}"


# Sidebar configuration
st.sidebar.title("Rossmann Sales Forecasting")
st.sidebar.markdown("---")

# API configuration
api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")

if st.sidebar.button("Test API Connection"):
    success, message = test_api_connection(api_url)
    if success:
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)

st.sidebar.markdown("---")

# Load data
historical_data = load_historical_data()

if historical_data is None:
    st.error("Processed data not found. Run training first: `python -m src.train`")
    st.stop()

# Store selection
available_stores = sorted(historical_data['Store'].unique())
store_id = st.sidebar.number_input(
    "Store ID",
    min_value=int(min(available_stores)),
    max_value=int(max(available_stores)),
    value=int(available_stores[0])
)

# Get store data
store_data = historical_data[historical_data['Store'] == store_id].copy()
if len(store_data) == 0:
    st.error(f"No data found for store {store_id}")
    st.stop()

store_data = store_data.sort_values('Date').reset_index(drop=True)
last_date = store_data['Date'].max()

# Forecast configuration
default_start = last_date + timedelta(days=1)
start_date = st.sidebar.date_input(
    "Forecast Start Date",
    value=default_start,
    min_value=default_start
)

horizon = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=7,
    max_value=56,
    value=28,
    step=1
)

generate_forecast = st.sidebar.button("Generate Forecast", type="primary")

# Main area
st.title("📈 Rossmann Sales Forecasting Dashboard")
st.markdown(f"**Store ID:** {store_id} | **Last Historical Date:** {last_date.date()}")
st.markdown("---")

# Generate forecast
if generate_forecast:
    with st.spinner("Generating forecast..."):
        # Call API
        api_response = call_api_predict(
            api_url,
            store_id,
            start_date.strftime('%Y-%m-%d'),
            horizon
        )

        if api_response:
            # Parse forecast data
            forecasts = api_response['forecasts']
            forecast_df = pd.DataFrame(forecasts)
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])

            # Display layout
            col1, col2 = st.columns(2)

            # Left panel: Historical sales
            with col1:
                st.subheader(f"Historical Sales (Last 180 Days)")

                # Get last 180 days
                cutoff_date = last_date - timedelta(days=180)
                history_display = store_data[store_data['Date'] >= cutoff_date].copy()

                # Create plot
                fig_history = go.Figure()
                fig_history.add_trace(go.Scatter(
                    x=history_display['Date'],
                    y=history_display['Sales'],
                    mode='lines',
                    name='Historical Sales',
                    line=dict(color='blue', width=2)
                ))

                fig_history.update_layout(
                    title=f"Store {store_id} - Historical Sales",
                    xaxis_title="Date",
                    yaxis_title="Sales",
                    hovermode='x unified',
                    height=500
                )

                st.plotly_chart(fig_history, use_container_width=True)

                # Save plot
                try:
                    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                    history_plot_path = REPORTS_DIR / f"dashboard_history_store{store_id}.png"
                    fig_history.write_image(str(history_plot_path))
                    st.success(f"Plot saved to {history_plot_path}")
                except Exception as e:
                    st.warning(f"Could not save plot: {e}")

            # Right panel: Forecast overlay
            with col2:
                st.subheader("Sales Forecast")

                # Get last 60 days of history for overlay
                overlay_cutoff = last_date - timedelta(days=60)
                history_overlay = store_data[store_data['Date'] >= overlay_cutoff].copy()

                # Create plot
                fig_forecast = go.Figure()

                # Historical line
                fig_forecast.add_trace(go.Scatter(
                    x=history_overlay['Date'],
                    y=history_overlay['Sales'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', width=2)
                ))

                # Forecast line
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_df['date'],
                    y=forecast_df['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', width=2, dash='dash')
                ))

                # Add vertical line at forecast start
                fig_forecast.add_vline(
                    x=start_date,
                    line_dash="dot",
                    line_color="gray",
                    annotation_text="Forecast Start"
                )

                fig_forecast.update_layout(
                    title=f"Store {store_id} - Forecast Overlay",
                    xaxis_title="Date",
                    yaxis_title="Sales",
                    hovermode='x unified',
                    height=500,
                    legend=dict(x=0.01, y=0.99)
                )

                st.plotly_chart(fig_forecast, use_container_width=True)

                # Save plot
                try:
                    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                    forecast_plot_path = REPORTS_DIR / f"dashboard_forecast_store{store_id}.png"
                    fig_forecast.write_image(str(forecast_plot_path))
                    st.success(f"Plot saved to {forecast_plot_path}")
                except Exception as e:
                    st.warning(f"Could not save plot: {e}")

            # Summary statistics
            st.markdown("---")
            st.subheader("Forecast Summary")

            col_a, col_b, col_c, col_d, col_e = st.columns(5)

            # Calculate metrics
            last_7_avg = history_display.tail(7)['Sales'].mean()
            forecast_7_avg = forecast_df.head(7)['yhat'].mean()
            forecast_total = forecast_df['yhat'].sum()
            forecast_min = forecast_df['yhat'].min()
            forecast_max = forecast_df['yhat'].max()

            col_a.metric("Last 7-Day Avg", f"{last_7_avg:,.0f}")
            col_b.metric("Forecast 7-Day Avg", f"{forecast_7_avg:,.0f}")
            col_c.metric(f"Forecast Total ({horizon}d)", f"{forecast_total:,.0f}")
            col_d.metric("Min Forecast", f"{forecast_min:,.0f}")
            col_e.metric("Max Forecast", f"{forecast_max:,.0f}")

            # Forecast data table
            st.markdown("---")
            st.subheader("Forecast Data")

            with st.expander("View Forecast Table"):
                display_df = forecast_df.copy()
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                display_df = display_df.rename(columns={'date': 'Date', 'yhat': 'Predicted Sales'})
                st.dataframe(display_df, use_container_width=True)

            # Download button
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="Download Forecast as CSV",
                data=csv,
                file_name=f"forecast_store{store_id}_{start_date}.csv",
                mime="text/csv"
            )

else:
    # Show placeholder
    st.info("Configure forecast parameters in the sidebar and click 'Generate Forecast' to view predictions.")

    # Show sample historical data
    st.subheader(f"Sample Historical Data - Store {store_id}")
    sample_data = store_data.tail(10)[['Date', 'Sales', 'Promo', 'SchoolHoliday']].copy()
    sample_data['Date'] = sample_data['Date'].dt.strftime('%Y-%m-%d')
    st.dataframe(sample_data, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Rossmann Sales Forecasting System | Built with Streamlit")
