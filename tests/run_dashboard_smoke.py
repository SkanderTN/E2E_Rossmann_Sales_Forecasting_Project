import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent.parent
features_path = BASE_DIR / 'data' / 'processed' / 'rossmann_features.parquet'

print('Loading features...')
df = pd.read_parquet(features_path)
store_id = sorted(df['Store'].unique())[0]
store_data = df[df['Store'] == store_id].sort_values('Date').reset_index(drop=True)
last_date = pd.to_datetime(store_data['Date']).max()

# create history_overlay (last 60 days)
overlay_cutoff = last_date - pd.Timedelta(days=60)
history_overlay = store_data[store_data['Date'] >= overlay_cutoff].copy()

# create fake forecast df
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq='D')
forecast_df = pd.DataFrame({'date': forecast_dates, 'yhat': [100]*7})

# create fig
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=history_overlay['Date'], y=history_overlay['Sales'], mode='lines', name='Historical'))
fig_forecast.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['yhat'], mode='lines', name='Forecast'))

# vline as ISO string
vline_x = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
print('vline_x', vline_x)

# add shape and annotation
fig_forecast.add_shape(type='line', x0=vline_x, x1=vline_x, y0=0, y1=1, xref='x', yref='paper', line=dict(color='gray', dash='dot'))
fig_forecast.add_annotation(x=vline_x, y=1.02, xref='x', yref='paper', showarrow=False, text='Forecast Start')

print('Added shape and annotation successfully')
# try to write image to ensure kaleido works
out = BASE_DIR / 'reports' / f'test_dashboard_store{store_id}.png'
fig_forecast.write_image(str(out))
print('Wrote image to', out)
