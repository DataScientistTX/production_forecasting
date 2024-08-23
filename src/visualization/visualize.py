import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.graph_objects as go
from plotly.io import write_image

def ensure_output_dir(output_dir):
    """Ensure the output directory exists."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def plot_oil_production(df, well_name, output_dir):
    ensure_output_dir(output_dir)
    df_vis = df[df['well_name'] == well_name]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_vis['months_since_first_production'], y=df_vis['oil'],
                             mode='lines', name=well_name))
    fig.update_layout(title=f'Oil Production Over Time for {well_name}',
                      xaxis_title='Months Since Production',
                      yaxis_title='Oil Production')
    fig.write_image(os.path.join(output_dir, f'oil_production_{well_name}.png'))

def plot_top_5_wells(df, output_dir):
    ensure_output_dir(output_dir)
    well_production = df.groupby('well_name')['oil'].sum()
    top_5_wells = well_production.nlargest(5)
    df_filtered = df[df.well_name.isin(top_5_wells.index)]
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_filtered, x='cumulative_oil_production', y='oil', hue='well_name')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Cumulative Oil Production (log scale)')
    plt.ylabel('Oil Production (log scale)')
    plt.title('Log-log Plot of Oil Production Over Time for Top 5 Wells')
    plt.savefig(os.path.join(output_dir, 'top_5_wells.png'))
    plt.close()

def plot_cumulative_production(df_filtered, output_dir):
    ensure_output_dir(output_dir)
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_filtered, x='months_since_first_production', y='cumulative_oil_production', hue='well_name')
    plt.xlabel('Months Since Production')
    plt.ylabel('Cumulative Oil Production')
    plt.title('Cumulative Oil Production Over Time for Top 5 Wells')
    plt.savefig(os.path.join(output_dir, 'cumulative_production.png'))
    plt.close()

def plot_total_production(series, production_type, output_dir):
    ensure_output_dir(output_dir)
    total_production = series[production_type].sum(axis=1)
    total_production = total_production[total_production != 0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=total_production.index, y=total_production.values, mode='lines', name=f'Total {production_type.capitalize()} Production'))
    fig.update_layout(title=f'Total {production_type.capitalize()} Production Over Time',
                      xaxis_title='Timestamp',
                      yaxis_title=f'Total {production_type.capitalize()} Production')
    fig.write_image(os.path.join(output_dir, f'total_{production_type}_production.png'))

def plot_producing_wells(series, output_dir):
    ensure_output_dir(output_dir)
    series['oil'] = series['oil'].fillna(0)
    non_zero_count = series['oil'].apply(lambda x: (x != 0).sum(), axis=1)
    non_zero_count = non_zero_count[non_zero_count != 0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=non_zero_count.index, y=non_zero_count.values, mode='lines', name='Number of producing wells'))
    fig.update_layout(title='Number of producing wells',
                      xaxis_title='Timestamp',
                      yaxis_title='Number of producing wells')
    fig.write_image(os.path.join(output_dir, 'producing_wells.png'))

def plot_gor(series, output_dir):
    ensure_output_dir(output_dir)
    GOR = series['gas_total'].sum(axis=1) / (series['oil'].sum(axis=1) + 1e-6)
    GOR = GOR[GOR != 0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=GOR.index, y=GOR.values, mode='lines', name='GOR'))
    fig.update_layout(title='Total GOR Over Time',
                      xaxis_title='Timestamp',
                      yaxis_title='Total GOR')
    fig.write_image(os.path.join(output_dir, 'gor.png'))

def plot_model_comparison(results_df, output_dir):
    ensure_output_dir(output_dir)
    plt.figure(figsize=(12, 8))
    results_df.boxplot()
    plt.title('Boxplot of RMSE Values')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.ylim(0, 40000)
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()