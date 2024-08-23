import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

def plot_oil_production(df, well_name):
    """
    Plot oil production over time for a specific well.
    """
    df_vis = df[df['well_name'] == well_name]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_vis['months_since_first_production'], y=df_vis['oil'],
                             mode='lines', name=well_name))
    fig.update_layout(title=f'Oil Production Over Time for {well_name}',
                      xaxis_title='Months Since Production',
                      yaxis_title='Oil Production')
    fig.show()

def plot_top_5_wells(df):
    """
    Plot oil production for the top 5 producing wells.
    """
    well_production = df.groupby('well_name')['oil'].sum()
    top_5_wells = well_production.nlargest(5)
    df_filtered = df[df.well_name.isin(top_5_wells.index)]
    
    sns.lineplot(data=df_filtered, x='cumulative_oil_production', y='oil', hue='well_name')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Cumulative Oil Production (log scale)')
    plt.ylabel('Oil Production (log scale)')
    plt.title('Log-log Plot of Oil Production Over Time for Top 5 Wells')
    plt.show()

def plot_cumulative_production(df_filtered):
    """
    Plot cumulative oil production for the top 5 wells.
    """
    sns.lineplot(data=df_filtered, x='months_since_first_production', y='cumulative_oil_production', hue='well_name')
    plt.xlabel('Months Since Production')
    plt.ylabel('Cumulative Oil Production')
    plt.title('Cumulative Oil Production Over Time for Top 5 Wells')
    plt.show()

def plot_total_production(series, production_type):
    """
    Plot total production (oil or gas) over time.
    """
    total_production = series[production_type].sum(axis=1)
    total_production = total_production[total_production != 0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=total_production.index, y=total_production.values, mode='lines', name=f'Total {production_type.capitalize()} Production'))
    fig.update_layout(title=f'Total {production_type.capitalize()} Production Over Time',
                      xaxis_title='Timestamp',
                      yaxis_title=f'Total {production_type.capitalize()} Production')
    fig.show()

def plot_producing_wells(series):
    """
    Plot the number of producing wells over time.
    """
    series['oil'] = series['oil'].fillna(0)
    non_zero_count = series['oil'].apply(lambda x: (x != 0).sum(), axis=1)
    non_zero_count = non_zero_count[non_zero_count != 0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=non_zero_count.index, y=non_zero_count.values, mode='lines', name='Number of producing wells'))
    fig.update_layout(title='Number of producing wells',
                      xaxis_title='Timestamp',
                      yaxis_title='Number of producing wells')
    fig.show()

def plot_gor(series):
    """
    Plot the Gas-Oil Ratio (GOR) over time.
    """
    GOR = series['gas_total'].sum(axis=1) / (series['oil'].sum(axis=1) + 1e-6)
    GOR = GOR[GOR != 0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=GOR.index, y=GOR.values, mode='lines', name='GOR'))
    fig.update_layout(title='Total GOR Over Time',
                      xaxis_title='Timestamp',
                      yaxis_title='Total GOR')
    fig.show()

def plot_model_comparison(results_df):
    """
    Plot a comparison of model performance using boxplots.
    """
    plt.figure(figsize=(10, 6))
    results_df.boxplot()
    plt.title('Boxplot of RMSE Values')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.ylim(0, 40000)
    plt.show()

# Add any other visualization functions here