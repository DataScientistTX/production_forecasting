import pandas as pd

def calculate_gas_decline_rate(gas_data):
    """
    Calculate the average month-over-month decline rate for gas.
    """
    gas_diff = gas_data.diff(periods=1).dropna()
    decline_rate = (gas_diff.values / (gas_data[:-1].values + 1e-6)) * 100
    return decline_rate.mean()

def calculate_well_characteristics(df):
    """
    Calculate various characteristics for each well.
    """
    return df.groupby('well_name').agg(
        average_gas_oil_ratio=pd.NamedAgg(column='gas_total', aggfunc=lambda x: x.mean() / df['oil'].mean()),
        num_months=pd.NamedAgg(column='period', aggfunc='count'),
        initial_oil_date=pd.NamedAgg(column='period', aggfunc='min'),
        average_gas_decline_rate=pd.NamedAgg(column='gas_total', aggfunc=calculate_gas_decline_rate)
    )

def filter_and_process_data(df, well_characteristics):
    """
    Filter and process the data based on well characteristics.
    """
    more_than_24 = well_characteristics[well_characteristics['num_months'] >= 24]
    well_list = more_than_24.index
    df_ = df[df.well_name.isin(well_list)]
    df_ = df_[['well_name', 'period', 'oil']]
    df_ = df_[df_.oil > 0]
    
    df_['period'] = pd.to_datetime(df_['period'])
    df_['months_since_first_production'] = df_.groupby('well_name')['period'].transform(lambda x: (x - x.min()) // pd.Timedelta('30D'))
    df_ = df_.sort_values(by=['well_name', 'period'])
    df_['cumulative_oil_production'] = df_.groupby('well_name')['oil'].cumsum()
    
    return df_, well_list

# Add any other feature engineering functions here