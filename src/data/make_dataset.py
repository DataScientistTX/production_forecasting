import pandas as pd
import os

def load_and_preprocess_data(file_path=None):
    """
    Load data from CSV and preprocess it.
    """
    if file_path is None:
        # Construct the path to the data file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        
        # Ensure the project root is named 'production_forecasting'
        if os.path.basename(project_root) != 'production_forecasting':
            raise RuntimeError("The project root directory must be named 'production_forecasting'")
        
        file_path = os.path.join(project_root, 'data', 'raw', 'test.csv')
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the file path.")
    
    print(f"Attempting to load data from: {file_path}")
    
    # Load the data
    try:
        df = pd.read_csv(file_path, index_col=0)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        raise
    
    # Preprocess the data
    df['period'] = pd.to_datetime(df['period'])
    df = df.sort_values(by='period').reset_index(drop=True)
    
    series = df.pivot_table(index='period', 
                            values=['oil', 'gas_total'], 
                            columns='well_name')
    
    return df, series

if __name__ == "__main__":
    # This allows you to run some tests directly on this file if needed
    df, series = load_and_preprocess_data()
    print(df.head())
    print(series.head())