import os
import sys

def find_project_root(current_path, project_name='production_forecasting'):
    """
    Find the project root by looking for a directory with 'src' and 'data' subdirectories.
    """
    while True:
        if os.path.basename(current_path) == project_name and \
           os.path.isdir(os.path.join(current_path, 'src')) and \
           os.path.isdir(os.path.join(current_path, 'data')):
            return current_path
        parent = os.path.dirname(current_path)
        if parent == current_path:  # We've reached the root directory
            raise RuntimeError(f"Could not find a directory named '{project_name}' with 'src' and 'data' subdirectories")
        current_path = parent

# Find the project root
try:
    project_root = find_project_root(__file__)
except RuntimeError:
    # If not found, try to find it starting from the user's desktop
    desktop_path = os.path.expanduser("~/Desktop")
    project_root = find_project_root(desktop_path)

print(f"Project root found: {project_root}")

# Add the project root to the Python path
sys.path.insert(0, project_root)

# Import functions from your modules
from src.data import load_and_preprocess_data
from src.features import calculate_well_characteristics, filter_and_process_data
from src.models import train_and_evaluate_models, load_chronos_pipeline, predict_oil_production
from src.visualization import (
    plot_oil_production,
    plot_top_5_wells,
    plot_cumulative_production,
    plot_total_production,
    plot_producing_wells,
    plot_gor,
    plot_model_comparison
)

def main():
    try:
        # Specify the correct path to your test.csv file
        data_path = os.path.join(project_root, 'data', 'raw', 'test.csv')
        output_dir = os.path.join(project_root, 'outputs', 'figures')
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created/confirmed: {output_dir}")
        
        # Load and preprocess data
        print(f"Loading and preprocessing data from {data_path}...")
        df, series = load_and_preprocess_data(data_path)

        # Calculate well characteristics
        print("Calculating well characteristics...")
        well_characteristics = calculate_well_characteristics(df)

        # Filter and process data
        print("Filtering and processing data...")
        df_filtered, well_list = filter_and_process_data(df, well_characteristics)

        # Plotting
        print("Generating plots...")
        plot_oil_production(df_filtered, 'FIELD92', output_dir)
        print(f"Oil production plot should be saved in: {os.path.join(output_dir, 'oil_production_FIELD92.png')}")
        
        plot_top_5_wells(df_filtered, output_dir)
        print(f"Top 5 wells plot should be saved in: {os.path.join(output_dir, 'top_5_wells.png')}")
        
        plot_cumulative_production(df_filtered, output_dir)
        print(f"Cumulative production plot should be saved in: {os.path.join(output_dir, 'cumulative_production.png')}")
        
        plot_total_production(series, 'oil', output_dir)
        print(f"Total oil production plot should be saved in: {os.path.join(output_dir, 'total_oil_production.png')}")
        
        plot_total_production(series, 'gas_total', output_dir)
        print(f"Total gas production plot should be saved in: {os.path.join(output_dir, 'total_gas_total_production.png')}")
        
        plot_producing_wells(series, output_dir)
        print(f"Producing wells plot should be saved in: {os.path.join(output_dir, 'producing_wells.png')}")
        
        plot_gor(series, output_dir)
        print(f"GOR plot should be saved in: {os.path.join(output_dir, 'gor.png')}")

        # Model training and evaluation
        print("Training and evaluating models...")
        use_subset = input("Do you want to use a subset of wells for faster processing? (y/n): ").lower() == 'y'
        subset_size = 10 if use_subset else len(well_list)
        results_df = train_and_evaluate_models(df_filtered, well_list[:subset_size], 
                                               cache_dir=os.path.join(project_root, 'outputs', 'model_cache'),
                                               use_subset=use_subset)
        plot_model_comparison(results_df, output_dir)
        print(f"Model comparison plot should be saved in: {os.path.join(output_dir, 'model_comparison.png')}")

        # Predict oil production for specific wells
        print("Predicting oil production for specific wells...")
        chronos_pipeline = load_chronos_pipeline()
        prediction_wells = ['FIELD4', 'FIELD55D', 'FIELD216', 'FIELD211']
        for well in prediction_wells:
            if well in well_list[:subset_size]:
                print(f"Predicting for {well}...")
                predict_oil_production(well, df, chronos_pipeline, output_dir)
                print(f"Prediction plot for {well} should be saved in: {os.path.join(output_dir, f'forecast_{well}.png')}")
            else:
                print(f"Skipping prediction for {well} as it's not in the processed subset.")

        print("Analysis complete! All figures should be saved in the outputs/figures directory.")
        
        print("Checking if files were actually saved:")
        for filename in os.listdir(output_dir):
            print(f"Found file: {filename}")

    except FileNotFoundError as e:
        print(f"Error: File not found. Please check if the file exists and the path is correct.")
        print(f"File path: {e.filename}")
    except ImportError as e:
        print(f"Error: Failed to import a required module. Please check if all dependencies are installed.")
        print(f"Import error details: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        import traceback
        print("Detailed error information:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()