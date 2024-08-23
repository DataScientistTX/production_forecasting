import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def predict_oil_production(well_name, df, pipeline, output_dir):
    """
    Predict oil production for a specific well using the Chronos pipeline and save the plot.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_well = df[df['well_name'] == well_name]
    df_well = df_well[df_well.oil > 0].reset_index(drop=True)
    
    context = torch.tensor(df_well["oil"].values, dtype=torch.float32)
    prediction_length = 6
    forecast = pipeline.predict(context, prediction_length, num_samples=24, temperature=1.0, top_k=50, top_p=1.0)
    
    forecast_index = range(len(df_well), len(df_well) + prediction_length)
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_well.index, df_well["oil"], color="royalblue", label="Historical data")
    plt.plot(forecast_index, median, color="tomato", label="Median forecast")
    plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
    plt.legend()
    plt.grid(True)
    plt.title(f"Oil Production Forecast for {well_name}")
    plt.xlabel("Time")
    plt.ylabel("Oil Production")
    plt.savefig(os.path.join(output_dir, f'forecast_{well_name}.png'))
    plt.close()

    print(f"Forecast for {well_name}:")
    print(f"Median forecast: {median}")
    print(f"80% prediction interval: [{low}, {high}]")