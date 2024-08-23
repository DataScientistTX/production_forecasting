import torch
import numpy as np
import matplotlib.pyplot as plt

def predict_oil_production(well_name, df, pipeline):
    """
    Predict oil production for a specific well using the Chronos pipeline.
    """
    df = df[df['well_name'] == well_name]
    df = df[df.oil > 0].reset_index(drop=True)
    
    context = torch.tensor(df["oil"])
    prediction_length = 6
    forecast = pipeline.predict(context, prediction_length, num_samples=24, temperature=1.0, top_k=50, top_p=1.0)
    
    forecast_index = range(len(df)-1, len(df)-1+prediction_length)
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
    
    plt.figure(figsize=(8, 4))
    plt.plot(df["oil"], color="royalblue", label="historical data")
    plt.plot(forecast_index, median, color="tomato", label="median forecast")
    plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
    plt.legend()
    plt.grid()
    plt.title(f"Oil Production Forecast for {well_name}")
    plt.show()

# Add any other prediction-related functions here