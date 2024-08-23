import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import torch
from chronos import ChronosPipeline

def load_chronos_pipeline():
    """
    Load the pre-trained Chronos pipeline.
    """
    return ChronosPipeline.from_pretrained("amazon/chronos-t5-small", torch_dtype=torch.bfloat16)

def train_and_evaluate_models(df, well_list):
    """
    Train and evaluate multiple models for each well.
    """
    models = {
        'Linear': LinearRegression(),
        'Polynomial (Degree 2)': (PolynomialFeatures(degree=2), LinearRegression()),
        'Polynomial (Degree 3)': (PolynomialFeatures(degree=3), LinearRegression()),
        'Chronos': load_chronos_pipeline()
    }
    
    results = {model: [] for model in models}
    
    for well in well_list:
        dat = df[df.well_name == well]
        X = dat.months_since_first_production.values.reshape(-1, 1)
        y = dat.oil.values.reshape(-1, 1)
        
        data_length = len(X)
        test_start_index = max(0, data_length - 12)
        train_end_index = test_start_index
        train_start_index = max(0, train_end_index - 24)
        
        X_train, X_test = X[train_start_index:train_end_index], X[test_start_index:]
        y_train, y_test = y[train_start_index:train_end_index], y[test_start_index:]
        
        for model_name, model in models.items():
            if model_name.startswith('Polynomial'):
                poly_features, regressor = model
                X_train_poly = poly_features.fit_transform(X_train)
                X_test_poly = poly_features.transform(X_test)
                regressor.fit(X_train_poly, y_train)
                y_pred = regressor.predict(X_test_poly)
            elif model_name == 'Chronos':
                # Ensure the data is in the correct format for Chronos
                context = torch.tensor(dat["oil"].iloc[:-12].values, dtype=torch.float32)
                forecast = model.predict(context, 12, num_samples=24, temperature=1.0, top_k=50, top_p=1.0)
                y_pred = np.quantile(forecast[0].numpy(), 0.5, axis=0).reshape(-1, 1)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            results[model_name].append(rmse)
    
    return pd.DataFrame(results, index=well_list)

# Add any other model training or evaluation functions here