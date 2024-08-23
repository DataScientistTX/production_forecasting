import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import torch
from chronos import ChronosPipeline
from joblib import Parallel, delayed
import os
import pickle

def load_chronos_pipeline():
    """
    Load the pre-trained Chronos pipeline.
    """
    return ChronosPipeline.from_pretrained("amazon/chronos-t5-small", torch_dtype=torch.bfloat16)

def train_and_evaluate_single_model(model, model_name, X_train, X_test, y_train, y_test):
    """
    Train and evaluate a single model.
    """
    if model_name.startswith('Polynomial'):
        poly_features, regressor = model
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        regressor.fit(X_train_poly, y_train)
        y_pred = regressor.predict(X_test_poly)
    elif model_name == 'Chronos':
        context = torch.tensor(y_train.flatten(), dtype=torch.float32)
        forecast = model.predict(context, len(y_test), num_samples=1, temperature=1.0, top_k=50, top_p=1.0)
        y_pred = forecast[0, 0, :].numpy().reshape(-1, 1)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

def train_and_evaluate_models(df, well_list, cache_dir='model_cache', use_subset=False, subset_size=10):
    """
    Train and evaluate multiple models for each well, with caching and optional subset usage.
    """
    models = {
        'Linear': LinearRegression(),
        'Polynomial (Degree 2)': (PolynomialFeatures(degree=2), LinearRegression()),
        'Polynomial (Degree 3)': (PolynomialFeatures(degree=3), LinearRegression()),
        'Chronos': load_chronos_pipeline()
    }
    
    if use_subset:
        well_list = well_list[:subset_size]
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'results_cache{"_subset" if use_subset else ""}.pkl')
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pd.DataFrame(pickle.load(f))
    
    results = {model: [] for model in models}
    
    def process_well(well):
        dat = df[df.well_name == well]
        X = dat.months_since_first_production.values.reshape(-1, 1)
        y = dat.oil.values.reshape(-1, 1)
        
        data_length = len(X)
        test_start_index = max(0, data_length - 12)
        train_end_index = test_start_index
        train_start_index = max(0, train_end_index - 24)
        
        X_train, X_test = X[train_start_index:train_end_index], X[test_start_index:]
        y_train, y_test = y[train_start_index:train_end_index], y[test_start_index:]
        
        well_results = {}
        for model_name, model in models.items():
            well_results[model_name] = train_and_evaluate_single_model(model, model_name, X_train, X_test, y_train, y_test)
        return well_results
    
    # Use parallel processing
    well_results = Parallel(n_jobs=-1)(delayed(process_well)(well) for well in well_list)
    
    for well_result in well_results:
        for model_name, rmse in well_result.items():
            results[model_name].append(rmse)
    
    # Cache the results
    with open(cache_file, 'wb') as f:
        pickle.dump(results, f)
    
    return pd.DataFrame(results, index=well_list)

# Add any other model training or evaluation functions here