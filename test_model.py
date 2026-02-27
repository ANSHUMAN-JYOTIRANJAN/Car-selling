import joblib
import pandas as pd

print("Loading best_model_pipeline.pkl...")
try:
    pipeline = joblib.load('best_model_pipeline.pkl')
    print("Pipeline loaded:", type(pipeline))
    if hasattr(pipeline, 'feature_names_in_'):
        print("Pipeline features:", list(pipeline.feature_names_in_))
except Exception as e:
    print("Error loading pipeline:", e)

print("\nLoading best_model.pkl...")
try:
    best_model = joblib.load('best_model.pkl')
    print("best_model loaded:", type(best_model))
    if hasattr(best_model, 'feature_names_in_'):
        print("best_model features:", list(best_model.feature_names_in_))
except Exception as e:
    print("Error loading best_model:", e)

