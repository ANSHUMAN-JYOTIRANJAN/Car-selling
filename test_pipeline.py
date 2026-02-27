import joblib
import pandas as pd

try:
    pipeline = joblib.load('best_model_pipeline.pkl')
    # Create dummy dataframe matching the expected features
    test_df = pd.DataFrame([[2015, 50000, 'Petrol', 'Individual', 'Manual', 'First Owner', 20.0, 1197.0, 82.0, 5.0]],
                           columns=['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])
    print("Testing pipeline predict:")
    pred = pipeline.predict(test_df)
    print("Pipeline Prediction successful:", pred)
except Exception as e:
    print("Pipeline failed:", e)

