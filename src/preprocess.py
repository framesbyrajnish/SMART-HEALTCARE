import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_diabetes(df, save_scaler_path=None):
    df = df.copy()
    cols_with_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    for c in cols_with_zero:
        if c in df.columns:
            df[c] = df[c].replace(0, pd.NA)

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    feature_cols = [c for c in df.columns if c != 'Outcome']
    X = df[feature_cols]
    y = df['Outcome']

    X[cols_with_zero] = imputer.fit_transform(X[cols_with_zero])
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    if save_scaler_path:
        joblib.dump({'imputer': imputer, 'scaler': scaler, 'columns': X.columns.tolist()}, save_scaler_path)

    return X_scaled, y
