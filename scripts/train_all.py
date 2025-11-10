"""Train Pima and Heart models. Usage:
python scripts/train_all.py --pima data/pima.csv --heart data/heart.csv --out models/
"""
import argparse
from pathlib import Path
from src.data_loader import load_pima, load_heart
from src.preprocess import preprocess_diabetes, preprocess_heart
from src.models import train_random_forest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pima', required=True)
    parser.add_argument('--heart', required=True)
    parser.add_argument('--out', default='models/')
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df_pima = load_pima(args.pima)
    Xp, yp = preprocess_diabetes(df_pima, save_scaler_path=out / 'pima_scaler.joblib')
    train_random_forest(Xp, yp, save_path=out / 'diabetes_rf.joblib')

    df_heart = load_heart(args.heart)
    # reuse preprocess_diabetes for simple pipeline if needed
    Xh, yh = preprocess_diabetes(df_heart, save_scaler_path=out / 'heart_scaler.joblib')
    train_random_forest(Xh, yh, save_path=out / 'heart_rf.joblib')

if __name__ == '__main__':
    main()
