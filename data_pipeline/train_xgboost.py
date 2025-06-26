import sys
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from data_pipeline.utils import load_config, get_logger

def train_model(config_path: str):
    # Load config & logger
    cfg = load_config(config_path)
    log = get_logger("train_xgboost")

    # Load features + raw labels
    df = pd.read_csv(cfg['data']['features_csv'])
    log.info("Loaded feature dataset")

    # Prepare X, y (map mss: -1→0, 0→1, 1→2)
    feature_cols = [c for c in df.columns if c not in ('time', 'mss')]
    X = df[feature_cols]
    y_raw = df['mss'].astype(int)
    y = (y_raw + 1).astype(int)
    log.info(f"Labels distribution (mapped): {dict(pd.Series(y).value_counts())}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg['training']['test_size'],
        random_state=cfg['training']['random_state'],
        stratify=y
    )
    log.info(f"Split data: train={len(X_train)}, test={len(X_test)}")

    # Build classifier
    xgb_params = cfg['training'].get('xgb_params', {
        "max_depth":      5,
        "learning_rate":  0.1,
        "n_estimators":   100,
        "objective":     "multi:softmax",
        "num_class":      3
    })
    clf = xgb.XGBClassifier(**xgb_params, use_label_encoder=False, eval_metric="mlogloss")

    # Optional GridSearchCV
    grid_cfg = cfg['training'].get('xgb_param_grid')
    if grid_cfg:
        grid = GridSearchCV(
            clf,
            param_grid=grid_cfg,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train)
        clf = grid.best_estimator_
        log.info(f"GridSearchCV best params: {grid.best_params_}, CV acc: {grid.best_score_:.4f}")
    else:
        clf.fit(X_train, y_train)
        log.info("Finished training XGBoost model")

    # Save booster to JSON
    booster_path = cfg['models']['xgb_model']
    clf.get_booster().dump_model(booster_path)
    log.info(f"Saved booster JSON to {booster_path}")

    # Save classifier (with classes_)
    model_path = cfg['models']['label_classes']
    joblib.dump(clf, model_path)
    log.info(f"Saved XGBClassifier to {model_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m data_pipeline.train_xgboost config.yaml")
        sys.exit(1)
    train_model(sys.argv[1])
