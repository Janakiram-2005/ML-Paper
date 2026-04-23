from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt


RANDOM_STATE = 42
TEST_RATIO = 0.2
SEQUENCE_LENGTH = 30
LAGS = [1, 2, 3, 7, 14, 30, 60]
TARGET = "price_silver"
PLOT_NAME = "Dig-silver.png"
MODEL_FILE = "silver_lstm_model.keras"


def load_data(data_dir: Path) -> pd.DataFrame:
    gold_path = data_dir / "gold_price.csv"
    silver_path = data_dir / "silver_price.csv"

    if not gold_path.exists() or not silver_path.exists():
        raise FileNotFoundError(
            f"Expected files not found in {data_dir}. "
            "Required: gold_price.csv and silver_price.csv"
        )

    gold_df = pd.read_csv(gold_path)
    silver_df = pd.read_csv(silver_path)

    for df in (gold_df, silver_df):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    gold_df = gold_df.dropna(subset=["date", "price"]).rename(columns={"price": "price_gold"})
    silver_df = silver_df.dropna(subset=["date", "price"]).rename(columns={"price": "price_silver"})

    merged_df = pd.merge(gold_df, silver_df, on="date", how="inner").sort_values("date")
    return merged_df.set_index("date")


def build_features(merged_df: pd.DataFrame) -> pd.DataFrame:
    df = merged_df.copy()

    df["gold_daily_return"] = df["price_gold"].pct_change()
    df["silver_daily_return"] = df["price_silver"].pct_change()
    df["silver_sma_5"] = df["price_silver"].rolling(window=5).mean()
    df["silver_sma_20"] = df["price_silver"].rolling(window=20).mean()
    df["silver_sma_50"] = df["price_silver"].rolling(window=50).mean()

    for lag in LAGS:
        df[f"price_silver_lag_{lag}"] = df["price_silver"].shift(lag)

    return df.dropna().copy()


def create_lstm_sequences(df_ml: pd.DataFrame):
    selected_features = [
        "price_gold",
        "price_silver",
        "gold_daily_return",
        "silver_daily_return",
        "silver_sma_5",
        "silver_sma_20",
        "silver_sma_50",
    ]

    for lag in LAGS:
        selected_features.append(f"price_silver_lag_{lag}")

    df_lstm = df_ml[selected_features].dropna().copy()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_lstm)

    target_idx = selected_features.index(TARGET)
    X_lstm, y_lstm = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_data)):
        X_lstm.append(scaled_data[i - SEQUENCE_LENGTH : i, :])
        y_lstm.append(scaled_data[i, target_idx])

    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)

    split_point = int(len(X_lstm) * (1 - TEST_RATIO))
    X_train, X_test = X_lstm[:split_point], X_lstm[split_point:]
    y_train, y_test = y_lstm[:split_point], y_lstm[split_point:]
    test_dates = df_lstm.index[SEQUENCE_LENGTH + split_point :]

    return X_train, X_test, y_train, y_test, scaler, selected_features, test_dates


def build_lstm_model(hp, input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(
        LSTM(
            units=hp.Int("units_1", min_value=32, max_value=128, step=32),
            return_sequences=True,
        )
    )
    model.add(Dropout(hp.Float("dropout_1", min_value=0.1, max_value=0.5, step=0.1)))
    model.add(
        LSTM(
            units=hp.Int("units_2", min_value=32, max_value=128, step=32),
            return_sequences=False,
        )
    )
    model.add(Dropout(hp.Float("dropout_2", min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(units=1))

    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss="mean_squared_error")
    return model


def build_default_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(96, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
    return model


def invert_scaled_target(values_scaled, scaler, selected_features, target_name):
    target_idx = selected_features.index(target_name)
    dummy = np.zeros((len(values_scaled), len(selected_features)))
    dummy[:, target_idx] = values_scaled.reshape(-1)
    inverted = scaler.inverse_transform(dummy)
    return inverted[:, target_idx]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    eps = 1e-9
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)
    within_2pct = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps)) <= 0.02) * 100)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE(%)": mape,
        "AccuracyWithin2%(%)": within_2pct,
    }


def plot_predictions(test_dates, y_true, y_pred, output_path: Path):
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_true, label="Actual Silver Price", color="blue")
    plt.plot(test_dates, y_pred, label="Predicted Silver Price (LSTM Tuned)", color="purple", linestyle="--")
    plt.title("Silver Price Prediction using Tuned LSTM")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()


def main():
    np.random.seed(RANDOM_STATE)

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "d2"

    print("Loading and preparing data...")
    merged_df = load_data(data_dir)
    df_ml = build_features(merged_df)

    X_train, X_test, y_train, y_test, scaler, selected_features, test_dates = create_lstm_sequences(df_ml)

    print(f"LSTM Training set shape: {X_train.shape}")
    print(f"LSTM Testing set shape: {X_test.shape}")

    model_path = project_root / MODEL_FILE
    if model_path.exists():
        print(f"\nLoading existing model from: {model_path}")
        model_lstm = load_model(model_path)
    else:
        print("\nBuilding and training LSTM model...")
        tuner = kt.Hyperband(
            lambda hp: build_lstm_model(hp, (X_train.shape[1], X_train.shape[2])),
            objective="val_loss",
            max_epochs=50,
            factor=3,
            hyperband_iterations=2,
            directory=str(project_root / "my_dir"),
            project_name="lstm_silver_price_tuning",
            overwrite=True,
        )

        early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

        print("\nStarting LSTM Hyperparameter Tuning...")
        try:
            tuner.search(
                X_train,
                y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1,
            )

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            print("\nLSTM Hyperparameter Tuning Complete.")
            print(f"Optimal number of units in the first LSTM layer: {best_hps.get('units_1')}")
            print(f"Optimal dropout rate in the first LSTM layer: {best_hps.get('dropout_1')}")
            print(f"Optimal number of units in the second LSTM layer: {best_hps.get('units_2')}")
            print(f"Optimal dropout rate in the second LSTM layer: {best_hps.get('dropout_2')}")
            print(f"Optimal learning rate for the Adam optimizer: {best_hps.get('learning_rate')}")
            model_lstm = tuner.hypermodel.build(best_hps)
        except (ModuleNotFoundError, RuntimeError) as exc:
            print(f"\nHyperparameter tuning skipped: {exc}")
            print("Using default LSTM settings. Install tensorboard to enable tuning.")
            model_lstm = build_default_lstm_model((X_train.shape[1], X_train.shape[2]))

        model_lstm.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1,
        )
        model_lstm.save(model_path)
        print(f"LSTM model training complete and saved to: {model_path}")

    y_pred_scaled = model_lstm.predict(X_test, verbose=0)
    y_test_inverse = invert_scaled_target(y_test, scaler, selected_features, TARGET)
    y_pred_inverse = invert_scaled_target(y_pred_scaled.reshape(-1), scaler, selected_features, TARGET)

    metrics = regression_metrics(y_test_inverse, y_pred_inverse)
    print("\nSilver LSTM Model Test Metrics")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    plot_path = project_root / PLOT_NAME
    plot_predictions(test_dates, y_test_inverse, y_pred_inverse, plot_path)
    print(f"\nPlot saved to: {plot_path}")


if __name__ == "__main__":
    main()
