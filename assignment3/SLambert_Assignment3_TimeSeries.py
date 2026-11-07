"""Assignment 3 experiments for the Jena weather time-series data."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
MAX_EPOCHS = 12
BATCH_SIZE = 256
SAMPLING_RATE = 6
SEQUENCE_LENGTH = 120
DELAY = SAMPLING_RATE * (SEQUENCE_LENGTH + 24 - 1)

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "jena_climate_2009_2016.csv"
DATA_URL = "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


def load_weather_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read the Jena CSV and return features and temperature targets."""
    if DATA_FILE.exists():
        data_file = DATA_FILE
    else:
        downloaded_path = Path(
            keras.utils.get_file(
                "jena_climate_2009_2016.csv.zip",
                origin=DATA_URL,
                extract=True,
            )
        )
        if downloaded_path.is_dir():
            data_file = next(downloaded_path.rglob("jena_climate_2009_2016.csv"))
        else:
            data_file = downloaded_path.with_suffix("")
    weather = pd.read_csv(data_file)
    feature_names = list(weather.columns[1:])
    raw_data = weather.iloc[:, 1:].to_numpy(dtype="float32")
    temperature = weather.iloc[:, 2].to_numpy(dtype="float32")
    return raw_data, temperature, feature_names


def prepare_datasets(raw_data: np.ndarray, temperature: np.ndarray):
    """Create chronological train, validation, and test datasets."""
    num_train_samples = int(0.50 * len(raw_data))
    num_val_samples = int(0.25 * len(raw_data))
    num_test_samples = len(raw_data) - num_train_samples - num_val_samples

    train_mean = raw_data[:num_train_samples].mean(axis=0)
    train_std = raw_data[:num_train_samples].std(axis=0)
    normalized_data = (raw_data - train_mean) / train_std

    common_args = dict(
        data=normalized_data[:-DELAY],
        targets=temperature[DELAY:],
        sampling_rate=SAMPLING_RATE,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
    )
    train_dataset = keras.utils.timeseries_dataset_from_array(
        **common_args,
        shuffle=True,
        seed=SEED,
        start_index=0,
        end_index=num_train_samples,
    )
    val_dataset = keras.utils.timeseries_dataset_from_array(
        **common_args,
        shuffle=False,
        start_index=num_train_samples,
        end_index=num_train_samples + num_val_samples,
    )
    test_dataset = keras.utils.timeseries_dataset_from_array(
        **common_args,
        shuffle=False,
        start_index=num_train_samples + num_val_samples,
    )

    options = tf.data.Options()
    options.experimental_deterministic = True
    train_dataset = train_dataset.with_options(options).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.with_options(options).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.with_options(options).prefetch(tf.data.AUTOTUNE)

    split_info = {
        "total_rows": int(len(raw_data)),
        "train_rows": int(num_train_samples),
        "validation_rows": int(num_val_samples),
        "test_rows": int(num_test_samples),
    }
    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_mean,
        train_std,
        split_info,
    )


def evaluate_naive(dataset, train_mean: np.ndarray, train_std: np.ndarray) -> float:
    """Predict that tomorrow's temperature will equal the current temperature."""
    total_absolute_error = 0.0
    samples_seen = 0
    for samples, targets in dataset:
        predictions = samples[:, -1, 1] * train_std[1] + train_mean[1]
        total_absolute_error += float(
            tf.reduce_sum(tf.abs(predictions - targets)).numpy()
        )
        samples_seen += int(samples.shape[0])
    return total_absolute_error / samples_seen


def build_gru_16(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.GRU(16)(inputs)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs, name="gru_16")


def build_gru_32(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.GRU(32)(inputs)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs, name="gru_32")


def build_lstm_32(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(32)(inputs)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs, name="lstm_32")


def build_lstm_dropout(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(32, recurrent_dropout=0.25)(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs, name="lstm_32_recurrent_dropout")


def build_stacked_gru(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.GRU(32, dropout=0.2, return_sequences=True)(inputs)
    x = layers.GRU(16, dropout=0.2)(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs, name="stacked_gru_32_16")


def build_conv_gru(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(32, 5, activation="relu")(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.GRU(32)(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs, name="conv1d_gru_32")


MODEL_BUILDERS = [
    ("GRU 16 units", build_gru_16, "1 GRU layer", "16", "None"),
    ("GRU 32 units", build_gru_32, "1 GRU layer", "32", "None"),
    ("LSTM 32 units", build_lstm_32, "1 LSTM layer", "32", "None"),
    (
        "LSTM 32 with recurrent dropout",
        build_lstm_dropout,
        "1 LSTM layer",
        "32",
        "0.25 recurrent, 0.50 output",
    ),
    (
        "Stacked GRU 32 and 16",
        build_stacked_gru,
        "2 GRU layers",
        "32, 16",
        "0.20 layer, 0.30 output",
    ),
    (
        "Conv1D plus GRU 32",
        build_conv_gru,
        "Conv1D + GRU",
        "32 filters, 32 GRU",
        "None",
    ),
]


def train_model(model_name, builder, input_shape, train_dataset, val_dataset):
    """Train one model and retain its best validation checkpoint."""
    set_seed()
    keras.backend.clear_session()
    model = builder(input_shape)
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("/", "_")
    )
    model_path = MODELS_DIR / f"{safe_name}.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_mae", mode="min", save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_mae", mode="min", patience=2, min_delta=0.01
        ),
        keras.callbacks.TerminateOnNaN(),
    ]

    start = time.perf_counter()
    history = model.fit(
        train_dataset,
        epochs=MAX_EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=2,
    )
    duration = time.perf_counter() - start

    best_epoch = int(np.argmin(history.history["val_mae"]) + 1)
    best_val_mae = float(np.min(history.history["val_mae"]))
    return model_path, history.history, best_epoch, best_val_mae, duration, model.count_params()


def plot_results(results: pd.DataFrame, histories: dict[str, dict]) -> None:
    """Save the two summary charts used in the report."""
    model_results = results[results["model"] != "Naive baseline"].copy()
    model_results = model_results.sort_values("best_validation_mae", ascending=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = ["#17365D" if i == 0 else "#9FBAD0" for i in range(len(model_results))]
    bars = ax.barh(
        model_results["model"],
        model_results["best_validation_mae"],
        color=colors,
        edgecolor="white",
    )
    ax.invert_yaxis()
    ax.axvline(
        results.loc[results["model"] == "Naive baseline", "best_validation_mae"].iloc[0],
        color="#A64132",
        linestyle="--",
        linewidth=1.8,
        label="Naive baseline",
    )
    ax.set_xlabel("Validation MAE (degrees C, lower is better)")
    ax.set_title("Validation MAE by Model")
    ax.legend(frameon=False)
    for bar, value in zip(bars, model_results["best_validation_mae"]):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "validation_mae_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    winner = model_results.iloc[0]["model"]
    winner_history = histories[winner]
    epochs = np.arange(1, len(winner_history["mae"]) + 1)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(epochs, winner_history["mae"], marker="o", label="Training MAE", color="#6C8EAD")
    ax.plot(epochs, winner_history["val_mae"], marker="o", label="Validation MAE", color="#17365D")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (degrees C)")
    ax.set_title(f"Training History for {winner}")
    ax.set_xticks(epochs)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "best_model_learning_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed()
    raw_data, temperature, feature_names = load_weather_data()
    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_mean,
        train_std,
        split_info,
    ) = prepare_datasets(raw_data, temperature)

    first_batch, _ = next(iter(train_dataset))
    input_shape = tuple(first_batch.shape[1:])
    naive_val_mae = evaluate_naive(val_dataset, train_mean, train_std)
    naive_test_mae = evaluate_naive(test_dataset, train_mean, train_std)
    print(f"Naive validation MAE: {naive_val_mae:.4f}")
    print(f"Naive test MAE: {naive_test_mae:.4f}")

    results = [
        {
            "model": "Naive baseline",
            "architecture": "Current temperature",
            "units": "N/A",
            "dropout": "N/A",
            "parameters": 0,
            "epochs_run": 0,
            "best_epoch": 0,
            "best_validation_mae": naive_val_mae,
            "test_mae": naive_test_mae,
            "training_minutes": 0.0,
        }
    ]
    histories: dict[str, dict] = {}
    checkpoints: dict[str, str] = {}

    for model_name, builder, architecture, units, dropout in MODEL_BUILDERS:
        print(f"\n{'=' * 72}\nTraining {model_name}\n{'=' * 72}")
        (
            model_path,
            history,
            best_epoch,
            best_val_mae,
            duration,
            parameters,
        ) = train_model(
            model_name, builder, input_shape, train_dataset, val_dataset
        )
        histories[model_name] = {
            key: [float(value) for value in values]
            for key, values in history.items()
        }
        checkpoints[model_name] = str(model_path)
        results.append(
            {
                "model": model_name,
                "architecture": architecture,
                "units": units,
                "dropout": dropout,
                "parameters": parameters,
                "epochs_run": len(history["mae"]),
                "best_epoch": best_epoch,
                "best_validation_mae": best_val_mae,
                "test_mae": np.nan,
                "training_minutes": duration / 60,
            }
        )

        pd.DataFrame(results).to_csv(RESULTS_DIR / "experiment_results_partial.csv", index=False)
        with open(RESULTS_DIR / "training_histories_partial.json", "w", encoding="utf-8") as handle:
            json.dump(histories, handle, indent=2)

    results_df = pd.DataFrame(results)
    model_rows = results_df[results_df["model"] != "Naive baseline"]
    winner_index = model_rows["best_validation_mae"].idxmin()
    winner_name = results_df.loc[winner_index, "model"]
    winner_model = keras.models.load_model(checkpoints[winner_name])
    winner_test_mae = float(winner_model.evaluate(test_dataset, verbose=0)[1])
    results_df.loc[winner_index, "test_mae"] = winner_test_mae

    results_df.to_csv(RESULTS_DIR / "experiment_results.csv", index=False)
    with open(RESULTS_DIR / "training_histories.json", "w", encoding="utf-8") as handle:
        json.dump(histories, handle, indent=2)
    with open(RESULTS_DIR / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": SEED,
                "max_epochs": MAX_EPOCHS,
                "batch_size": BATCH_SIZE,
                "sampling_rate": SAMPLING_RATE,
                "sequence_length": SEQUENCE_LENGTH,
                "delay_steps": DELAY,
                "feature_count": len(feature_names),
                "feature_names": feature_names,
                "input_shape": input_shape,
                "splits": split_info,
                "winner": winner_name,
                "winner_test_mae": winner_test_mae,
                "tensorflow_version": tf.__version__,
            },
            handle,
            indent=2,
        )

    plot_results(results_df, histories)
    print("\nFinal results")
    print(results_df.to_string(index=False))
    print(f"\nWinner: {winner_name}")
    print(f"Winner test MAE: {winner_test_mae:.4f} degrees C")


if __name__ == "__main__":
    main()
