"""
Alzheimer's Disease Classification — fixed training pipeline.
Key stability fixes:
1) EarlyStopping monitors val_auc (mode=max)
2) Increased warmup/fine-tune patience
3) Gentler ReduceLROnPlateau
4) Consistent checkpoint monitor (val_auc)
5) Correct fine-tune order for distributed training: unfreeze -> compile (inside scope)
"""

import os
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, Model, Input
from keras.applications import EfficientNetV2B3
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


@dataclass
class Config:
    OUTPUT_DIR: str = "./outputs"
    IMG_SIZE: int = 224
    CHANNELS: int = 3
    CLASSES: Tuple[str, ...] = (
        "MildDemented",
        "ModerateDemented",
        "NonDemented",
        "VeryMildDemented",
    )
    NUM_CLASSES: int = 4
    BATCH_SIZE: int = 32
    EPOCHS_WARMUP: int = 15
    EPOCHS_FINETUNE: int = 25
    N_FOLDS: int = 3
    LR_WARMUP: float = 3e-4
    LR_FINETUNE: float = 1e-4
    MIN_LR: float = 1e-6
    WEIGHT_DECAY: float = 1e-4
    LABEL_SMOOTHING: float = 0.1
    DROPOUT: float = 0.45
    UNFREEZE_LAST_N: int = 60

    # stop-too-early fix
    PATIENCE_WARMUP: int = 12
    PATIENCE_FT: int = 15
    REDUCE_PATIENCE: int = 6
    REDUCE_FACTOR: float = 0.5


cfg = Config()
Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

strategy = (
    tf.distribute.MirroredStrategy()
    if len(gpus) > 1
    else tf.distribute.get_strategy()
)


def _parse(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=cfg.CHANNELS, expand_animations=False)
    img.set_shape([None, None, cfg.CHANNELS])
    img = tf.image.resize(img, [cfg.IMG_SIZE, cfg.IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def _augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.image.random_saturation(img, 0.85, 1.15)
    img = tf.image.random_hue(img, 0.05)
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = img + tf.random.normal(tf.shape(img), stddev=0.015)
    return tf.clip_by_value(img, 0.0, 1.0), label


def build_dataset(paths: np.ndarray, labels_oh: np.ndarray, augment: bool = False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels_oh))
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(min(len(paths), 4096), seed=42)
    return ds.batch(cfg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def load_backbone_weights(backbone: Model) -> None:
    fname = "efficientnetv2-b3_notop.h5"
    for d in [Path.home() / ".keras" / "models", Path("/root/.keras/models"), Path("/tmp")]:
        p = d / fname
        if p.exists():
            backbone.load_weights(str(p), by_name=True, skip_mismatch=True)
            print(f"Loaded cached weights: {p}")
            return
    print("No cached ImageNet weights found. Continuing with random init.")


def build_model(backbone_trainable: bool = False) -> Model:
    inp = Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.CHANNELS))
    x = layers.Rescaling(255.0)(inp)

    backbone = EfficientNetV2B3(
        include_top=False,
        weights=None,
        input_shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.CHANNELS),
        include_preprocessing=True,
    )
    load_backbone_weights(backbone)
    backbone.trainable = backbone_trainable

    x = backbone(x, training=backbone_trainable)
    x_avg = layers.GlobalAveragePooling2D()(x)
    x_max = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([x_avg, x_max])
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(cfg.DROPOUT)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(cfg.DROPOUT / 2)(x)
    out = layers.Dense(cfg.NUM_CLASSES, activation="softmax", dtype="float32")(x)
    return Model(inp, out, name="AlzheimerNet_fixed")


def compile_model(model: Model, lr: float) -> Model:
    model.compile(
        optimizer=AdamW(learning_rate=lr, weight_decay=cfg.WEIGHT_DECAY),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=cfg.LABEL_SMOOTHING),
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def unfreeze_last_n(model: Model, n: int):
    bb = next((l for l in model.layers if "efficientnetv2" in l.name.lower()), None)
    if bb is None:
        raise RuntimeError("Backbone not found")
    bb.trainable = True
    for lyr in bb.layers[:-n]:
        lyr.trainable = False


def make_callbacks(fold: int, phase: str, total_epochs: int):
    patience = cfg.PATIENCE_WARMUP if phase == "warm" else cfg.PATIENCE_FT
    return [
        ModelCheckpoint(
            filepath=f"{cfg.OUTPUT_DIR}/ckpt_f{fold}_{phase}.weights.h5",
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            start_from_epoch=3,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=cfg.REDUCE_FACTOR,
            patience=cfg.REDUCE_PATIENCE,
            min_lr=cfg.MIN_LR,
            verbose=1,
        ),
    ]


def safe_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    yb = label_binarize(y_true, classes=list(range(cfg.NUM_CLASSES)))
    return float(roc_auc_score(yb, probs, multi_class="ovr", average="weighted"))


def train_kfold(df_train: pd.DataFrame):
    paths = df_train["path"].values
    labels = df_train["label_idx"].values
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=42)
    metrics = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(paths, labels), 1):
        print(f"\nFold {fold}/{cfg.N_FOLDS}")
        tr_p, tr_l = paths[tr_idx], labels[tr_idx]
        va_p, va_l = paths[val_idx], labels[val_idx]
        tr_oh = keras.utils.to_categorical(tr_l, cfg.NUM_CLASSES)
        va_oh = keras.utils.to_categorical(va_l, cfg.NUM_CLASSES)
        train_ds = build_dataset(tr_p, tr_oh, augment=True)
        val_ds = build_dataset(va_p, va_oh, augment=False)

        with strategy.scope():
            model = build_model(backbone_trainable=False)
            model = compile_model(model, cfg.LR_WARMUP)

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.EPOCHS_WARMUP,
            callbacks=make_callbacks(fold, "warm", cfg.EPOCHS_WARMUP),
            verbose=1,
        )

        # Critical fix: do both unfreeze and compile in scope before fine-tune fit
        with strategy.scope():
            unfreeze_last_n(model, cfg.UNFREEZE_LAST_N)
            model = compile_model(model, cfg.LR_FINETUNE)

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.EPOCHS_FINETUNE,
            callbacks=make_callbacks(fold, "fine", cfg.EPOCHS_FINETUNE),
            verbose=1,
        )

        probs = model.predict(val_ds, verbose=0)
        preds = np.argmax(probs, axis=1)
        acc = float(np.mean(preds == va_l))
        f1 = float(f1_score(va_l, preds, average="weighted"))
        auc = safe_auc(va_l, probs)
        metrics.append({"fold": fold, "val_acc": acc, "val_f1": f1, "val_auc": auc})
        print(f"Fold {fold} => acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")

        del model
        gc.collect()

    return pd.DataFrame(metrics)


if __name__ == "__main__":
    print("This script contains the fixed training pipeline.")
    print("Load your dataframe with columns: path, label_idx then call train_kfold(df_train).")
