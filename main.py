import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import keras



file_path = "train_pandas.csv"


def text_pipeline(file_path = file_path, max_tokens = 15000):
    df = pd.read_csv(file_path)

    train_dataset_test = tf.data.Dataset.from_tensor_slices((df['FB_WSW'].values, df['label'].values))
    train_dataset_test = train_dataset_test.shuffle(buffer_size=16).batch(16)

    test_vectorization = layers.TextVectorization(
    ngrams=2,
    max_tokens=max_tokens,
    output_mode="tf_idf")
    train_test_train_ds = train_dataset_test.map(lambda x, y: x)
    test_vectorization.adapt(train_test_train_ds)

    return test_vectorization

