import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import xarray as xr

# DATA
BATCH_SIZE = 8
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (64, 64, 32, 1)
NUM_CLASSES = 11

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 60

# TUBELET EMBEDDING
PATCH_SIZE = (2, 2, 2)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 2
NUM_LAYERS = 8

class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=3,
            strides=patch_size,
            padding="same",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        return projected_patches

class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens


def create_vivit_classifier(
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES,
):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    x = layers.Conv3D(32,1, padding = "same",kernel_regularizer='l1', activation = "relu")(inputs)
    x = layers.BatchNormalization()(x)

    x1 = TubeletEmbedding(32,2)(x)
    x1 = layers.BatchNormalization()(x1)
    patches1 = layers.Conv3D(32,1, padding = "same",kernel_regularizer='l1', activation = "relu")(x1)
    patches1 = layers.BatchNormalization()(patches1)

    x2 = TubeletEmbedding(32,2)(patches1)
    x2 = layers.BatchNormalization()(x2)
    patches2 = layers.Conv3D(32,1, padding = "same",kernel_regularizer='l1', activation = "relu")(x2)
    patches2 = layers.BatchNormalization()(patches2)

    p2 = layers.UpSampling3D((2,2,2))(patches2)
    patches1 = layers.Add()([patches1,p2])

    p2 = layers.UpSampling3D((2,2,2))(patches1)
    patches0 = layers.Add()([x,p2])

    patches01 = layers.Reshape((-1,32))(patches0)
    patches11 = layers.Reshape((-1,32))(patches1)
    patches21 = layers.Reshape((-1,32))(patches2)


    # Encode patches.
    encoded_patches0 = PositionalEncoder(32)(patches01)
    encoded_patches1 = PositionalEncoder(32)(patches11)
    encoded_patches2 = PositionalEncoder(32)(patches21)

    list = []

    # Create multiple layers of the Transformer block.
    encoded_patches = encoded_patches0
    for _ in range(1):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=32 // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=32 * 4, activation=tf.nn.gelu),
                layers.Dense(units=32, activation=tf.nn.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])
    list.append(encoded_patches)
    


    encoded_patches = encoded_patches1
    for _ in range(4):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=32 // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=32 * 4, activation=tf.nn.gelu),
                layers.Dense(units=32, activation=tf.nn.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])
    list.append(encoded_patches)



    encoded_patches = encoded_patches2
    for _ in range(8):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=32 // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=32 * 4, activation=tf.nn.gelu,kernel_regularizer='l1'),
                layers.Dense(units=32, activation=tf.nn.gelu,kernel_regularizer='l1'),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])
    list.append(encoded_patches)



    # Layer normalization and Global average pooling.
    x0 = layers.LayerNormalization(epsilon=layer_norm_eps)(list[0])
    x1 = layers.LayerNormalization(epsilon=layer_norm_eps)(list[1])
    x2 = layers.LayerNormalization(epsilon=layer_norm_eps)(list[2])
    x0 = layers.Reshape((64,64,32,32))(x0)
    x1 = layers.Reshape((32,32,16,32))(x1)
    x2 = layers.Reshape((16,16,8,32))(x2)
    patches01 = layers.Reshape((64,64,32,32))(patches01)

    x0 = layers.Concatenate(axis=-1)([patches01,x0])
    x0 = layers.Conv3D(32,3,padding = "same", kernel_regularizer='l1', activation = "relu")(x0)
    x0 = layers.BatchNormalization()(x0)

    x1 = layers.Concatenate(axis=-1)([patches1,x1])#try
    x1 = layers.Conv3D(32,3,padding = "same", kernel_regularizer='l1', activation = "relu")(x1)
    x1 = layers.BatchNormalization()(x1)

    x2 = layers.Concatenate(axis=-1)([patches2,x2])#try
    x2 = layers.Conv3D(32,3,padding = "same", kernel_regularizer='l1', activation = "relu")(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.UpSampling3D((2,2,2))(x2)
    x2 = layers.Conv3D(32,3,padding = "same", kernel_regularizer='l1', activation = "relu")(x2)
    x2 = layers.BatchNormalization()(x2)

    x1 = layers.UpSampling3D((2,2,2))(x1)
    x1 = layers.Conv3D(32,3,padding = "same", kernel_regularizer='l1', activation = "relu")(x1)
    x2 = layers.UpSampling3D((2,2,2))(x2)
    x2 = layers.Conv3D(32,3,padding = "same", kernel_regularizer='l1', activation = "relu")(x2)

    #representation = layers.Add()([x1,x2])
    representation = layers.Concatenate(axis=-1)([x0,x1,x2])

    representation = layers.Conv3D(32,1, padding = "same",kernel_regularizer='l1', activation = "relu")(representation)
    representation = layers.BatchNormalization()(representation)
    outputs = layers.Conv2D(11,1,1)(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = create_vivit_classifier()

model.summary()

train_x = xr.open_dataset("train_x").to_array().to_numpy()[0]
train_y = xr.open_dataset("train_y").to_array().to_numpy()[0]
test_x = xr.open_dataset("test_x").to_array().to_numpy()[0]
test_y = xr.open_dataset("test_y").to_array().to_numpy()[0]
valid_x = xr.open_dataset("valid_x").to_array().to_numpy()[0]
valid_y = xr.open_dataset("valid_y").to_array().to_numpy()[0]



import csv
import numpy as np
 
with open('train_x.csv', 'r') as f:
    data = list(csv.reader(f, delimiter=";"))
 
np.array(data.to_array()).shape

@tf.function
def preprocess(frames: tf.Tensor, label: tf.Tensor):
    """Preprocess the frames tensors and parse the labels."""
    # Preprocess images
    frames = tf.image.convert_image_dtype(
        frames[
            ..., tf.newaxis
        ],  # The new axis is to help for further processing with Conv3D layers
        tf.float32,
    )
    # Parse label
    label = tf.cast(label, tf.float32)
    return frames, label


def prepare_dataloader(
    videos: np.ndarray,
    labels: np.ndarray,
    loader_type: str = "train",
    batch_size: int = BATCH_SIZE,
):
    """Utility function to prepare the dataloader."""
    dataset = tf.data.Dataset.from_tensor_slices((videos, labels))

    if loader_type == "train":
        dataset = dataset.shuffle(BATCH_SIZE * 2)

    dataloader = (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataloader

trainloader = prepare_dataloader(train_x, train_y, "train")
validloader = prepare_dataloader(valid_x, valid_y, "valid")
testloader = prepare_dataloader(test_x, test_y, "test")

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    )


log_dir = fr"C:/Users/Leander/Skole/H2022/logs/fit/64x64x32_1epochtest" 
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

_ = model.fit(trainloader, epochs=1, validation_data=validloader, callbacks=[tensorboard_callback])

model.save("Transformer_64x64x32")
