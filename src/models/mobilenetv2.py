from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from configs.config import NUM_CLASSES, LEARNING_RATE


def build_mobilenetv2():

    # Load pretrained MobileNetV2
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze pretrained layers
    base_model.trainable = False

    # Custom classification head
    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation="relu")(x)

    x = Dropout(0.5)(x)

    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(
        inputs=base_model.input,
        outputs=outputs
    )

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model