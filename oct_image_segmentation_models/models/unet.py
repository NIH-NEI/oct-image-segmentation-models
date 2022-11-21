from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    concatenate,
    Conv2D,
    Dropout,
    Input,
    MaxPooling2D,
    UpSampling2D,
)
from typeguard import typechecked
from typing import Tuple


def batch_activate(x):
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def convolution_block(x, filters, kernel):
    x = Conv2D(
        filters, kernel, strides=(1, 1), padding="same", dilation_rate=1
    )(x)
    x = batch_activate(x)
    return x


def unet_enc_block(inp, size, enc_kernel=(3, 3), conv_layers=2, pool="max"):
    for _ in range(conv_layers):
        inp = convolution_block(inp, filters=size, kernel=enc_kernel)
    c = inp
    if pool == "max":
        inp = MaxPooling2D(pool_size=(2, 2))(inp)
    return [inp, c]


def upsample_conv(inp, size, kernel):
    x = UpSampling2D()(inp)
    x = convolution_block(x, filters=size, kernel=kernel)
    return x


def unet_dec_block(
    inp, size, concat_map, enc_kernel=(3, 3), dec_kernel=(2, 2), conv_layers=2
):
    x = upsample_conv(inp, size, dec_kernel)

    x = concatenate([x, concat_map])
    [x, _] = unet_enc_block(
        x, size, enc_kernel=enc_kernel, conv_layers=conv_layers, pool=False
    )

    return x


@typechecked
def unet(
    *,
    input_channels: int,
    output_channels: int,
    start_neurons: int = 8,
    pool_layers: int = 4,
    conv_layers: int = 2,
    enc_kernel: tuple = (3, 3),
    dec_kernel: tuple = (2, 2),
) -> Tuple[Model, dict]:
    inp = Input(batch_shape=(None, None, None, input_channels))

    x = inp

    enc = []

    for i in range(pool_layers):
        [x, c] = unet_enc_block(
            x,
            start_neurons * (2**i),
            enc_kernel=enc_kernel,
            conv_layers=conv_layers,
        )

        enc.append(c)

    [x, _] = unet_enc_block(
        x,
        start_neurons * (2**pool_layers),
        enc_kernel=enc_kernel,
        conv_layers=conv_layers,
        pool=False,
    )
    x = Dropout(0.5)(x)

    for i in range(pool_layers):
        x = unet_dec_block(
            x,
            start_neurons * (2 ** (pool_layers - 1 - i)),
            enc[pool_layers - 1 - i],
            enc_kernel=enc_kernel,
            dec_kernel=dec_kernel,
            conv_layers=conv_layers,
        )

    o = Conv2D(
        filters=output_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation="softmax",
    )(x)

    hyperparameters = {
        "input_channels": input_channels,
        "output_channels": output_channels,
        "start_neurons": start_neurons,
        "pool_layers": pool_layers,
        "conv_layers": conv_layers,
        "enc_kernel": enc_kernel,
        "dec_kernel": dec_kernel,
    }

    return Model(inputs=inp, outputs=o), hyperparameters
