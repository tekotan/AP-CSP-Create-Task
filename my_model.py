import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


class Generator:
    def __init__(self):
        pass

    def downsample(self, inputs, filters, batch_norm=True):
        conv = tf.keras.layers.Conv2D(
            filters=filters, strides=2, kernel_size=4, padding="same"
        )(inputs)
        batch_norm = tf.keras.layers.BatchNormalization()(conv)
        relu = tf.keras.layers.LeakyReLU(alpha=0.2)(batch_norm)
        return relu

    def upsample(self, inputs, filters, dropout=True):
        conv = tf.keras.layers.Conv2DTranspose(
            filters=filters, strides=2, kernel_size=2, padding="same"
        )(inputs)
        batch_norm = tf.keras.layers.BatchNormalization()(conv)
        if dropout:
            dropout = tf.keras.layers.Dropout(0.5)(batch_norm)
            relu = tf.keras.layers.LeakyReLU(alpha=0.2)(dropout)
        else:
            relu = tf.keras.layers.LeakyReLU(alpha=0.2)(batch_norm)
        return relu

    def get_model(self):
        inputs = tf.keras.layers.Input(shape=[None, None, 3])
        # encoder
        encoder_conv64 = self.downsample(inputs, 64, batch_norm=False)
        encoder_conv128 = self.downsample(encoder_conv64, 128)
        encoder_conv256 = self.downsample(encoder_conv128, 256)
        encoder_conv512_1 = self.downsample(encoder_conv256, 512)
        encoder_conv512_2 = self.downsample(encoder_conv512_1, 512)
        encoder_conv512_3 = self.downsample(encoder_conv512_2, 512)
        encoder_conv512_4 = self.downsample(encoder_conv512_3, 512)
        encoder_conv512_5 = self.downsample(encoder_conv512_4, 512)

        decoder_conv1024 = tf.keras.layers.concatenate(
            [self.upsample(encoder_conv512_5, 512), encoder_conv512_4]
        )
        decoder_conv1024 = tf.keras.layers.concatenate(
            [self.upsample(decoder_conv1024, 512), encoder_conv512_3]
        )
        decoder_conv1024 = tf.keras.layers.concatenate(
            [self.upsample(decoder_conv1024, 512), encoder_conv512_2]
        )
        decoder_conv1024 = tf.keras.layers.concatenate(
            [self.upsample(decoder_conv1024, 512), encoder_conv512_1]
        )
        decoder_conv512 = tf.keras.layers.concatenate(
            [self.upsample(decoder_conv1024, 256), encoder_conv256]
        )
        decoder_conv256 = tf.keras.layers.concatenate(
            [self.upsample(decoder_conv512, 128), encoder_conv128]
        )
        decoder_conv128 = tf.keras.layers.concatenate(
            [self.upsample(decoder_conv256, 256), encoder_conv64]
        )

        output_image = tf.keras.layers.Conv2DTranspose(
            3, (4, 4), strides=(2, 2), padding="same", activation="tanh"
        )(decoder_conv128)
        return tf.keras.Model(inputs=inputs, outputs=output_image)


class Discriminator:
    def __init__(self):
        pass

    def downsample(self, inputs, filters, batch_norm=True):
        conv = tf.keras.layers.Conv2D(
            filters=filters, strides=(2, 2), kernel_size=(4, 4)
        )(inputs)
        batch_norm = tf.keras.layers.BatchNormalization()(conv)
        relu = tf.keras.layers.LeakyReLU(alpha=0.2)(batch_norm)
        return relu

    def upsample(self, inputs, filers, dropout=True):
        conv = tf.keras.layers.Conv2DTranspose(
            filters=filters, strides=(2, 2), kernel_size=(4, 4)
        )(inputs)
        batch_norm = tf.keras.layers.BatchNormalization()(conv)
        if dropout:
            dropout = tf.keras.layers.Dropout(0.5)(batch_norm)
            relu = tf.keras.layers.LeakyReLU(alpha=0.2)(dropout)
        else:
            relu = tf.keras.layers.LeakyReLU(alpha=0.2)(batch_norm)
        return relu

    def get_model(self):
        inp = tf.keras.layers.Input(shape=[None, None, 3], name="input_image")
        tar = tf.keras.layers.Input(shape=[None, None, 3], name="target_image")

        inputs = tf.keras.layers.concatenate(
            [inp, tar], axis=-1
        )  # (bs, 256, 256, channels*2)
        print(inputs.get_shape())
        down1 = self.downsample(inputs, 64, False)  # (bs, 128, 128, 64)
        down2 = self.downsample(down1, 128)  # (bs, 64, 64, 128)
        down3 = self.downsample(down2, 256)  # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1, use_bias=False)(
            zero_pad1
        )  # (bs, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

        output_image = tf.keras.layers.Conv2D(1, 4, strides=1)(
            zero_pad2
        )  # (bs, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=output_image)


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
LAMBDA = 100


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(
        tf.zeros_like(disc_generated_output), disc_generated_output
    )

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss


def generate_images(model, test_input, tar):
    # the training=True is intentional here since
    # we want the batch statistics while running the model
    # on the test dataset. If we use training=False, we will get
    # the accumulated statistics learned from the training dataset
    # (which we don't want)
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ["Input Image", "Ground Truth", "Predicted Image"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")
    plt.savefig("best_so_far.png")
