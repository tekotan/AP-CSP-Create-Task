import os
import tensorflow as tf

_URL = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz"

path_to_zip = tf.keras.utils.get_file("facades.tar.gz", origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), "facades/")

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


def load_face(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    return tf.cast(image, tf.float32)


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(
        input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    real_image = tf.image.resize(
        real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    return input_image, real_image


def resize_face(input_image, height, width):
    input_image = tf.image.resize(
        input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return input_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3]
    )

    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def normalize_face(input_image):
    input_image = (input_image / 127.5) - 1
    return input_image


@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_face_train(image_file):
    image = load_face(image_file)
    image = normalize_face(image)

    return image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_face_test(image_file):
    image = load_face(image_file)
    image = normalize_face(image)

    return image


def get_datasets():
    train_dataset = tf.data.Dataset.list_files(PATH + "train/*.jpg")
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.map(
        load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_dataset = train_dataset.batch(1)

    test_dataset = tf.data.Dataset.list_files(PATH + "test/*.jpg")
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(1)
    return train_dataset, test_dataset


def get_datasets_face():
    train_dataset_man = tf.data.Dataset.list_files(PATH + "train/man/*.jpg")
    train_dataset_man = train_dataset_man.shuffle(BUFFER_SIZE)
    train_dataset_man = train_dataset_man.map(
        load_image_face_train, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_dataset_man = train_dataset_man.batch(1)

    train_dataset_woman = tf.data.Dataset.list_files(PATH + "train/woman/*.jpg")
    train_dataset_woman = train_dataset_woman.shuffle(BUFFER_SIZE)
    train_dataset_woman = train_dataset_woman.map(
        load_image_face_train, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_dataset_woman = train_dataset_woman.batch(1)

    test_dataset_man = tf.data.Dataset.list_files(PATH + "test/man/*.jpg")
    test_dataset_man = test_dataset_man.shuffle(BUFFER_SIZE)
    test_dataset_man = test_dataset_man.map(
        load_image_face_train, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    test_dataset_man = test_dataset_man.batch(1)

    test_dataset_woman = tf.data.Dataset.list_files(PATH + "test/woman/*.jpg")
    test_dataset_woman = test_dataset_woman.shuffle(BUFFER_SIZE)
    test_dataset_woman = test_dataset_woman.map(
        load_image_face_train, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    test_dataset_woman = test_dataset_woman.batch(1)

    train_dataset = tf.data.Dataset.zip((train_dataset_man, train_dataset_woman))
    test_dataset = tf.data.Dataset.zip((test_dataset_man, test_dataset_woman))
    return train_dataset, test_dataset
