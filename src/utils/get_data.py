import numpy as np
import tensorflow as tf

SAMPLE_IMAGE_URLS = [
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg",
]


def load_real_sample_data(num_samples: int = 64) -> np.ndarray:
    image_list = []

    for i, url in enumerate(SAMPLE_IMAGE_URLS):
        file_name = f"sample_{i}.jpg"
        img_path = tf.keras.utils.get_file(file_name, origin=url)

        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        image_list.append(img_array)

    images = np.array(image_list)

    # Tile/cycle through the unique downloaded images to reach target batch size
    repeats = (num_samples // len(images)) + 1
    batched_images = np.tile(images, (repeats, 1, 1, 1))[:num_samples]

    # MobileNetV2 preprocessing (scales pixel values to [-1, 1])
    preprocessed_images: np.ndarray = tf.keras.applications.mobilenet_v2.preprocess_input(
        batched_images
    )
    return preprocessed_images
