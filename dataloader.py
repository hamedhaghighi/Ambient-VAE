import tensorflow as tf
from tensorflow.python.framework.ops import convert_to_tensor
from utils import save_images, get_image
import numpy as np

class Dataloader():
    def __init__(self, repeat_num, batch_size, output_size):
        self.repeat_num = repeat_num
        self.batch_size = batch_size
        self.output_size = output_size

    def make_dataset(self, data_ob, dataset):
        if dataset == 'celebA':
            self.next_train_batch, self.init_train = self.make_CelebA(data_ob.train_data_list)
            self.next_val_batch, self.init_val = self.make_CelebA(data_ob.val_data_list)
        elif dataset == 'mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            self.next_train_batch, self.init_train = self.create_mnist_dataset(x_train, y_train, self.batch_size)
            self.next_val_batch, self.init_val = self.create_mnist_dataset(x_test, y_test, self.batch_size)
        return self.init_train, self.init_val

    def get_next_train_batch(self , sess):
        return sess.run(self.next_train_batch)

    def get_next_val_batch(self, sess):
        return sess.run(self.next_val_batch)


    def create_mnist_dataset(self, data, labels, batch_size):
        def gen():
            for image, _ in zip(data, labels):
                yield np.expand_dims(image, axis=-1)/255.0
        ds = tf.data.Dataset.from_generator(gen, tf.float32, (28,28,1))
        ds = ds.repeat().batch(batch_size)
        iterator = tf.data.Iterator.from_structure(
            ds.output_types, ds.output_shapes)
        next_x = iterator.get_next()
        init_op = iterator.make_initializer(ds)
        return next_x, init_op
    

    def make_CelebA(self, data_list):
        dataset = tf.data.Dataset.from_tensor_slices(
            convert_to_tensor(data_list, dtype=tf.string))
        dataset = dataset.map(lambda filename: tuple(tf.py_func(self._read_by_function,
                                                                [filename], [tf.double])), num_parallel_calls=16)
        dataset = dataset.repeat(self.repeat_num)
        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(self.batch_size))

        iterator = tf.data.Iterator.from_structure(
            dataset.output_types, dataset.output_shapes)
        next_x = tf.squeeze(iterator.get_next())
        init_op = iterator.make_initializer(dataset)
        return next_x, init_op

    def _parse_function(self, images_filenames):

        image_string = tf.read_file(images_filenames)
        image_decoded = tf.image.decode_and_crop_jpeg(
            image_string, crop_window=[218 / 2 - 54, 178 / 2 - 54, 108, 108], channels=3)
        image_resized = tf.image.resize_images(
            image_decoded, [self.output_size, self.output_size])
        image_resized = image_resized / 127.5 - 1

        return image_resized

    def _read_by_function(self, filename):

        array = get_image(filename, 108, is_crop=True, resize_w=self.output_size,
                          is_grayscale=False)
        real_images = np.array(array)
        return real_images
