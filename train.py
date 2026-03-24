import argparse
import h5py
import numpy as np
import os
import argparse as _argparse

_pre = _argparse.ArgumentParser(add_help=False)
_pre.add_argument('--omp-threads', type=int, default=2)
_pre.add_argument('--intra-threads', type=int, default=2)
_pre.add_argument('--inter-threads', type=int, default=1)
_pre.add_argument('--tf-onednn', choices=['0','1'], default='0')
_pre.add_argument('--gpu-memory', type=int, default=0)
_pre.add_argument('--low-priority', action='store_true')
_pre_args, _remaining = _pre.parse_known_args()

os.environ['TF_ENABLE_ONEDNN_OPTS'] = _pre_args.tf_onednn
os.environ['OMP_NUM_THREADS'] = str(_pre_args.omp_threads)
os.environ['TF_NUM_INTRAOP_THREADS'] = str(_pre_args.intra_threads)
os.environ['TF_NUM_INTEROP_THREADS'] = str(_pre_args.inter_threads)

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence


class HDF5Sequence(Sequence):
    def __init__(self, x_path, y_path, batch_size, datagen=None, shuffle=True):
        self.xf = h5py.File(x_path, 'r')
        self.yf = h5py.File(y_path, 'r')
        self.x = self.xf['x']
        self.y = self.yf['y']
        self.batch_size = batch_size
        self.datagen = datagen
        self.indexes = np.arange(self.x.shape[0])
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(self.x.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        if len(batch_idx) == 0:
            return np.array([]), np.array([])

        sorted_idx = np.sort(batch_idx)
        batch_x_sorted = self.x[sorted_idx].astype('float32') / 255.0
        batch_y_sorted = np.squeeze(self.y[sorted_idx]).astype('float32')

        positions = np.searchsorted(sorted_idx, batch_idx)
        batch_x = batch_x_sorted[positions]
        batch_y = batch_y_sorted[positions]

        if self.datagen is not None:
            gen = self.datagen.flow(batch_x, batch_y, batch_size=batch_x.shape[0], shuffle=False)
            batch_x, batch_y = next(gen)

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def close(self):
        try:
            self.xf.close()
            self.yf.close()
        except Exception:
            pass


def build_model(input_shape=(96, 96, 3), fine_tune_at=None):
    base = DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)

    if fine_tune_at is None:
        base.trainable = False
    else:
        base.trainable = True
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False

    inp = layers.Input(shape=input_shape)
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def main(args):
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        preprocessing_function=lambda x: x
    )

    if _pre_args.gpu_memory and _pre_args.gpu_memory > 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=_pre_args.gpu_memory)]
            )

    if _pre_args.low_priority:
        try:
            if os.name == 'nt':
                import ctypes
                kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
                BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
                kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
            else:
                os.nice(10)
        except Exception:
            pass

    def safe_fit(model, *fargs, **fkwargs):
        try:
            return model.fit(*fargs, **fkwargs)
        except TypeError:
            fkwargs.pop('workers', None)
            fkwargs.pop('use_multiprocessing', None)
            return model.fit(*fargs, **fkwargs)

    if args.test_samples and args.test_samples > 0:

        with h5py.File(args.train_x, 'r') as fx:
            x_small = fx['x'][: args.test_samples].astype('float32') / 255.0

        with h5py.File(args.train_y, 'r') as fy:
            y_small = np.squeeze(fy['y'][: args.test_samples]).astype('float32')

        model = build_model(
            input_shape=(x_small.shape[1], x_small.shape[2], x_small.shape[3])
        )

        safe_fit(
            model,
            datagen.flow(x_small, y_small, batch_size=args.batch_size),
            steps_per_epoch=max(1, len(x_small) // args.batch_size),
            epochs=args.epochs,
            workers=0,
            use_multiprocessing=False
        )

    else:

        seq = HDF5Sequence(
            args.train_x,
            args.train_y,
            args.batch_size,
            datagen=datagen,
            shuffle=True
        )

        model = build_model(
            input_shape=(seq.x.shape[1], seq.x.shape[2], seq.x.shape[3])
        )

        safe_fit(
            model,
            seq,
            epochs=args.epochs,
            workers=0,
            use_multiprocessing=False
        )

        seq.close()

    out_path = args.output or 'densenet201_model.h5'
    model.save(out_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train-x', required=True)
    p.add_argument('--train-y', required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--test-samples', type=int, default=0)
    p.add_argument('--output', type=str, default=None)

    args = p.parse_args(_remaining)
    main(args)