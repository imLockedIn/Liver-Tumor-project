# Liver-Tumor-project
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import nibabel as nib
import numpy as np
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, Callback
from tensorflow.keras import regularizers


# Data Generator with Augmentation
def normalize_ct_volume(volume):
    """Improved CT normalization with histogram clipping"""
    # Clip to liver/tumor HU range (-100 to 200)
    volume = np.clip(volume, -100, 200)

    # Apply windowing for better contrast
    liver_window_center = 50
    liver_window_width = 150
    min_val = liver_window_center - liver_window_width/2
    max_val = liver_window_center + liver_window_width/2
    volume = np.clip(volume, min_val, max_val)

    # Normalize to 0-1
    return (volume - min_val) / (max_val - min_val)

class NiiDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_filenames, mask_filenames, batch_size, image_size, data_augmentation=False, **kwargs):
        super().__init__(**kwargs)
        self.image_filenames = image_filenames
        self.mask_filenames = mask_filenames
        self.batch_size = batch_size
        self.image_size = image_size
        self.data_augmentation = data_augmentation

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.mask_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        x = np.zeros((self.batch_size, *self.image_size, 1), dtype=np.float32)
        y = np.zeros((self.batch_size, *self.image_size, 3), dtype=np.float32)

        for i, (image_filename, mask_filename) in enumerate(zip(batch_x, batch_y)):
            # Load and normalize CT image
            im = nib.load(image_filename).get_fdata().astype(np.float32)
            im = normalize_ct_volume(im)  # REPLACED NORMALIZATION

            # Load mask (no normalization needed)
            ma = nib.load(mask_filename).get_fdata().astype(np.float32)

            # Select random slice
            slice_index = random.randint(0, im.shape[2] - 1)

            # Store image slice
            x[i, :, :, 0] = im[:, :, slice_index]

            # Create one-hot encoded mask
            y[i, :, :, 0] = (ma[:, :, slice_index] == 0).astype(np.float32)  # Background
            y[i, :, :, 1] = (ma[:, :, slice_index] == 1).astype(np.float32)  # Liver
            y[i, :, :, 2] = (ma[:, :, slice_index] == 2).astype(np.float32)  # Tumor

        if self.data_augmentation:
            # Apply random transformations
            for i in range(self.batch_size):
                x[i] = self._augment_image(x[i])
                y[i] = self._augment_image(y[i])

        return x, y

    def _augment_image(self, image):
        """Apply random augmentations"""
        # Random rotation
        if random.random() > 0.5:
            image = tf.image.rot90(image, k=random.randint(1, 3))

        # Random flip
        if random.random() > 0.5:
            image = tf.image.flip_left_right(image)
        if random.random() > 0.5:
            image = tf.image.flip_up_down(image)

        # Random zoom and resize
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            new_size = int(self.image_size[0] * scale)
            image = tf.image.resize(image, (new_size, new_size))
            image = tf.image.resize_with_crop_or_pad(image, self.image_size[0], self.image_size[1])

        return image


# UNet Model with Dropout and Regularization
def encoder(inputs, filters, pool_size):
    conv_pool = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    conv_pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(conv_pool)
    return conv_pool


def decoder(inputs, concat_input, filters, transpose_size):
    up = tf.keras.layers.Concatenate()(
        [tf.keras.layers.Conv2DTranspose(filters, transpose_size, strides=(2, 2), padding='same')(inputs),
         concat_input])
    up = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(up)
    up = tf.keras.layers.Dropout(0.5)(up)  # Dropout layer for regularization
    return up


def UNet(img_size=(512, 512, 1)):
    inputs = tf.keras.Input(shape=img_size)
    conv_pool1 = encoder(inputs, 32, (2, 2))
    conv_pool2 = encoder(conv_pool1, 64, (2, 2))
    conv_pool3 = encoder(conv_pool2, 128, (2, 2))
    conv_pool4 = encoder(conv_pool3, 256, (2, 2))
    bridge = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv_pool4)
    up6 = decoder(bridge, conv_pool3, 256, (2, 2))
    up7 = decoder(up6, conv_pool2, 128, (2, 2))
    up8 = decoder(up7, conv_pool1, 64, (2, 2))
    up9 = decoder(up8, inputs, 32, (2, 2))
    outputs = tf.keras.layers.Conv2D(3, (1, 1), activation='softmax')(up9)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model


# Custom Combined Loss (Dice + Binary Cross-Entropy)
def combined_loss(y_true, y_pred):
    weights = tf.constant([0.1, 1.0, 3.0])  # Background, liver, tumor
    weighted_bce = tf.keras.losses.binary_crossentropy(y_true, y_pred) * weights
    return tf.reduce_mean(weighted_bce) + (1 - dice_coef(y_true, y_pred))


# Dice Coefficient
def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    y_true_f = tf.cast(y_true_f, tf.float32)
    y_pred_f = tf.cast(y_pred_f, tf.float32)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


# Custom Callback for Plotting
class PlotMetricsCallback(Callback):
    def __init__(self):
        super(PlotMetricsCallback, self).__init__()
        self.epochs = []
        self.losses = []
        self.val_losses = []
        self.dice_scores = []
        self.val_dice_scores = []

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        self.losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])
        self.dice_scores.append(logs['dice_coef'])
        self.val_dice_scores.append(logs['val_dice_coef'])

        plt.clf()
        plt.subplot(1, 2, 1)
        plt.plot(self.epochs, self.losses, label="Train Loss", color='red')
        plt.plot(self.epochs, self.val_losses, label="Validation Loss", color='orange')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.epochs, self.dice_scores, label="Train Dice", color='blue')
        plt.plot(self.epochs, self.val_dice_scores, label="Validation Dice", color='cyan')
        plt.xlabel("Epoch")
        plt.ylabel("Dice Coefficient")
        plt.title("Dice Coefficient Curve")
        plt.legend()

        plt.tight_layout()
        plt.pause(0.1)


# Dataset Paths
images = 'Data/Task03_Liver'
train_images = sorted(glob(os.path.join(images, 'imagesTr', '*.nii.gz')))
train_masks = sorted(glob(os.path.join(images, 'labelsTr', '*.nii.gz')))
test_images = sorted(glob(os.path.join(images, 'imagesTsReal', '*.nii.gz')))
test_masks = sorted(glob(os.path.join(images, 'labelsTsReal', '*.nii.gz')))
batch_size = 1
image_size = (512, 512)

train_generator = NiiDataGenerator(train_images[:10], train_masks[:10], batch_size, image_size, data_augmentation=True)
val_generator = NiiDataGenerator(test_images[10:], test_masks[10:], batch_size, image_size)

# Model Training with Learning Rate Scheduling
model = UNet()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=combined_loss, metrics=[dice_coef])

checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
plot_metrics_callback = PlotMetricsCallback()

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_images) // batch_size,
    epochs=200,
    validation_data=val_generator,
    validation_steps=len(test_images) // batch_size,
    callbacks=[checkpoint, early_stopping, tensorboard, lr_scheduler, plot_metrics_callback]
)

model.save('liver_segmentation_model.h5')


