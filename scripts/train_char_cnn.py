
import argparse, os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.char_cnn import build_cnn
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train(data_dir, epochs=20, batch_size=64, save_path='models/char_cnn.h5', img_size=28):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15,
                                 rotation_range=5, width_shift_range=0.05,
                                 height_shift_range=0.05, shear_range=0.02,
                                 zoom_range=0.05)
    train_gen = datagen.flow_from_directory(data_dir, target_size=(img_size,img_size),
                                            color_mode='grayscale', class_mode='categorical',
                                            batch_size=batch_size, subset='training')
    val_gen = datagen.flow_from_directory(data_dir, target_size=(img_size,img_size),
                                          color_mode='grayscale', class_mode='categorical',
                                          batch_size=batch_size, subset='validation')
    num_classes = train_gen.num_classes
    model = build_cnn(input_shape=(img_size,img_size,1), num_classes=num_classes)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    callbacks = [
        ModelCheckpoint(save_path, save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(patience=6, monitor='val_accuracy', mode='max', restore_best_weights=True)
    ]
    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    print('Training finished. Model saved to', save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Path to character dataset (ImageFolder format)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--save-path', default='models/char_cnn.h5')
    args = parser.parse_args()
    train(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, save_path=args.save_path)
