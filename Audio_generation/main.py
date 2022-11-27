
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from audio import load_generic_audio, get_audio_from_model, load_train_valid_filenames

import time
from scipy.io.wavfile import write
import tqdm
import tensorflow as tf
import os
import pickle
from pathlib import Path
#from oldcode.old2.wavenet_model import WaveNet
from wavenet_model import WaveNet
import numpy as np
import logging
from tensorflow.python.client import device_lib

# PATHS
LJ_DIRECTORY = Path('./data_wav/')  # Dataset Directory
GENERATED_AUDIO_OUTPUT_DIRECTORY = Path('./saved_data/output/generated/')
MODEL_OUTPUT_DIRECTORY = Path('./saved_data/output/model/')
CHECKPOINTDIRECTORY = Path('./saved_data/output/model/')
LOG_DIRECTORY = Path('./saved_data/model_logs/')



def convertToFrames(audio, frame_size, frame_shift):
  X = []
  Y = []
  audio_len = len(audio)
  for i in range(0, audio_len - frame_size - 1, frame_shift):
    frame = audio[i:i + frame_size]
    if len(frame) < frame_size:
      break
    if i + frame_size >= audio_len:
        break
    temp = audio[i + frame_size]
    target_val = int((np.sign(temp) * (np.log(1 + 256 * abs(temp)) / (np.log(1 + 256))) + 1) / 2.0 * 255)
    X.append(frame.reshape(frame_size, 1))
    Y.append((np.eye(256)[target_val]))
  return np.array(X), np.array(Y)


def createDataset(audio_data, batch_size, frame_size, frame_shift):
  data_frames = convertToFrames(audio_data, frame_size, frame_shift)
  #print("data_frames: ", data_frames[0].shape)

  ds = tf.data.Dataset.from_tensor_slices(data_frames)
  
  ds = ds.repeat()

  ds = ds.batch(batch_size)
  #ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return ds



def generateAudioFromModel(model, model_id, sr=16000, frame_size=256, num_files=1, generated_seconds=1, validation_audio=None):
    audio_context = validation_audio[:frame_size]

    for i in range(num_files):
        new_audio = get_audio_from_model(model, sr, generated_seconds, audio_context, frame_size)
        audio_context = validation_audio[i:i + frame_size]
        log_dir = Path(LOG_DIRECTORY / model_id)
        wavname = (model_id + "_sample_" + str(i) + '.wav')
        outputPath = "saved_data/"+ wavname
        print("Saving File", outputPath)
        write(outputPath, sr, new_audio)


def trainModel():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.33

    session = tf.compat.v1.InteractiveSession(config=config)
    # Initialize Variables
    hyperparameters = {"frame_size": 256,
                       "frame_shift": 128,
                       "sample_rate": 16000,
                       "batch_size": 128,
                       "epochs": 5,
                       "num_filters": 64,
                       "filter_size": 2,
                       "dilation_rate": 2,
                       "num_layers": 40}



    # Get Audio
    print("Retrieving Audio")
    training_files, validation_files = load_train_valid_filenames(LJ_DIRECTORY, num_samples=100,
                                                                  percent_training=0.9)
    #validation_files = training_files
    print("Training files",len(training_files))
    print("Validation files",len(validation_files))
    print("Concatting Audio")
    training_audio, validation_audio = load_generic_audio(training_files, validation_files, sample_rate=hyperparameters["sample_rate"])
    training_audio_length = len(training_audio)
    validation_audio_length = len(validation_audio)
    print("Audio Retrieved")
    print("Training Audio Length:", training_audio_length)
    print("Valdiation Audio Length:", validation_audio_length)

    training_dataset = createDataset(training_audio, hyperparameters["batch_size"], hyperparameters["frame_size"], hyperparameters["frame_shift"])
    validation_dataset = createDataset(validation_audio, hyperparameters["batch_size"], hyperparameters["frame_size"], hyperparameters["frame_shift"])


    # CALLBACKS
    model_id = str(int(time.time()))
    #p = Path.mkdir(LOG_DIRECTORY / model_id)
    #p.mkdir(parents=True)
    log_dir = Path(LOG_DIRECTORY / model_id)
    log_dir.mkdir(parents=True, exist_ok=True)
    full_path = log_dir.absolute()
    logdir_path_string = full_path.as_posix()
    print(logdir_path_string)
    checkpoint_filepath = MODEL_OUTPUT_DIRECTORY / model_id / "checkpoint.ckpt"

    tensorboard_callback = TensorBoard(log_dir=logdir_path_string,
                                       histogram_freq=0)
    earlystopping_callback = EarlyStopping(monitor='val_accuracy',
                                           min_delta=0.01,
                                           patience=10,
                                           verbose=0,
                                           restore_best_weights=True)
 
    checkpoint_path = Path(CHECKPOINTDIRECTORY / model_id / 'checkpoint' / (model_id+"_checkpoint.hdf5"))
    temp_path = Path(CHECKPOINTDIRECTORY / model_id / 'checkpoint')
    temp_path.mkdir(parents=True, exist_ok=True)


    full_path = checkpoint_path.absolute()
    checkpoint_path_string = full_path.as_posix()
    print(checkpoint_path_string)

    checkpoint_callback = ModelCheckpoint(
        checkpoint_path_string, monitor='val_accuracy', verbose=1,
        save_best_only=False, save_weights_only=False,
        save_frequency=1)


    def saveToFile(filepath, object_to_save, pickle_true = True):
      print("Writing mandatory Hyperparameter logs to {}\n".format(filepath))
      Path.touch(filepath)
      with open(filepath, 'wb') as f:
          if(pickle_true):
            pickle.dump(object_to_save, f, pickle.HIGHEST_PROTOCOL)
          else:
            pickle.dump(object_to_save, f)


    print("Writing mandatory Hyperparameter logs to {}\n".format(log_dir / "hyperparameters.pkl"))
    # Write hyper parameters to log file
    hyperparamter_filename = log_dir / "hyperparameters.pkl"
    Path.touch(hyperparamter_filename)
    with open(hyperparamter_filename, 'wb') as f:
        pickle.dump(hyperparameters, f, pickle.HIGHEST_PROTOCOL)

    # Write validation file names to file
    validation_filename = log_dir / "validation_files.pkl"
    #Path.touch(validation_filename)
    with open(validation_filename, 'wb') as fp:
        pickle.dump(validation_files, fp)

    # Write training file names to file
    training_filename = log_dir / "training_files.pkl"
    #print(training_files)
    with open(training_filename, 'wb') as fp:
        pickle.dump(training_files, fp)

    print("Starting Model Training...\n")
    print("Model ID", model_id)


    sub = WaveNet(num_filters=hyperparameters["num_filters"],
                  filter_size=hyperparameters["filter_size"],
                  dilation_rate=hyperparameters["dilation_rate"],
                  num_layers=hyperparameters["num_layers"],
                  input_size=hyperparameters["frame_size"])
    model = sub.model()
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])


    print(device_lib.list_local_devices())

    model.fit(training_dataset,
              epochs=hyperparameters["epochs"],
              steps_per_epoch=training_audio_length // hyperparameters["batch_size"],
              validation_data=validation_dataset,
              validation_steps= validation_audio_length // hyperparameters["batch_size"],
              verbose=1,
              callbacks=[tensorboard_callback, earlystopping_callback,checkpoint_callback])

    print('Saving model...')
    model.save(MODEL_OUTPUT_DIRECTORY / model_id / ('final_model_' + model_id + '_' + '.h5'))
    print("Model saved.", model_id)

    print("Generating Audio.")
    generateAudioFromModel(model, model_id, sr=hyperparameters["sample_rate"], frame_size=hyperparameters["frame_size"],
                           num_files=1, generated_seconds=1, validation_audio=validation_audio)
    print("Program Complete.")
    return model_id

if __name__ == '__main__':
    trainModel()