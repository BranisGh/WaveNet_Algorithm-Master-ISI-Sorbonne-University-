import librosa
import random
import os
import numpy as np
import fnmatch
import tqdm
import tensorflow as tf

"""
This file contains function necessary for working with audio data and input and outputting audio from Wavenet..
"""

LJ_DIRECTORY = "C:\\Users\\pasca\\Desktop\\audioProject\\data\\ljdataset"


# Gets all names of files within a directory
def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def load_train_valid_filenames(directory, num_samples=None, percent_training=0.9):
    randomized_files = find_files(directory)
    random.shuffle(randomized_files)
    if num_samples is not None:
        randomized_files = randomized_files[:num_samples]
    number_of_training_samples = int(round(percent_training * len(randomized_files)))
    training_files, validation_files = randomized_files[:number_of_training_samples], randomized_files[
                                                                                      number_of_training_samples:]
    return training_files, validation_files

# Reads the training/validation audio and concats it into a single array for the NN
def load_generic_audio(training_files, validation_files, sample_rate=16000):
    '''Generator that yields audio waveforms from the directory.'''

    # Concat training data
    training_data = []
    for training_filename in training_files:
        audio, _ = librosa.load(training_filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        training_data = training_data + audio.tolist()

    # Concat validation data
    validation_data = []
    for validation_filename in validation_files:
        audio, _ = librosa.load(validation_filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        validation_data = validation_data + audio.tolist()

    return np.array(training_data), np.array(validation_data)

# Generates an audio clip from the NN. After each sample is collected, the inverse of the softmax is taken to normalize the sound
def get_audio_from_model(model, sr, duration, seed_audio, frame_size):
    new_audio = np.zeros((sr * duration))
    for curr_sample_idx in tqdm.tqdm(range(new_audio.shape[0])):
        distribution = np.array(model.predict(seed_audio.reshape(1, frame_size, 1)), dtype=float).reshape(256)
        distribution /= distribution.sum().astype(float)
        predicted_val = np.random.choice(range(256), p=distribution)
        ampl_val_8 = predicted_val / 255.0
        ampl_val_16 = (np.sign(ampl_val_8) * (1/255.0) * ((1 + 256.0)**abs(
            ampl_val_8) - 1)) * 2**15
        new_audio[curr_sample_idx] = ampl_val_16
        seed_audio[:-1] = seed_audio[1:]
        seed_audio[-1] = ampl_val_16
    return new_audio.astype(np.int16)