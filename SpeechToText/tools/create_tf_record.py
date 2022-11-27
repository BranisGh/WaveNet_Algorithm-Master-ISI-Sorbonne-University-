import os
import json
import glog
import utils
import random
import tensorflow as tf
from tqdm import tqdm

flags = tf.compat.v1.app.flags
flags.DEFINE_string('input_dir', r'C:\Users\Brani\OneDrive\Bureau\speech-to-text-wavenet2-master', 'Directory to mmc dataset.')
flags.DEFINE_string('list_path', 'data/list.json', 'Path to list.')
flags.DEFINE_string('config_path', 'config/english-28.json', 'Directory to config.')
flags.DEFINE_integer('train_test_ratio', 20, 'The ratio of train to test.')
flags.DEFINE_string('output_dir', 'data/v28', 'Path of output.')
FLAGS = flags.FLAGS


def _read_vctk(root, ratio=0.2):
  ids = []
  for line in open(root + '/VCTK-Corpus/speaker-info.txt', 'r').readlines()[1:]:
    ids.append('/' + line.split(' ')[0])
  train_filenames = []
  test_filenames = []
  
  for i, id in enumerate(ids):
    
    
    txt_dir = '/VCTK-Corpus/txt' + id
    wav_dir = '/VCTK-Corpus/wav48' + id
    
    for filename in os.listdir(root + wav_dir):
      stem, ext = os.path.splitext(filename)
      
      nametxt=stem.split('_')
      #print(txt_dir + '/' + str(nametxt[0]) +'_'+str(nametxt[1])+'.txt')
      if ext == '.flac':
        wav_name = wav_dir + '/' + filename
        
        txt_name = txt_dir + '/' + str(nametxt[0]) +'_'+str(nametxt[1])+'.txt'
        
        if os.path.exists(root + txt_name):
          if random.uniform(0, 1) < ratio:
            test_filenames.append((wav_name, txt_name))
          else:
            train_filenames.append((wav_name, txt_name))
        else:
          glog.warning("Skip %s, the label file is not found." % filename)
  return train_filenames, test_filenames


def _create_dataset(input_dir, filenames, output_path):
  count = 0
  writer = tf.io.TFRecordWriter(output_path+'.cache')
  random.shuffle(filenames)
  for i, filename in tqdm(enumerate(filenames)):
    wave_path = input_dir + filename[0]
    txt_path = input_dir + filename[1]
    stem = os.path.splitext(os.path.split(filename[0])[-1])[0]
    wave = utils.read_wave(wave_path)
    text = utils.read_txt(txt_path)
    if len(wave) >= len(text):
      data = tf.train.Example(features=tf.train.Features(feature={
        'uid': tf.train.Feature(bytes_list=tf.train.BytesList(value=[stem.encode('utf-8')])),
        'audio/data': tf.train.Feature(float_list=tf.train.FloatList(value=wave.reshape([-1]).tolist())),
        'audio/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=wave.shape)),
        'text': tf.train.Feature(int64_list=tf.train.Int64List(value=text)),
      }))
      writer.write(data.SerializeToString())
    else:
      glog.error("length of label(%d) is greater than feature(%d) at %s." % (len(text), len(wave), stem))

    count = i + 1
    if count % 1000 == 0:
      glog.info('processed %d/%d files.' % (count, len(filenames)))
  if count % 1000 != 0:
    glog.info('processed %d/%d files.' % (count, len(filenames)))
  if os.path.exists(output_path):
    os.remove(output_path)
  if os.path.exists(output_path+'.cache'):
    os.renames(output_path+'.cache', output_path)


def main(_):
  utils.load(FLAGS.config_path)
  if os.path.exists(FLAGS.list_path):
    train_filenames, test_filenames = json.load(open(FLAGS.list_path, 'r', encoding='utf-8'))
  else:
    train_filenames = []
    test_filenames = []
    for dataset_name in os.listdir(FLAGS.input_dir):
      if dataset_name == 'VCTK-Corpus':
        # glog.warning('Reading the dataset --------------------------')
        dataset = _read_vctk(FLAGS.input_dir, FLAGS.train_test_ratio)
      else:
        print(dataset_name)
        glog.warning('Dataset named %s is not supported now.')
        continue
      train_filenames += dataset[0]
      test_filenames += dataset[1]
    json.dump((train_filenames, test_filenames), open(FLAGS.list_path, 'w', encoding='utf-8'), ensure_ascii=False)
  train_path = FLAGS.output_dir + '/train.record'
  test_path = FLAGS.output_dir + '/test.record'
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)
  glog.info('processing train dataset...')
  _create_dataset(FLAGS.input_dir, train_filenames, train_path)
  glog.info('processing test dataset...')
  _create_dataset(FLAGS.input_dir, test_filenames, test_path)
  glog.info('End.')


if __name__ == '__main__':
  tf.compat.v1.app.run()
