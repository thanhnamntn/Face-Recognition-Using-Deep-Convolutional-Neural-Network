#Thư viên
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading
import numpy as np
import tensorflow as tf

#Đường dẫn file
PATH_MAIN = './'
PATH_TRAIN = PATH_MAIN + 'TrainData'
PATH_TEST = PATH_MAIN + 'TestData'
PATH_OUTPUT = PATH_MAIN + 'Output'
PATH_LABEL = PATH_MAIN + 'label.txt'
PATH_MODEL = PATH_MAIN + 'Model6/'

TRAIN_SHARDS = 2
VALIDATION_SHARDS = 2
NUM_THREADS = 2

#Kích thước hình ảnh
IMAGE_SIZE = 128

#Số lần train và test
TRAINING_SET_SIZE = 100
TESTING_SET_SIZE = 10


#Số lượng hình ảnh train và test
BATCH_SIZE_TRAIN = 15
BATCH_SIZE_TEST = 10


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, text, height, width):
  
  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/class/label': _int64_feature(label), 
      'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
      'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
      'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
  return example

class _image_object:
    def __init__(self):
    		self.image = tf.Variable([], dtype = tf.string)
    		self.height = tf.Variable([], dtype = tf.int64)
    		self.width = tf.Variable([], dtype = tf.int64)
    		self.filename = tf.Variable([], dtype = tf.string)
    		self.label = tf.Variable([], dtype = tf.int32)

class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _is_png(filename):
  return filename.endswith('.png')


def _process_image(filename, coder):
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()

  # Convert any PNG to JPEG's for consistency.
  if _is_png(filename):
    print('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards):
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(PATH_OUTPUT, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]
      text = texts[i]

      try:
        image_buffer, height, width = _process_image(filename, coder)
      except Exception as e:
        print(e)
        print('SKIPPED: Unexpected error while decoding %s.' % filename)
        continue

      example = _convert_to_example(filename, image_buffer, label,
                                    text, height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, filenames, texts, labels, num_shards):
  assert len(filenames) == len(texts)
  assert len(filenames) == len(labels)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), NUM_THREADS + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (NUM_THREADS, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames,
            texts, labels, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_image_files(data_dir, labels_file):
  print('Determining list of input files and labels from %s.' % data_dir)
  unique_labels = [l.strip() for l in tf.gfile.FastGFile(
      labels_file, 'r').readlines()]

  labels = []
  filenames = []
  texts = []

  # Leave label index 0 empty as a background class.
  label_index = 1

  # Construct the list of JPEG files and labels.
  for text in unique_labels:
    jpeg_file_path = '%s/%s/*' % (data_dir, text)
    matching_files = tf.gfile.Glob(jpeg_file_path)

    labels.extend([label_index] * len(matching_files))
    texts.extend([text] * len(matching_files))
    filenames.extend(matching_files)

    if not label_index % 100:
      print('Finished finding files in %d of %d classes.' % (
          label_index, len(labels)))
    label_index += 1

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  texts = [texts[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]

  print('Found %d JPEG files across %d labels inside %s.' %
        (len(filenames), len(unique_labels), data_dir))
  return filenames, texts, labels


def _process_dataset(name, directory, num_shards, labels_file):
    
  filenames, texts, labels = _find_image_files(directory, labels_file)
  _process_image_files(name, filenames, texts, labels, num_shards)


#Kiểm tra file tfrecord đã tồn tại chưa ?
def _dataset_exists(output_filename):
  if(os.listdir(output_filename) == []):
      return False
  return True

#Đọc dữ liệu và chuyển đổi
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features = {
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/class/label": tf.FixedLenFeature([], tf.int64),})
    
    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=3)

    image_object = _image_object()
    image_object.image = tf.image.resize_image_with_crop_or_pad(image_raw, IMAGE_SIZE, IMAGE_SIZE)
    image_object.height = features["image/height"]
    image_object.width = features["image/width"]
    image_object.filename = features["image/filename"]
    image_object.label = tf.cast(features["image/class/label"], tf.int64)
    return image_object

def flower_input(if_random = True, if_training = True,BATCH_SIZE=BATCH_SIZE_TRAIN):
    if(if_training):
        filenames = [os.path.join(PATH_OUTPUT, "train-0000%d-of-00002" % i) for i in range(0, 1)]
    else:
        filenames = [os.path.join(PATH_OUTPUT, "validation-0000%d-of-00002" % i) for i in range(0, 1)]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError("Failed to find file: " + f)
    filename_queue = tf.train.string_input_producer(filenames)
    image_object = read_and_decode(filename_queue)
    image = tf.image.per_image_standardization(image_object.image)
#    image = image_object.image
#    image = tf.image.adjust_gamma(tf.cast(image_object.image, tf.float32), gamma=1, gain=1) # Scale image to (0, 1)
    label = image_object.label
    filename = image_object.filename

    if(if_random):
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(TRAINING_SET_SIZE * min_fraction_of_examples_in_queue)
        print("Filling queue with %d images before starting to train. " "This will take a few minutes." % min_queue_examples)
        num_preprocess_threads = 1
        image_batch, label_batch, filename_batch = tf.train.shuffle_batch(
            [image, label, filename],
            batch_size = BATCH_SIZE,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 3 * BATCH_SIZE,
            min_after_dequeue = min_queue_examples)
        return image_batch, label_batch, filename_batch
    else:
        image_batch, label_batch, filename_batch = tf.train.batch(
            [image, label, filename],
            batch_size = BATCH_SIZE,
            num_threads = 1)
        return image_batch, label_batch, filename_batch


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.02, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def CNN(image_batch):
    W_conv1 = weight_variable([5, 5, 3, 64])
    b_conv1 = bias_variable([64])

    x_image = tf.reshape(image_batch, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1) # 64

    W_conv2 = weight_variable([5, 5, 64, 128])
    b_conv2 = bias_variable([128])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2) # 32

    W_conv3 = weight_variable([5, 5, 128, 256])
    b_conv3 = bias_variable([256])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3) # 16

    W_conv4 = weight_variable([5, 5, 256, 512])
    b_conv4 = bias_variable([512])

    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4) # 8

    W_conv5 = weight_variable([5, 5, 512, 1024])
    b_conv5 = bias_variable([1024])

    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    h_pool5 = max_pool_2x2(h_conv5) # 4

    W_fc1 = weight_variable([4*4*1024, 1024])
    b_fc1 = bias_variable([1024])

    h_pool5_flat = tf.reshape(h_pool5, [-1, 4*4*1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

    #h_fc1_drop = tf.nn.dropout(h_fc1, 1.0)

    W_fc2 = weight_variable([1024, 4])
    b_fc2 = bias_variable([4])

    #h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    #W_fc3 = weight_variable([512, 64])
    #b_fc3 = bias_variable([64])

    #h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    #W_fc4 = weight_variable([64, 4])
    #b_fc4 = bias_variable([4])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
#    y_conv = tf.matmul(h_fc3, W_fc4) + b_fc4

    return y_conv

def Train():
    image_batch_out, label_batch_out, filename_batch = flower_input(if_random = False, if_training = True,BATCH_SIZE=BATCH_SIZE_TRAIN)

    image_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE_TRAIN, IMAGE_SIZE, IMAGE_SIZE, 3])
    image_batch = tf.reshape(image_batch_out, (BATCH_SIZE_TRAIN, IMAGE_SIZE, IMAGE_SIZE, 3))

    label_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE_TRAIN, 4])
    label_offset = -tf.ones([BATCH_SIZE_TRAIN], dtype=tf.int64, name="label_batch_offset")
    label_batch_one_hot = tf.one_hot(tf.add(label_batch_out, label_offset), depth=4, on_value=1.0, off_value=0.0)

    logits_out = CNN(image_batch_placeholder)
#    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=label_batch_one_hot, logits=logits_out))
    loss = tf.losses.mean_squared_error(labels=label_batch_placeholder, predictions=logits_out)

    train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(loss)

    if (os.path.isfile(PATH_MODEL + 'Model.ckpt.meta') == True):
        saver = tf.train.import_meta_graph(PATH_MODEL + 'Model.ckpt.meta')
    else:
        saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        var = tf.Variable(42, name='var')
        
        file_writer = tf.summary.FileWriter("./logs", sess.graph)
        
        
        sess.run(tf.global_variables_initializer())
        
        if (os.path.isfile(PATH_MODEL + 'Model.ckpt.meta') == True):
            saver.restore(sess,PATH_MODEL + "Model.ckpt")
        
        coord = tf.train.Coordinator()
        
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)

        for i in range(TRAINING_SET_SIZE + 1):
            image_out, label_out, label_batch_one_hot_out, filename_out = sess.run([image_batch, label_batch_out, label_batch_one_hot, filename_batch])

            _, infer_out, loss_out = sess.run([train_step, logits_out, loss], feed_dict={image_batch_placeholder: image_out, label_batch_placeholder: label_batch_one_hot_out})

            print(i)
            print("label_out: ")
            print(filename_out)
            print(label_out)
            print("loss: ")
            print(loss_out)
            if(i % 10 == 0 and i >= 10):
                saver.save(sess, PATH_MODEL + "/Model.ckpt")

        coord.request_stop()
        coord.join(threads)
        sess.close()


def Eval():
    image_batch_out, label_batch_out, filename_batch = flower_input(if_random = False, if_training = False,BATCH_SIZE=BATCH_SIZE_TEST)

    image_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE_TEST, IMAGE_SIZE, IMAGE_SIZE, 3])
    image_batch = tf.reshape(image_batch_out, (BATCH_SIZE_TEST, IMAGE_SIZE, IMAGE_SIZE, 3))

    label_tensor_placeholder = tf.placeholder(tf.int64, shape=[BATCH_SIZE_TEST])
    label_offset = -tf.ones([BATCH_SIZE_TEST], dtype=tf.int64, name="label_batch_offset")
    label_batch = tf.add(label_batch_out, label_offset)

    logits_out = tf.reshape(CNN(image_batch_placeholder), [BATCH_SIZE_TEST, 4]) 
    logits_batch = tf.to_int64(tf.arg_max(logits_out, dimension = 1))

    correct_prediction = tf.equal(logits_batch, label_tensor_placeholder)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.import_meta_graph(PATH_MODEL + 'model.ckpt.meta')
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, PATH_MODEL + 'Model.ckpt')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)

        accuracy_accu = 0

        for i in range(TESTING_SET_SIZE):
            image_out, label_out, filename_out = sess.run([image_batch, label_batch, filename_batch])

            accuracy_out, logits_batch_out = sess.run([accuracy, logits_batch], feed_dict={image_batch_placeholder: image_out, label_tensor_placeholder: label_out})
            accuracy_accu += accuracy_out

            print(i)
            print("label_out: ")
            print(filename_out)
            print(label_out)
            print(logits_batch_out)
            print(accuracy_out)
            
        print("Accuracy: ")
        print((accuracy_accu / TESTING_SET_SIZE)*100)

        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(unused_argv):
  if _dataset_exists(PATH_OUTPUT) == False:
      # Run it!
      _process_dataset('validation', PATH_TEST,VALIDATION_SHARDS, PATH_LABEL)
      _process_dataset('train', PATH_TRAIN,TRAIN_SHARDS, PATH_LABEL)
    
  Train()
  Eval()

if __name__ == '__main__':
  tf.app.run()