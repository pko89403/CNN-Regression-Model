import tensorflow as tf
import os
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_dir = ['./data/test.csv']
batch_size = 1

def create_file_reader_ops(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = reader.read(filename_queue)
    record_defaults = [[""], [""], [0.0]]
    #onTargetSEQ, offTargetSEQ, label = tf.decode_csv(csv_row, record_defaults=record_defaults, field_delim=",")
    data = tf.decode_csv(csv_row, record_defaults=record_defaults, field_delim=",")

    #onTargetSEQ = tf.reshape(onTargetSEQ, [1])
    #offTargetSEQ = tf.reshape(offTargetSEQ, [1])
    #label = tf.reshape(label, [1])
    #return onTargetSEQ, offTargetSEQ, label
    return data
filename_queue = tf.train.string_input_producer(data_dir)
data = create_file_reader_ops(filename_queue)

#onTarget, offTarget, label = create_file_reader_ops(filename_queue)
# batch_onTarget, batch_offTarget, batch_label = tf.train.batch([onTarget, offTarget, label],
#                                                              shapes=[[1], [1], [1]],
#                                                              batch_size=batch_size)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.tables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    i = 1
    while (True):
        try:
            print(i, " Step : ", sess.run([data]))
            i = i + 1
        except tf.errors.OutOfRangeError:
            print("WTF")
            break
    coord.request_stop()
    coord.join(threads)