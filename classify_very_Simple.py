import tensorflow as tf
import os
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_dir = ['./data/train_class_bias0.csv']
model_saveDir = './save_model/'
batch_size = 100
DROPOHT_RATE = 1.0
LEARNING_RATE = 0.01

letters = "ACGT"
onTargetLen = 20
offTargetLen = 23
CLASS = 2
mapping_letters = tf.string_split([letters], delimiter="")


def init_weights(shape, stddev=0.03):
    return tf.Variable(tf.truncated_normal(shape, stddev))
def init_bias(shape):
    return tf.Variable(tf.zeros(shape))

# Weights & Bias For Model
def seq_processing(seq):
    table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_letters.values, default_value=0)
    seq_char = tf.string_split(seq, delimiter="")
    encoded = tf.one_hot(table.lookup(seq_char.values), len(letters), dtype=tf.float32)
    return encoded
def create_file_reader_ops(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=0)
    _, csv_row = reader.read(filename_queue)
    record_defaults = [[""], [""], [0.0]]
    onTargetSEQ, offTargetSEQ, label = tf.decode_csv(csv_row, record_defaults=record_defaults, field_delim=",")

    onTargetSEQ = tf.reshape(onTargetSEQ, [1])
    onTarget = seq_processing(onTargetSEQ)
    offTargetSEQ = tf.reshape(offTargetSEQ, [1])
    offTarget = seq_processing(offTargetSEQ)
    label = tf.reshape(label, [1])
    #label_hot = tf.one_hot(label, CLASS)

    return onTarget, offTarget, label


def model(onTarget, offTarget, labels, w1, b1, w2, b2, w3, b3, w4, b4):
    onTarget_Flat = tf.layers.flatten(onTarget)
    offTarget_Flat = tf.layers.flatten(offTarget)

    targetSeq = tf.concat([onTarget_Flat, offTarget_Flat], axis=-1)

    h1 = tf.sigmoid(tf.matmul(targetSeq, w1) +  b1)

    h2 = tf.sigmoid(tf.matmul(h1, w2) +  b2)
    h3 = tf.sigmoid(tf.matmul(h2, w3) +  b3)
    h4 = tf.sigmoid(tf.matmul(h3, w4) +  b4)



    hypothesis = tf.sigmoid(h4)

    return hypothesis, labels


filename_queue = tf.train.string_input_producer(data_dir)
onTarget, offTarget, label = create_file_reader_ops(filename_queue)
batch_onTarget, batch_offTarget, batch_label = tf.train.batch([onTarget, offTarget, label],
                                                              shapes=[[onTargetLen, 4], [offTargetLen, 4], [1]],
                                                              batch_size=batch_size)
w1 = init_weights([172, 344])
b1 = init_weights([344])

w2 = init_weights([344, 344])
b2 = init_weights([344])

w3 = init_weights([344, 344])
b3 = init_weights([344])

w4 = init_weights([344, 1])
b4 = init_weights([1])

h, y = model(batch_onTarget, batch_offTarget, batch_label, w1, b1, w2, b2, w3, b3, w4, b4)
cost = -tf.reduce_mean(y * tf.log(h) + (1-y)*tf.log(1-h))
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
train_step = optimizer.minimize(cost)


predicted = tf.cast(h > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.tables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    i = 1
    while (True):
        try:

            opt, acc = sess.run([train_step, accuracy])
            print(i, " Step - AdamOpt : ", opt, " acc : ", acc* 100)

            i = i + 1
        except tf.errors.OutOfRangeError:
            break
    coord.request_stop()
    coord.join(threads)