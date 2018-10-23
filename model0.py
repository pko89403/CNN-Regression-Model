import tensorflow as tf
import os
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_dir = ['./data/train.csv']
model_saveDir = './save_model/'
batch_size = 5996
DROPOHT_RATE = 0.3
LEARNING_RATE = 0.0001

letters = "ACGT"
onTargetLen = 20
offTargetLen = 23

mapping_letters = tf.string_split([letters], delimiter="")


def init_weights(shape, stddev=0.03):
    return tf.Variable(tf.truncated_normal(shape, stddev))
def init_bias(shape):
    return tf.Variable(tf.zeros(shape))

# Weights & Bias For Model
conICh = 4
convOCH = 40
onTargetFilter = 5 # 20 - 5
offTargetFilter = 8 # 23 - 8

onTargetW1 = init_weights(shape=[onTargetFilter, conICh, convOCH], stddev=0.01)
onTargetB1 = init_bias(shape=[convOCH])

offTargetW1 = init_weights(shape=[offTargetFilter, conICh, convOCH], stddev=0.01)
offTargetB1 = init_bias(shape=[convOCH])

fc1_W = init_weights(shape=[15 * 80, 80], stddev=0.01)
fc1_B = init_bias(shape=[80])

fc2_W = init_weights(shape=[80, 40], stddev=0.01)
fc2_B = init_bias(shape=[40])

fc3_W = init_weights(shape=[40, 40], stddev=0.01)
fc3_B = init_bias(shape=[40])

fc4_W = init_weights(shape=[40, 1], stddev=0.01)
fc4_B = init_bias(shape=[1])

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
    return onTarget, offTarget, label


def model():
    onTargetConv = tf.nn.conv1d(batch_onTarget, onTargetW1, stride=1, padding="VALID")  # (1, 18, 80)
    onTargetConv_Relu = tf.nn.sigmoid(onTargetConv + onTargetB1)
    onTargetConv_Relu_Pool = tf.nn.pool(onTargetConv_Relu, window_shape=[2], padding="VALID",
                                        pooling_type="AVG")  # (1, 17, 80)

    offTargetConv = tf.nn.conv1d(batch_offTarget, offTargetW1, stride=1, padding="VALID")  # (1, 18, 80)
    offTargetConv_Relu = tf.nn.sigmoid(offTargetConv + offTargetB1)
    offTargetConv_Relu_Pool = tf.nn.pool(offTargetConv_Relu, window_shape=[2], padding="VALID",
                                         pooling_type="AVG")  # (1, 17, 80)

    targetConcat = tf.concat([onTargetConv_Relu_Pool, offTargetConv_Relu_Pool], axis=-1)  # (1, 17, 160)

    targetConcat_Flat = tf.contrib.layers.flatten(targetConcat)
    targetConcat_Flat_Drop = tf.nn.dropout(targetConcat_Flat, DROPOHT_RATE)

    fc1 = tf.nn.sigmoid(tf.matmul(targetConcat_Flat_Drop, fc1_W) + fc1_B)
    fc1_Drop = tf.nn.dropout(fc1, DROPOHT_RATE)

    fc2 = tf.nn.sigmoid(tf.matmul(fc1_Drop, fc2_W) + fc2_B)
    fc2_Drop = tf.nn.dropout(fc2, DROPOHT_RATE)

    fc3 = tf.nn.sigmoid(tf.matmul(fc2_Drop, fc3_W) + fc3_B)
    fc3_Drop = tf.nn.dropout(fc3, DROPOHT_RATE)

    result = tf.nn.sigmoid(tf.add(tf.matmul(fc3_Drop, fc4_W), fc4_B))

    return result


filename_queue = tf.train.string_input_producer(data_dir)
onTarget, offTarget, label = create_file_reader_ops(filename_queue)
batch_onTarget, batch_offTarget, batch_label = tf.train.batch([onTarget, offTarget, label],
                                                              shapes=[[onTargetLen, 4], [offTargetLen, 4], [1]],
                                                              batch_size=batch_size)
model_Pred = model()

loss = tf.reduce_mean(tf.square(model_Pred-batch_label))
adamOpt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_step = adamOpt.minimize(loss)

l, p = confusion_matrix.remove_squeezable_dimensions(batch_label, model_Pred)
s = tf.square(p - l)
mean_t = tf.reduce_mean(s)
saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.tables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    i = 1
    while (True):
        try:

            opt, mse, l, p = sess.run([train_step, mean_t, batch_label, model_Pred])
            print(i, " Step - AdamOpt : ", opt, " MSE : ", mse, " l : ", l, " p : ", p)

            i = i + 1
        except tf.errors.OutOfRangeError:
            break
    coord.request_stop()
    coord.join(threads)