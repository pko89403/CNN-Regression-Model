import tensorflow as tf
import os
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_dir = ['./data/train.csv']
model_saveDir = './save_model/'
batch_size = 5996
DROPOHT_RATE = 0.3
LEARNING_RATE = 0.001

letters = "ACGT"
onTargetLen = 20
offTargetLen = 23

mapping_letters = tf.string_split([letters], delimiter="")

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
def init_weights(shape, stddev=0.03):
    return tf.Variable(tf.truncated_normal(shape, stddev))

filename_queue = tf.train.string_input_producer(data_dir)
onTarget, offTarget, label = create_file_reader_ops(filename_queue)
batch_onTarget, batch_offTarget, batch_label = tf.train.batch([onTarget, offTarget, label],
                                                              shapes=[[onTargetLen, 4], [offTargetLen, 4], [1]],
                                                              batch_size=batch_size)

# Convolutional Weights & Bias For Model
conICh = 4
convOCH = 80
conv0CH2 = 120
convOCH3 = 160
convOCH4 = 200
onTargetFilter = 1 # 20 - 2 + 1 = 19
onTargetW1 = init_weights(shape=[onTargetFilter, conICh, convOCH], stddev=0.01)
onTargetB1 = init_weights(shape=[convOCH], stddev=0.01)
onTargetConv = tf.nn.conv1d(batch_onTarget, onTargetW1, stride=1, padding="SAME")  # (1, 20, 4)
onTargetConv_Relu = tf.nn.relu(onTargetConv + onTargetB1)

onTargetW2 = init_weights(shape=[onTargetFilter, convOCH, conv0CH2], stddev=0.01)
onTargetB2 = init_weights(shape=[conv0CH2], stddev=0.01)
onTargetConv2 = tf.nn.conv1d(onTargetConv_Relu, onTargetW2, stride=1, padding="SAME")  # (1, 20, 4)
onTargetConv_Relu2 = tf.nn.relu(onTargetConv2 + onTargetB2)

onTargetConv_Relu2_Pool = tf.nn.pool(onTargetConv_Relu2, window_shape=[2], padding="SAME", strides=[2],
                                    pooling_type="AVG")  # (1, 20, 256)

onTargetFilter2 = 1 # 18 - 2 + 1
onTargetW3 = init_weights(shape=[onTargetFilter2, conv0CH2, convOCH3], stddev=0.01)
onTargetB3 = init_weights(shape=[convOCH3], stddev=0.01)
onTargetConv3 = tf.nn.conv1d(onTargetConv_Relu2_Pool, onTargetW3, stride=1, padding="SAME")  # (1, 10, 160)
onTargetConv3_Relu = tf.nn.relu(onTargetConv3 + onTargetB3)

onTargetW4 = init_weights(shape=[onTargetFilter2, convOCH3, convOCH4], stddev=0.01)
onTargetB4 = init_weights(shape=[convOCH4], stddev=0.01)
onTargetConv4 = tf.nn.conv1d(onTargetConv3_Relu, onTargetW4, stride=1, padding="SAME")  # (1, 10, 160)
onTargetConv4_Relu = tf.nn.relu(onTargetConv4 + onTargetB4)

onTargetConv4_Relu_Pool = tf.nn.pool(onTargetConv4_Relu, window_shape=[2], padding="SAME", strides=[2],
                                    pooling_type="AVG")  # (1, 16, 512)
onTargetConcat_Flat = tf.contrib.layers.flatten(onTargetConv4_Relu_Pool)

# Fully-Connected Weights & Bias For Model
on_fc1_W = init_weights(shape=[200 * 5, 500], stddev=0.01)
on_fc1_B = init_weights(shape=[500], stddev=0.01)

on_fc2_W = init_weights(shape=[500, 250], stddev=0.01)
on_fc2_B = init_weights(shape=[250], stddev=0.01)

on_fc3_W = init_weights(shape=[250, 125], stddev=0.01)
on_fc3_B = init_weights(shape=[125], stddev=0.01)

on_fc4_W = init_weights(shape=[125, 125], stddev=0.01)
on_fc4_B = init_weights(shape=[125], stddev=0.01)

on_fc5_W = init_weights(shape=[125, 1], stddev=0.01)

# Model
on_fc1 = tf.nn.relu(tf.matmul(onTargetConcat_Flat, on_fc1_W) + on_fc1_B)
on_fc1_Drop = tf.nn.dropout(on_fc1, DROPOHT_RATE)

on_fc2 = tf.nn.relu(tf.matmul(on_fc1_Drop, on_fc2_W) + on_fc2_B)
on_fc2_Drop = tf.nn.dropout(on_fc2, DROPOHT_RATE)

on_fc3 = tf.nn.relu(tf.matmul(on_fc2_Drop, on_fc3_W) + on_fc3_B)
on_fc3_Drop = tf.nn.dropout(on_fc3, DROPOHT_RATE)

on_fc4 = tf.nn.relu(tf.matmul(on_fc3_Drop, on_fc4_W) + on_fc4_B)
on_fc4_Drop = tf.nn.dropout(on_fc4, DROPOHT_RATE)

onTargetX = tf.matmul(on_fc4_Drop, on_fc5_W)


conICh_off = 4
convOCH_off = 92
conv0CH2_off = 128
convOCH3_off = 192
convOCH4_off = 256
offTargetFilter = 1 # 23 - 3 + 1
offTargetW1 = init_weights(shape=[offTargetFilter, conICh_off, convOCH_off], stddev=0.01)
offTargetB1 = init_weights(shape=[convOCH_off], stddev=0.01)
offTargetConv = tf.nn.conv1d(batch_offTarget, offTargetW1, stride=1, padding="SAME")  # (1, 21, 256)
offTargetConv_Relu = tf.nn.relu(offTargetConv + offTargetB1)

offTargetW2 = init_weights(shape=[offTargetFilter, convOCH_off, conv0CH2_off], stddev=0.01)
offTargetB2 = init_weights(shape=[conv0CH2_off], stddev=0.01)
offTargetConv2 = tf.nn.conv1d(offTargetConv_Relu, offTargetW2, stride=1, padding="SAME")  # (1, 21, 256)
offTargetConv2_Relu = tf.nn.relu(offTargetConv2 + offTargetB2)

offTargetConv_Relu2_Pool = tf.nn.pool(offTargetConv2_Relu, window_shape=[2], padding="SAME", strides=[2],
                                     pooling_type="MAX")  # (1, 20, 256)

offTargetFilter2 = 1 # 20 - 3 + 1
offTargetW3 = init_weights(shape=[offTargetFilter2, conv0CH2_off, convOCH3_off], stddev=0.01)
offTargetB3 = init_weights(shape=[convOCH3_off], stddev=0.01)
offTargetConv3 = tf.nn.conv1d(offTargetConv_Relu2_Pool, offTargetW3, stride=1, padding="SAME")  # (1, 18, 256)
offTargetConv3_Relu = tf.nn.relu(offTargetConv3 + offTargetB3)

offTargetW4 = init_weights(shape=[offTargetFilter2, convOCH3_off, convOCH4_off], stddev=0.01)
offTargetB4 = init_weights(shape=[convOCH4_off], stddev=0.01)
offTargetConv4 = tf.nn.conv1d(offTargetConv3_Relu, offTargetW4, stride=1, padding="SAME")  # (1, 18, 256)
offTargetConv4_Relu = tf.nn.relu(offTargetConv4 + offTargetB4)

offTargetConv4_Relu_Pool = tf.nn.pool(offTargetConv4_Relu, window_shape=[2], padding="SAME", strides=[2],
                                     pooling_type="MAX")  # (1, 16, 256)


offTargetConcat_Flat = tf.contrib.layers.flatten(offTargetConv4_Relu_Pool)

# Fully-Connected Weights & Bias For Model
off_fc1_W = init_weights(shape=[256 * 6, 500], stddev=0.01)
off_fc1_B = init_weights(shape=[500], stddev=0.01)

off_fc2_W = init_weights(shape=[500, 250], stddev=0.01)
off_fc2_B = init_weights(shape=[250], stddev=0.01)

off_fc3_W = init_weights(shape=[250, 125], stddev=0.01)
off_fc3_B = init_weights(shape=[125], stddev=0.01)

off_fc4_W = init_weights(shape=[125, 125], stddev=0.01)
off_fc4_B = init_weights(shape=[125], stddev=0.01)

off_fc5_W = init_weights(shape=[125, 1], stddev=0.01)

# Model
off_fc1 = tf.nn.relu(tf.matmul(offTargetConcat_Flat, off_fc1_W) + off_fc1_B)
off_fc1_Drop = tf.nn.dropout(off_fc1, DROPOHT_RATE)

off_fc2 = tf.nn.relu(tf.matmul(off_fc1_Drop, off_fc2_W) + off_fc2_B)
off_fc2_Drop = tf.nn.dropout(off_fc2, DROPOHT_RATE)

off_fc3 = tf.nn.relu(tf.matmul(off_fc2_Drop, off_fc3_W) + off_fc3_B)
off_fc3_Drop = tf.nn.dropout(off_fc3, DROPOHT_RATE)

off_fc4 = tf.nn.relu(tf.matmul(off_fc3_Drop, off_fc4_W) + off_fc4_B)
off_fc4_Drop = tf.nn.dropout(off_fc4, DROPOHT_RATE)

offTargetX = tf.matmul(off_fc4_Drop, off_fc5_W)

onT_W = init_weights(shape=[1], stddev=0.03)
offT_W = init_weights(shape=[1], stddev=0.03)
merge_B = init_weights(shape=[1], stddev=0.03)

model_Pred = (onTargetX * onT_W) + (offTargetX * offT_W) + merge_B


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

            opt, mse = sess.run([train_step, mean_t])
            print(i, " Step - AdamOpt : ", opt, " MSE : ", mse)
            i = i + 1
        except tf.errors.OutOfRangeError:
            break
    coord.request_stop()
    coord.join(threads)