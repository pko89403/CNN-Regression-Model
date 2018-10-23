import tensorflow as tf
import os
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_dir = ['./data/train2.csv']
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
    reader = tf.TextLineReader(skip_header_lines=1)
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
convOCH = 256
conv0CH2 = 512

onTargetFilter = 5 # 20 - 5 + 1 = 16
onTargetW1 = init_weights(shape=[onTargetFilter, conICh, convOCH], stddev=0.01)
onTargetB1 = init_weights(shape=[convOCH], stddev=0.01)
onTargetConv = tf.nn.conv1d(batch_onTarget, onTargetW1, stride=1, padding="VALID")  # (1, 16, 256)
onTargetConv_Relu = tf.nn.sigmoid(onTargetConv + onTargetB1)
onTargetConv_Relu_Pool = tf.nn.pool(onTargetConv_Relu, window_shape=[2], padding="VALID",
                                    pooling_type="AVG")  # (1, 15, 256)

onTargetFilter2 = 5 # 15 - 5 + 1
onTargetW2 = init_weights(shape=[onTargetFilter2, convOCH, conv0CH2], stddev=0.01)
onTargetB2 = init_weights(shape=[conv0CH2], stddev=0.01)
onTargetConv2 = tf.nn.conv1d(onTargetConv_Relu_Pool, onTargetW2, stride=1, padding="VALID")  # (1, 16, 256)
onTargetConv2_Relu = tf.nn.sigmoid(onTargetConv2 + onTargetB2)
onTargetConv2_Relu_Pool = tf.nn.pool(onTargetConv2_Relu, window_shape=[2], padding="VALID",
                                    pooling_type="AVG")  # (1, 10, 512)

offTargetFilter = 8 # 23 - 8 + 1
offTargetW1 = init_weights(shape=[offTargetFilter, conICh, convOCH], stddev=0.01)
offTargetB1 = init_weights(shape=[convOCH], stddev=0.01)
offTargetConv = tf.nn.conv1d(batch_offTarget, offTargetW1, stride=1, padding="VALID")  # (1, 16, 256)
offTargetConv_Relu = tf.nn.sigmoid(offTargetConv + offTargetB1)
offTargetConv_Relu_Pool = tf.nn.pool(offTargetConv_Relu, window_shape=[2], padding="VALID",
                                     pooling_type="AVG")  # (1, 15, 256)


offTargetFilter2 = 5 # 15 - 5 + 1
offTargetW2 = init_weights(shape=[offTargetFilter2, convOCH, conv0CH2], stddev=0.01)
offTargetB2 = init_weights(shape=[conv0CH2], stddev=0.01)
offTargetConv2 = tf.nn.conv1d(offTargetConv_Relu_Pool, offTargetW2, stride=1, padding="VALID")  # (1, 16, 256)
offTargetConv2_Relu = tf.nn.sigmoid(offTargetConv2 + offTargetB2)
offTargetConv2_Relu_Pool = tf.nn.pool(offTargetConv2_Relu, window_shape=[2], padding="VALID",
                                     pooling_type="AVG")  # (1, 10, 256)

targetConcat = tf.concat([onTargetConv2_Relu_Pool, offTargetConv2_Relu_Pool], axis=-1)  # (1, 9, 512)
targetConcat_Flat = tf.contrib.layers.flatten(targetConcat)

targetConcat_Flat_Drop = tf.nn.dropout(targetConcat_Flat, DROPOHT_RATE)


# Fully-Connected Weights & Bias For Model
fc1_W = init_weights(shape=[1024 * 10, 2048], stddev=0.01)
fc1_B = init_weights(shape=[2048], stddev=0.01)

fc2_W = init_weights(shape=[2048, 256], stddev=0.01)
fc2_B = init_weights(shape=[256], stddev=0.01)

fc3_W = init_weights(shape=[256, 256], stddev=0.01)
fc3_B = init_weights(shape=[256], stddev=0.01)

fc4_W = init_weights(shape=[256, 1], stddev=0.01)
fc4_B = init_weights(shape=[1], stddev=0.01)

# Model
fc1 = tf.nn.relu(tf.matmul(targetConcat_Flat_Drop, fc1_W) + fc1_B)
fc1_Drop = tf.nn.dropout(fc1, DROPOHT_RATE)

fc2 = tf.nn.relu(tf.matmul(fc1_Drop, fc2_W) + fc2_B)
fc2_Drop = tf.nn.dropout(fc2, DROPOHT_RATE)

fc3 = tf.nn.relu(tf.matmul(fc2_Drop, fc3_W) + fc3_B)
fc3_Drop = tf.nn.dropout(fc3, DROPOHT_RATE)

model_Pred = tf.add(tf.matmul(fc3_Drop, fc4_W), fc4_B)


loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_Pred, labels=batch_label))
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

            if (i % 100 == 0):
                print(i, " Step - AdamOpt : ", opt, " MSE : ", mse)
            i = i + 1
        except tf.errors.OutOfRangeError:
            break
    coord.request_stop()
    coord.join(threads)