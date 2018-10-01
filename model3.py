import tensorflow as tf
import os
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_dir = ['./data/train.csv']
model_saveDir = './save_model/'
batch_size = 64
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
convOCH = 80
conv0CH2 = 80
convOCH3 = 160
convOCH4 = 160
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
                                    pooling_type="MAX")  # (1, 20, 256)

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
                                    pooling_type="MAX")  # (1, 16, 512)


offTargetFilter = 1 # 23 - 3 + 1
offTargetW1 = init_weights(shape=[offTargetFilter, conICh, convOCH], stddev=0.01)
offTargetB1 = init_weights(shape=[convOCH], stddev=0.01)
offTargetConv = tf.nn.conv1d(batch_offTarget, offTargetW1, stride=1, padding="SAME")  # (1, 21, 256)
offTargetConv_Relu = tf.nn.relu(offTargetConv + offTargetB1)

offTargetW2 = init_weights(shape=[offTargetFilter, convOCH, conv0CH2], stddev=0.01)
offTargetB2 = init_weights(shape=[conv0CH2], stddev=0.01)
offTargetConv2 = tf.nn.conv1d(offTargetConv_Relu, offTargetW2, stride=1, padding="SAME")  # (1, 21, 256)
offTargetConv2_Relu = tf.nn.relu(offTargetConv2 + offTargetB2)

offTargetConv_Relu2_Pool = tf.nn.pool(offTargetConv2_Relu, window_shape=[3], padding="SAME", strides=[2],
                                     pooling_type="AVG")  # (1, 20, 256)

offTargetFilter2 = 1 # 20 - 3 + 1
offTargetW3 = init_weights(shape=[offTargetFilter2, conv0CH2, convOCH3], stddev=0.01)
offTargetB3 = init_weights(shape=[convOCH3], stddev=0.01)
offTargetConv3 = tf.nn.conv1d(offTargetConv_Relu2_Pool, offTargetW3, stride=1, padding="SAME")  # (1, 18, 256)
offTargetConv3_Relu = tf.nn.relu(offTargetConv3 + offTargetB3)

offTargetW4 = init_weights(shape=[offTargetFilter2, convOCH3, convOCH4], stddev=0.01)
offTargetB4 = init_weights(shape=[convOCH4], stddev=0.01)
offTargetConv4 = tf.nn.conv1d(offTargetConv3_Relu, offTargetW4, stride=1, padding="SAME")  # (1, 18, 256)
offTargetConv4_Relu = tf.nn.relu(offTargetConv4 + offTargetB4)

offTargetConv4_Relu_Pool = tf.nn.pool(offTargetConv4_Relu, window_shape=[3], padding="VALID", strides=[2],
                                     pooling_type="AVG")  # (1, 16, 256)

targetConcat = tf.concat([onTargetConv4_Relu_Pool, offTargetConv4_Relu_Pool], axis=-1)  # (1, 9, 512)
targetConcat_Flat = tf.contrib.layers.flatten(targetConcat)



# Fully-Connected Weights & Bias For Model
fc1_W = init_weights(shape=[320 * 5, 320], stddev=0.01)
fc1_B = init_weights(shape=[320], stddev=0.01)

fc2_W = init_weights(shape=[320, 80], stddev=0.01)
fc2_B = init_weights(shape=[80], stddev=0.01)

fc3_W = init_weights(shape=[80, 40], stddev=0.01)
fc3_B = init_weights(shape=[40], stddev=0.01)

fc4_W = init_weights(shape=[40, 40], stddev=0.01)
fc4_B = init_weights(shape=[40], stddev=0.01)

fc5_W = init_weights(shape=[40, 1], stddev=0.01)
fc5_B = init_weights(shape=[1], stddev=0.01)




# Model
fc1 = tf.nn.relu(tf.matmul(targetConcat_Flat, fc1_W) + fc1_B)
fc1_Drop = tf.nn.dropout(fc1, DROPOHT_RATE)

fc2 = tf.nn.relu(tf.matmul(fc1_Drop, fc2_W) + fc2_B)
fc2_Drop = tf.nn.dropout(fc2, DROPOHT_RATE)

fc3 = tf.nn.relu(tf.matmul(fc2_Drop, fc3_W) + fc3_B)
fc3_Drop = tf.nn.dropout(fc3, DROPOHT_RATE)

fc4 = tf.nn.relu(tf.matmul(fc3_Drop, fc4_W) + fc4_B)
fc4_Drop = tf.nn.dropout(fc4, DROPOHT_RATE)

fc5 = tf.matmul(fc4_Drop, fc5_W) + fc5_B

#model_Pred = tf.nn.(tf.add(tf.matmul(fc3_Drop, fc4_W), fc4_B))sigmoid
model_Pred = fc5


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