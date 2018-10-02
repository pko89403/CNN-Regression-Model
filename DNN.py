import tensorflow as tf
import os
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_dir = ['./data/train.csv']
model_saveDir = './save_model/'
batch_size = 64
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

onT1_W = init_weights(shape=[80, 128], stddev=0.01)
onT1_B = init_weights(shape=[128])

onT2_W = init_weights(shape=[128, 128], stddev=0.01)
onT2_B = init_weights(shape=[128])

offT1_W = init_weights(shape=[92, 128], stddev=0.01)
offT1_B = init_weights(shape=[128])

offT2_W = init_weights(shape=[128, 128], stddev=0.01)
offT2_B = init_weights(shape=[128])

fc3_W = init_weights(shape=[256, 256], stddev=0.01)
fc3_B = init_weights(shape=[256])

fc4_W = init_weights(shape=[256, 256], stddev=0.01)
fc4_B = init_weights(shape=[256])

fc5_W = init_weights(shape=[256, 32], stddev=0.01)
fc5_B = init_weights(shape=[32])

fc6_W = init_weights(shape=[32, 32], stddev=0.01)
fc6_B = init_weights(shape=[32])

fc7_W = init_weights(shape=[32, 16], stddev=0.01)
fc7_B = init_weights(shape=[16])

fc8_W = init_weights(shape=[16, 16], stddev=0.01)
fc8_B = init_weights(shape=[16])

fc9_W = init_weights(shape=[16, 1], stddev=0.01)


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
    onTarget_Flat = tf.contrib.layers.flatten(batch_onTarget) #(-1, 80)
    offTarget_Flat = tf.contrib.layers.flatten(batch_offTarget) #(-1, 92)

    onTarget1 = tf.nn.relu(tf.matmul(onTarget_Flat, onT1_W) + onT1_B)
    onTarget1_Drop = tf.nn.dropout(onTarget1, DROPOHT_RATE)
    onTarget2 = tf.nn.relu(tf.matmul(onTarget1_Drop, onT2_W) + onT2_B)

    offTarget1 = tf.nn.relu(tf.matmul(offTarget_Flat, offT1_W) + offT1_B)
    offTarget1_Drop = tf.nn.dropout(offTarget1, DROPOHT_RATE)
    offTarget2 = tf.nn.relu(tf.matmul(offTarget1_Drop, offT2_W) + offT2_B)

    target = tf.concat([onTarget2, offTarget2], axis=-1)
    target_Drop = tf.nn.dropout(target, DROPOHT_RATE)

    target1 = tf.nn.relu(tf.matmul(target_Drop, fc3_W)+ fc3_B )
    target1_Drop = tf.nn.dropout(target1, DROPOHT_RATE)

    target2 = tf.nn.relu(tf.matmul(target1_Drop, fc4_W)+ fc4_B)
    target2_Drop = tf.nn.dropout(target2, DROPOHT_RATE)

    target3 = tf.nn.relu(tf.matmul(target2_Drop, fc5_W)+ fc5_B)
    target3_Drop = tf.nn.dropout(target3, DROPOHT_RATE)

    target4 = tf.nn.relu(tf.matmul(target3_Drop, fc6_W)+ fc6_B)
    target4_Drop = tf.nn.dropout(target4, DROPOHT_RATE)

    target5 = tf.nn.relu(tf.matmul(target4_Drop, fc7_W)+ fc7_B)
    target5_Drop = tf.nn.dropout(target5, DROPOHT_RATE)


    target6 = tf.matmul(target5_Drop, fc8_W)+ fc8_B
    result = tf.matmul(target6, fc9_W)

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

            opt, mse = sess.run([train_step, mean_t])
            print(i, " Step - AdamOpt : ", opt, " MSE : ", mse)

            i = i + 1
        except tf.errors.OutOfRangeError:
            break
    coord.request_stop()
    coord.join(threads)