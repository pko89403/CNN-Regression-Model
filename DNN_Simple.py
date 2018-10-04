import tensorflow as tf
import os
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_dir = ['./data/train2.csv']
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

onT1_W = init_weights(shape=[80, 160], stddev=0.03)
onT1_B = init_weights(shape=[160])

onT2_W = init_weights(shape=[160, 320], stddev=0.03)
onT2_B = init_weights(shape=[320])

onT3_W = init_weights(shape=[320, 1], stddev=0.03)
onT3_B = init_weights(shape=[1])

offT1_W = init_weights(shape=[92, 184], stddev=0.03)
offT1_B = init_weights(shape=[184])

offT2_W = init_weights(shape=[184, 368], stddev=0.03)
offT2_B = init_weights(shape=[368])

offT3_W = init_weights(shape=[368, 1], stddev=0.03)
offT3_B = init_weights(shape=[1])



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

    onTarget1 = tf.nn.sigmoid(tf.matmul(onTarget_Flat, onT1_W) + onT1_B)
    onTarget1_Drop = tf.nn.dropout(onTarget1, DROPOHT_RATE)

    onTarget2 = tf.nn.sigmoid(tf.matmul(onTarget1_Drop, onT2_W) + onT2_B)
    onTarget2_Drop = tf.nn.dropout(onTarget2, DROPOHT_RATE)

    onTarget3 = tf.matmul(onTarget2_Drop, onT3_W) + onT3_B

    offTarget1 = tf.nn.sigmoid(tf.matmul(offTarget_Flat, offT1_W) + offT1_B)
    offTarget1_Drop = tf.nn.dropout(offTarget1, DROPOHT_RATE)

    offTarget2 = tf.nn.sigmoid(tf.matmul(offTarget1_Drop, offT2_W) + offT2_B)
    offTarget2_Drop = tf.nn.dropout(offTarget2, DROPOHT_RATE)

    offTarget3 = tf.matmul(offTarget2_Drop, offT3_W) + offT3_B

    result = tf.add(onTarget3, offTarget3)

    return result


filename_queue = tf.train.string_input_producer(data_dir)
onTarget, offTarget, label = create_file_reader_ops(filename_queue)
batch_onTarget, batch_offTarget, batch_label = tf.train.batch([onTarget, offTarget, label],
                                                              shapes=[[onTargetLen, 4], [offTargetLen, 4], [1]],
                                                              batch_size=batch_size)

model_Pred = model()

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

            opt, mse, l, p = sess.run([train_step, loss, batch_label, model_Pred])
            print(i, " Step - AdamOpt : ", opt, " MSE : ", mse, " l : ", l, " p : ", p)

            i = i + 1
        except tf.errors.OutOfRangeError:
            break
    coord.request_stop()
    coord.join(threads)