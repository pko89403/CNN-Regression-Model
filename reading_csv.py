import tensorflow as tf
import os
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_dir = ['./data/train.csv']
batch_size = 50
DROPOHT_RATE = 0.5
LEARNING_RATE = 0.00001

letters = "ACGT"
onTargetLen = 20
offTargetLen = 23

mapping_letters = tf.string_split([letters], delimiter="")

def init_weights(shape, stddev = 0.03):
    return tf.Variable(tf.random_normal(shape, stddev))
def seq_processing(seq):
    table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_letters.values, default_value=0)
    seq_char = tf.string_split(seq, delimiter="")
    encoded = tf.one_hot(table.lookup(seq_char.values), len(letters), dtype=tf.float32)
    return encoded

def create_file_reader_ops(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = reader.read(filename_queue)
    record_defaults = [[""], [""], [000.000]]
    onTargetSEQ, offTargetSEQ, label = tf.decode_csv(csv_row, record_defaults=record_defaults, field_delim=",")

    onTargetSEQ = tf.reshape(onTargetSEQ, [1])
    onTarget = seq_processing(onTargetSEQ)
    offTargetSEQ = tf.reshape(offTargetSEQ, [1])
    offTarget = seq_processing(offTargetSEQ)
    label = tf.reshape(label, [1])
    return onTarget, offTarget, label

filename_queue = tf.train.string_input_producer(data_dir)
onTarget, offTarget, label = create_file_reader_ops(filename_queue)

def model():
    conICh= 4
    convOCH = 80
    onTargetFilter = 3
    offTargetFilter = 6
    # Convolution1D(80, 5, relu) -> AveragePooling(2)
    # Flatten()
    # FullyConnected(80) -> DropOut(0.3)
    # FullyConnected(40) -> DropOut(0.3)
    # FullyConnected(40) -> DropOut(0.3)
    # FullyConnected(1, linear) -> Result

    onTargetW1 = init_weights(shape = [onTargetFilter, conICh, convOCH], stddev=0.03)
    onTargetB1 = init_weights(shape = [convOCH], stddev=0.03)
    onTargetConv = tf.nn.conv1d(batch_onTarget, onTargetW1, stride=1, padding="VALID")  # (1, 18, 80)
    onTargetConv_Relu = tf.nn.relu(onTargetConv+onTargetB1)
    onTargetConv_Relu_Pool = tf.nn.pool(onTargetConv_Relu, window_shape=[2], padding="VALID", pooling_type="AVG") # (1, 17, 80)

    offTargetW1 = init_weights(shape = [offTargetFilter, conICh, convOCH], stddev=0.03)
    offTargetB1 = init_weights(shape=[convOCH], stddev=0.03)
    offTargetConv = tf.nn.conv1d(batch_offTarget, offTargetW1, stride=1, padding="VALID") # (1, 18, 80)
    offTargetConv_Relu = tf.nn.relu(offTargetConv+offTargetB1)
    offTargetConv_Relu_Pool = tf.nn.pool(offTargetConv_Relu, window_shape=[2], padding="VALID", pooling_type="AVG") # (1, 17, 80)

    targetConcat = tf.concat([onTargetConv_Relu_Pool, offTargetConv_Relu_Pool], axis = -1) # (1, 17, 160)

    targetConcat_Flat = tf.contrib.layers.flatten(targetConcat)
    targetConcat_Flat_Drop = tf.nn.dropout(targetConcat_Flat,DROPOHT_RATE)

    fc1_W = init_weights(shape=[17*160, 80], stddev=0.03)
    fc1_B = init_weights(shape=[80], stddev=0.03)
    fc1 = tf.nn.relu(tf.matmul(targetConcat_Flat_Drop, fc1_W) + fc1_B)
    fc1_Drop = tf.nn.dropout(fc1, DROPOHT_RATE)

    fc2_W = init_weights(shape=[80, 40], stddev=0.03)
    fc2_B = init_weights(shape=[40], stddev=0.03)
    fc2 = tf.nn.relu(tf.matmul(fc1_Drop, fc2_W) + fc2_B)
    fc2_Drop = tf.nn.dropout(fc2, DROPOHT_RATE)

    fc3_W = init_weights(shape=[40, 40], stddev=0.03)
    fc3_B = init_weights(shape=[40], stddev=0.03)
    fc3 = tf.nn.relu(tf.matmul(fc2_Drop, fc3_W) + fc3_B)
    fc3_Drop = tf.nn.dropout(fc3, DROPOHT_RATE)

    fc4_W = init_weights(shape=[40, 1], stddev=0.03)
    fc4_B = init_weights(shape=[1], stddev=0.03)
    result = tf.add(tf.matmul(fc3_Drop, fc4_W), fc4_B)

    return result

filename_queue = tf.train.string_input_producer(data_dir)
onTarget, offTarget, label = create_file_reader_ops(filename_queue)
batch_onTarget, batch_offTarget, batch_label = tf.train.batch([onTarget, offTarget, label], shapes=[[onTargetLen,4], [offTargetLen,4], [1]], batch_size=batch_size)
model_Pred = model()

l, p = confusion_matrix.remove_squeezable_dimensions(batch_label, model_Pred)
s = math_ops.square(p - l)
mean_t = math_ops.reduce_mean(s)

mse = tf.losses.mean_squared_error(model_Pred, batch_label)

adamOpt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_step = adamOpt.minimize(mse)

saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.tables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    i=1
    while (True):
        try:
            onT, lab = sess.run([batch_onTarget, batch_label])

            sess.run([train_step])
            e_val = sess.run(mse)
            print("STEP", i, ": LABEL ", len(lab), " MSE : ", e_val)

            i = i+1
            if(i % 2000):
                saver.save(sess, './save_model/MSEmodel', global_step=i)
        except tf.errors.OutOfRangeError:
            break


    coord.request_stop()
    coord.join(threads)