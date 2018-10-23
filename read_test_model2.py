import tensorflow as tf
from tensorflow.data import Dataset, Iterator

train_ON_Path = './data/onTarget_Train_filt.txt'
train_OFF_Path = './data/offTarget_Train_filt.txt'
train_LAB_Path = './data/label_Train_filt.txt'

test_ON_Path = './data/test_on.txt'
test_OFF_Path = './data/test_off.txt'
test_LAB_Path = './data/test_label.txt'

onList = open(train_ON_Path, 'r')
offList = open(train_OFF_Path, 'r')
labelList = open(train_LAB_Path, 'r')

train_ON = []
for line in onList:
    train_ON.append(line.strip())

train_OFF = []
for line in offList:
    train_OFF.append(line.strip())

train_Label = []
for line in labelList:
    train_Label.append(line.strip())

onTList = open(test_ON_Path, 'r')
offTList = open(test_OFF_Path, 'r')
labelTList = open(test_LAB_Path, 'r')

test_ON = []
for line in onTList:
    test_ON.append(line.strip())

test_OFF = []
for line in offTList:
    test_OFF.append(line.strip())

test_Label = []
for line in labelTList:
    test_Label.append(line.strip())


letters = "ACGT"
onTargetLen = 20
offTargetLen = 23
mapping_letters = tf.string_split([letters], delimiter="")
table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_letters.values, default_value=0)

def seq_processing(seq):
    seq = tf.reshape(seq, [1])
    seq_char = tf.string_split(seq, delimiter="")
    encoded = tf.one_hot(table.lookup(seq_char.values), len(letters), dtype=tf.float32)
    return encoded

def input_parser(onTarget, offTarget, label):
    onTarget_enc= seq_processing(onTarget)
    offTarget_enc = seq_processing(offTarget)
    label = tf.string_to_number(label)
    return onTarget_enc, offTarget_enc, label


train_data = Dataset.from_tensor_slices((train_ON, train_OFF, train_Label))
train_data = train_data.map(input_parser)
train_data = train_data.repeat(10).shuffle(10000).batch(100)

test_data = Dataset.from_tensor_slices((test_ON, test_OFF, test_Label))
test_data = test_data.map(input_parser)
test_data = test_data.batch(100)


iterator = Iterator.from_structure(train_data.output_types,
                                   train_data.output_shapes)

input_onT, input_offT, input_Labels = iterator.get_next()
train_init_op = iterator.make_initializer(train_data)
test_init_op = iterator.make_initializer(test_data)


def init_weights(shape,stddev=0.05):
    return tf.Variable(tf.random_normal(shape, stddev=0.05))

def model(onTarget, offTarget, labels,
          onTargetW1,onTargetB1, onTargetW2,onTargetB2, onTargetW3,onTargetB3, onTargetW4,onTargetB4,
          offTargetW1,offTargetB1, offTargetW2,offTargetB2, offTargetW3,offTargetB3, offTargetW4,offTargetB4,
          fc1_W,fc1_B, fc2_W,fc2_B, fc3_W,fc3_B, fc4_W,fc4_B, fc5_W,fc5_B):
    onTargetConv = tf.nn.conv1d(onTarget, onTargetW1, stride=1, padding="SAME")  # (1, 20, 4)
    onTargetConv_Relu = tf.nn.relu(onTargetConv + onTargetB1)

    onTargetConv2 = tf.nn.conv1d(onTargetConv_Relu, onTargetW2, stride=1, padding="SAME")  # (1, 20, 4)
    onTargetConv_Relu2 = tf.nn.relu(onTargetConv2 + onTargetB2)

    onTargetConv_Relu2_Pool = tf.nn.pool(onTargetConv_Relu2, window_shape=[2], padding="SAME", strides=[2],
                                         pooling_type="MAX")  # (1, 20, 256)


    onTargetConv3 = tf.nn.conv1d(onTargetConv_Relu2_Pool, onTargetW3, stride=1, padding="SAME")  # (1, 10, 160)
    onTargetConv3_Relu = tf.nn.relu(onTargetConv3 + onTargetB3)

    onTargetConv4 = tf.nn.conv1d(onTargetConv3_Relu, onTargetW4, stride=1, padding="SAME")  # (1, 10, 160)
    onTargetConv4_Relu = tf.nn.relu(onTargetConv4 + onTargetB4)

    onTargetConv4_Relu_Pool = tf.nn.pool(onTargetConv4_Relu, window_shape=[2], padding="SAME", strides=[2],
                                         pooling_type="MAX")  # (1, 16, 512)

    offTargetConv = tf.nn.conv1d(offTarget, offTargetW1, stride=1, padding="SAME")  # (1, 21, 256)
    offTargetConv_Relu = tf.nn.relu(offTargetConv + offTargetB1)

    offTargetConv2 = tf.nn.conv1d(offTargetConv_Relu, offTargetW2, stride=1, padding="SAME")  # (1, 21, 256)
    offTargetConv2_Relu = tf.nn.relu(offTargetConv2 + offTargetB2)

    offTargetConv_Relu2_Pool = tf.nn.pool(offTargetConv2_Relu, window_shape=[3], padding="SAME", strides=[2],
                                          pooling_type="MAX")  # (1, 20, 256)
    offTargetConv3 = tf.nn.conv1d(offTargetConv_Relu2_Pool, offTargetW3, stride=1, padding="SAME")  # (1, 18, 256)
    offTargetConv3_Relu = tf.nn.relu(offTargetConv3 + offTargetB3)

    offTargetConv4 = tf.nn.conv1d(offTargetConv3_Relu, offTargetW4, stride=1, padding="SAME")  # (1, 18, 256)
    offTargetConv4_Relu = tf.nn.relu(offTargetConv4 + offTargetB4)

    offTargetConv4_Relu_Pool = tf.nn.pool(offTargetConv4_Relu, window_shape=[3], padding="VALID", strides=[2],
                                          pooling_type="MAX")  # (1, 16, 256)

    print(offTargetConv4_Relu_Pool.get_shape())
    targetConcat = tf.concat([onTargetConv4_Relu_Pool, offTargetConv4_Relu_Pool], axis=-1)  # (1, 9, 512)
    targetConcat_Flat = tf.contrib.layers.flatten(targetConcat)
    # Model
    fc1 = tf.nn.relu(tf.matmul(targetConcat_Flat, fc1_W) + fc1_B)
    fc1_Drop = tf.nn.dropout(fc1, 0.8)

    fc2 = tf.nn.relu(tf.matmul(fc1_Drop, fc2_W) + fc2_B)
    fc2_Drop = tf.nn.dropout(fc2, 0.8)

    fc3 = tf.nn.relu(tf.matmul(fc2_Drop, fc3_W) + fc3_B)
    fc3_Drop = tf.nn.dropout(fc3, 0.8)

    fc4 = tf.nn.relu(tf.matmul(fc3_Drop, fc4_W) + fc4_B)

    fc5 = tf.matmul(fc4, fc5_W) + fc5_B

    # model_Pred = tf.nn.(tf.add(tf.matmul(fc3_Drop, fc4_W), fc4_B))sigmoid
    model_Pred = fc5

    hypothesis = tf.nn.sigmoid(model_Pred)

    return hypothesis, labels

# Weights & Bias For Model
conICh = 4
convOCH = 80
conv0CH2 = 80
convOCH3 = 160
convOCH4 = 160
onTargetFilter = 1  # 20 - 2 + 1 = 19
onTargetW1 = init_weights(shape=[onTargetFilter, conICh, convOCH], stddev=0.01)
onTargetB1 = init_weights(shape=[convOCH], stddev=0.01)

onTargetW2 = init_weights(shape=[onTargetFilter, convOCH, conv0CH2], stddev=0.01)
onTargetB2 = init_weights(shape=[conv0CH2], stddev=0.01)

onTargetFilter2 = 1  # 18 - 2 + 1
onTargetW3 = init_weights(shape=[onTargetFilter2, conv0CH2, convOCH3], stddev=0.01)
onTargetB3 = init_weights(shape=[convOCH3], stddev=0.01)

onTargetW4 = init_weights(shape=[onTargetFilter2, convOCH3, convOCH4], stddev=0.01)
onTargetB4 = init_weights(shape=[convOCH4], stddev=0.01)

offTargetFilter = 1  # 23 - 3 + 1
offTargetW1 = init_weights(shape=[offTargetFilter, conICh, convOCH], stddev=0.01)
offTargetB1 = init_weights(shape=[convOCH], stddev=0.01)

offTargetW2 = init_weights(shape=[offTargetFilter, convOCH, conv0CH2], stddev=0.01)
offTargetB2 = init_weights(shape=[conv0CH2], stddev=0.01)

offTargetFilter2 = 1  # 20 - 3 + 1
offTargetW3 = init_weights(shape=[offTargetFilter2, conv0CH2, convOCH3], stddev=0.01)
offTargetB3 = init_weights(shape=[convOCH3], stddev=0.01)

offTargetW4 = init_weights(shape=[offTargetFilter2, convOCH3, convOCH4], stddev=0.01)
offTargetB4 = init_weights(shape=[convOCH4], stddev=0.01)

# model
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


h, y = model(input_onT, input_offT, input_Labels,
             onTargetW1,onTargetB1, onTargetW2,onTargetB2, onTargetW3,onTargetB3, onTargetW4,onTargetB4,
             offTargetW1,offTargetB1, offTargetW2,offTargetB2, offTargetW3,offTargetB3, offTargetW4,offTargetB4
             ,fc1_W,fc1_B,fc2_W,fc2_B,fc3_W,fc3_B,fc4_W,fc4_B,fc5_W,fc5_B)

cost = -tf.reduce_mean(y * tf.log(h) + (1-y)*tf.log(1-h))
optimizer = tf.train.GradientDescentOptimizer(0.0001)
train_step = optimizer.minimize(cost)

predicted = tf.cast(h > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

init_op = tf.global_variables_initializer()
table_op = tf.tables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(table_op)
    sess.run(train_init_op)
    epoch = 0
    while(True):
        try:
            epoch = epoch+1
            l, _, acc = sess.run([cost, train_step, accuracy])
            if(epoch % 50 == 0):
                print(epoch, l, acc)
        except tf.errors.OutOfRangeError:
            print("Train end of dataset")

            break
    sess.run(test_init_op)

    while(True):
        try:
            epoch = epoch+1
            acc = sess.run([accuracy])
            print("epoch : ", epoch ,"// acc : ", acc)
        except tf.errors.OutOfRangeError:
            print("Test end of dataset")

            break

