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
    line= line[0:20]
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

    line= line[0:20]
    test_OFF.append(line.strip())

test_Label = []
for line in labelTList:
    test_Label.append(line.strip())


letters = "ACGT"
onTargetLen = 20
offTargetLen = 20
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
train_data = train_data.repeat(100).shuffle(10000).batch(100)

test_data = Dataset.from_tensor_slices((test_ON, test_OFF, test_Label))
test_data = test_data.map(input_parser)
test_data = test_data.batch(100)


iterator = Iterator.from_structure(train_data.output_types,
                                   train_data.output_shapes)

input_onT, input_offT, input_Labels = iterator.get_next()
train_init_op = iterator.make_initializer(train_data)
test_init_op = iterator.make_initializer(test_data)


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.05))

def model(onTarget, offTarget, labels, w0, bO, wOF, bOF, wMO, wMOF, bM, w1, b1, w2, b2, w3, b3, w4, b4):

    onTarget_Flat = tf.layers.flatten(onTarget)
    offTarget_Flat = tf.layers.flatten(offTarget)
    print(onTarget_Flat)
    on = tf.matmul(onTarget_Flat, w0) + bO
    off = tf.matmul(offTarget_Flat, wOF) + bOF

    merge = tf.add(tf.add(tf.multiply(on, wMO), tf.multiply(off, wMOF)), bM)
    h1 = tf.matmul(merge, w1) +  b1
    h1 = tf.nn.dropout(h1, 0.8)

    h2 = tf.matmul(h1, w2) +  b2
    h2 = tf.nn.dropout(h2, 0.8)

    h3 = tf.matmul(h2, w3) +  b3
    h3 = tf.nn.dropout(h3, 0.8)

    h4 = tf.matmul(h3, w4)
    hypothesis = tf.sigmoid(h4)

    return hypothesis, labels

wO = init_weights([80, 160])
bO = init_weights([160])

wOF = init_weights([80, 160])
bOF = init_weights([160])

wMO = init_weights([160])
wMOF = init_weights([160])
bM = init_weights([160])

w1 = init_weights([160, 320])
b1 = init_weights([320])

w2 = init_weights([320, 320])
b2 = init_weights([320])

w3 = init_weights([320, 320])
b3 = init_weights([320])

w4 = init_weights([320, 1])
b4 = init_weights([1])

h, y = model(input_onT, input_offT, input_Labels, wO, bO, wOF, bOF, wMO, wMOF, bM, w1, b1, w2, b2, w3, b3, w4, b4)

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

