import os
import tensorflow as tf

letters = "ACGT"
mapping_letters = tf.string_split([letters], delimiter="")
letter = ['A','C','G','T']
batch_size = 10
BATCH_SIZE = 1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
data_dir = ['./data/train.csv']

"""
def nuc_proc(sequence):
    table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_letters.values, default_value=0)
    test_char = tf.string_split(sequence, delimiter="")
    encoded = tf.one_hot(table.lookup(test_char.values), len(letters), dtype=tf.int8)
    encoded = tf.transpose(encoded)
    return encoded


filename_queue = tf.train.string_input_producer(data_dir, shuffle=False, name = 'filename_queue')

# set tensorflow reader
reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)

# set record_defaults corresponding to data form
seq1, seq2, mr = tf.decode_csv(value, record_defaults = [[''], [''], [0.]])
# set collecting data and batch option
# seq1, seq2, mr = tf.train.batch( [data[0], data[1], data[-1] ], batch_size=BATCH_SIZE)

#data
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    mapping_letters = tf.string_split([letters], delimiter="")
    test = sess.run(seq1)

    table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_letters.values, default_value=0)
    test_char = tf.string_split(test], delimiter="")
    encoded = tf.one_hot(table.lookup(test_char.values), len(letters), dtype=tf.int8)


        tf.InteractiveSession().as_default()
    

    for i in range(1):
        x = sess.run([seq1])
        encoded = tf.map_fn(nuc_proc(),seq1)
        print(encoded)


    coord.request_stop()
    coord.join(threads)


    encoded = tf.reshape(encoded, [BATCH_SIZE, -1,4])
    print(encoded)
    print(encoded.eval())
    print(tf.shape(encoded))
    
"""

def seq_processing(example1):
    table = tf.contrib.lookup.index_table_from_tensor(mapping=letter, default_value=0)
    test_char1 = tf.string_split([example1], delimiter="")
    encoded1 = tf.one_hot(table.lookup(test_char1.values), len(letters), dtype=tf.int8)
    return encoded1

def read_my_file_format(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    example1, example2, label = tf.decode_csv(value, record_defaults = [[""],[""], [0.]])

    example1 = tf.reshape(example1, shape=[1])
    print("check example1 ", type(example1), example1.get_shape())

    #processed_example1, = seq_processing(example1)
    processed_example1 = tf.map_fn(fn = seq_processing, elems = example1)
    processed_example1 = tf.reshape(processed_example1, [1, 4,20])

    print(processed_example1.get_shape())
    #processed_example1= example1
    return processed_example1, label

def input_pipeline(filenames, batch_size, read_threads, num_epoches = None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epoches, shuffle=True)

    example1_batch, label_batch = read_my_file_format(filename_queue)
    print(example1_batch, label_batch)
    example_list = [read_my_file_format(filename_queue) for _ in range(read_threads)]
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_List = tf.train.shuffle_batch_join(
        example_list, shapes=[[batch_size, 4,20],()], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)

    return example_List


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    table_op = tf.tables_initializer()
    example1_batch, label_batch = input_pipeline(data_dir,batch_size=1,read_threads=1)

    print(sess.run(example1_batch))
    print(sess.run(label_batch))