import tensorflow as tf

def index_matrix_to_pairs(index_matrix):
    # [[3,1,2], [2,3,1]] -> [[[0, 3], [1, 1], [2, 2]],
    #                        [[0, 2], [1, 3], [2, 1]]]
    replicated_first_indices = tf.range(tf.shape(index_matrix)[0])
    rank = len(index_matrix.get_shape())
    if rank == 2:
        replicated_first_indices = tf.tile(
            tf.expand_dims(replicated_first_indices, axis=1),
            [1, tf.shape(index_matrix)[1]])
        replicated_first_indices = tf.cast(replicated_first_indices, dtype=tf.int64)
    return tf.stack([replicated_first_indices, index_matrix], axis=rank)

def string_hash_to_index(tensor, bucket=1<<22):
    return tf.strings.to_hash_bucket_fast(tensor, bucket)

def int_to_string_with_key(tensor, key):
    return key + "_" + tf.strings.as_string(tensor)

def float_to_string_with_key(tensor, key, precision=1):
    return key + "_" + tf.strings.as_string(tensor, precision)

def float_to_int(tensor, order):
    wc = 10 ** order   
    return tf.cast(tensor * wc, tf.int64)

def float_custom_hash(tensor, key, precision=0, bucket=1<<22):
    tensor = float_to_string_with_key(tensor, key, precision)
    tensor = string_hash_to_index(tensor, bucket)
    return tensor

def int_custom_hash(tensor, key, bucket=1<<22):
    tensor = int_to_string_with_key(tensor, key)
    tensor = string_hash_to_index(tensor, bucket)
    return tensor
