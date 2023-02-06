import tensorflow as tf

def DIN(seq, seq_len, target, conf, scope="DIN"):
    # seq BATCH_SIZE * SEQ_LEN * FEAT_NUM : N * M * H
    # target BATCH_SIZE * FEAT_NUM : N * H
    # return : BATCH_SIZE * H

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        seq_shape = tf.shape(seq)
        target = tf.tile(target, [1, seq_shape[1], 1])

        input = tf.concat([seq, target, seq - target, seq * target], axis=-1)

        layers = conf.get("layers", [64, 32])
        for layer in layers:
            input = tf.layers.dense(input, layer, activation=tf.nn.sigmoid, name="att_"+str(layer))

        input = tf.squeeze(tf.layers.dense(input, 1, activation=None, name="att_final"), axis=-1) # N * M

        # Mask
        seq_mask = tf.squeeze(tf.sequence_mask(seq_len, seq_shape[1]))
        # seq_mask = tf.Print(seq_mask, [seq_mask], message="seq_mask", summarize=100)
        padding = tf.ones_like(input) * (-2 ** 32 + 1)
        attention = tf.nn.softmax(tf.where(seq_mask, input, padding), axis=-1) # N * M
        # attention = tf.Print(attention, [attention], message="attention", summarize=100)
        attention = tf.tile(tf.expand_dims(attention, axis=2), [1, 1, seq_shape[2]])
        output = tf.reduce_sum(tf.transpose(attention * seq, [0, 2, 1]), axis=1)
    return output
