# coding: utf-8 -*-
from data_input import *
from tools import *
from config import *
from util import *
import numpy as np
import tensorflow as tf
from layers import *
import os
from collections import namedtuple
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class DNN:
    def __init__(self):
        pass

    # env_feature = > dense_feature
    # cxr_feature = > screen_predict_feature
    # cat_feature = > screen_cate_feature
    # dense_feature = > screen_dense_feature


    def _inter_poi_self_attention(self,input):
        encoder_Q = tf.matmul(tf.reshape(input, (-1, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1] + 1)),
                              self.inter_poi_self_attention_weights['weight_Q'])  # (batch * 5) * 8
        encoder_K = tf.matmul(tf.reshape(input, (-1, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1] + 1)),
                              self.inter_poi_self_attention_weights['weight_K'])  # (batch * 5) * 8
        encoder_V = tf.matmul(tf.reshape(input, (-1, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1] + 1)),
                              self.inter_poi_self_attention_weights['weight_V'])  # (batch * 5) * 8

        encoder_Q = tf.reshape(encoder_Q,(tf.shape(input)[0], POI_NUM, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1] + 1))  # batch * 5 * 4
        encoder_K = tf.reshape(encoder_K,(tf.shape(input)[0], POI_NUM, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1] + 1))  # batch * 5 * 4
        encoder_V = tf.reshape(encoder_V,(tf.shape(input)[0], POI_NUM, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1] + 1))  # batch * 5 * 4

        attention_map = tf.matmul(encoder_Q, tf.transpose(encoder_K, [0, 2, 1]))  # batch * 5 * 5

        attention_map = attention_map / 8
        attention_map = tf.nn.softmax(attention_map)  # batch * 5 * 5

        output = tf.reshape(tf.matmul(attention_map, encoder_V),
                            (-1, POI_NUM * MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1] + 1))  # batch * 5 * 4

        return output

    def _extr_network(self, features):
        # 自然结果内部做self-attention
        self.nature_poi = tf.gather(self.input_embedding, list([0, 2, 4]), axis=1)
        self.ad_poi = tf.gather(self.input_embedding, list([1, 3]), axis=1)
        emb_size = CATE_FEATURE_EMBEDDINGS_SHAPE[-1] * FEATURE_CATE_NUM

        mask = tf.squeeze(tf.sequence_mask(features['fea_mask'], FEATURE_NATURE_POI))
        nature_poi = multihead_attention(queries=self.nature_poi,
                                         keys=self.nature_poi,
                                         values=self.nature_poi,
                                         key_masks=mask,
                                         num_heads=4,
                                         dropout_rate=0.1,
                                         training=self.train
                                         )

        nature_poi = feedforward(nature_poi,
                                 num_units=[emb_size,
                                            emb_size],
                                 is_training=self.train)

        # 与ad_poi做target-attention
        target_att = multihead_attention(queries=self.ad_poi,
                                         keys=nature_poi,
                                         values=nature_poi,
                                         key_masks=mask,
                                         num_heads=4,
                                         dropout_rate=0.1,
                                         training=self.train,
                                         causality=True,
                                         scope="target_att"
                                         )

        target_att = feedforward(target_att,
                                 num_units=[emb_size,
                                            emb_size],
                                 is_training=self.train,
                                 scope="target_att"
                                 )
        return target_att


    def _build_model(self, features, labels, mode, params):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('dnn_model'):
            self.extr_out = self._extr_network(features)
            self.extr_out = tf.reshape(self.extr_out, [-1, 2 * self.single_input_size])
            self.feat_predict = tf.reshape(self.feat_predict, [-1, 5 * 1])
            fc_out = self.input_embedding

            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "MLP_A" + str(i)
                fc_out = tf.layers.dense(fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i], activation=None,
                                         name=dense_name)
                fc_out = tf.nn.swish(fc_out)
            fc_out = tf.reshape(fc_out, [-1, POI_NUM * MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1]])
            fc_out = tf.concat([self.extr_out, self.feat_predict, fc_out], axis=1)
            fc_out_ctr, fc_out_imp = fc_out, fc_out

            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'])):
                dense_name = "MLP_B" + str(i)
                fc_out_ctr = tf.layers.dense(fc_out_ctr, MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'][i], activation=None, name=dense_name)
                fc_out_ctr = tf.nn.swish(fc_out_ctr)
            ctr_out = tf.layers.dense(fc_out_ctr, OUT_NUM, activation=None, name="final_out_ctr")

            if not self.train:
                ctr_out = tf.nn.sigmoid(ctr_out)

            self.out = tf.reshape(ctr_out, [-1, POI_NUM])
            self.Q_network_output = self.out


    def _create_loss(self, labels):
        with tf.name_scope('loss'):
            self.label = labels['ctr_label']
            self.mask = labels['mask']
            # ctr_loss
            self.loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.label, self.out, weights=self.mask))

    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999,
                                                epsilon=1e-8)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
    
    def _create_weights(self):
        with tf.name_scope('feature_emb_weights'):
            self.feature_weights = {
                'embedding': tf.get_variable('embedding',
                                             shape=CATE_FEATURE_EMBEDDINGS_SHAPE,
                                             initializer=tf.zeros_initializer())
            }

            glorot = np.sqrt(2.0 / (4 + 1))

            self.inter_poi_self_attention_weights = {
                'weight_Q': tf.get_variable(name='inter_poi_self_attention_weights_Q',
                                            shape=[MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1] + 1, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1] + 1],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_K': tf.get_variable(name='inter_poi_self_attention_weights_K',
                                            shape=[MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1] + 1, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1] + 1],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_V': tf.get_variable(name='inter_poi_self_attention_weights_V',
                                            shape=[MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1] + 1, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1] + 1],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
            }

    def _process_features(self, features):
        # env_feature = > dense_feature
        # cxr_feature = > screen_predict_feature
        # cat_feature = > screen_cate_feature
        # dense_feature = > screen_dense_feature

        # N * M * K
        # N * D ( D <= M )
        self.cate_feature_embeddings = tf.reshape(tf.nn.embedding_lookup(
            self.feature_weights['embedding'], features['cate_feature']),
            [-1, POI_NUM, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        
        self.input_embedding = tf.concat(
            [self.cate_feature_embeddings], axis=2)
        self.single_input_size = FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]
        self.feat_predict = features['dense_feature']
        return features

    def _get_attr_hash(self, tensors, emb_table, num):
        first_cate, second_cate, thrid_cate, _ = tf.split(tensors,[1, 1, 1, 3], axis=2)
        first_cate, second_cate, thrid_cate  = tf.squeeze(first_cate, axis=2), tf.squeeze(second_cate, axis=2),  tf.squeeze(thrid_cate, axis=2)
        first_cate = tf.reshape(tf.nn.embedding_lookup(
            emb_table, first_cate),
            [-1, num, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        second_cate = tf.reshape(tf.nn.embedding_lookup(
            emb_table, second_cate),
            [-1, num, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        thrid_cate = tf.reshape(tf.nn.embedding_lookup(
            emb_table, thrid_cate),
            [-1, num, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        return first_cate, second_cate, thrid_cate


    def _delivery_hash(self, tensors):
        feat_fei, feat_juli, feat_shijian, feat_qisongjia = tf.split(tensors, [1, 1, 1, 1], axis=2)
        feat_fei, feat_juli, feat_shijian, feat_qisongjia = float_custom_hash(feat_fei, "feat_fei", 1),  float_custom_hash(feat_juli, "feat_juli"), float_custom_hash(feat_shijian, "feat_shijian"), float_custom_hash(feat_qisongjia, "feat_qisongjia", 2)
        ad_delivery = tf.concat([feat_fei, feat_juli, feat_shijian, feat_qisongjia], axis=-1)
        return ad_delivery

    def _create_indicator(self, labels):
        ctr_out = self.out
        ctr_out = tf.reduce_mean(tf.nn.sigmoid(ctr_out))

        # All gradients of loss function wrt trainable variables
        '''
        grads = tf.gradients(self.loss, tf.trainable_variables())
        for grad, var in list(zip(grads, tf.trainable_variables())):
            tf.summary.histogram(var.name + '/gradient', grad)
        '''

        def format_log(tensors):
            log0 = "train info: step {}, loss={:.4f}, ctr_loss={:.4f}, " \
                   "ctr_out={:.4f}".format(
                tensors["step"], tensors["loss"], tensors["loss"],
                tensors["ctr_out"],
            )
            return log0

        self.logging_hook = tf.train.LoggingTensorHook({"step": tf.train.get_global_step(),
                                                        "loss": self.loss,
                                                        "ctr_out": ctr_out 
                                                        },
                                                       every_n_iter=5,
                                                       formatter=format_log)

    def model_fn_estimator(self, features, labels, mode, params):
        self._create_weights()
        self._process_features(features)
        self._build_model(features, labels, mode, params)
        if self.train:
            self._create_loss(labels)
            self._create_optimizer()
            self._create_indicator(labels)
            return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, train_op=self.train_op, training_hooks=[self.logging_hook])
        else:
            if 'save_model' in list(params.keys()):
                outputs = {
                    "Q_network_output": tf.identity(self.Q_network_output, "Q_network_output"),
                    "out": tf.identity(self.out, "out")
                    }
            else:
                ctr_out = self.out
                # gmv
                outputs = {'out': self.out,
                           'mask': features['mask'],
                           'ctr_out': ctr_out,
                           'ctr_label': features['ctr_label'],
                           'q_out': self.Q_network_output,
                           'cxr_feature': features['dense_feature'],
                            }
            export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                      tf.estimator.export.PredictOutput(outputs)}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=outputs, export_outputs=export_outputs)


    def tf_print(self, var, varStr='null'):
        return tf.Print(var, [var], message=varStr, summarize=100)
