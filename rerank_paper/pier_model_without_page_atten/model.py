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

    def inter_attr_self_attention(self,input):

        encoder_Q = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.inter_attr_self_attention_weights['weight_Q'])  # (batch * 5) * 8
        encoder_K = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.inter_attr_self_attention_weights['weight_K'])  # (batch * 5) * 8
        encoder_V = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.inter_attr_self_attention_weights['weight_V'])  # (batch * 5) * 8

        encoder_Q = tf.reshape(encoder_Q,
                               (tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_K = tf.reshape(encoder_K,
                               (tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_V = tf.reshape(encoder_V,
                               (tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        attention_map = tf.matmul(encoder_Q, tf.transpose(encoder_K, [0, 2, 1]))  # batch * 5 * 5

        attention_map = attention_map / 8
        attention_map = tf.nn.softmax(attention_map)  # batch * 5 * 5

        output = tf.reshape(tf.matmul(attention_map, encoder_V),
                            (-1, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        output = tf.matmul(tf.reshape(input,[-1, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]]) + output, self.inter_attr_self_attention_weights['weight_MLP'])

        return output

    def intra_attr_self_attention(self,input):
        encoder_Q = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_attr_self_attention_weights['weight_Q'])  # (batch * 5) * 8
        encoder_K = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_attr_self_attention_weights['weight_K'])  # (batch * 5) * 8
        encoder_V = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_attr_self_attention_weights['weight_V'])  # (batch * 5) * 8

        encoder_Q = tf.reshape(encoder_Q,(tf.shape(input)[0], POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_K = tf.reshape(encoder_K,(tf.shape(input)[0], POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_V = tf.reshape(encoder_V,(tf.shape(input)[0], POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        attention_map = tf.matmul(encoder_Q, tf.transpose(encoder_K, [0, 2, 1]))  # batch * 5 * 5

        attention_map = attention_map / 8
        attention_map = tf.nn.softmax(attention_map)  # batch * 5 * 5

        output = tf.reshape(tf.matmul(attention_map, encoder_V), (-1, POI_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        output = tf.nn.swish(output)

        return output

    def _oam(self,input):

        self.intra_attr_output = tf.identity(input)
        intra_attr_output_each_poi = tf.identity(input)
        for i in range(FEATURE_CATE_NUM):
            self.tmp_intra_attr_input = tf.reshape(tf.gather(input,list(range(i,i+1)),axis=2),[-1,POI_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.tmp_intra_attr_output = self.intra_attr_self_attention(self.tmp_intra_attr_input)
            self.tmp_intra_attr_output_mlp = tf.matmul(self.tmp_intra_attr_input + self.tmp_intra_attr_output, self.intra_attr_self_attention_weights['weight_MLP'])
            self.tmp_intra_attr_output_mlp = tf.nn.swish(self.tmp_intra_attr_output_mlp)
            if i == 0:
                self.intra_attr_output = tf.expand_dims(self.tmp_intra_attr_output_mlp,axis=1)
                intra_attr_output_each_poi = tf.expand_dims(tf.reshape(self.tmp_intra_attr_output,[-1,POI_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]]),axis=2)
            else:
                self.intra_attr_output = tf.concat([self.intra_attr_output,tf.expand_dims(self.tmp_intra_attr_output_mlp, axis=1)],axis=1)
                intra_attr_output_each_poi = tf.concat([intra_attr_output_each_poi,tf.expand_dims(tf.reshape(self.tmp_intra_attr_output, [-1, POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]]), axis=2)],axis=2)

        inter_attr_output = self.inter_attr_self_attention(self.intra_attr_output)

        # inter_attr_output: -1 * 8
        # intra_attr_output_each_poi : -1 * POI_NUM * feanum * CATE_FEATURE_EMBEDDINGS_SHAPE[1]]
        return inter_attr_output,intra_attr_output_each_poi

    def _user_interest_target_attention(self,target,behavior,mask):
        target = tf.tile(tf.expand_dims(target,axis=1),[1,tf.shape(behavior)[1],1])
        paddings = tf.ones_like(mask) * (-2 ** 32 + 1)
        weights = tf.reduce_sum(tf.multiply(target, behavior), axis=2)
        weights = tf.where(tf.equal(mask, 1.0), weights, paddings)
        weights = tf.nn.softmax(weights,axis=1) # batch * 5
        #
        # weights = tf.Print(weights, [weights],'weights:', summarize=100)
        # tf.identity(weights)

        user_interest = tf.reduce_sum(tf.expand_dims(weights,axis=2) * behavior,axis=1)
        return user_interest # batch * 8

    def _intra_poi_self_attention(self,input):
        encoder_Q = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_poi_self_attention_weights['weight_Q'])  # (batch * 5) * 8
        encoder_K = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_poi_self_attention_weights['weight_K'])  # (batch * 5) * 8
        encoder_V = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_poi_self_attention_weights['weight_V'])  # (batch * 5) * 8

        encoder_Q = tf.reshape(encoder_Q,(tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_K = tf.reshape(encoder_K,(tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_V = tf.reshape(encoder_V,(tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        attention_map = tf.matmul(encoder_Q, tf.transpose(encoder_K, [0, 2, 1]))  # batch * 5 * 5

        attention_map = attention_map / 8
        attention_map = tf.nn.softmax(attention_map)  # batch * 5 * 5

        output = tf.reshape(tf.matmul(attention_map, encoder_V),
                            (-1, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        # output = tf.matmul(tf.reshape(input, [-1, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]]) + output,
        #                    self.intra_poi_self_attention_weights['weight_MLP'])

        return output

    def _build_model(self, features, labels, mode, params):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('dnn_model'):
            self.cur_page_oam_output,self.intra_attr_output_each_poi = self._oam(self.cate_feature_embeddings)

            self.behavior_page_oam_output = tf.identity(self.cur_page_oam_output)
            for i in range(PAGE_NUM):
                self.tmp_behavior_input = tf.reshape(tf.gather(self.behavior_cate_feature_embeddings,list(range(i,i+1)),axis=1),[-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.tmp_behavior_oam_output,_ = self._oam(self.tmp_behavior_input)
                if i == 0:
                    self.behavior_page_oam_output = tf.expand_dims(self.tmp_behavior_oam_output,axis=1)
                else:
                    self.behavior_page_oam_output = tf.concat([self.behavior_page_oam_output,tf.expand_dims(self.tmp_behavior_oam_output,axis=1)],axis=1)

            # self.behavior_page_oam_output = tf.Print(self.behavior_page_oam_output, [self.behavior_page_oam_output],
            #                                          'self.behavior_page_oam_output:', summarize=100)
            # tf.identity(self.behavior_page_oam_output)
            self.user_interest_emb = self._user_interest_target_attention(self.cur_page_oam_output,self.behavior_page_oam_output,features['page_mask'])

            self.poi_rep_emb = self.intra_attr_output_each_poi # -1 * poi * 8
            for i in range(POI_NUM):
                self.tmp_poi_input = tf.reshape(tf.gather(self.intra_attr_output_each_poi, list(range(i, i + 1)), axis=1),[-1, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.tmp_poi_output = self._intra_poi_self_attention(self.tmp_poi_input)
                if i == 0:
                    self.poi_rep_emb = tf.expand_dims(self.tmp_poi_output, axis=1)
                else:
                    self.poi_rep_emb = tf.concat([self.poi_rep_emb, tf.expand_dims(self.tmp_poi_output, axis=1)], axis=1)

            self.cur_page_oam_output = tf.tile(tf.expand_dims(self.cur_page_oam_output,axis=1),[1,POI_NUM,1])
            self.user_interest_emb = tf.tile(tf.expand_dims(self.user_interest_emb, axis=1), [1, POI_NUM, 1])

            # self.cur_page_oam_output = tf.Print(self.cur_page_oam_output, [self.cur_page_oam_output],
            #                                     'self.cur_page_oam_output:', summarize=100)
            # tf.identity(self.cur_page_oam_output)
            #
            #
            # self.user_interest_emb = tf.Print(self.user_interest_emb, [self.user_interest_emb],
            #                                     'self.user_interest_emb:', summarize=100)
            # tf.identity(self.user_interest_emb)
            # self.poi_rep_emb = tf.Print(self.poi_rep_emb, [self.poi_rep_emb],
            #                                     'self.poi_rep_emb:', summarize=100)
            # tf.identity(self.poi_rep_emb)

            fc_out = tf.reshape(self.poi_rep_emb,[-1,POI_NUM,FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])  # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "MLP_A" + str(i)
                fc_out = tf.layers.dense(fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i], activation=None,
                                         name=dense_name)
                fc_out = tf.nn.swish(fc_out)

            # fc_out_org = tf.reshape(self.cate_feature_embeddings, [-1, POI_NUM, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])  # Batch_size * POI_NUM * FEAT_NUM
            # for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_C'])):
            #     dense_name = "MLP_C" + str(i)
            #     fc_out_org = tf.layers.dense(fc_out_org, MODEL_PARAMS['INPUT_TENSOR_LAYERS_C'][i], activation=None,name=dense_name)
            #     fc_out_org = tf.nn.swish(fc_out_org)

           # fc_out = tf.concat([fc_out, self.feat_predict], axis=2)

            # fc_input = tf.concat([self.cur_page_oam_output, self.user_interest_emb, fc_out,self.feat_predict], axis=2)
            fc_input = tf.concat([self.cur_page_oam_output, fc_out,self.feat_predict], axis=2)

            fc_out_ctr, fc_out_imp = tf.reshape(fc_input, [-1, MLP_INPUT_DIM]), tf.reshape(fc_input, [-1, MLP_INPUT_DIM])

            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'])):
                dense_name = "MLP_B" + str(i)
                fc_out_ctr = tf.layers.dense(fc_out_ctr, MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'][i], activation=None, name=dense_name)
                fc_out_ctr = tf.nn.swish(fc_out_ctr)

            ctr_out = tf.layers.dense(fc_out_ctr, OUT_NUM, activation=None, name="final_out_ctr")

            if not self.train:
                ctr_out = tf.nn.sigmoid(ctr_out)

            self.out = tf.reshape(ctr_out,[-1,POI_NUM])
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
                                             initializer=tf.zeros_initializer(),
                                            )
            }

            glorot = np.sqrt(2.0 / (4 + 1))

            self.intra_attr_self_attention_weights = {
                'weight_Q':tf.get_variable(name='intra_attr_self_attention_weights_Q',shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1],CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                           initializer=tf.random_normal_initializer(0.0, glorot),
                                           dtype=np.float32),
                'weight_K': tf.get_variable(name='intra_attr_self_attention_weights_K',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_V': tf.get_variable(name='intra_attr_self_attention_weights_V',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_MLP': tf.get_variable(name='intra_attr_self_attention_weights_MLP',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1] * POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32)
            }

            self.inter_attr_self_attention_weights = {
                'weight_Q': tf.get_variable(name='inter_attr_self_attention_weights_Q',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_K': tf.get_variable(name='inter_attr_self_attention_weights_K',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_V': tf.get_variable(name='inter_attr_self_attention_weights_V',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_MLP': tf.get_variable(name='inter_attr_self_attention_weights_MLP',
                                              shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1] * FEATURE_CATE_NUM,
                                                     CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                              initializer=tf.random_normal_initializer(0.0, glorot),
                                              dtype=np.float32)
            }

            self.intra_poi_self_attention_weights = {
                'weight_Q': tf.get_variable(name='intra_poi_self_attention_weights_Q',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_K': tf.get_variable(name='intra_poi_self_attention_weights_K',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_V': tf.get_variable(name='intra_poi_self_attention_weights_V',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_MLP': tf.get_variable(name='intra_poi_self_attention_weights_MLP',
                                              shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1] * FEATURE_CATE_NUM,
                                                     CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                              initializer=tf.random_normal_initializer(0.0, glorot),
                                              dtype=np.float32)
            }



    def _process_features(self, features):
        # env_feature = > dense_feature
        # cxr_feature = > screen_predict_feature
        # cat_feature = > screen_cate_feature
        # dense_feature = > screen_dense_feature

        # N * M * K
        # N * D ( D <= M )
        cate_fea = tf.reshape(features['cate_feature'],[-1,POI_NUM * FEATURE_CATE_NUM])
        self.cate_feature_embeddings = tf.reshape(tf.nn.embedding_lookup(
            self.feature_weights['embedding'], cate_fea),
            [-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        
        # self.input_embedding = tf.concat(
        #     [self.cate_feature_embeddings], axis=2)
        self.feat_predict = features['dense_feature']

        behavior_cate_fea = tf.reshape(features['behavior_cate_feature'],[-1,PAGE_NUM * POI_NUM * FEATURE_CATE_NUM])
        self.behavior_cate_feature_embeddings = tf.reshape(tf.nn.embedding_lookup(
            self.feature_weights['embedding'], behavior_cate_fea),
            [-1, PAGE_NUM , POI_NUM, FEATURE_CATE_NUM ,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])

        self.behavior_input_embedding = tf.concat([self.behavior_cate_feature_embeddings], axis=3)
        self.behavior_feat_predict = features['behavior_dense_feature']

        return features


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
