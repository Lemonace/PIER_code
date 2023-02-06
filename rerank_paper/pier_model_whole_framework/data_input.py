# -*- coding: utf-8 -*-
import tensorflow as tf
from config import *
import numpy as np
from tools import tick_tock,allPermutation


def generate_parse_tfrecord_local_fn():
    def _parse_function(batch_examples):
        common_features, sequence_features = feature_parse_scheme()
        parsed_features = tf.parse_example(
            serialized=batch_examples,
            features=common_features
        )
        features = feature_product(parsed_features)
        labels = label_product(parsed_features)
        return features, labels

    return _parse_function


def generate_parse_valid_tfrecord_local_fn():
    def _parse_function(batch_examples):
        common_features, sequence_features = feature_parse_scheme()
        parsed_features = tf.parse_example(
            serialized=batch_examples,
            features=common_features
        )
        features = feature_product(parsed_features)
        labels = label_product(parsed_features)
        return features, labels

    return _parse_function


def feature_parse_scheme():
    label_len = POI_NUM * 2 + PAGE_NUM
    feature_len = POI_NUM * FEATURE_NUM + PAGE_NUM * POI_NUM * FEATURE_NUM_FOR_PAGE
    common_features = {
        "label": tf.FixedLenFeature([label_len], dtype=tf.float32),
        "feature": tf.FixedLenFeature([feature_len], dtype=tf.float32),
    }

    sequence_features = {}
    return common_features, sequence_features


def label_product(parsed_features):
    labels = parsed_features['label']

    labels_result = {
        # ctr_label
        'ctr_label': tf.gather(labels, list(range(0, POI_NUM)), axis=1),
        'mask': tf.gather(labels, list(range(POI_NUM, 2 * POI_NUM)), axis=1),
        'page_mask': tf.gather(labels, list(range(2 * POI_NUM, 2 * POI_NUM + PAGE_NUM)), axis=1),
    }
    return labels_result


def feature_product(parsed_features):
    feature_buffer = parsed_features['feature']
    labels = parsed_features['label']
    # 获取特征
    # FEATURE_CATE_NUM：品类相关特征
    # FEATURE_DENSE_NUM：连续值特征
    # FEATURE_CXR_NUM：模型预估值特征

    full_permuation_index = allPermutation(POI_NUM)

    # current page
    current_page_start = 0
    current_page_end = current_page_start + POI_NUM * FEATURE_NUM

    pre_page_start = current_page_end
    pre_page_end = pre_page_start + PAGE_NUM * POI_NUM * FEATURE_NUM_FOR_PAGE

    
    cur_page_features = tf.reshape(tf.gather(feature_buffer,list(range(current_page_start,current_page_end)),axis=1), [-1, POI_NUM, FEATURE_NUM])
    pre_page_features = tf.reshape(tf.gather(feature_buffer,list(range(pre_page_start,pre_page_end)),axis=1), [-1, PAGE_NUM, POI_NUM, FEATURE_NUM_FOR_PAGE])
    
    position_fea = tf.gather(cur_page_features, list(range(0, 1)), axis=2)
    adid_fea = tf.gather(cur_page_features, list(range(1, 2)), axis=2)
    obj_type_fea = tf.gather(cur_page_features, list(range(2, 3)), axis=2)
    hist_ctr_fea = tf.gather(cur_page_features, list(range(3, 4)), axis=2)
    locationid_fea = tf.gather(cur_page_features, list(range(4, 5)), axis=2)
    categoryid_fea = tf.gather(cur_page_features, list(range(5, 6)), axis=2)
    price_fea =  tf.gather(cur_page_features, list(range(6, 7)), axis=2)
    iscontext_fea = tf.gather(cur_page_features, list(range(7, 8)), axis=2)
    userid_fea = tf.gather(cur_page_features, list(range(8, 9)), axis=2)

    pre_position_fea = tf.gather(pre_page_features, list(range(0, 1)), axis=3)
    pre_adid_fea = tf.gather(pre_page_features, list(range(1, 2)), axis=3)
    pre_obj_type_fea = tf.gather(pre_page_features, list(range(2, 3)), axis=3)
    pre_hist_ctr_fea = tf.gather(pre_page_features, list(range(3, 4)), axis=3)
    pre_locationid_fea = tf.gather(pre_page_features, list(range(4, 5)), axis=3)
    pre_categoryid_fea = tf.gather(pre_page_features, list(range(5, 6)), axis=3)
    pre_price_fea = tf.gather(pre_page_features, list(range(6, 7)), axis=3)
    pre_iscontext_fea = tf.gather(pre_page_features, list(range(7, 8)), axis=3)
    pre_userid_fea = tf.gather(pre_page_features, list(range(8, 9)), axis=3)

    random_indices = np.random.randint(0,PERMUATION_SIZE,size=[1,TOP_K],dtype=np.int32)

    features_result = {
        
        'dense_feature': hist_ctr_fea,
        'permuation_mask' : tf.gather(labels, list(range(POI_NUM, 2 * POI_NUM)), axis=1),
        # 离散特征(品类)
        'cate_feature': tf.cast(tf.concat([position_fea, adid_fea, obj_type_fea, locationid_fea, iscontext_fea, categoryid_fea, userid_fea], axis=2), tf.int64),
        # ctr_label
        'ctr_label': tf.gather(labels, list(range(0, POI_NUM)), axis=1),
        'mask': tf.gather(labels, list(range(POI_NUM, 2 * POI_NUM)), axis=1),
        'page_mask':tf.gather(labels, list(range(2 * POI_NUM, 2 * POI_NUM + PAGE_NUM)), axis=1),
        'behavior_dense_feature': pre_hist_ctr_fea,
        # 离散特征(品类)
        'behavior_cate_feature': tf.cast(tf.concat([pre_position_fea, pre_adid_fea, pre_obj_type_fea, pre_locationid_fea,
                                                    pre_iscontext_fea, pre_categoryid_fea, pre_userid_fea],axis=3), tf.int64),

        'hash_vector' : tf.constant(HASH_VECTOR,tf.float32),
        'position_encoding': tf.constant(POSITION_ENCODING, tf.float32),
        'disantce_weight' : tf.constant(TIME_AWARE_WEIGHT,tf.float32),
        'random_indices' : tf.constant(random_indices,tf.int32)
    }
    features_result['full_permuation_index'] = tf.constant(full_permuation_index,tf.int32)
    return features_result


# num_parallel 表示cpu的核数,用于控制 map的并行度
def input_fn_maker(file_names, is_train, batch_size, epoch=None, num_parallel=4):
    def input_fn():
        _parse_fn = generate_parse_tfrecord_local_fn() if is_train else generate_parse_valid_tfrecord_local_fn()
        files = tf.data.Dataset.list_files(file_names)
        # print(files)
        dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4 * 10))
        dataset = dataset.prefetch(buffer_size=batch_size * 10)
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(_parse_fn, num_parallel_calls=num_parallel)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    return input_fn


# 从hive表统计得到均值和方差文件
def get_normalization_parameter(mean_var_path):
    with tf.gfile.Open(mean_var_path) as f:
        fea_mean = f.readline().strip().split(' ')
        fea_var = f.readline().strip().split(' ')
        cont_fea_mean = list(map(float, fea_mean))
        cont_fea_var = list(map(float, fea_var))
    f.close()
    return cont_fea_mean, cont_fea_var


def get_bias_weight_parameter(bias_weight_path):
    with tf.gfile.Open(bias_weight_path) as f2:
        fea_mean = f2.readline().strip().split('\t')
        cont_fea_mean = list(map(float, fea_mean))
    f2.close()
    return cont_fea_mean



if __name__ == '__main__':
    train_file = TRAIN_FILE
    # train_file = ["/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yangfan129/train_data/avito_v1_new/avito_v1_new/train_data/part-r-00000"]
    train_input_fn = input_fn_maker(train_file, is_train=True, batch_size=16, epoch=1)
    features, labels = train_input_fn()

    sess = tf.Session()
    try:
        with tick_tock("DATA_INPUT") as _:
            features_np, labels_np = sess.run([features, labels])

        print("*" * 100, "features_np")
        for key in features_np:
            print("=" * 50, key, np.shape(features_np[key]))
            print(features_np[key])


        print("*" * 100, "labels_np")
        for key in labels_np:
            print("=" * 50, key, np.shape(labels_np[key]))
            print(labels_np[key])

    except Exception as e:
        print(e)
