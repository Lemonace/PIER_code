# coding: utf-8 -*-
import shutil
import time
import os

# result
RANDOM_SEED = 2021
BATCH_SIZE = 1024
IMP_LOSS_WEIGHT = 0.02
# basic config
EPOCH = 1
LEARNING_RATE = 0.005
DATA_MODE = 3 # 1:local train，2:local test, 3:docker evaluate
TRAIN_MODE = 3
MODEL_NAME = "extr_model_v3"
# poi类别特征
FEATURE_CATE_NUM = 7 # v1r3:19
# dense特征
FEATURE_DENSE_NUM = 5  # v1:28 v1r2:79 v1r3:83
# 预估值特征
FEATURE_CXR_NUM = 3
# 环境特征
FEATURE_ENV_NUM = 2
# 自然poi
FEATURE_NATURE_POI = 3

# N: Cut Number of POI For Train
POI_NUM = 5
FEATURE_NUM = 9
PAGE_NUM = 5
FEATURE_NUM_FOR_PAGE = 11
# 属性特征：KA AOR BRAND
FEATURE_ATTR_NUM = 3

# DELIVERY_FEAT
DELIVERY_FEAT_NUM = 4

# OUT NUM
OUT_NUM = 5

PLACE_HOLDER_NUM = 11
DENSE_FEAT_NUM = 439


# embedding_look_up维度
CATE_FEATURE_EMBEDDINGS_SHAPE = [1 << 22, 8]

# 网络结构参数
MODEL_PARAMS = {
    'INPUT_TENSOR_LAYERS_A': [60, 32, 20],
    'INPUT_TENSOR_LAYERS_B': [128, 32],
    'INPUT_TENSOR_LAYERS_C': [50, 20],
    'INPUT_TENSOR_LAYERS_D': [50, 20],
    'INPUT_TENSOR_LAYERS_E': [50, 20]
}
A_INPUT_DIM = (MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1] + 1)

DIN_CONF = {}

# train data
# /users/lemonace/Downloads/tfrecord-rl-limit5-v1
if DATA_MODE == 1:
    # TRAIN_FILE = ['/users/meituan_sxw/Downloads/part-r-00046']
    TRAIN_FILE = ['/Users/lemonace/Downloads/docker_data/part-r-00049']
    VALID_FILE = TRAIN_FILE
    PREDICT_FILE = VALID_FILE
    TEST_FILE = PREDICT_FILE
elif DATA_MODE == 2:
    TRAIN_FILE = ['/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yangfan129/train_data/avito_v1_new/avito_v1_new/test_data/part-r-*']
    VALID_FILE = TRAIN_FILE
    TEST_FILE = VALID_FILE
elif DATA_MODE == 3:
    TRAIN_FILE = [
        "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yangfan129/train_data/avito_v3_new/train_data/part-r-*"]
    VALID_FILE = [
        "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yangfan129/train_data/avito_v3_new/test_data/part-r-*"]
    TEST_FILE = [
        "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yangfan129/train_data/avito_v3_new/test_data/part-r-*"]
elif DATA_MODE == 4:
    DATA_FILE = "/home/hadoop-hmart-waimaiad/cephfs/data/yangfan129/train_data/tfrecord-multi-channel-v1/"
    #TRAIN_LIST = ["20211222", "20211223", "20211224", "20211225"]
    TRAIN_LIST = ["20220123"]
    VALID_LIST = ["20220124"]
    TRAIN_FILE = [DATA_FILE + x + "/part-r-*" for x in TRAIN_LIST]
    VALID_FILE = [DATA_FILE + x + "/part-r-0001*" for x in VALID_LIST]
    TEST_FILE = [DATA_FILE + x + "/part-r-00011" for x in TRAIN_LIST]

# 辅助脚本
MEAN_VAR_PATH_POI = "./avg_std/poi"
MEAN_VAR_PATH_DELIVERY = "./avg_std/delivery"
MODEL_SAVE_PATH = "../model/" + MODEL_NAME
MODEL_SAVE_PB_EPOCH_ON = False
MODEL_SAVE_PB_EPOCH_PATH = MODEL_SAVE_PATH + "_pbs"
