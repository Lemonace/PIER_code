import tensorflow

from config import *
import tensorflow_core.contrib.predictor as predictor

def load_listwise_model():
    model_filename_dir = MODEL_SAVE_PATH + "_pbs/ep0/"
    predict_fn = predictor.from_saved_model(model_filename_dir)
    # predict_fn = tf.compat.v2.saved_model.load(model_filename_dir)

    # env_feature = > dense_feature
    # cxr_feature = > screen_predict_feature
    # cat_feature = > screen_cate_feature
    # dense_feature = > screen_dense_feature
    predictions = predict_fn({
        'screen_predict_feature': [[[0.036115, 0.05427262, 0.09489095, 0.2],
                                    [0.027565, 0.07474336, 0.04988268, 0.53],
                                    [0.024815, 0.1775544, 0.12052802, 0.24],
                                    [0.023316, 0.12283709, 0.10298113, 0.1]]],
        # dense 特征 (价格，评分)
        'screen_dense_feature': [[[1359., 30.146147, 26., 5., 4.85],
                                  [318., 14.675659, 0., 5., 4.94],
                                  [637., 24.784016, 0., 5., 4.65],
                                  [185., 25.333273, 0., 5., 4.75]]],
        # 离散特征(品类)
        'screen_cate_feature': [[[2638824, 4148885, 432243, 3985407, 3385100, 3019284],
                                 [2638824, 3905410, 3212599, 3985407, 1997821, 3019284],
                                 [2638824, 4148885, 3622545, 3985407, 1997821, 3019284],
                                 [2638824, 4148885, 432243, 3985407, 1997821, 3019284]]],
        # 环境特征（是否有铂金）
        'dense_feature': [[0., 0.]]
    })

    print('Q_network_output:', predictions['Q_network_output'])
    print('out:', predictions['out'])

if __name__ == '__main__':
    # load_pg_model()
    load_listwise_model()