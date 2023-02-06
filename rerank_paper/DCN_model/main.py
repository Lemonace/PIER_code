# -*- coding: utf-8 -*-
from config import *
from model import *
from sklearn import metrics


def create_estimator():
    tf.logging.set_verbosity(tf.logging.INFO)
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(
        tf_random_seed=RANDOM_SEED,
        save_summary_steps=100,
        save_checkpoints_steps=1000,
        model_dir=MODEL_SAVE_PATH,
        keep_checkpoint_max=2,
        log_step_count_steps=1000,
        session_config=session_config)
    nn_model = DNN()
    estimator = tf.estimator.Estimator(model_fn=nn_model.model_fn_estimator, config=config)
    return estimator, nn_model


def save_model_pb_with_estimator(estimator, params, export_dir_base):
    estimator._params['save_model'] = params['save_model']

    def _serving_input_receiver_fn():
        # env_feature = > dense_feature
        # cxr_feature = > screen_predict_feature
        # cat_feature = > screen_cate_feature
        # dense_feature = > screen_dense_feature
        receiver_tensors = {
            # ctr cvr gmv预估值 && bid
            'screen_predict_feature': tf.placeholder(tf.float32, [None, POI_NUM, FEATURE_CXR_NUM],
                                                     name='screen_predict_feature'),
            # dense 特征 (价格，评分)
            'screen_dense_feature': tf.placeholder(tf.float32, [None, POI_NUM, FEATURE_DENSE_NUM],
                                                   name='screen_dense_feature'),
            # 离散特征(品类)
            'screen_cate_feature': tf.placeholder(tf.int64, [None, POI_NUM, FEATURE_CATE_NUM],
                                                  name='screen_cate_feature'),
            # 环境特征（是否有铂金）
            'dense_feature': tf.placeholder(tf.float32, [None, DENSE_FEAT_NUM],
                                            name='dense_feature')
        }
        return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors, features=receiver_tensors)

    export_dir = estimator.export_saved_model(export_dir_base=export_dir_base,
                                              serving_input_receiver_fn=_serving_input_receiver_fn)
    estimator._params.pop('save_model')
    return export_dir.decode()


def calculate_result(result_generator):
    y_ctr, pred_ctr, ctr = [], [], []
    for result in result_generator:
        cxr_feature = result['cxr_feature']
        mask = result['mask']
        # ctr_label
        idx = np.where(mask.reshape(-1) == 1)
        y_ctr += result['ctr_label'].reshape(-1)[idx].tolist()
        pred_ctr += result['ctr_out'].reshape(-1)[idx].tolist()
        ctr += cxr_feature[:, 0].reshape(-1)[idx].tolist()

    ctr_auc, ctr_auc_jp, ctr_cb, ctr_cb_jp = metrics.roc_auc_score(y_ctr, pred_ctr), metrics.roc_auc_score(y_ctr,
                                                                                                           ctr), np.sum(
        pred_ctr) / np.sum(y_ctr), np.sum(ctr) / np.sum(y_ctr)
    print("ctr_auc:{}, ctr_auc_jp:{}, ctr_cb:{}, ctr_cb_jp:{}".format(ctr_auc, ctr_auc_jp, ctr_cb, ctr_cb_jp))


if __name__ == '__main__':

    estimator, nn_model = create_estimator()

    with tick_tock("DATA_INPUT") as _:
        valid_input_fn = input_fn_maker(VALID_FILE, False, batch_size=1024, epoch=1)
        test_input_fn = input_fn_maker(TEST_FILE, False, batch_size=1024, epoch=1)

    if TRAIN_MODE == 1:
        for i in range(EPOCH):
            for idx, data in enumerate(TRAIN_FILE):
                with tick_tock("DATA_INPUT") as _:
                    train_input_fn = input_fn_maker([data], True, batch_size=BATCH_SIZE, epoch=1)
                with tick_tock("TRAIN") as _:
                    estimator.train(train_input_fn)
                if MODEL_SAVE_PB_EPOCH_ON:
                    export_dir = save_model_pb_with_estimator(estimator, params={'save_model': 'listwise'},
                                                              export_dir_base=MODEL_SAVE_PB_EPOCH_PATH)
                    ep_insert_index = i * len(TRAIN_FILE) + idx
                    target_dir = export_dir + "/../ep" + str(ep_insert_index)
                    while os.path.exists(target_dir):
                        target_dir = export_dir + "/../ep" + str(ep_insert_index)
                    shutil.move(export_dir, target_dir)
                    print(time.strftime("%m-%d %H:%M:%S ",
                                        time.localtime(time.time())) + "export model PB: " + target_dir)
                # with tick_tock("PREDICT") as _:
                # result_generator = estimator.predict(input_fn=valid_input_fn, yield_single_examples=False)
                # calculate_result(result_generator)



    elif TRAIN_MODE == 2:
        with tick_tock("PREDICT") as _:
            result_generator = estimator.predict(input_fn=valid_input_fn, yield_single_examples=False)
            calculate_result(result_generator)

    elif TRAIN_MODE == 3:
        for i in range(EPOCH):
            for idx, data in enumerate(TRAIN_FILE):
                with tick_tock("DATA_INPUT") as _:
                    train_input_fn = input_fn_maker([data], True, batch_size=BATCH_SIZE, epoch=1)
                with tick_tock("TRAIN") as _:
                    estimator.train(train_input_fn)
                with tick_tock("PREDICT") as _:
                    result_generator = estimator.predict(input_fn=valid_input_fn, yield_single_examples=False)
                    print("valid_data")
                    calculate_result(result_generator)
                    # result_generator = estimator.predict(input_fn=test_input_fn, yield_single_examples=False)
                    print("train_data")
                    # calculate_result(result_generator)
                    # save pb


    elif TRAIN_MODE == 4:
        export_dir = save_model_pb_with_estimator(estimator, params={'save_model': 'listwise'},
                                                  export_dir_base=MODEL_SAVE_PB_EPOCH_PATH)
        ep_insert_index = 0
        target_dir = export_dir + "/../ep" + str(ep_insert_index)
        while os.path.exists(target_dir):
            target_dir = export_dir + "/../ep" + str(ep_insert_index)
        shutil.move(export_dir, target_dir)
        print(time.strftime("%m-%d %H:%M:%S ",
                            time.localtime(time.time())) + "export model PB: " + target_dir)

