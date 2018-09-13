import tensorflow as tf
import numpy as np
import logging
import toulouse_dataset
import cv2
model_params = {
    'input_shape': (320, 50, 3),
    'batch_size': 100
}

hyper_params = {
    'learning_rate': 0.01,
    'drop_out': 0.25
}

tf.logging.set_verbosity(tf.logging.INFO)


def conv2d_fn(input_tensor, k_size, n_out):
    return tf.layers.conv2d(inputs= input_tensor, \
                            filters= n_out, \
                            kernel_size= k_size, \
                            activation= tf.nn.relu, \
                            use_bias= True)
def maxpool2d_fn(input_tensor, p_size, strides):
    return tf.layers.max_pooling2d(inputs= input_tensor, pool_size= p_size, strides= strides)
def model_fn(features, labels, mode):
    features_tensor = tf.cast(features, tf.float32, name="input_tensor")
    net = conv2d_fn(features_tensor, 3, 32)
    net = maxpool2d_fn(net, 2, 2)
    net = conv2d_fn(features_tensor, 3, 64)
    net = maxpool2d_fn(net, 2, 2)
    net = tf.layers.flatten(net)
    # net = tf.layers.dense(inputs= features_tensor, units= 512, activation=tf.nn.relu)
    # net = tf.layers.dense(inputs= net, units= 256)
    out_put = tf.layers.dense(inputs= net, units= 2, name="out_put")
    prediction = {
        'coordinate' : tf.cast(out_put, tf.int32)
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = prediction)
    labels = tf.cast(labels, tf.int32)
    loss = tf.losses.mean_squared_error(labels= labels, predictions= out_put)
    tf.summary.scalar('loss', loss)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate= hyper_params['learning_rate'])
        train_op = optimizer.minimize(loss = loss, global_step= tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)
    rmse = tf.metrics.root_mean_squared_error(labels, prediction['coordinate'])

    # Add the rmse to the collection of evaluation metrics.
    eval_metrics = {"rmse": rmse}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        # Report sum of error for compatibility with pre-made estimators
        loss=loss,
        eval_metric_ops=eval_metrics)
def preprocess_data(img_list, width= 320, height=50):
    # image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    res = img_list
    # create a big 1D-array
    
    # for img in img_list:
    #     # img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    #     # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     # mask_white = cv2.inRange(img, 140, 255)
    #     res.append(cv2.resize(img, (int(width/2), int(height/2))))
        
    # res = np.array(res)
    # data_shape = res.shape
    # res = np.reshape(res, [data_shape[0], data_shape[1], data_shape[2], -1])
    # print(res.shape)
    # Normalize
    res = res / 255. # values in [0, 1]
    res -= 0.5 # values in [-0.5, 0.5]
    res *= 2 # values in [-1, 1]
    return res
# x_train, y_train, x_test, y_test = toulouse_dataset.load_toulouse_dataset()
# x_train = preprocess_data(x_train)
# x_test = preprocess_data(x_test)

model_classifier = tf.estimator.Estimator(
    model_fn = model_fn, \
    model_dir= 'CheckPoint2')
# print(model_classifier)
# train_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x = x_train,
#     y = y_train,
#     num_epochs= None,
#     batch_size= model_params['batch_size'],
#     shuffle= True)
# model_classifier.train(
#     input_fn = train_input_fn,\
#     steps= 2000)

# eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x = x_train,
#     y = y_train,
#     num_epochs= 1,
#     shuffle= False)
# eval_result = model_classifier.evaluate(input_fn = eval_input_fn)
# print(eval_result)
def serving_input_receiver_fn():
    inputs = tf.placeholder(dtype = tf.float32, shape=[None, 50, 320, 3])
    return tf.estimator.export.TensorServingInputReceiver(inputs, inputs)

model_classifier.export_savedmodel(export_dir_base="model", serving_input_receiver_fn= serving_input_receiver_fn)
