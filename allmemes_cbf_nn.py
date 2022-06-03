import os
import tensorflow as tf
import numpy as np
import shutil

tags_list = open('tags').read().splitlines()
meme_ids_list = open('meme_ids').read().splitlines()

meme_id_column = tf.feature_column.categorical_column_with_hash_bucket(key='meme_id',
                                                                       hash_bucket_size=len(meme_ids_list) + 1)

tag_column_categorical = tf.feature_column.categorical_column_with_vocabulary_list(key='tag',
                                                                                   vocabulary_list=tags_list,
                                                                                   num_oov_buckets=1)

tag_column = tf.feature_column.indicator_column(tag_column_categorical)

feature_columns = [
    meme_id_column,
    tag_column,
]

recorde_defaults = []
column_keys = []
label_key = []


def read_dataset(filename, mode, batch_size=512):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.io.decode_csv(value_column)
            features = dict(zip(column_keys, columns))
            label = features.pop(label_key)
            return features, label

        file_list = tf.io.gfile.glob(filename)

        dataset = tf.data.TextLineDataset(file_list).map(decode_csv)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None
            dataset = dataset.shuffle(buffer_size=10 * batch_size)
        else:
            num_epochs = 1
        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()

    return _input_fn


def model_fn(features, labels, mode, params):
    net = tf.compat.v1.feature_column.input_layer(features, params['feature_columns'])

    for units in params['hidden_units']:
        net = tf.keras.layers.dense(net, units=units, activation=tf.nn.relu)

    logits = tf.keras.layers.dense(net, params['n_classes'], activation=None)
    predicted_classes = tf.argmax(logits, 1)

    from tensorflow.python.lib.io import file_io

    with file_io.FileIO('meme_ids', mode='r') as ifp:
        content = tf.constant([x.rstrip() for x in ifp])
    predicted_class_names = tf.gather(content, predicted_classes)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'class_names': predicted_class_names[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    table = tf.contrib.lookup.index_table_from_file(vocabulary_file='meme_ids')
    labels = table.lookup(labels)

    loss = tf.losses.sparse.softmax_cross_entropy(labels=labels, logits=logits)

    accuracy = tf.metrics.accuracy(labels=labels,
                                   prediction=predicted_classes,
                                   name='acc_op')
    top_10_accuracy = tf.metrics.mean(tf.nn.in_top_k(predictions=logits, targets=labels, k=10))

    metrics = {
        'accuracy': accuracy,
        'top_10_accuracy': top_10_accuracy,
    }

    tf.summary.scalar('accuracy', accuracy[1])
    tf.summary.scalar('top_10-accuracy', top_10_accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics
        )

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
