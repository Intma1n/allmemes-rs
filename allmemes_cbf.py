import numpy as np
import tensorflow as tf
from pprint import pprint

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.reset_default_graph()

print(tf.__version__)

users = ['User1', 'User2', 'User3', 'User4', 'User5', ]
memes = ['Meme1', 'Meme2', 'Meme3', 'Meme4', 'Meme5', ]
tags = ['Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5', ]

num_users = len(users)
num_memes = len(memes)
num_tags = len(tags)

users_memes = [[0, 0, 0, 1, 1],
               [1, 0, 2, 0, 1],
               [1, 1, 0, 1, 0],
               [1, 2, 1, 1, 1],
               [0, 2, 0, 0, 0], ]

memes_feats = [[1, 1, 1, 1, 1],
               [1, 1, 1, 0, 1],
               [1, 0, 0, 1, 1],
               [0, 1, 1, 1, 0],
               [1, 0, 0, 0, 1], ]

users_memes = tf.constant(users_memes, dtype=tf.float32)
memes_feats = tf.constant(memes_feats, dtype=tf.float32)

wgtd_feature_matrices = [tf.expand_dims(tf.transpose(users_memes)[:, i], axis=1) * memes_feats for i in
                         range(num_users)]
pprint(wgtd_feature_matrices)

user_memes_feats = tf.stack(wgtd_feature_matrices)

users_memes_feats_sums = tf.reduce_sum(user_memes_feats, axis=1)
users_memes_feats_totals = tf.reduce_sum(users_memes_feats_sums, axis=1)

users_feats = tf.stack([users_memes_feats_sums[i, :] / users_memes_feats_totals[i] for i in range(num_users)], axis=0)


def find_user_top_feats(user_index):
    feats_ind = tf.nn.top_k(users_feats[user_index], num_tags)[1]
    return tf.gather_nd(tags, tf.expand_dims(feats_ind, axis=1))


with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    users_top_feats = {}
    for i in range(num_users):
        top_feats = sess.run(find_user_top_feats(i))
        users_top_feats[users[i]] = list(top_feats)

pprint(users_top_feats)

users_ratings = [tf.map_fn(lambda x: tf.tensordot(users_feats[i], x, axes=1), memes_feats) for i in range(num_users)]

all_users_ratings = tf.stack(users_ratings)

all_users_ratings_new = tf.where(tf.equal(users_memes, tf.zeros_like(users_memes)),
                                 all_users_ratings,
                                 -np.inf * tf.ones_like(tf.cast(users_memes, tf.float32)))


def find_user_top_memes(user_index, num_to_recommend):
    memes_ind = tf.nn.top_k(all_users_ratings_new[user_index], num_to_recommend)[1]
    return tf.gather_nd(memes, tf.expand_dims(memes_ind, axis=1))


with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    users_top_memes = {}
    num_to_recommend = tf.reduce_sum(tf.cast(tf.equal(users_memes, tf.zeros_like(users_memes)), dtype=tf.float32),
                                     axis=1)
    for ind in range(num_users):
        top_memes = sess.run(find_user_top_memes(ind, tf.cast(num_to_recommend[ind], dtype=tf.int32)))
        users_top_memes[users[ind]] = list(top_memes)

pprint(users_top_memes)
