import numpy as np
import tensorflow as tf
from pprint import pprint
from random import randint

users = []
memes = []
tags = []


def fill_objs(objs_arr, obj_name, your_range):
    for i in range(your_range):
        objs_arr.append(f'{obj_name}{i}')
    return objs_arr


fill_objs(users, 'User', 100)
fill_objs(tags, 'Tag', 100)
fill_objs(memes, 'Meme', 100)

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.reset_default_graph()

print(tf.__version__)

users_memes = np.random.randint(3, size=(len(users), len(memes)))
memes_tags = np.random.randint(2, size=(len(memes), len(tags)))

num_users = len(users)
num_tags = len(tags)

users_memes = tf.constant(users_memes, dtype=tf.float32)
memes_feats = tf.constant(memes_tags, dtype=tf.float32)

print('===============users_memes=======================')
print(users_memes)
print('===============memes_feats=======================')
print(memes_feats)

wgtd_feature_matrices = [tf.expand_dims(tf.transpose(users_memes)[:, i], axis=1) * memes_feats for i in
                         range(num_users)]
# print(wgtd_feature_matrices)

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

# print(users_top_feats)

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
    number_to_recommend = tf.reduce_sum(tf.cast(tf.equal(users_memes, tf.zeros_like(users_memes)), dtype=tf.float32),
                                        axis=1)
    for ind in range(num_users):
        top_memes = sess.run(find_user_top_memes(ind, tf.cast(number_to_recommend[ind], dtype=tf.int32)))
        users_top_memes[users[ind]] = list(top_memes)

for i in range(len(users) - 1):
    print(f'User{i}', users_top_memes[f'User{i}'])
    print('----------------------------------')
