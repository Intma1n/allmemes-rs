import pandas as pd
import numpy as np
from random import randint

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsRegressor

user_ratings = pd.read_csv('user_ratings.csv')
user_ratings_df = pd.DataFrame(user_ratings)
print(user_ratings_df)

user_ratings_pivot = user_ratings_df.pivot(index='userId',
                                           columns='movieId',
                                           values='rating')
print('===========================================================')
print(user_ratings_pivot)

avg_ratings = user_ratings_pivot.mean(axis=1)
user_ratings_pivot = user_ratings_pivot.sub(avg_ratings, axis=0)
user_ratings_pivot = user_ratings_pivot.fillna(0)

similarities = cosine_similarity(user_ratings_pivot)
cosine_similarity_df = pd.DataFrame(user_ratings_pivot,
                                    index=user_ratings_pivot.index,
                                    columns=user_ratings_pivot.index)
print('=================cosine_similarity_df=======================')
print(cosine_similarity_df)

user_similarity_series = cosine_similarity_df.loc[1]
ordered_similarities = user_similarity_series.sort_values(ascending=False)
nearest_neighbors = ordered_similarities[1:4].index
print('=================nearest_neighbors========================')
print(nearest_neighbors)

user_ratings_table = user_ratings_df.pivot(index='userId',
                                           columns='movieId',
                                           values='rating')

neighbors_rating = user_ratings_table.reindex(nearest_neighbors)
print('===============neighbors_rating[1].mean()======================')
print(neighbors_rating[1].mean())

user_ratings_pivot.drop(1, axis=1, inplace=True)

target_user_x = user_ratings_pivot.loc[[1]]
print('===============target_user_x====================')
print(target_user_x)

other_users_y = user_ratings_table[1]
print('================other_users_y========================')
print(other_users_y)

other_users_x = user_ratings_pivot[other_users_y.notnull()]
print('===============other_users_x=======================')
print(other_users_x)

other_users_y.dropna(inplace=True)
print('================other_users_y========================')
print(other_users_y)

user_knn = KNeighborsRegressor(metric='cosine', n_neighbors=5)
user_knn.fit(other_users_x, other_users_y)
user_user_pred = user_knn.predict(target_user_x)

print('=================user_user_pred=========================')
print(user_user_pred)
