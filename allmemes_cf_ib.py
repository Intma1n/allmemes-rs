import pandas as pd
import numpy as np
from random import randint
from sklearn.metrics.pairwise import cosine_similarity

user_ratings = pd.read_csv('user_ratings.csv')
user_ratings_df = pd.DataFrame(user_ratings)

user_ratings_pivot = user_ratings_df.pivot(index='userId',
                                           columns='movieId',
                                           values='rating')
print('================user_ratings_pivot===================')
print(user_ratings_pivot)

avg_ratings = user_ratings_pivot.mean(axis=1)
user_ratings_pivot = user_ratings_pivot.sub(avg_ratings, axis=0)

print('===================user_ratings_pivot====================')
print(user_ratings_pivot)

user_ratings_pivot = user_ratings_pivot.fillna(0)

print('================user_ratings_pivot=====================')
print(user_ratings_pivot)

movie_ratings_pivot = user_ratings_pivot.T
print('================movie_ratings_pivot=================')
print(movie_ratings_pivot)

similarities = cosine_similarity(movie_ratings_pivot)
cosine_similarity_df = pd.DataFrame(movie_ratings_pivot,
                                    index=movie_ratings_pivot.index,
                                    columns=movie_ratings_pivot.index)

print('================cosine_similarity_df===================')
print(cosine_similarity_df)

cosine_similarity_series = cosine_similarity_df.loc[1]
ordered_similarities = cosine_similarity_series.sort_values(ascending=False)

print('==============ordered_similarities.head()=================')
print(ordered_similarities.head())
