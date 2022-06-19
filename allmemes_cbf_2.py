import pandas as pd
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import pdist, squareform

movie_genre = pd.read_csv('movies.csv')
movie_genre_df = pd.DataFrame(movie_genre)

genres_array_df = pd.crosstab(movie_genre_df['movieId'], movie_genre_df['genres'])

print('===========================================================')
print(genres_array_df)

jaccard_distances = pdist(movie_genre_df.values, metric='jaccard')

print('===========================================================')
print(jaccard_distances)

square_jaccard_distances = squareform(jaccard_distances)

print('===========================================================')
print(square_jaccard_distances)

jaccard_similarity_array = 1 - square_jaccard_distances

print('===========================================================')
print(jaccard_similarity_array)

distance_df = pd.DataFrame(jaccard_similarity_array,
                           index=genres_array_df['title'],
                           columns=genres_array_df['title'])

print('===========================================================')
print(distance_df.head())

print(distance_df['Toy Story (1995)'].sort_values(ascending=False))
