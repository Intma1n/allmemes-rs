from sklearn.metrics.pairwise import cosine_similarity

# Assign the arrays to variables
sw_IV = movie_ratings_centered.loc['Star Wars: Episode IV - A New Hope (1977)', :].values.reshape(1, -1)
sw_V = movie_ratings_centered.loc['Star Wars: Episode V - The Empire Strikes Back (1980)', :].values.reshape(1, -1)

# Find the similarity between two Star Wars movies
similarity_A = cosine_similarity(sw_IV, sw_V)

# Assign the arrays to variables
jurassic_park = movie_ratings_centered.loc['Jurassic Park (1993)', :].values.reshape(1, -1)
pulp_fiction = movie_ratings_centered.loc['Pulp Fiction (1994)', :].values.reshape(1, -1)

# Find the similarity between Pulp Fiction and Jurassic Park
similarity_B = cosine_similarity(jurassic_park, pulp_fiction)
print(similarity_B)