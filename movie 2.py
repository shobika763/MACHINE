import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie data (Movie Title and ID)
movies = pd.DataFrame({
    'movieId': [1, 2, 3, 4, 5, 6],
    'title': ['joe', 'Jumanji', 'Grumpier Old Men', 'Waiting to Exhale', 'Father of the Bride', 'Twelve Monkeys']
})

# Sample user ratings data (User ID, Movie ID, and Rating)
ratings = pd.DataFrame({
    'userId': [1, 1, 1, 2, 2, 3, 3],
    'movieId': [1, 2, 3, 2, 3, 4, 5],
    'rating': [5, 4, 3, 5, 2, 4, 4]
})

# Merge movies with ratings
movie_ratings = pd.merge(ratings, movies, on='movieId')

# Create a pivot table for users and movies
user_movie_ratings = movie_ratings.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# Calculate similarity between movies using cosine similarity
cosine_sim = cosine_similarity(user_movie_ratings.T)  # Transpose to get movies as rows
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_movie_ratings.columns, columns=user_movie_ratings.columns)

def recommend_movies(movie_title, cosine_sim_df, top_n=3):
    if movie_title not in cosine_sim_df:
        print(f"Movie '{movie_title}' not found in the dataset!")
        return []
    
    sim_scores = cosine_sim_df[movie_title]
    sim_scores = sim_scores.sort_values(ascending=False)
    
    # Return top_n most similar movies, excluding the movie itself
    similar_movies = sim_scores.index[1:top_n+1]
    return similar_movies.tolist()

# User input for movie recommendation
movie_to_recommend = input("Enter the movie title to get recommendations: ")

# Get recommended movies
recommended_movies = recommend_movies(movie_to_recommend, cosine_sim_df, top_n=3)

if recommended_movies:
    print(f"\nMovies similar to '{movie_to_recommend}':")
    for idx, movie in enumerate(recommended_movies, 1):
        print(f"{idx}. {movie}")
else:
    print("No recommendations available.")