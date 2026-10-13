
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Create a small movie dataset
movies = {
    "title": [
        "The Dark Knight",
        "Batman Begins",
        "Iron Man",
        "The Avengers",
        "Spider-Man",
        "Interstellar",
        "Inception",
        "The Matrix",
        "Titanic",
        "The Notebook",
        "Toy Story",
        "Finding Nemo",
        "Jurassic Park",
        "The Lion King",
        "John Wick"
    ],

    "genres": [
        "action crime drama superhero",
        "action crime drama superhero",
        "action science fiction superhero",
        "action science fiction superhero",
        "action adventure superhero",
        "science fiction space drama",
        "science fiction action thriller",
        "science fiction action thriller",
        "romance drama disaster",
        "romance drama",
        "animation comedy family adventure",
        "animation family adventure",
        "science fiction adventure thriller",
        "animation drama adventure family",
        "action crime thriller"
    ],

    "description": [
        "A hero fights crime and protects Gotham City.",
        "A young Bruce Wayne becomes a masked crime fighter.",
        "A genius inventor builds a powerful armored suit.",
        "Superheroes unite to protect Earth from a dangerous enemy.",
        "A young hero uses special powers to fight criminals.",
        "Astronauts travel through space searching for a new home.",
        "A skilled thief enters dreams to steal secrets.",
        "A hacker discovers a simulated world and fights machines.",
        "Two strangers fall in love aboard a famous ocean liner.",
        "Two people fall in love despite family and life challenges.",
        "Toys come alive and experience exciting adventures.",
        "A father travels across the ocean to find his son.",
        "Scientists bring dinosaurs back to life in a theme park.",
        "A young lion learns responsibility and becomes a king.",
        "A skilled assassin seeks revenge against dangerous criminals."
    ]
}

df = pd.DataFrame(movies)

# Step 2: Combine genres and descriptions
df["features"] = df["genres"] + " " + df["description"]

# Step 3: Convert text into numerical vectors
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["features"])

# Step 4: Calculate similarity between every pair of movies
similarity_matrix = cosine_similarity(tfidf_matrix)

# Step 5: Create the recommendation function
def recommend_movies(movie_name, number=5):

    # Find the movie entered by the user
    matching_movies = df[
        df["title"].str.lower() == movie_name.strip().lower()
    ]

    if matching_movies.empty:
        print("Movie not found in the database.")
        print("Available movies:")
        for title in df["title"]:
            print("-", title)
        return

    # Get the position of the selected movie
    movie_index = matching_movies.index[0]

    # Get similarity scores for this movie
    scores = list(enumerate(similarity_matrix[movie_index]))

    # Sort from highest similarity to lowest
    scores = sorted(
        scores,
        key=lambda item: item[1],
        reverse=True
    )

    # Exclude the selected movie itself
    scores = [
        item for item in scores
        if item[0] != movie_index
    ]

    # Display the top recommendations
    print("\nMovies similar to:", df.iloc[movie_index]["title"])
    print("-" * 45)

    for index, score in scores[:number]:
        print(
            df.iloc[index]["title"],
            "| Similarity:",
            round(score * 100, 2),
            "%"
        )


# Step 6: Let the user try the system
print("===== MOVIE RECOMMENDATION SYSTEM =====")
print("\nAvailable movies:")

for title in df["title"]:
    print("-", title)

movie_name = input("\nEnter a movie you like: ")

recommend_movies(movie_name)
