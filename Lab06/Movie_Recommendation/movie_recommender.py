import gradio as gr
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import os

warnings.filterwarnings('ignore')

print("Loading data and training K-Means clustering model. Please wait...")

# Load Data
try:
    df = pd.read_csv('movie_metadata.csv')
except FileNotFoundError:
    df = pd.read_csv('https://raw.githubusercontent.com/tisage/CISC339/refs/heads/main/Lab06/movie_metadata.csv')

# 1. Feature Selection
numeric_features = [
    'duration', 'budget', 'gross', 'imdb_score', 'num_voted_users',
    'num_user_for_reviews', 'director_facebook_likes',
    'actor_1_facebook_likes', 'movie_facebook_likes'
]
numeric_features = [col for col in numeric_features if col in df.columns]
display_cols = ['movie_title', 'director_name', 'genres', 'imdb_score', 'title_year']
display_cols = [col for col in display_cols if col in df.columns]

all_cols = numeric_features + [col for col in display_cols if col not in numeric_features]
df_clean = df[all_cols].dropna()

# Outlier Removal
if 'budget' in df_clean.columns:
    df_clean = df_clean[df_clean['budget'] < 400_000_000]
if 'gross' in df_clean.columns:
    df_clean = df_clean[df_clean['gross'] < 800_000_000]
if 'duration' in df_clean.columns:
    df_clean = df_clean[(df_clean['duration'] > 40) & (df_clean['duration'] < 250)]

if 'movie_title' in df_clean.columns:
    df_clean['movie_title'] = df_clean['movie_title'].str.strip()

df_clean = df_clean.reset_index(drop=True)

# 2. Categorical Encoding
# Genres
all_genres = set()
for genres_str in df_clean['genres'].dropna():
    genres_list = str(genres_str).split('|')
    all_genres.update([g.strip() for g in genres_list])

for genre in sorted(all_genres):
    df_clean[f'genre_{genre}'] = df_clean['genres'].apply(
        lambda x: 1 if genre in str(x) else 0
    )
genre_features = [f'genre_{g}' for g in sorted(all_genres)]

# Director
if 'director_name' in df_clean.columns:
    director_encoder = LabelEncoder()
    df_clean['director_encoded'] = director_encoder.fit_transform(df_clean['director_name'].astype(str))
    director_features = ['director_encoded']
else:
    director_features = []

# Combine features for clustering
all_feature_cols = numeric_features + genre_features + director_features
X_combined = df_clean[all_feature_cols].values

# 3. Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# 4. Train K-Means (Optimal K=8 pre-determined from Silhouette Analysis)
best_k = 8
optimal_kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
df_clean['cluster'] = optimal_kmeans.fit_predict(X_scaled)

print("Model training complete! Launching Gradio UI...")

# Prepare List of Movie Titles for Dropdown
movie_list = sorted(df_clean['movie_title'].unique().tolist())

def recommend_movies(movie_title, num_recommendations):
    if not movie_title:
        return "Please select a movie from the dropdown."
        
    # Find movie
    matches = df_clean[df_clean['movie_title'] == movie_title]
    if matches.empty:
        return f"Error: '{movie_title}' not found in database."
        
    target_movie = matches.iloc[0]
    target_idx = df_clean.index.get_loc(target_movie.name)
    target_cluster = target_movie['cluster']
    
    # Calculate similarity within the same cluster
    same_cluster = df_clean[df_clean['cluster'] == target_cluster].copy()
    target_features = X_scaled[target_idx].reshape(1, -1)
    
    cluster_positions = [df_clean.index.get_loc(idx) for idx in same_cluster.index]
    cluster_features = X_scaled[cluster_positions]
    
    distances = np.linalg.norm(cluster_features - target_features, axis=1)
    same_cluster['distance'] = distances
    
    # Get top N recommendations
    recommendations = same_cluster[same_cluster.index != target_movie.name].nsmallest(int(num_recommendations), 'distance')
    
    # Format Output
    output = f"### 🎬 You Selected: **{target_movie['movie_title']}**\n"
    output += f"- **Genres**: {target_movie['genres']}\n"
    output += f"- **Rating**: {target_movie['imdb_score']:.1f}/10\n"
    output += f"- **Year**: {int(target_movie['title_year'])}\n\n"
    output += f"--- \n\n"
    output += f"### 💡 Recommended Movies (Cluster {target_cluster}):\n"
    
    for idx, (_, movie) in enumerate(recommendations.iterrows(), 1):
        similarity = 1 / (1 + movie['distance'])
        score = f"{movie['imdb_score']:.1f}★" if pd.notnull(movie['imdb_score']) else "N/A"
        year = int(movie['title_year']) if pd.notnull(movie['title_year']) else 'Unknown'
        
        output += f"**{idx}. {movie['movie_title']}** ({year})\n"
        output += f"- **Rating**: {score}\n"
        output += f"- **Genres**: {movie['genres']}\n"
        output += f"- **Similarity**: {similarity*100:.1f}%\n\n"
        
    return output

# Build Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🍿 AI Movie Recommendation System")
    gr.Markdown("This app uses **K-Means Clustering** (Unsupervised Learning) to group movies by similarity in features (Budget, Rating, Genres, Director, etc.). Select a movie to see what the AI recommends from the same cluster!")
    
    with gr.Row():
        with gr.Column(scale=1):
            movie_dropdown = gr.Dropdown(choices=movie_list, label="Select a Movie Title", value="Avatar")
            rec_count = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Recommendations")
            btn = gr.Button("Get Recommendations", variant="primary")
            
        with gr.Column(scale=2):
            output_box = gr.Markdown(label="Recommendations")
            
    btn.click(fn=recommend_movies, inputs=[movie_dropdown, rec_count], outputs=output_box)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", inbrowser=True)
