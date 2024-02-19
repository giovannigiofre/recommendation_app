import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from difflib import get_close_matches

# Load and preprocess the dataset
def load_data(filepath):
    """
    Load the Spotify songs dataset.
    """
    data = pd.read_csv(filepath)
    # Add any necessary preprocessing steps here
    return data

# Feature selection based on user input or default settings
def select_features(user_selected_features=None):
    """
    Select features for similarity comparison.
    """
    default_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
    return user_selected_features if user_selected_features else default_features

# Search for songs based on partial input
def search_songs(data, partial_song_name, limit=5):
    """
    Search for song names that closely match the partial input.
    """
    # Ensure all song names are strings and handle NaN values
    all_song_names = data['track_name'].dropna().astype(str).unique()
    close_matches = get_close_matches(partial_song_name, all_song_names, n=limit, cutoff=0.3)
    return close_matches

# Recommend songs based on sound similarity without needing the artist name
def recommend_song_by_sound_similarity(data, song, selected_features=None):
    """
    Recommend songs based on sound similarity, not requiring the artist name.
    """
    try:
        # Flexible matching for song titles
        matched_songs = data[data['track_name'].str.lower().str.contains(song.lower(), na=False)]

        if matched_songs.empty:
            print(f"No matches found for '{song}'. Please check the spelling or try a different song.")
            return None

        similar_songs = data.copy()
        features = select_features(selected_features)
        sound_properties = normalize(similar_songs[features])

        # Calculate similarity based on the first match
        song_index = matched_songs.index[0]  # Safe to assume non-empty due to check above
        similar_songs['Similarity with song'] = cosine_similarity(sound_properties, sound_properties[song_index, None]).flatten()

        similar_songs = similar_songs.sort_values(by='Similarity with song', ascending=False)
        similar_songs = similar_songs[['track_name', 'artist_name', 'popularity'] + features]
        similar_songs.reset_index(drop=True, inplace=True)

        return similar_songs.iloc[1:11]
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    filepath = 'data/SpotifyAudioFeaturesApril2019.csv'
    data = load_data(filepath)
    
    # Example search for song names
    partial_song_name = "Shape of You"
    song_matches = search_songs(data, partial_song_name)
    print("Song matches:", song_matches)
    
    # Assuming the user selects the first match
    selected_song = song_matches[0]
    recommended_songs = recommend_song_by_sound_similarity(data, selected_song)
    print(recommended_songs)