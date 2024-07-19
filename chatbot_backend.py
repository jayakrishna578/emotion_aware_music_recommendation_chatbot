from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
import requests
import os
from typing import List
from huggingface_hub import InferenceClient

app = FastAPI()

# Environmental Variables
HUGGING_FACE_API_KEY = os.getenv('HUGGING_FACE_API_KEY')
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
SPOTIFY_USER_ID = os.getenv('SPOTIFY_USER_ID')

class EmotionDetector:
    def __init__(self, hf_token, endpoint_url):
        self.client = InferenceClient(endpoint_url, token=hf_token)
    
    def detect_emotion(self, user_input):
        prompt = f"You are an AI assistant, you detect the user emotion based on their input and respond in a single word that is the emotion you have detected. The following sentence is the user input: '{user_input}'"
        
        # Use the streaming API to process the prompt
        stream = self.client.text_generation(prompt)
        
        parts = stream.split(':')
        if len(parts) > 1:
            detected_emotion = parts[1].strip().split(' ')[0]
        else:
            detected_emotion = "Unknown"
        
        return detected_emotion
    
    
class AssistantResponder:
    def __init__(self, hf_token, endpoint_url):
        self.client = InferenceClient(endpoint_url, token=hf_token)
    
    def generate_response(self, user_input, memory, temperature, top_p, max_length):
        prompt = f"You are a helpful and kind assistant. Respond to the user query based on the conversation history.\n\n{memory}User: {user_input}\nAssistant:"
        
        # Adjust generation parameters as needed
        gen_kwargs = dict(
            max_new_tokens=max_length,
            top_k=5,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=1.02,
            stop_sequences=["\n"],
        )
        
        # Use the streaming API to process the prompt
        stream = self.client.text_generation(prompt, **gen_kwargs)
        
        return stream


class SpotifyManager:
    def __init__(self, client_id: str = SPOTIFY_CLIENT_ID, client_secret: str = SPOTIFY_CLIENT_SECRET, user_id: str = SPOTIFY_USER_ID):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_id = user_id

    def get_access_token(self) -> str:
        # This method should ideally handle user-level OAuth tokens, not just client credentials
        response = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
        )
        response.raise_for_status()
        return response.json()["access_token"]

    def search_tracks(self, access_token: str, mood: str, limit: int = 10) -> List[str]:
        query = f"genre:{mood}"
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(f"https://api.spotify.com/v1/search?type=track&limit={limit}&q={query}", headers=headers)
        response.raise_for_status()
        tracks = response.json()["tracks"]["items"]
        return [track["uri"] for track in tracks]

    def create_playlist(self, access_token: str, playlist_name: str) -> str:
        headers = {"Authorization": f"Bearer {access_token}"}
        payload = {
            "name": playlist_name,
            "public": False  # Make playlist private
        }
        response = requests.post(f"https://api.spotify.com/v1/users/{self.user_id}/playlists", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["id"]

    def add_tracks_to_playlist(self, access_token: str, playlist_id: str, track_uris: List[str]):
        headers = {"Authorization": f"Bearer {access_token}"}
        payload = {"uris": track_uris}
        response = requests.post(f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks", headers=headers, json=payload)
        response.raise_for_status()

class ChatInput(BaseModel):
    text: str
    consent: bool

@app.post("/chat")
async def chat(input: ChatInput, emotion_detector: EmotionDetector = Depends(), spotify_manager: SpotifyManager = Depends()):
    if not input.consent:
        return {"error": "User consent required for processing."}

    # Detect emotion from user input
    detected_emotion = emotion_detector.detect_emotion(input.text)

    # Get user-level access token (needs proper implementation)
    access_token = spotify_manager.get_access_token()

    # Create a new playlist for the detected emotion
    playlist_name = f"Playlist for {detected_emotion} Mood"
    playlist_id = spotify_manager.create_playlist(access_token, playlist_name)

    # Search for tracks based on the detected emotion
    track_uris = spotify_manager.search_tracks(access_token, detected_emotion)

    # Add tracks to the created playlist
    spotify_manager.add_tracks_to_playlist(access_token, playlist_id, track_uris)

    # Construct response with playlist details
    playlist_url = f"https://open.spotify.com/playlist/{playlist_id}"
    response_message = f"Created a '{playlist_name}' playlist based on the detected emotion: {detected_emotion}."
    return {
        "message": response_message,
        "playlist_name": playlist_name,
        "playlist_url": playlist_url,
        "tracks_added": len(track_uris)
    }
