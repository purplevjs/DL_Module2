import os
import asyncio
from flask import Flask, request
from flask_socketio import SocketIO 
from google.cloud import speech 
from google.cloud.speech_v1 import types 
import queue
import threading 
import base64
from google.api_core.exceptions import DeadlineExceeded
from flask_socketio import join_room

SILENCE_CHUNK_DURATION = 0.128  # 100ms
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2         # 16-bit
NUM_CHANNELS = 1
chunk_size = int(SILENCE_CHUNK_DURATION * SAMPLE_RATE * NUM_CHANNELS * BYTES_PER_SAMPLE)
silence_chunk = b"\x00" * chunk_size


# Define the socketio instance
app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")
# Set the environment variable for the Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key_gst.json"
# Create the Google Cloud Speech client
client = speech.SpeechClient()
audio_queue = queue.Queue()

Finished = None
stream_started = False
transcript_text = ""



# Move configuration to global scope
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code="en-US",
    enable_automatic_punctuation=True,
)

streaming_config = speech.StreamingRecognitionConfig(
    config=config,
    interim_results=True,
)

def audio_stream_generator():
    chunk_counter = 0  # Add counter
    
    while True:
        try:
            # Set timeout to 1 second
            chunk = audio_queue.get(timeout=1)
            if chunk is None:
                print("Received stop signal in audio stream generator")
                break
                
            chunk_counter += 1  # Increment counter
            if chunk_counter % 200 == 0:
                print(f"Processing audio chunk #{chunk_counter}")
                
            yield types.StreamingRecognizeRequest(audio_content=chunk)
        except queue.Empty:
            # Continue waiting when queue is empty
            print("Queue empty, sending silence to keep connection")
            yield types.StreamingRecognizeRequest(audio_content=silence_chunk)
            continue
        except Exception as e:
            print(f"Audio stream generator error: {e}")
            break


def process_stream():
    """Speech recognition process running in a separate thread."""
    global transcript_text
    global stream_started
    global Finished
    
    Finished = False
    
    try:
        requests = audio_stream_generator()
        # Timeout can be adjusted as needed
        responses = client.streaming_recognize(config=streaming_config, requests=requests, timeout=120)
        
        for response in responses:
            if not response.results:
                continue
                
            result = response.results[0]
            
            # Skip non-final results, don't send to frontend
            if not result.is_final:
                continue
                
            # Only process final results
            transcript = result.alternatives[0].transcript
            transcript_text = "".join([transcript_text, transcript])
            print(f"Final transcription result: {transcript}")
            socketio.emit("transcription", transcript_text, namespace="/ws")
            
        # Mark as finished when the stream ends naturally
        Finished = True
        print("Stream recognition complete")

    except DeadlineExceeded:
        # If the caught exception type is timeout
        print("Recognition timeout: DeadlineExceeded")
        socketio.emit("timeout", {"message": "Speech recognition timed out."}, namespace="/ws")
        Finished = True
    except Exception as e:
        print(f"Recognition process error: {e}")
        Finished = True
    finally:
        # After recognition ends, reset the flag for next start
        stream_started = False
        print("Speech recognition stream ended, reset stream_started to False")
