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
from Function.sptt import process_stream, audio_queue, app, socketio
import Function.sptt as sptt
from Function.NLP_preprocessor import simplify_text
from Function.AI_conversation import gen_answer, paraphrase
import time
from Function.Store_data import distinguish_data

# Define global variables
cleaned_text = ""
token_count = 0

# Add a background task queue for storing text that needs to be processed
background_task_queue = queue.Queue()

# Add background thread function to process tasks in the queue
def background_worker():
    print("Background data processing thread started")
    while True:
        try:
            # Get task from queue, will block if queue is empty
            text = background_task_queue.get()
            if text is None:  # None value used as termination signal
                break
                
            print(f"Background processing text data: {text[:50]}...")  # Print first 50 characters of text as log
            
            # Call distinguish_data function to process text
            try:
                distinguish_data(text)
                print("Text data processing completed and index updated")
            except Exception as e:
                print(f"Error processing text data: {e}")
            
            # Mark task as complete
            background_task_queue.task_done()
        except Exception as e:
            print(f"Background worker thread error: {e}")

# Start background worker thread
background_thread = threading.Thread(target=background_worker, daemon=True)
background_thread.start()

@socketio.on("connect", namespace="/ws")
def handle_connect():
    print("Client connected")

@socketio.on("disconnect", namespace="/ws")
def handle_disconnect():
    print("Client disconnected")
    audio_queue.put(None)  # End audio stream when connection is closed
    sptt.stream_started = False  # Reset flag

@socketio.on("start_recording", namespace="/ws")
def handle_start_recording():
    print("Recording started")
    sptt.Finished = False
    sptt.transcript_text = ""  # Reset transcription text

@socketio.on("stop_recording", namespace="/ws")
def handle_stop_recording():
    global cleaned_text
    global token_count
    
    audio_queue.put(None)
    
    processing_lock = threading.Event()
    
    def process_transcription():
        global cleaned_text
        global token_count
        
        if processing_lock.is_set():
            return
        
        processing_lock.set()
        
        try:
            max_wait_time = 60
            wait_count = 0
            
            while wait_count < max_wait_time:
                if sptt.Finished:
                    if sptt.transcript_text:
                        cleaned_text, token_count = simplify_text(sptt.transcript_text)
                        paraphrased_text = paraphrase(cleaned_text, token_count)
                        
                        # Add original text to background processing queue
                        background_task_queue.put(sptt.transcript_text)
                        
                        # Generate answer directly without any prompts or markers
                        ai_answer = gen_answer(paraphrased_text)
                        
                        socketio.emit("answer", ai_answer, namespace="/ws")
                        socketio.emit("answer_complete", namespace="/ws")
                    break
                
                wait_count += 0.1
                time.sleep(0.1)
        finally:
            processing_lock.clear()

    threading.Thread(target=process_transcription, daemon=True).start()

@socketio.on("join_room", namespace="/ws")
def handle_join_room(data):
    room_name = data.get("room")
    if room_name:
        join_room(room_name)
        print(f"Client {request.sid} joined room {room_name}")
        # Optional: Send message back to client that they've joined
        socketio.emit("joined_room", {"room": room_name}, to=request.sid, namespace="/ws")

@socketio.on("audio_chunk", namespace="/ws")
def handle_audio_chunk(data):
    # Display transcription text in real-time
    socketio.emit("transcription", sptt.transcript_text, namespace="/ws")

    try:
        # Convert ArrayBuffer to bytes
        if isinstance(data, dict) and 'data' in data:
            # If data is wrapped in a dictionary
            audio_data = bytes(data['data'])
        elif isinstance(data, (bytes, bytearray)):
            # If already in bytes format
            audio_data = bytes(data)
        else:
            # Other cases, try direct conversion
            audio_data = bytes(data)
        
        audio_queue.put(audio_data)

        if not sptt.stream_started:
            sptt.stream_started = True
            print("Starting speech recognition process (thread)")
            # Use native thread to execute process_stream
            t = threading.Thread(target=process_stream, daemon=True)
            t.start()
            
    except Exception as e:
        print(f"Error processing audio data: {e}")

if __name__ == "__main__":
    # Use absolute path to ensure correct file location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_dir)
    credentials_path = os.path.join(current_dir, "key_gst.json")
    print(credentials_path)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    socketio.run(app, host="0.0.0.0", port=5000)