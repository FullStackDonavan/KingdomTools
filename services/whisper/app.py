from flask import Flask, request, jsonify
import whisper
import io
import tempfile
import os
import traceback

app = Flask(__name__)
model = whisper.load_model("base")  # You can choose a larger model if needed

@app.route('/inference', methods=['POST'])
def inference():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # Check if the file is an audio file
    if not file.filename.endswith(('.mp3', '.wav', '.m4a')):
        return jsonify({"error": "Invalid file type. Please upload an audio file."}), 400
    
    try:
        # Save the audio file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
            print(f"Temporary file created at: {temp_file_path}")  # Debugging line
        
        # Check if the file exists at the path
        if not os.path.exists(temp_file_path):
            return jsonify({"error": f"Temporary file not found: {temp_file_path}"}), 500
        
        # Load the audio file using Whisper's method
        audio = whisper.load_audio(temp_file_path)
        
        # Perform inference using the Whisper model
        result = model.transcribe(audio)
        
        # Optionally, clean up the temporary file
        os.remove(temp_file_path)
        
        return jsonify({"transcription": result['text']})
    
    except Exception as e:
        # Print the full traceback for better error understanding
        error_message = f"Failed to process the audio: {str(e)}\n{traceback.format_exc()}"
        print(error_message)  # Log the error to the console
        return jsonify({"error": error_message}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)  # Exposing port 9000
