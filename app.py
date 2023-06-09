import os
import whisper
import librosa
import soundfile as sf
import openai
from flask import Flask, request, send_file

app = Flask(__name__)

# You can replace this with your key stored as an environment variable for security
openai.api_key = os.environ.get('OPENAI_API_KEY')



model = whisper.load_model("base") 

@app.route('/')
def home():
    return '''
        <form method=post enctype=multipart/form-data>
             <input type=file name=file>
             <input type=submit>
        </form>
    '''

@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join('uploads', filename)
    file.save(filepath)

    # Process the file
    # Replace the code in this section with your code for processing the file
    audio_path = filepath[:-4] + ".wav" 
    y, sr = librosa.load(filepath, sr=16000)
    sf.write(audio_path, y, sr)

    result = model.transcribe(audio_path)
    text = result["text"].strip()
    text = text.replace(". ", ".\n\n")

    text_file = filename[:-4] + ".txt"
    text_path = os.path.join('uploads', text_file)

    with open(text_path, "w") as f:
        f.write(text)

    print(f"Processed {filename} and saved the transcription as {text_file}")
    
    model_message = {
        'role': 'system',
        'content': 'You are a helpful assistant that summarizes audio transcriptions.'
    }

    user_message = {
        'role': 'user',
        'content': f'{text}'
    }

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[model_message, user_message],
    )

    summary = response['choices'][0]['message']['content']
    summary_file = text_file[:-4] + "_summary.txt" 
    summary_path = os.path.join('uploads', summary_file)

    with open(summary_path, "w") as f:
        f.write(summary)

    print(f"Summarized {text_file} and saved the summary as {summary_file}")

    # At this point, you have two text files in your 'uploads' folder: the transcription and the summary
    # You can either return these files directly or provide links to download them
    return send_file(summary_path)

if __name__ == '__main__':
    app.run(port=5000)
