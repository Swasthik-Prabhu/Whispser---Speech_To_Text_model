import whisper
import warnings
import os
import pandas as pd




d = {'title':[],'Text':[]}
for file in os.listdir("whisper/audio"):
    if file.endswith("mp3"):
        file_name = file
        
        warnings.filterwarnings("ignore", category=UserWarning, message="FP16 is not supported on CPU; using FP32 instead")
        warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

        try:
            # Load the medium model
            model = whisper.load_model("medium")

            # Transcribe the audio file
            result = model.transcribe(f"whisper/audio/{file_name}")
            print(result["text"])
        except Exception as e:
            print(f"Error occurred while loading the model or transcribing the audio file: {e}")


        d['title'].append(file_name)
        d['Text'].append(result["text"])
        df = pd.DataFrame(data = d)
        df.to_csv("audio.csv")  

