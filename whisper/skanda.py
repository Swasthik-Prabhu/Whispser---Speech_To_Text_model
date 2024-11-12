import whisper
import warnings
import os
import pandas as pd
from transformers import LEDTokenizer, LEDForConditionalGeneration
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", category=UserWarning, message="huggingface_hub cache-system uses symlinks")

# Initialize Whisper and LED
model_whisper = whisper.load_model("medium")
tokenizer_led = LEDTokenizer.from_pretrained("allenai/led-base-16384")
model_led = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")

# Data structure to hold titles, transcripts, and summaries
data = {'title': [], 'Text': [], 'Summary': []}

# Process each mp3 file
for file in os.listdir("whisper/audio"):
    if file.endswith("mp3"):
        file_name = file

        # Suppress warnings
        warnings.filterwarnings("ignore", category=UserWarning, message="FP16 is not supported on CPU; using FP32 instead")
        warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

        try:
            # Transcribe audio using Whisper
            result = model_whisper.transcribe(f"whisper/audio/{file_name}")
            transcript = result["text"]
            print("Transcript : " ,transcript)

            # Summarize transcript using LED
            inputs = tokenizer_led(transcript, return_tensors="pt", truncation=True, max_length=16384)
            summary_ids = model_led.generate(inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer_led.decode(summary_ids[0], skip_special_tokens=True)

        except Exception as e:
            print(f"Error occurred while processing file {file_name}: {e}")
            summary = "Error during summarization."

        # Add to data dictionary
        data['title'].append(file_name)
        data['Text'].append(transcript)
        data['Summary'].append(summary)

# Save data to CSV
df = pd.DataFrame(data)
df.to_csv("audio_summary.csv", index=False)
print("Transcriptions and summaries saved to audio_summary.csv")
print("Summarized text : ",summary)
