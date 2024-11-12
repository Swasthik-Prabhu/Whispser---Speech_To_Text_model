import whisper
import warnings
import os
import pandas as pd
from transformers import LEDTokenizer, LEDForConditionalGeneration

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", category=UserWarning, message="huggingface_hub cache-system uses symlinks")

# Initialize Whisper and LED
model_whisper = whisper.load_model("medium")
tokenizer_led = LEDTokenizer.from_pretrained("allenai/led-base-16384")
model_led = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")

# Data structure to hold titles, transcripts, and summaries
data = {'title': [], 'Text': [], 'Summary': []}

# Function to chunk text
def chunk_text(text, max_length=4096):
    words = text.split()
    for i in range(0, len(words), max_length):
        yield ' '.join(words[i:i + max_length])

# Process each mp3 file
for file in os.listdir("whisper/audio"):
    if file.endswith("mp3"):
        file_name = file

        try:
            # Transcribe audio using Whisper
            result = model_whisper.transcribe(f"whisper/audio/{file_name}")
            transcript = result["text"]
            print("Transcript:", transcript)

            # Summarize transcript in chunks
            chunked_summaries = []
            for chunk in chunk_text(transcript, max_length=1024):  # 1024 to fit model context
                inputs = tokenizer_led(chunk, return_tensors="pt", truncation=True, max_length=16384)
                summary_ids = model_led.generate(inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
                chunked_summary = tokenizer_led.decode(summary_ids[0], skip_special_tokens=True)
                chunked_summaries.append(chunked_summary)

            # Combine chunked summaries into one summary
            summary = " ".join(chunked_summaries)

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
print("Summarized text:", summary)
