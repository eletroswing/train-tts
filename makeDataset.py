import csv
import os

import zipfile
import shutil

import torch
import torchaudio

import gc
import pandas
from faster_whisper import WhisperModel

from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners

torch.set_num_threads(16)


##########################################33
#edit here, your data must be in format: file_path, speaker_name, language 
############################################3
BaseDataset = 'data.csv'
zip_path = 'dataset.zip'

Refine = 0.15
Buffer = 0.2

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print('''
Try to read the raw dataset, expected format:
audio_path,speaker_name,lang
audio_path,speaker_name,lang
audio_path,speaker_name,lang
audio_path,speaker_name,lang
''')

if not os.path.exists(BaseDataset):
    raise FileNotFoundError(f"{BaseDataset} not found")

if not os.path.exists('output'):
  os.mkdir('output')

device = "cuda" if torch.cuda.is_available() else "cpu" 
print("Loading Whisper Model!")
asr_model = WhisperModel("large-v2", device=device, compute_type="float16")
metadata = {"audio_file": [], "text": [], "speaker_name": []}

audio_total_size = 0
lines = 0

with open(BaseDataset, mode='r', newline='') as archive:
    reader = csv.reader(archive, delimiter=",")
    for line in reader:
      print(f'Processing line: {lines}')
      lines += 1;
      audio_path, speaker_name, lang = line
      wav, sr = torchaudio.load(audio_path)
      
      # stereo to mono if needed
      if wav.size(0) != 1:
          wav = torch.mean(wav, dim=0, keepdim=True)
      
      wav = wav.squeeze()
      audio_total_size += (wav.size(-1) / sr)

      segments, _ = asr_model.transcribe(audio_path, word_timestamps=True, language=lang)
      segments = list(segments)
      i = 0
      sentence = ""
      sentence_start = None
      first_word = True
      # added all segments words in a unique list
      words_list = []
      for _, segment in enumerate(segments):
          words = list(segment.words)
          words_list.extend(words)

      # process each word
      for word_idx, word in enumerate(words_list):
          if first_word:
              sentence_start = word.start
              # If it is the first sentence, add buffer or get the begining of the file
              if word_idx == 0:
                  sentence_start = max(sentence_start - Buffer, 0)  # Add buffer to the sentence start
              else:
                  # get previous sentence end
                  previous_word_end = words_list[word_idx - 1].end
                  # add buffer or get the silence midle between the previous sentence and the current one
                  sentence_start = max(sentence_start - Buffer, (previous_word_end + sentence_start)/2)

              sentence = word.word
              first_word = False
          else:
              sentence += word.word

          if word.word[-1] in ["!", ".", "?"]:
              sentence = sentence[1:]
              # Expand number and abbreviations plus normalization
              sentence = multilingual_cleaners(sentence, lang)
              audio_file_name, _ = os.path.splitext(os.path.basename(audio_path))

              audio_file = f"wavs/{audio_file_name}_{str(i).zfill(8)}.wav"

              # Check for the next word's existence
              if word_idx + 1 < len(words_list):
                  next_word_start = words_list[word_idx + 1].start
              else:
                  # If don't have more words it means that it is the last sentence then use the audio len as next word start
                  next_word_start = (wav.shape[0] - 1) / sr

              # Average the current word end and next word start
              word_end = min((word.end + next_word_start) / 2, word.end + Buffer)
                
              absoulte_path = os.path.join('output', audio_file)
              os.makedirs(os.path.dirname(absoulte_path), exist_ok=True)
              i += 1
              first_word = True

              audio = wav[int(sr*sentence_start):int(sr*word_end)].unsqueeze(0)
              # if the audio is too short ignore it (i.e < 0.33 seconds)
              if audio.size(-1) >= sr/3:
                  torchaudio.save(absoulte_path,
                      audio,
                      sr
                  )
              else:
                  continue

              metadata["audio_file"].append(audio_file)
              metadata["text"].append(sentence)
              metadata["speaker_name"].append(speaker_name)
      clear_gpu_cache()
    
    df = pandas.DataFrame(metadata)
    df = df.sample(frac=1)
    num_val_samples = int(len(df)*Refine)

    df_eval = df[:num_val_samples]
    df_train = df[num_val_samples:]

    df_train = df_train.sort_values('audio_file')
    train_metadata_path = os.path.join('output', "metadata_train.csv")
    df_train.to_csv(train_metadata_path, sep="|", index=False)

    eval_metadata_path = os.path.join('output', "metadata_eval.csv")
    df_eval = df_eval.sort_values('audio_file')
    df_eval.to_csv(eval_metadata_path, sep="|", index=False)

    # deallocate VRAM and RAM
    del asr_model, df_train, df_eval, df, metadata
    gc.collect()
'''
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
  for root, dirs, files in os.walk('output'):
      for file in files:
          caminho_completo = os.path.join(root, file)
          caminho_relativo = os.path.relpath(caminho_completo, os.path.join('output', '..'))
          zipf.write(caminho_completo, caminho_relativo)

shutil.rmtree('output')
'''