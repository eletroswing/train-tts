from TTS.TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt
import os

train_csv = "/content/output/metadata_train.csv"
eval_csv = "/content/output/metadata_eval.csv"
max_audio_length = 11
language = "pt"
num_epochs = 10
batch_size = 4
grad_acumm = 1
output_path = "out"

if not train_csv or not eval_csv:
    print("You need to run the data processing step or manually set `Train CSV` and `Eval CSV` fields!")
try:
    # convert seconds to waveform frames
    max_audio_length = int(max_audio_length * 22050)
    train_gpt(language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path=output_path, max_audio_length=max_audio_length)
except Exception as e:
    print(e)
    print("The training was interrupted due to an error! Please check the console to see the full error message.")
