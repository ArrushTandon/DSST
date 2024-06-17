import whisper
import os
import Levenshtein
import csv
import librosa


def transcribe_audio(file_path, model):
    audio, sr = librosa.load(file_path, sr=None)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    return result.text


def calculate_cer(transcription, ground_truth):
    transcription = transcription.lower()
    ground_truth = ground_truth.lower()
    return Levenshtein.distance(transcription, ground_truth) / len(ground_truth)


def process_subdirectory(subdirectory_path, model, parent_directory, inner_subdirectory):
    aggregated_transcription = ""
    cer_results = []
    for root, dirs, files in os.walk(subdirectory_path):
        for file_name in files:
            if file_name.endswith(('.mp3', '.wav', '.m4a', '.flac')):
                file_path = os.path.join(root, file_name)
                transcription = transcribe_audio(file_path, model)
                file_id = os.path.splitext(file_name)[0]
                aggregated_transcription += f"{file_id} {transcription}\n"
    subdirectory_name = os.path.basename(subdirectory_path)
    aggregated_transcription_file_path = os.path.join(subdirectory_path, f"{parent_directory}-{subdirectory_name}.trans")
    with open(aggregated_transcription_file_path, "w", encoding="utf-8") as f:
        f.write(aggregated_transcription)
    ground_truth_file_path = os.path.join(subdirectory_path, f"{parent_directory}-{subdirectory_name}.trans")
    if os.path.exists(ground_truth_file_path):
        with open(ground_truth_file_path, "r", encoding="latin-1") as f:
            ground_truth = f.read().strip()
        cer = calculate_cer(aggregated_transcription, ground_truth)
        cer_results.append(cer)
    else:
        print(f"Ground truth file {ground_truth_file_path} not found for {subdirectory_name}")
    return cer_results

if __name__ == "__main__":
    model = whisper.load_model("base")
    directory = r"C:\Users\arrus\PycharmProjects\DSST Using Whisper\LibriSpeech\train-clean-100"
    all_cer_results = []
    for subdirectory in os.listdir(directory):
        subdirectory_path = os.path.join(directory, subdirectory)
        if os.path.isdir(subdirectory_path):
            print(f"Processing directory: {subdirectory}")
            directory_cer_results = []
            for inner_subdirectory in os.listdir(subdirectory_path):
                inner_subdirectory_path = os.path.join(subdirectory_path, inner_subdirectory)
                if os.path.isdir(inner_subdirectory_path):
                    print(f"Processing Sub-Directory: {inner_subdirectory}")
                    cer_results = process_subdirectory(inner_subdirectory_path, model, subdirectory, inner_subdirectory)
                    directory_cer_results.extend(cer_results)
            if directory_cer_results:
                average_cer = sum(directory_cer_results) / len(directory_cer_results)
                all_cer_results.append((subdirectory, average_cer))

            print("Transcribed")

    cer_output_file = r"C:\Users\arrus\OneDrive\Desktop\train_clean_100_cer_results.csv"
    with open(cer_output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Directory", "Average CER"])
        for result in all_cer_results:
            csv_writer.writerow(result)

    print("Transcription and CER calculation completed.")
