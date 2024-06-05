import os

import numpy as np
from matplotlib import pyplot as plt

from mediapipe.tasks import python
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio
from scipy.io import wavfile
from moviepy.editor import VideoFileClip
from scipy.signal import stft


def run_NN(video_path, max_results=100):
    video_clip = VideoFileClip(video_path)

    # Extract the audio for the first 30 seconds
    audio_clip = video_clip.audio.subclip(0, 30)
    audio_clip.write_audiofile('temp.wav')

    # Customize and associate model for Classifier
    base_options = python.BaseOptions(model_asset_path='weights/yamnet.tflite')
    options = audio.AudioClassifierOptions(
        base_options=base_options, max_results=max_results)

    # Create classifier, segment audio clips, and classify
    with audio.AudioClassifier.create_from_options(options) as classifier:
        sample_rate, wav_data = wavfile.read('temp.wav')
        audio_clip = containers.AudioData.create_from_array(
            wav_data.astype(float) / np.iinfo(np.int16).max, sample_rate)
        classification_result_list = classifier.classify(audio_clip)
        print(sample_rate)

        detected_timestamps_alarm = []
        detected_timestamps_speech = []

        for idx, classification_result in enumerate(classification_result_list):
            for category in classification_result.classifications[0].categories:
                if category.category_name == 'Speech':
                    print(
                        f"Speech detected in classification result {idx + 1} at time: {classification_result.timestamp_ms}ms")
                    detected_timestamps_speech.append(classification_result.timestamp_ms)
                if category.category_name == 'Alarm':
                    print("Category:", category)
                    print(
                        f"Alarm detected in classification result {idx + 1} at time: {classification_result.timestamp_ms}ms")
                    detected_timestamps_alarm.append(classification_result.timestamp_ms)
                else:
                    pass
        os.remove('temp.wav')
        return detected_timestamps_alarm[-1]

def get_spectrogram_timestamp(video_path, frequency_q, ground_timestamp=None, visualise=True):
    video_clip = VideoFileClip(video_path)

    # Extract the audio for the first 30 seconds
    audio_clip = video_clip.audio.subclip(0, 30)
    audio_clip.write_audiofile('temp.wav')

    sample_rate, data = wavfile.read('temp.wav')

    # If stereo, select a single channel (e.g., left channel)
    if len(data.shape) > 1:
        data = data[:, 0]

    window_size = 512  # Size of each window
    overlap = 0.5  # Overlap between consecutive windows (50%)
    noverlap = int(window_size * overlap)

    frequencies, times, Zxx = stft(data, fs=sample_rate, nperseg=window_size, noverlap=noverlap)

    freq_index = np.argmin(np.abs(frequencies - frequency_q))
    time_index = np.argmax(np.abs(Zxx[freq_index]))

    time_with_max_magnitude = times[time_index]

    if visualise:
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(times, frequencies, np.abs(Zxx), shading='gouraud')
        plt.title(f'STFT buzzer detection')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='Magnitude')
        plt.axvline(x=time_with_max_magnitude, color='r', linestyle='-', linewidth=2)
        if ground_timestamp is not None:
            plt.axvline(x=ground_timestamp / 1000, color='g', linestyle='--', linewidth=2)
        plt.ylim(0, 10000)  # Limit frequency range for better visualization
        plt.show()
        # plt.savefig(f'STFT Magnitude start detection {video_path[0]}.jpg')
    if ground_timestamp is not None:
        print(f'Deviation: {abs(ground_timestamp - time_with_max_magnitude * 1000)}')
    os.remove('temp.wav')
    return time_with_max_magnitude