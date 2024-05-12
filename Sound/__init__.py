import os

import numpy as np
from matplotlib import pyplot as plt

from mediapipe.tasks import python
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio
from scipy.io import wavfile
from moviepy.editor import VideoFileClip


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