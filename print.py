import pickle
from helper import *
import cv2
import matplotlib.pyplot as plt
import pandas as pd

import Sound


def group_ones_with_zeroes(lst):
    groups = [[]]
    for num in lst:
        groups[-1].append(num)
        if num == 0:
            groups.append([])
    return [group + [0] * (10 - len(group)) for group in groups if group]


def find_group_starts(lst):
    return [i for i, (a, b) in enumerate(zip(lst, lst[1:]), 1) if a == 1 and b == 0]


def group_starting_points_dynamic(start_points):
    if len(start_points) == 0:
        return []

    # Calculate the distances between consecutive starting points
    distances = [start_points[i + 1] - start_points[i] for i in range(len(start_points) - 1)]

    # Calculate the average distance
    avg_distance = sum(distances) / len(distances)

    # Set the threshold as a multiple of the average distance
    threshold = 1.3 * avg_distance  # Adjust the multiplier as needed

    # Group starting points based on the dynamic threshold
    grouped_start_points = []
    current_group = []

    for i, start_point in enumerate(start_points):
        if i == 0:
            current_group.append(start_point)
        else:
            if start_point - current_group[-1] <= threshold:
                current_group.append(start_point)
            else:
                grouped_start_points.append(current_group)
                current_group = [start_point]

    grouped_start_points.append(current_group)  # Append the last group

    return grouped_start_points


with open('output/run_60fps/swimmer_cross_segment_list.pkl', 'rb') as file:
    # Use pickle to load the list from the file
    loaded_data = pickle.load(file)

video_path = 'data/video/Florine Gaspard 100m BREASTSTROKE FIN BK Open 2023.MTS'
print("List loaded from file:", len(loaded_data))

filt_data = get_filtered_list(loaded_data)
filt_max_indices = local_maximum(filt_data, window_size=100)
filt_max_indices.append(len(loaded_data))
plt.plot(filt_data)
for indice in filt_max_indices:
    plt.axvline(x=indice, color='r', linestyle='--', linewidth=0.25, label=f'{indice}s')
plt.show()

grouped_data = group_ones_with_zeroes(loaded_data)
group_starts = find_group_starts(loaded_data)
grouped_start_points_dynamic = group_starting_points_dynamic(group_starts)

fps = (cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS))

print(len(grouped_start_points_dynamic))
print(grouped_start_points_dynamic)

time_intervals = []
interval_labels = [
    "0m-5m", "5m-15m", "15m-25m", "25m-35m",
    "35m-45m", "45m-55m", "55m-65m", "65m-75m",
    "75m-85m", "85m-95m"
]

# Calculate the first time interval
# time_intervals.append(grouped_start_points_dynamic[0][0] * 1 / fps)
# for i in range(1, 10):
#     time_interval = (grouped_start_points_dynamic[i][0] - grouped_start_points_dynamic[i - 1][0]) / fps
#     time_intervals.append(time_interval)

time_intervals.append(filt_max_indices[0] * 1 / fps)
for i in range(1, 10):
    time_interval = (filt_max_indices[i] - filt_max_indices[i - 1]) / fps
    time_intervals.append(time_interval)

buzzer_frequency = 1722.65625
buzzer_timestamp = Sound.get_spectrogram_timestamp('data/video/Florine Gaspard 100m BREASTSTROKE FIN BK Open 2023.MTS', buzzer_frequency, visualise=False)

total_sum = sum(time_intervals)-buzzer_timestamp+2.38

for label, interval in zip(interval_labels, time_intervals):
    print(f'{label} -> {interval:.2f}s')

minutes, seconds = divmod(int(total_sum), 60)
milliseconds = int((total_sum - int(total_sum)) * 1000)
# Time_25 = sum([Time0_5, Time5_15, Time15_25])

print(f"{minutes} minutes, {seconds} seconds, {milliseconds} milliseconds")

average_speed = 100 / total_sum
print(average_speed)

total_10 = ""

"""
Results in same manner as manual report
Dynamically decide heat distance every 50m contains 5 segments --> 50 * (len(time_intervals)//5)
Most important information 0->25 25->50 0->50 50->75 75->100 50->100 0->100  
"""
# TODO Find a way to detect the end of the swimming pool (lap time)

heat_length = 50 * (len(time_intervals) // 5)
print(f'Detected heat length: {heat_length}m ')
print(f"start timestamp: {buzzer_timestamp}")

LAP1 = {"0_25m": grouped_start_points_dynamic[2][0]*1/fps - buzzer_timestamp,
        "25_35m": (grouped_start_points_dynamic[3][0]-grouped_start_points_dynamic[2][0])*1/fps,
        "35_45m": (grouped_start_points_dynamic[4][0]-grouped_start_points_dynamic[3][0])*1/fps}

LAP2 = {"55_65m": (grouped_start_points_dynamic[6][0] - grouped_start_points_dynamic[5][0])*1/fps,
        "65_75m": (grouped_start_points_dynamic[7][0]-grouped_start_points_dynamic[6][0])*1/fps,
        "75_95m": (grouped_start_points_dynamic[9][0]-grouped_start_points_dynamic[8][0])*1/fps}

print(LAP1)
print(LAP2)

print(f"Heat timing: {len(loaded_data)*1/fps-buzzer_timestamp}s")

data = {
    "LAP": ["1", "1", "1", "2", "2", "2"],
    "Sections": ["Section 1", "Section 2", "Section 3", "Section 4", "Section 5", "Section 6"],
    "Segment": ["15-25m", "25-35m", "35-50m", "65-75m", "75-85m", "85-95m"],
    "25m Sections (video)": [14.32, 17.36, 17.28, 17.28, 18.98, 18.98],
    "SF": [50.6, 48.1, 47.4, 50.6, 49.7, 49.5],
    "SL": [1.76, 1.83, 1.85, 1.66, 1.67, 1.60],
    "Str. Index": [2.63, 2.70, 2.71, 2.33, 2.12, 2.12],
    "# Strokes": [7, 14, 21, 10, 16, 26],
    "Clean m/s": [1.49, 1.47, 1.46, 1.40, 1.38, 1.32]
}

df = pd.DataFrame(data)
df.plot()
print(df)

# plt.plot(loaded_data)
# plt.show()
