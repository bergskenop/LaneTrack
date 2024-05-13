import pickle
import time

import cv2
from matplotlib import pyplot as plt

import Pool
import Sound
from helper import *

# path = 'data/video/Fanny Lecluyse 50m SS Reeksen EK25m 2015_1 - Swimming.avi'
# path = 'data/video/Jérôme_Florent_Manaudou_50mNl_FinA.MTS'
# path = 'data/video/Bere Waerniers 200m VL FIN FSC 2024.MP4'
path = 'data/video/Florine Gaspard 100m BREASTSTROKE FIN BK Open 2023.MTS'
#
run_time = time.time()
p = Pool.Pool('50')

cap = cv2.VideoCapture(path)
fps_divider = 4
fps = int(cap.get(cv2.CAP_PROP_FPS))/fps_divider
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))/fps_divider
writer = None
p.initialise_lanes(path, fps_div=fps_divider)
for idx, frame in enumerate(p.process(path, fps_div=fps_divider)):
    if writer is None:
        writer = cv2.VideoWriter(f"Florine_Gaspard_25m.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (frame.shape[1], frame.shape[0]))
    writer.write(frame)
print(f'\nTotal run time: {time.time()-run_time}')
print(f"\nRun finished at {round(idx/total_frames*100,2)}%, generating results")


print(p.frame_instance.swimmer.box_width_list)

plt.plot(get_filtered_list(interpolate_nan(p.frame_instance.swimmer.box_height_list)))
plt.xlabel('Frame index')
plt.ylabel('Height')
plt.title('Plot of Height')
plt.show()

plt.plot(get_filtered_list(interpolate_nan(p.frame_instance.swimmer.box_width_list)))
plt.xlabel('Frame index')
plt.ylabel('Width')
plt.title('Plot of Width')
plt.show()

frame_numbers = [remarkable_frame[0] for remarkable_frame in p.remarkable_frames]

# Determine the maximum frame number
max_frame_number = int(total_frames)
# Create a list initialized with zeros
frame_indicator = [0] * (max_frame_number + 1)

# Update the list with frame numbers present in the tuples
for frame_number in frame_numbers:
    frame_indicator[frame_number] = 1

with open("cluster_list.pkl", 'wb') as file:
    # Use pickle to dump the list into the file
    pickle.dump(frame_indicator, file)

# Plot the frame indicators
plt.plot(frame_indicator)
plt.title("Frame Indicator Plot")
plt.xlabel("Frame Number")
plt.ylabel("Presence (1) or Absence (0)")
plt.show()

buzzer_time = Sound.run_NN(path)
print(f'Buzzer detected at: {buzzer_time}ms')


