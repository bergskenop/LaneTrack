import pickle
import shutil
import time

import cv2
from matplotlib import pyplot as plt

import Pool
import Sound
import Stroke
from Stroke import *
import Swimmer
from helper import *

# path = 'data/video/Fanny Lecluyse 50m SS Reeksen EK25m 2015_1 - Swimming.avi'
# path = 'data/video/Jérôme_Florent_Manaudou_50mNl_FinA.MTS'
# path = 'data/video/Bere Waerniers 200m VL FIN FSC 2024.MP4'
path = 'data/video/Florine Gaspard 100m BREASTSTROKE FIN BK Open 2023.MTS'
# path = 'data/video/Florine Gaspard 100m Breaststroke Mirrored.mp4'
#
run_name = input('Enter run name: ')
run_time = time.time()

cap = cv2.VideoCapture(path)
fps_divider = 1
fps = int(cap.get(cv2.CAP_PROP_FPS))/fps_divider
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))/fps_divider
writer = None

p = Pool.Pool('50', total_frames)
if os.path.exists(f'output/{run_name}') and input("Run exists, load data? ").upper() == 'Y':
    print("Load files from workspace...")
    with open(f"output/{run_name}/height_list.pkl", 'rb') as f:
        box_height_list = pickle.load(f)
    with open(f"output/{run_name}/width_list.pkl", 'rb') as f:
        box_width_list = pickle.load(f)
    with open(f"output/{run_name}/lane_angle_list.pkl", 'rb') as f:
        rotation_list_median = pickle.load(f)
    with open(f"output/{run_name}/swimmer_cross_segment_list.pkl", 'rb') as f:
        frame_indicator = pickle.load(f)
    print('Successfully loaded: box height, box width, angle list and frame indicator')

    plot_results(box_height_list, box_width_list, rotation_list_median, frame_indicator)

    s = Stroke(get_filtered_list(interpolate_nan(box_height_list)))

    data = []
    for idx, val in enumerate(s.height_lst):
        data.append((idx*1/fps, val))

    x_values = [point[0] for point in data]
    y_values = [point[1] for point in data]
    frame_indices = list(range(len(s.height_lst)))

    plt.plot(x_values, y_values, linewidth=.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Height')
    plt.title('Plot of Local maximum Height')
    for x in s.stroke_indices:
        print(f'Stroke at: {x / 50}s')
        plt.axvline(x=x/50, color='r', linestyle='--', linewidth=0.25, label=f'{x/50}s')
    plt.savefig(f'output/{run_name}/stroke_plot_1000.png', dpi=1000)
    plt.show()


else:
    if os.path.exists(f"output/{run_name}"):
        print(f"Removing all contents within {run_name}")
        shutil.rmtree(f"output/{run_name}")
    os.mkdir(f"output/{run_name}")
    p.initialise_lanes(path, fps_div=fps_divider)
    for idx, frame in enumerate(p.process(path, fps_div=fps_divider)):
        if writer is None:
            writer = cv2.VideoWriter(f"output/{run_name}/Florine_Gaspard_25m.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    print(f'\nTotal run time: {time.time() - run_time}')
    print(f"\nRun finished at {round(idx / total_frames * 100, 2)}%, generating results")
    with open(f"output/{run_name}/height_list.pkl", 'wb') as f:
        pickle.dump(p.frame_instance.swimmer.box_height_list, f)
    with open(f"output/{run_name}/width_list.pkl", 'wb') as f:
        pickle.dump(p.frame_instance.swimmer.box_width_list, f)
    with open(f"output/{run_name}/lane_angle_list.pkl", 'wb') as f:
        pickle.dump(p.LD.get_median_rotation(), f)
    with open(f"output/{run_name}/swimmer_cross_segment_list.pkl", 'wb') as f:
        pickle.dump(p.frame_indicator, f)


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

    with open("cluster_list.pkl", 'wb') as file:
        # Use pickle to dump the list into the file
        pickle.dump(p.frame_indicator, file)

    # Plot the frame indicators
    plt.plot(p.frame_indicator)
    plt.title("Frame Indicator Plot")
    plt.xlabel("Frame Number")
    plt.ylabel("Presence (1) or Absence (0)")
    plt.show()

    buzzer_time = Sound.run_NN(path)
    print(f'Buzzer detected at: {buzzer_time}ms')



