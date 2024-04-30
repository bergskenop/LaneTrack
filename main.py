import time

import cv2

import Pool

# path = 'data/video/Fanny Lecluyse 50m SS Reeksen EK25m 2015_1 - Swimming.avi'
# path = 'data/video/Jérôme_Florent_Manaudou_50mNl_FinA.MTS'
# path = 'data/video/Bere Waerniers 200m VL FIN FSC 2024.MP4'
path = 'data/video/Florine Gaspard 100m BREASTSTROKE FIN BK Open 2023.MTS'

run_time = time.time()
p = Pool.Pool('50')

# p.test_lanemaskdetection(path)

writer = None
cap = cv2.VideoCapture(path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
for idx, frame in enumerate(p.process(path)):
    if writer is None:
        writer = cv2.VideoWriter(f"Florine_Gaspard_SwimmerTrackingAndSegmentCrossing.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (frame.shape[1], frame.shape[0]))
    writer.write(frame)
print(f'Total run time: {time.time()-run_time}')

