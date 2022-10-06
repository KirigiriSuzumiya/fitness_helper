import json
import cv2
import subprocess
import threading
import numpy as np
import os


def camera_thread():
    shell = "python python/det_keypoint_unite_infer.py --det_model_dir=model/picodet_v2_s_192_pedestrian " \
            "--keypoint_model_dir=model/tinypose_128x96 --camera_id=0 --device=GPU --save_res=True"
    subprocess.run(shell)


def video_player(video_name):
    json_path = os.path.join('video', video_name[:video_name.rfind('.')], video_name[:video_name.rfind('.')]+".json")
    video_path = os.path.join('output', video_name)
    data = json.load(open(json_path, "r"))
    video = cv2.VideoCapture(video_path)
    frame_id = 0
    fp = open("temp.json", "w")
    fp.close()
    while True:
        ret, frame = video.read()
        if not ret:
            raise "视频读取错误"
        frame_id += 1
        frame_res = data[frame_id]
        peo_boxes = frame_res[1]
        skel_list = frame_res[2][0]
        peo_area = []
        for i, box in enumerate(peo_boxes):
            peo_area.append((box[2] - box[0]) * (box[3] - box[1]))
        main_boxes = peo_boxes[peo_area.index(max(peo_area))]
        cv2.rectangle(frame,
                      (main_boxes[0], main_boxes[1]), (main_boxes[2], main_boxes[3]),
                      color=[0, 0, 255], thickness=3)
        for i, skel_points in enumerate(skel_list[peo_area.index(max(peo_area))]):
            cv2.circle(frame,
                       (int(skel_points[0]), int(skel_points[1])),
                       radius=4, color=[255, 0, 0], thickness=2)

        frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        cv2.imshow("test", frame)
        k = cv2.waitKey(20)
        # q键退出
        if (k & 0xff == ord('q')):
            break
        while True:
            camera_res = None
            therhold = 0.97
            video_KPT = skel_list[peo_area.index(max(peo_area))]
            video_box = peo_boxes[peo_area.index(max(peo_area))]
            while camera_res is None:
                try:
                    camera_res = json.load(open("temp.json", "r"))
                except:
                    pass
            camera_KPT = camera_res[2][0][0]
            camera_box = camera_res[1][0]
            # camera_KPT = skel_list[peo_area.index(min(peo_area))]
            video_vec = np.array(video_KPT)[5:17, 0:2]
            camera_vec = np.array(camera_KPT)[5:17, 0:2]
            for i in range(len(video_vec)):
                video_vec[i][0] = (video_vec[i][0]-video_box[0])/(video_box[2] - video_box[0])
                video_vec[i][1] = (video_vec[i][1] - video_box[1]) / (video_box[3] - video_box[1])
                camera_vec[i][0] = (camera_vec[i][0] - camera_box[0]) / (camera_box[2] - camera_box[0])
                camera_vec[i][1] = (camera_vec[i][1] - camera_box[1]) / (camera_box[3] - camera_box[1])
            video_vec = video_vec.reshape(-1)
            camera_vec = camera_vec.reshape(-1)
            cos_sim = video_vec.dot(camera_vec) / (np.linalg.norm(video_vec) * np.linalg.norm(camera_vec))
            print(cos_sim)
            if cos_sim >= therhold:
                break


t = threading.Thread(target=camera_thread, args=())
t.setDaemon(True)
t.start()
while True:
    video_player("taiji.mp4")
