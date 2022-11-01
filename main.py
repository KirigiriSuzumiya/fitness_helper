import json
import cv2
import subprocess
import threading
import numpy as np
import os

base_dir = os.path.dirname(__file__)


def camera_thread():
    infer_path = os.path.join(base_dir, "python", "det_keypoint_unite_infer.py")
    det_model_dir = os.path.join(base_dir, "model", "picodet_v2_s_192_pedestrian")
    kpt_model_dir = os.path.join(base_dir, "model", "tinypose_128x96")
    shell = "python %s --det_model_dir=%s --keypoint_model_dir=%s --camera_id=0 --device=GPU --save_res=True" % \
            (infer_path, det_model_dir, kpt_model_dir)
    subprocess.run(shell)


def video_player(video_name):
    json_path = os.path.join(base_dir, 'video', video_name[:video_name.rfind('.')],
                             video_name[:video_name.rfind('.')] + ".json")
    video_path = os.path.join(base_dir, 'video', video_name[:video_name.rfind('.')], video_name)
    print(video_path)
    print(json_path)
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

        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        cv2.imshow("test", frame)
        k = cv2.waitKey(20)
        # q键退出
        if (k & 0xff == ord('q')):
            break
        im = None
        while True:
            try:
                im = np.load(os.path.join(base_dir, "python", "frame.npy"))
            except:
                continue
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
            video_kpt = np.array(video_KPT)[0:17, 0:2]
            camera_kpt = np.array(camera_KPT)[0:17, 0:2]
            cosine = []
            attention_index = []
            kpt_tuple = [[6, 8], [5, 7], [8, 10], [7, 9], [5, 6], [12, 14], [11, 13], [14, 16], [13, 15]]
            kpt_tuple_name = ["左上臂", "右上臂", "左下臂", "右下臂", "两肩", "左大腿", "右大腿", "左小腿", "右小腿"]
            kpt_tuple_name_en = ["Left upper arm", "Right upper arm", "Left lower arm", "Right lower arm",
                                 "Two shoulders", "Left thigh", "Right thigh", "Left calf", "Right calf"]
            for i in range(len(kpt_tuple)):
                kpt_index = kpt_tuple[i]
                video_vec = [video_kpt[kpt_index[0]][0] - video_kpt[kpt_index[1]][0],
                             video_kpt[kpt_index[0]][1] - video_kpt[kpt_index[1]][1]]
                camera_vec = [camera_kpt[kpt_index[0]][0] - camera_kpt[kpt_index[1]][0],
                              camera_kpt[kpt_index[0]][1] - camera_kpt[kpt_index[1]][1]]
                video_vec = np.array(video_vec)
                camera_vec = np.array(camera_vec)
                cos_sim = video_vec.dot(camera_vec) / (np.linalg.norm(video_vec) * np.linalg.norm(camera_vec))
                cosine.append(cos_sim)
                if cosine[i] >= 0.95:
                    cv2.line(im,
                             (int(camera_kpt[kpt_index[0]][0]), int(camera_kpt[kpt_index[0]][1])),
                             (int(camera_kpt[kpt_index[1]][0]), int(camera_kpt[kpt_index[1]][1])),
                             color=[0, 255, 0],
                             thickness=2)
                else:
                    cv2.line(im,
                             (int(camera_kpt[kpt_index[0]][0]), int(camera_kpt[kpt_index[0]][1])),
                             (int(camera_kpt[kpt_index[1]][0]), int(camera_kpt[kpt_index[1]][1])),
                             color=[0, 0, 255],
                             thickness=2)
                    attention_index.append(i)
            text_w = 3
            text_h = 1
            line_inter = im.shape[0] / 40.
            text_scale = max(0.5, im.shape[0] / 3000.)
            for i in attention_index:
                text_h += int(line_inter)
                text_loc = (text_w, text_h)
                cv2.putText(
                    im,
                    kpt_tuple_name_en[i],
                    text_loc,
                    cv2.FONT_ITALIC,
                    text_scale, (0, 255, 255),
                    thickness=1)
            cv2.imshow("camera rec", im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if sum(cosine) > (0.95*9):
                break




t = threading.Thread(target=camera_thread, args=())
t.setDaemon(True)
t.start()
while True:
    video_player("taiji.mp4")
