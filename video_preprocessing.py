import shutil
import subprocess
import os
video_path = r"C:\Users\boyif\Desktop\paddle\fitness_helper\FormatFactoryPart26.mp4"
shell = r'python python/det_keypoint_unite_infer.py --det_model_dir=model/picodet_v2_s_192_pedestrian ' \
        r'--keypoint_model_dir=model/tinypose_128x96 --save_res=True --video_file=%s --device=GPU' % video_path
subprocess.run(shell)
print("视频处理完成")
video_name = os.path.basename(video_path)[:os.path.basename(video_path).rfind('.')]
target_path = os.path.join("video", video_name)
os.mkdir(target_path)
shutil.copy(video_path, os.path.join(target_path, os.path.basename(video_path)))
shutil.copy("det_keypoint_unite_video_results.json", os.path.join(target_path, video_name+".json"))
os.remove("det_keypoint_unite_video_results.json")
print("json处理完成")