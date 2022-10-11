import shutil
import sys
import os
import subprocess
import traceback
from time import sleep

from PyQt5.QtCore import QThread, pyqtSignal
import ui.fitness as fitness
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog


class MainDialog(QDialog):
    progressChanged = pyqtSignal(int)
    progressChanged1 = pyqtSignal(str)
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)
        self.ui = fitness.Ui_fitness_helper()
        self.ui.setupUi(self)
        self.progressChanged.connect(self.ui.progressBar.setValue)
        self.progressChanged1.connect(self.ui.textBrowser.append)
        self.video_thread = video_process('', self.progressChanged1)
        self.info_thread = info_process(self.ui.textBrowser, self.progressChanged)


    def accept(self):
        file_path = self.ui.file_path_input.text()
        self.ui.textBrowser.append("开始处理:%s\n" % file_path)
        self.video_thread.file_path = file_path
        self.video_thread.start()
        self.info_thread.start()


    def reject(self):
        self.ui.textBrowser.setText("")
        self.ui.file_path_input.setText("")
        pass

    def browser(self):
        file_path, file_type = QFileDialog.getOpenFileName(self, '浏览文件')
        self.ui.file_path_input.setText(file_path)
        self.ui.file_path_input.selectAll()
        self.ui.file_path_input.setFocus()


class video_process(QThread):
    def __init__(self, file_path, info_box):
        super(video_process, self).__init__()
        self.file_path = file_path
        self.info_box = info_box

    def run(self):
        video_path = self.file_path
        shell = r'python python/det_keypoint_unite_infer.py --det_model_dir=model/picodet_v2_s_192_pedestrian ' \
                r'--keypoint_model_dir=model/tinypose_128x96 --save_res=True --video_file=%s --device=GPU' % video_path
        self.info_box.emit("开始处理视频，请耐心等待……")
        subprocess.run(shell, stdout=open('log.txt', 'w'))
        self.info_box.emit("视频处理完成")
        video_name = os.path.basename(video_path)[:os.path.basename(video_path).rfind('.')]
        target_path = os.path.join("video", video_name)
        os.mkdir(target_path)
        self.info_box.emit("视频复制完成")
        shutil.copy(video_path, os.path.join(target_path, os.path.basename(video_path)))
        shutil.copy("det_keypoint_unite_video_results.json", os.path.join(target_path, video_name + ".json"))
        os.remove("det_keypoint_unite_video_results.json")
        self.info_box.emit("Json复制完成")
        self.info_box.emit("视频与Json保存于%s" % target_path)


class info_process(QThread):
    def __init__(self, info_box, progressBar):
        super().__init__()
        self.info_box = info_box
        self.progressBar = progressBar

    def run(self):
        self.progressBar.emit(0)
        last_line = -1
        frame_all = None
        frame_now = None
        while True:
            try:
                fp = open("log.txt", 'r', encoding="utf-8")
                info = fp.readlines()
                fp.close()
                for i, line in enumerate(info):
                    if i < last_line:
                        continue
                    if line[0:3] == 'fps':
                        print(line[line.find("frame_count:")+13:-1])
                        frame_all = eval(line[line.find("frame_count:")+13:-1])
                    if line[0:13] =="detect frame:":
                        print(line[14:-1])
                        frame_now = eval(line[14:-1])
                    if frame_now and frame_all:
                        self.progressBar.emit(int(frame_now/frame_all*100))
                    if int(frame_now / frame_all * 100) == 100:
                        break
            except Exception as e:
                traceback.print_exc()
                pass
            sleep(1)


if __name__ == '__main__':
    myapp = QApplication(sys.argv)
    myDlg = MainDialog()
    myDlg.show()
    sys.exit(myapp.exec_())
