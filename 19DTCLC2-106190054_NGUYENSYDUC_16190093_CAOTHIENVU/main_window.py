import sys

from outwindow2 import Ui_MainWindow
import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from main_layout_du_an import Ui_MainWindow
import mediapipe as mp
import tensorflow
import pyttsx3
import threading
import time
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


from DEF_import import *
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)

        self.uic.Button_start.clicked.connect(self.start_capture_video)
        self.uic.Button_stop.clicked.connect(self.stop_capture_video)

        self.thread = {}

        #Nút trở về
        self.uic.Button_return.clicked.connect(self.return_menu)
        #Nút cchon du lieu
        self.uic.Button_input.clicked.connect(self.input_video)
        #Nút output
        self.uic.Button_close.clicked.connect(self.close)
        #Nút output
        self.uic.button_output.clicked.connect(self.start_video)
        #Nút logon
        self.uic.btn_login.clicked.connect(self.login)

    def login(self):
        self.uic.stackedWidget.setCurrentWidget(self.uic.display_main)

    def return_menu(self):
        self.uic.stackedWidget.setCurrentWidget(self.uic.menu)


    #Hàm input video
    def input_video(self):
        print("Hello")
        link_data = QFileDialog.getOpenFileName(filter="*.mp4 *.mkv")
        f = open("input_video.txt",mode = "w",encoding="utf-8-sig")
        f.write(f"{link_data[0]}")
        f.close()
    def closeEvent(self, event):
        self.stop_capture_video()

    def stop_capture_video(self):
        if 1 in self.thread:
            self.thread[1].stop()

    def start_capture_video(self):
        f = open("input_video.txt",mode="w",encoding="utf-8-sig")
        f.write("0")
        f.close()

        self.thread[1] = CaptureVideoThread(index=1)
        self.thread[1].start()
        self.thread[1].signal.connect(self.show_webcam)
        self.thread[1].signal_2.connect(self.show_text)
    def start_video(self):
        self.thread[1] = CaptureVideoThread(index=1)
        self.thread[1].start()
        self.thread[1].signal.connect(self.show_webcam)
        self.thread[1].signal_2.connect(self.show_text)
    def show_text(self,mang):
        print("data la ",mang)
        if len(mang) ==0:
            self.uic.DU_DOAN.setText(f"")
        else:
            self.uic.DU_DOAN.setText(f"{mang[-1]}")

    def show_webcam(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.uic.Screen.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(681, 700, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class CaptureVideoThread(QThread):
    signal = pyqtSignal(np.ndarray)
    signal_2 = pyqtSignal(object)
    def __init__(self, index):
        self.index = index
        super(CaptureVideoThread, self).__init__()
        # self._run_flag = True
        self.gg = True
    def stop(self):
        self.gg = False
        self.wait()
        cv2.destroyAllWindows()
        self.terminate()  # Hàm dừng luồng

    def run(self):
        def speak(text):
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[1].id)
            engine.say(text)
            try:
                engine.runAndWait()
            except Exception as ep:
                print(ep)

        f = open("input_video.txt", mode="r", encoding="utf-8-sig")
        nguon = f.readline()
        print("Nguồn video là", nguon)

        if nguon == "0" or nguon == "1":
            nguon = int(nguon)
        else:
            pass
        f.close()
        cap = cv2.VideoCapture(nguon)  # 'D:/8.Record video/My Video.mp4'
        pTime = 0
        colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 117, 16), (15, 117, 245), (244, 117, 16),
                  (25, 117, 245), (240, 117, 16), (116, 117, 245), (235, 117, 16)]
        actions = np.array(['Hello', 'How are you', 'Thanks', 'My name is', 'Bye', 'A', 'B', 'C', 'D', 'V', 'U'])
        sequence = []
        sentence = []
        threshold = 0.8
        text = " "
        cTime,pTime = 0,0

        model = tensorflow.keras.models.load_model('LSTM.h5')
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                # Đọc camera
                ret, frame = cap.read()
                # Lấy toạ độ các điểm
                image, results = mediapipe_detection(frame, holistic)
                # print(results)
                # Vẽ các điểm lên cơ thể
                draw_styled_landmarks(image, results)
                keypoints = extract_keypoints(results)
                # print(keypoints)
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(image, "FPS:" + str(int(fps)), (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 190), 2,
                            cv2.LINE_AA)
                if sum(keypoints) != 0:
                    sequence.append(keypoints)
                    sequence = sequence[-30:]
                #
                else:
                    sequence.clear()
                    print(len(sequence))
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    # print(actions[np.argmax(res)])

                            #3. Viz logic
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                threading.Thread(target=speak, args=(actions[np.argmax(res)],)).start()

                        else:
                            sentence.append(actions[np.argmax(res)])
                            threading.Thread(target=speak, args=(actions[np.argmax(res)],)).start()

                    if len(sentence) > 3:
                        sentence = sentence[-3:]
                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, text.join(sentence), (3, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)




                self.signal.emit(image)
                # cv2.imshow("Camera",image)
                # if cv2.waitKey(1) == ord("q"):
                #     break
                # if sentence[0] != "" and len(sentence) > 1:
                self.signal_2.emit(sentence)
                if self.gg == False:
                    break

        cap.release()
        cv2.destroyAllWindows()
        # self.quit()
        # self.wait()






if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())





