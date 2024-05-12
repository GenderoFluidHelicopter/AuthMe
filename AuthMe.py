from PyQt6.QtCore import QSize, Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtWidgets import QWidget, QLabel, QGridLayout, QApplication, QMainWindow, QPushButton, QMessageBox, QVBoxLayout, QFileDialog
from PyQt6 import QtGui
from PyQt6.QtGui import QPixmap
import sys 
import cv2 as cv
import numpy as np
import sys
import os
import shutil
from FR import FaceRecognition
from Preparer import ImageProcessor
from Processor import TextProcessor
from PDF import PDF2JPG
from Cleaner import Cleaner

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.run_flag = False
        self.last_frame = None
        self.camera = cv.VideoCapture(0)

    def run(self):
        self.run_flag = True
        while self.run_flag:
            ret, cv_img = self.camera.read()
            if ret:
                self.last_frame = cv_img
                self.change_pixmap_signal.emit(cv_img)

    def stop(self):
        self.run_flag = False
        self.wait()
        self.camera.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.im1_width = 800
        self.im1_height = 600

        self.setWindowTitle("AuthMe")
        self.setFixedSize(QSize(1820, 1080))

        self.new_button = QPushButton("Включить поток с камеры")
        self.new_button.setCheckable(True)
        self.new_button.clicked.connect(self.new_clicked)

        self.stop_button = QPushButton("Сделать снимок с камеры")
        self.stop_button.setCheckable(True)
        self.stop_button.clicked.connect(self.stop_clicked)

        self.quit_button = QPushButton("Выход")
        self.quit_button.setCheckable(True)
        self.quit_button.clicked.connect(self.quit_clicked)

        self.pdf_button = QPushButton("Конвертировать пдф")
        self.pdf_button.setCheckable(True)
        self.pdf_button.clicked.connect(self.pdf_clicked)

        self.prepare_fr_button = QPushButton("Подготовить кандидатов")
        self.prepare_fr_button.clicked.connect(self.prepare_fr_clicked)

        self.evaluate_button = QPushButton("Проверка")
        self.evaluate_button.clicked.connect(self.evaluate_clicked)
        
        self.clean_button = QPushButton("Очистить папки")
        self.clean_button.setCheckable(True)
        self.clean_button.clicked.connect(self.clean_clicked)

        self.select_folder_button = QPushButton("Выбрать папку")
        self.select_folder_button.clicked.connect(self.select_folder)

        self.im_cam_label = QLabel(self)
        self.im_cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.candidate_image_label = QLabel(self)
        self.candidate_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.candidate_image_label.resize(600, 800)

        self.fio_label = QLabel(self)

        camera_image_layout = QVBoxLayout()
        candidate_image_layout = QVBoxLayout()
        camera_image_layout.addWidget(self.im_cam_label)
        candidate_image_layout.addWidget(self.candidate_image_label)

        layout = QGridLayout()
        layout.addLayout(camera_image_layout, 0, 0, 1, 2)
        layout.addLayout(candidate_image_layout, 0, 2, 1, 2)

        layout.addWidget(self.fio_label, 1, 0, 1, 4)
        self.fio_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fio_label.setFixedHeight(60)
        self.fio_label.setFont(QtGui.QFont("Arial", 20))

        layout.addWidget(self.new_button, 2, 0)
        layout.addWidget(self.stop_button, 2, 1)
        layout.addWidget(self.quit_button, 2, 2)
        layout.addWidget(self.prepare_fr_button, 2, 3)
        layout.addWidget(self.evaluate_button, 3, 1)
        layout.addWidget(self.pdf_button, 3, 2)
        layout.addWidget(self.clean_button, 3, 3)
        layout.addWidget(self.select_folder_button, 3, 0)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.camera_thread = VideoThread()

        self.cleaner = Cleaner()
        self.pdf_processor = PDF2JPG()
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.face_recognition = FaceRecognition()  

    def closeEvent(self, event):
        self.camera_thread.stop()
        self.camera_thread.camera.release()
        event.accept()
        
    def new_clicked(self):
        if not self.camera_thread.isRunning():
            self.camera_thread.change_pixmap_signal.connect(self.update_image)
            self.camera_thread.start()
            self.new_button.setDisabled(True)
            self.stop_button.setEnabled(True)
            print("New button!")

    def stop_clicked(self):
        if self.camera_thread.isRunning():
            # self.new_button.setEnabled(True)
            # self.stop_button.setDisabled(True)
            ret, cv_img = self.camera_thread.camera.read()
            if ret:
                self.last_frame = cv_img
                if self.last_frame is not None:
                    cv.imwrite('./data/ex00.jpg', self.last_frame)
                    # self.camera_thread.stop()
                    # self.camera_thread.camera.release()
            print("Stop button!")

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Выбрать папку", ".")
        if folder_path:
            print("Выбранная папка:", folder_path)
            destination_folder = "./set"

            try:
                for file_name in os.listdir(folder_path):
                    source_file_path = os.path.join(folder_path, file_name)
                    if os.path.isfile(source_file_path):
                        destination_file_path = os.path.join(destination_folder, file_name)
                        shutil.copy(source_file_path, destination_file_path)
                print("Файлы скопированы успешно.")
            except Exception as e:
                print("Ошибка при копировании файлов:", e)

    def quit_clicked(self):
        self.stop_clicked()
        self.close()
        print("Quit!")
    
    def pdf_clicked(self):
        try:
            self.pdf_processor.convert_pdf_to_images()
            print("PDFs ready!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during PDF processing: {e}")

    def clean_clicked(self):
        try:
            self.cleaner.delete_folder_contents()
            print("All folders empty!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during FOLDERS processing: {e}")

    def prepare_fr_clicked(self):
        try:
            self.image_processor.yolo_segment()
            self.image_processor.process_image_with_mtcnn()
            self.image_processor.yolo_predict()
            self.image_processor.apply_lanczos_filter()
            print("Images processed successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during IMAGE processing: {e}")

        try:
            self.text_processor.process_text()
            print("Text processed successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during TEXT processing: {e}")
        
        try:
            self.face_recognition.prepare_embeddings()
            self.save_embeddings()
            print("Face Recognition embeddings prepared.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during FR processing: {e}")

    def evaluate_clicked(self):
        try:
            img = cv.imread('./data/ex00.jpg')
            if img is not None:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img_cropped = self.face_recognition.mtcnn(img)
                if img_cropped is not None:
                    img_cropped = img_cropped.to(self.face_recognition.device)
                    img_embedding = self.face_recognition.resnet(img_cropped.unsqueeze(0)).detach().cpu().numpy()
                    self.load_embeddings()
                    
                    best_candidate, cosine_similarity, euclidean_similarity, manhattan_similarity = self.face_recognition.calculate_similarity(img_embedding)
                    
                    if best_candidate is not None:
                        print(f"The best candidate is {best_candidate} with a cosine similarity of {cosine_similarity:.2f}, Euclidean similarity of {euclidean_similarity:.2f}, and Manhattan similarity of {manhattan_similarity:.2f}")

                        # Загрузка изображения кандидата
                        candidate_image_path = os.path.join('./rotated', best_candidate)
                        pixmap = QPixmap(candidate_image_path)
                        pixmap = pixmap.scaled(600, 800, Qt.AspectRatioMode.KeepAspectRatio)
                        self.candidate_image_label.setPixmap(pixmap)

                        other_file_path = os.path.join('./processed_text', os.path.splitext(best_candidate)[0] + '.txt')
                    if os.path.exists(other_file_path):
                        with open(other_file_path, 'r', encoding='utf-8') as file:
                            fio = file.read().strip()
                            self.fio_label.setText(f"Перед Вами {fio}")
                    else:
                        self.fio_label.setText("Не удалось найти соответствующий файл с ФИО в другой папке.")
                                
                else:
                    print("No candidate found.")
            else:
                print("No face detected in the input image.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.im_cam_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        # convert from an opencv image to QPixmap
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.im1_width, self.im1_height, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    def save_embeddings(self):
        self.face_recognition.save_embeddings('./embeddings/embeddings.pkl')

    def load_embeddings(self):
        try:
            self.face_recognition.load_embeddings('./embeddings/embeddings.pkl')
            print("Embeddings loaded successfully.")
        except Exception as e:
            print(f"Error loading embeddings: {e}")

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())