import cv2
from PIL import Image, ImageEnhance
import os
from ultralytics import YOLO
import pathlib
import numpy as np
from facenet_pytorch import MTCNN
import torch

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class ImageProcessor:
    def __init__(self):
        self.input_folder = './dataset'
        self.out_fol = './segmented'
        self.out_fol1 = './rotated'
        self.output_folder = './cropped'
        self.output_folder1 = './prepared'
        self.model1 = YOLO('./data/best1.pt')
        self.model2 = YOLO('./data/best-seg.pt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=self.device)

    def yolo_segment(self):
        if not os.path.exists(self.out_fol):
            os.makedirs(self.out_fol)
        
        for file_name in os.listdir(self.input_folder):
            if file_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                img_path = os.path.join(self.input_folder, file_name)
                img = cv2.imread(img_path)
                
                prediction = self.model2(img)
                boxes = prediction[0].boxes.xyxy.tolist()
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
                    cv2.imwrite(os.path.join(self.out_fol, file_name), cropped_img)
                else:
                    print(f"Object detected in {file_name}")

    def process_image_with_mtcnn(self):
        if not os.path.exists(self.out_fol1):
            os.makedirs(self.out_fol1)
        num_processed = 0
        num_saved = 0
    
        for filename in os.listdir(self.out_fol):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.out_fol, filename)
                output_path = os.path.join(self.out_fol1, filename)
                
                img = Image.open(image_path)
                
                rotation_angle = 0
                
                while rotation_angle < 360:
                    rotated_img = img.rotate(rotation_angle, expand=True)
                    
                    rotated_img_np = np.array(rotated_img)
                    
                    height, width = rotated_img_np.shape[:2]
                    quadrant = rotated_img_np[height//2:, :width//2]
                    
                    boxes, probabilities = self.detector.detect(quadrant)
                    
                    if boxes is not None and len(boxes) > 0:
                        rotated_img.save(output_path)
                        num_saved += 1
                        print(f"Processed {filename}, saved at {output_path}")
                        break
                    
                    rotation_angle += 90
                    
                if rotation_angle == 360:
                    print(f"No faces detected in {filename}, skipping...")
                    
                num_processed += 1
        
        print(f"Processed {num_processed} images, saved {num_saved} images.")

    def yolo_predict(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        for file_name in os.listdir(self.out_fol1):
            if file_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                img_path = os.path.join(self.out_fol1, file_name)
                img = cv2.imread(img_path)
                
                prediction = self.model1(img)
                boxes = prediction[0].boxes.xyxy.tolist()
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
                    cv2.imwrite(os.path.join(self.output_folder, file_name), cropped_img)
                else:
                    print(f"Object detected in {file_name}")

    def apply_lanczos_filter(self, contrast_factor=3.0, brightness_factor=1.1):
        if not os.path.exists(self.output_folder1):
            os.makedirs(self.output_folder1)
        
        for file_name in os.listdir(self.output_folder):
            input_image_path = os.path.join(self.output_folder, file_name)
            img = Image.open(input_image_path)

            img_grayscale = img.convert('L')

            brightness_enhancer = ImageEnhance.Brightness(img_grayscale)
            img_brightened = brightness_enhancer.enhance(brightness_factor)

            contrast_enhancer = ImageEnhance.Contrast(img_brightened)
            img_contrasted = contrast_enhancer.enhance(contrast_factor)

            output_image_path = os.path.join(self.output_folder1, file_name)
            img_contrasted.save(output_image_path)