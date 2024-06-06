from ultralytics import YOLO
import cv2
import os
# Load a model
model = YOLO("weights//best.pt")  # load a custom model

folder = "D:\\VisDrone2019-DET-val\\images"  # folder of images
output_folder = f"{folder}_output"  # folder of images

file_list = os.listdir(folder)

# Filter out the image files
image_files = [file for file in file_list if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

# Perform inference on each image
for image_file in image_files:
    print(image_file)
    image_path = os.path.join(folder, image_file)
    output_path = os.path.join(output_folder, image_file)
    print(output_path)
    #im = cv2.imread(image_path)
    result = model.predict(image_path)
    result[0].save(output_path)