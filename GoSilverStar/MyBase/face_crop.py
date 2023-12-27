from facenet_pytorch import MTCNN
import cv2
import os
from PIL import Image
import torch

class AdvancedFaceDetector:
    def __init__(self):
        self.detector = MTCNN(keep_all=True, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def detect_and_crop(self, image):
        # Detect faces
        boxes, _ = self.detector.detect(image)

        # If a face is detected, crop the image around the first face
        if boxes is not None and len(boxes) > 0:
            x, y, w, h = boxes[0]
            cropped_image = image.crop((x, y, w, h))
            return cropped_image

        # If no face is detected, return None
        return None

# Image path and output directory
image_path = 'C:/Users/82109/Downloads/MyBase/mask1.jpg'
output_directory = 'C:/Users/82109/Downloads/MyBase/output'

def test_face_detection_and_save(image_path, output_directory):
    # Load the image using PIL
    image = Image.open(image_path)

    # Initialize the advanced face detector
    face_detector = AdvancedFaceDetector()

    # Detect and crop the face
    cropped_image = face_detector.detect_and_crop(image)

    if cropped_image is not None:
        # Ensure the output directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Create the output file path
        output_file_path = os.path.join(output_directory, 'cropped_face.jpg')

        # Save the cropped image
        cropped_image.save(output_file_path)
        print(f"Saved cropped image to {output_file_path}")
    else:
        print("No face detected in the image.")

# Run the test and save the output
test_face_detection_and_save(image_path, output_directory)
