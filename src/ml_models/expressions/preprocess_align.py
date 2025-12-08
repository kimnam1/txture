import os
import argparse
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Error: MediaPipe is not installed, please install first: pip install mediapipe")
    exit(1)


class FaceAligner:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def align_face(self, image):
        """Face alignment based on MediaPipe"""
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return image
        
        landmarks = results.multi_face_landmarks[0]
        h, w = image_rgb.shape[:2]
        
        # Extract eye landmarks
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[362]
        
        left_eye_center = np.array([left_eye.x * w, left_eye.y * h])
        right_eye_center = np.array([right_eye.x * w, right_eye.y * h])
        
        # Compute angle between the eyes
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Compute eye center point
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        
        # Apply affine transformation
        aligned = cv2.warpAffine(image_rgb, M, (w, h))
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(aligned)


def process_dataset(input_dir, output_dir):
    """Process the entire dataset"""
    aligner = FaceAligner()
    
    # Check whether there are train/test subdirectories
    subdirs = ['train', 'test', 'Train', 'Test']
    has_subdirs = any(os.path.isdir(os.path.join(input_dir, subdir)) for subdir in subdirs)
    
    total_processed = 0
    total_failed = 0
    
    if has_subdirs:
        # Process train/test structure
        for subdir in ['train', 'test', 'Train', 'Test']:
            subdir_path = os.path.join(input_dir, subdir)
            if not os.path.exists(subdir_path):
                continue
                
            print(f"\nProcessing subdirectory: {subdir}")
            
            # Get all classes in this subdirectory
            classes = [d for d in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, d))]
            print(f"Found classes: {classes}")
            
            for class_name in classes:
                input_class_dir = os.path.join(subdir_path, class_name)
                output_class_dir = os.path.join(output_dir, subdir, class_name)
                
                # Create output directory
                os.makedirs(output_class_dir, exist_ok=True)
                
                # Get all images in this class
                image_files = [f for f in os.listdir(input_class_dir) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                print(f"Processing class: {class_name} ({len(image_files)} images)")
                
                class_processed = 0
                class_failed = 0
                
                for image_file in tqdm(image_files, desc=f"Processing {subdir}/{class_name}"):
                    input_path = os.path.join(input_class_dir, image_file)
                    output_path = os.path.join(output_class_dir, image_file)
                    
                    # Read image
                    image = Image.open(input_path).convert('RGB')
                    
                    # Face alignment
                    aligned_image = aligner.align_face(image)
                    
                    # Save aligned image
                    aligned_image.save(output_path)
                    
                    class_processed += 1
                    total_processed += 1
                
                print(f"{subdir}/{class_name}: Successfully aligned {class_processed} images")
    
    else:
        # Directly process class directory structure
        classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        print(f"Found classes: {classes}")
        
        for class_name in classes:
            input_class_dir = os.path.join(input_dir, class_name)
            output_class_dir = os.path.join(output_dir, class_name)
            
            # Create output directory
            os.makedirs(output_class_dir, exist_ok=True)
            
            # Get all images in this class
            image_files = [f for f in os.listdir(input_class_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"Processing class: {class_name} ({len(image_files)} images)")
            
            class_processed = 0
            
            for image_file in tqdm(image_files, desc=f"Processing {class_name}"):
                input_path = os.path.join(input_class_dir, image_file)
                output_path = os.path.join(output_class_dir, image_file)
                
                # Read image
                image = Image.open(input_path).convert('RGB')
                
                # Face alignment
                aligned_image = aligner.align_face(image)
                
                # Save aligned image
                aligned_image.save(output_path)
                
                class_processed += 1
                total_processed += 1
            
            print(f"{class_name}: Successfully aligned {class_processed} images")
    
    print(f"\nTotal: Successfully aligned {total_processed} images")
    print(f"Preprocessing completed! Aligned data saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Face alignment preprocessing tool')
    parser.add_argument('--input', type=str, required=True, help='Input dataset path')
    parser.add_argument('--output', type=str, required=True, help='Output dataset path')
    args = parser.parse_args()
    
    print(f"Input dataset: {args.input}")
    print(f"Output dataset: {args.output}")
    print("Starting preprocessing...")
    
    process_dataset(args.input, args.output)


if __name__ == '__main__':
    main()
