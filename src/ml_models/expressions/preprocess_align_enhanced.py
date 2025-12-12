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
    def __init__(self, face_detection_confidence=0.5, face_mesh_confidence=0.5):
        # Initialize face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range model (within 2 meters), 1 for long-range model
            min_detection_confidence=face_detection_confidence
        )
        
        # Initialize face mesh (for landmark detection and alignment)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=face_mesh_confidence
        )
    
    def detect_and_crop_face(self, image):
        """
        Detect and crop the face region
        Returns: (cropped_face_image, success_flag)
        """
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = image_rgb.shape[:2]
        
        # Use face detection to obtain face bounding box
        detection_results = self.face_detection.process(image_rgb)
        
        if not detection_results.detections:
            return image, False  # No face detected
        
        # Select the face with the highest confidence
        best_detection = max(detection_results.detections, 
                           key=lambda x: x.score[0])
        
        # Get bounding box
        bbox = best_detection.location_data.relative_bounding_box
        
        # Convert to pixel coordinates
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Expand bounding box to include more context
        # This is important for expression recognition, as a full facial area is needed
        expand_ratio = 0.3  # Expand outward by 30%
        expand_w = int(width * expand_ratio)
        expand_h = int(height * expand_ratio)
        
        # Compute expanded bounding box
        x_expanded = max(0, x - expand_w)
        y_expanded = max(0, y - expand_h)
        x2_expanded = min(w, x + width + expand_w)
        y2_expanded = min(h, y + height + expand_h)
        
        # Crop face region
        face_region = image_rgb[y_expanded:y2_expanded, x_expanded:x2_expanded]
        
        # Convert back to RGB format
        face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        cropped_face = Image.fromarray(face_rgb)
        
        return cropped_face, True
    
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
    
    def process_image(self, image):
        """
        Complete image processing pipeline: detection -> cropping -> alignment
        Returns: (processed_image, processing_info)
        """
        # Step 1: Try face detection and cropping
        cropped_face, crop_success = self.detect_and_crop_face(image)
        
        if not crop_success:
            # If detection fails, try aligning the whole image directly
            # (may already be a cropped face image)
            aligned_image = self.align_face(image)
            return aligned_image, "direct_align"
        
        # Step 2: Align the cropped face
        aligned_face = self.align_face(cropped_face)
        
        return aligned_face, "crop_and_align"


def process_dataset(input_dir, output_dir, face_detection_confidence=0.5, 
                   face_mesh_confidence=0.5, min_face_size=50):
    """Process the entire dataset"""
    aligner = FaceAligner(face_detection_confidence, face_mesh_confidence)
    
    # Check whether there are train/test subdirectories
    subdirs = ['train', 'test', 'Train', 'Test']
    has_subdirs = any(os.path.isdir(os.path.join(input_dir, subdir)) for subdir in subdirs)
    
    total_processed = 0
    total_failed = 0
    total_cropped = 0
    total_direct_aligned = 0
    
    if has_subdirs:
        # Process train/test directory structure
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
                class_cropped = 0
                class_direct = 0
                
                for image_file in tqdm(image_files, desc=f"Processing {subdir}/{class_name}"):
                    try:
                        input_path = os.path.join(input_class_dir, image_file)
                        output_path = os.path.join(output_class_dir, image_file)
                        
                        # Read image
                        image = Image.open(input_path).convert('RGB')
                        
                        # Check image size
                        if min(image.size) < min_face_size:
                            print(f"Warning: Image {image_file} too small ({image.size}), skipping...")
                            continue
                        
                        # Complete processing: detection -> cropping -> alignment
                        processed_image, process_info = aligner.process_image(image)
                        
                        # Save processed image
                        processed_image.save(output_path)
                        
                        class_processed += 1
                        total_processed += 1
                        
                        if process_info == "crop_and_align":
                            class_cropped += 1
                            total_cropped += 1
                        else:
                            class_direct += 1
                            total_direct_aligned += 1
                    
                    except Exception as e:
                        print(f"Error processing {image_file}: {e}")
                        class_failed += 1
                        total_failed += 1
                        continue
                
                print(f"{subdir}/{class_name}: Successfully processed {class_processed} images")
                print(f"  - Cropped and aligned: {class_cropped}")
                print(f"  - Direct aligned: {class_direct}")
                print(f"  - Failed: {class_failed}")
    
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
            class_failed = 0
            class_cropped = 0
            class_direct = 0
            
            for image_file in tqdm(image_files, desc=f"Processing {class_name}"):
                try:
                    input_path = os.path.join(input_class_dir, image_file)
                    output_path = os.path.join(output_class_dir, image_file)
                    
                    # Read image
                    image = Image.open(input_path).convert('RGB')
                    
                    # Check image size
                    if min(image.size) < min_face_size:
                        print(f"Warning: Image {image_file} too small ({image.size}), skipping...")
                        continue
                    
                    # Complete processing: detection -> cropping -> alignment
                    processed_image, process_info = aligner.process_image(image)
                    
                    # Save processed image
                    processed_image.save(output_path)
                    
                    class_processed += 1
                    total_processed += 1
                    
                    if process_info == "crop_and_align":
                        class_cropped += 1
                        total_cropped += 1
                    else:
                        class_direct += 1
                        total_direct_aligned += 1
                
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                    class_failed += 1
                    total_failed += 1
                    continue
            
            print(f"{class_name}: Successfully processed {class_processed} images")
            print(f"  - Cropped and aligned: {class_cropped}")
            print(f"  - Direct aligned: {class_direct}")
            print(f"  - Failed: {class_failed}")
    
    print(f"\n{'='*50}")
    print(f"Processing Summary:")
    print(f"Total processed: {total_processed} images")
    print(f"  - Cropped and aligned (full body): {total_cropped}")
    print(f"  - Direct aligned (face images): {total_direct_aligned}")
    print(f"Total failed: {total_failed}")
    print(f"{'='*50}")
    print(f"Preprocessing completed! Enhanced data saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced face alignment preprocessing tool with face detection')
    parser.add_argument('--input', type=str, required=True, help='Input dataset path')
    parser.add_argument('--output', type=str, required=True, help='Output dataset path')
    parser.add_argument('--face_detection_confidence', type=float, default=0.5, 
                       help='Face detection confidence threshold (0.0-1.0)')
    parser.add_argument('--face_mesh_confidence', type=float, default=0.5,
                       help='Face mesh detection confidence threshold (0.0-1.0)')
    parser.add_argument('--min_face_size', type=int, default=50,
                       help='Minimum face size in pixels')
    args = parser.parse_args()
    
    print(f"Enhanced Face Alignment Preprocessing")
    print(f"Input dataset: {args.input}")
    print(f"Output dataset: {args.output}")
    print(f"Face detection confidence: {args.face_detection_confidence}")
    print(f"Face mesh confidence: {args.face_mesh_confidence}")
    print(f"Minimum face size: {args.min_face_size}")
    print("Starting preprocessing...")
    print("\nFeatures:")
    print("- Automatic face detection and cropping for full body images")
    print("- Face alignment based on eye landmarks")
    print("- Graceful handling of both full body and face-only images")
    print("- Detailed processing statistics")
    print()
    
    process_dataset(
        args.input, 
        args.output,
        face_detection_confidence=args.face_detection_confidence,
        face_mesh_confidence=args.face_mesh_confidence,
        min_face_size=args.min_face_size
    )


if __name__ == '__main__':
    main()
