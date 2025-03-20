from django.shortcuts import render
import os
import cv2
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Base directory where model files are stored
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load pre-trained models
faceProto = os.path.join(BASE_DIR, "opencv_face_detector.pbtxt")
faceModel = os.path.join(BASE_DIR, "opencv_face_detector_uint8.pb")
ageProto = os.path.join(BASE_DIR, "age_deploy.prototxt")
ageModel = os.path.join(BASE_DIR, "age_net.caffemodel")
genderProto = os.path.join(BASE_DIR, "gender_deploy.prototxt")
genderModel = os.path.join(BASE_DIR, "gender_net.caffemodel")

# Model configurations
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Padding factor for better face detection
FACE_PADDING_RATIO = 0.20  # 20% padding around the face

# Load models
try:
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    models_loaded = True
    logger.info("Models loaded successfully")
except Exception as e:
    models_loaded = False
    logger.error(f"Error loading models: {str(e)}")

def index(request):
    """Render the main page with the webcam interface."""
    return render(request, 'index.html')

def get_face_with_padding(frame, x1, y1, x2, y2, padding_ratio=FACE_PADDING_RATIO):
    """Extract face with padding for better age/gender prediction."""
    height, width = frame.shape[:2]
    
    # Calculate padding
    pad_w = int((x2 - x1) * padding_ratio)
    pad_h = int((y2 - y1) * padding_ratio)
    
    # Apply padding with boundary checks
    face_x1 = max(0, x1 - pad_w)
    face_y1 = max(0, y1 - pad_h)
    face_x2 = min(width, x2 + pad_w)
    face_y2 = min(height, y2 + pad_h)
    
    # Extract padded face region
    return frame[face_y1:face_y2, face_x1:face_x2]

def preprocess_face_for_age(face):
    """Apply specific preprocessing for age detection."""
    # Ensure the face is not empty
    if face.size == 0:
        return None
        
    # Convert to grayscale for better age prediction
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to improve contrast
    equalized_face = cv2.equalizeHist(gray_face)
    
    # Convert back to BGR for the model
    enhanced_face = cv2.cvtColor(equalized_face, cv2.COLOR_GRAY2BGR)
    
    # Apply slight Gaussian blur to reduce noise
    enhanced_face = cv2.GaussianBlur(enhanced_face, (3, 3), 0)
    
    return enhanced_face

@csrf_exempt
def detect(request):
    """Detect age and gender from uploaded image."""
    if not models_loaded:
        return JsonResponse({"error": "Models not loaded properly"})
        
    if request.method == 'POST':
        file = request.FILES.get('image')
        if not file:
            logger.warning("No image file uploaded")
            return JsonResponse({"error": "No image uploaded"})

        try:
            # Read and decode image
            np_arr = np.frombuffer(file.read(), np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Check if image was loaded correctly
            if frame is None:
                logger.warning("Invalid image format")
                return JsonResponse({"error": "Invalid image format"})

            # Detect faces
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), 
                                        [104, 117, 123], True, False)
            faceNet.setInput(blob)
            detections = faceNet.forward()

            # Track the best face detection (highest confidence)
            best_detection = None
            highest_confidence = -1
            
            # Process all detections to find the best one
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # Only process faces with confidence above threshold
                if confidence > 0.5 and confidence > highest_confidence:
                    # Get coordinates of face
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], 
                                                              frame.shape[0], 
                                                              frame.shape[1], 
                                                              frame.shape[0]])
                    (x1, y1, x2, y2) = box.astype("int")
                    
                    # Ensure detection is within image boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    # Extract face with padding for better prediction
                    face = get_face_with_padding(frame, x1, y1, x2, y2)
                    
                    # Skip if extracted face is empty
                    if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
                        logger.warning(f"Empty face detected at coords: ({x1},{y1})-({x2},{y2})")
                        continue
                        
                    # We found a better detection
                    highest_confidence = confidence
                    
                    # Store coordinates for later processing
                    best_detection = {
                        'face': face,
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'confidence': confidence
                    }
            
            # Process the best detection if found
            if best_detection:
                # Access stored data
                face = best_detection['face']
                x1 = best_detection['x1']
                y1 = best_detection['y1']
                x2 = best_detection['x2']
                y2 = best_detection['y2']
                confidence = best_detection['confidence']
                
                # Make a copy for gender prediction (original color)
                gender_face = face.copy()
                
                # Preprocess face specifically for age detection
                enhanced_face = preprocess_face_for_age(face)
                if enhanced_face is None:
                    return JsonResponse({"error": "Could not process face for age detection"})

                # Prepare blob for gender classification (use original face)
                gender_blob = cv2.dnn.blobFromImage(gender_face, 1.0, (227, 227), 
                                               MODEL_MEAN_VALUES, swapRB=False)
                
                # Predict gender
                genderNet.setInput(gender_blob)
                genderPreds = genderNet.forward()
                gender_idx = genderPreds[0].argmax()
                gender = genderList[gender_idx]
                gender_confidence = float(genderPreds[0].max()) * 100
                
                # Prepare blob for age classification (use enhanced face)
                age_blob = cv2.dnn.blobFromImage(enhanced_face, 1.0, (227, 227), 
                                            MODEL_MEAN_VALUES, swapRB=False)
                
                # Predict age
                ageNet.setInput(age_blob)
                agePreds = ageNet.forward()
                
                # Calculate all age confidences for better analysis
                all_age_confidences = []
                for idx, conf in enumerate(agePreds[0]):
                    all_age_confidences.append({
                        "age_group": ageList[idx],
                        "confidence": f"{float(conf) * 100:.2f}%"
                    })
                
                # Get the highest confidence age
                age_idx = agePreds[0].argmax()
                age = ageList[age_idx]
                age_confidence = float(agePreds[0].max()) * 100
                
                # Get second highest confidence for age (for borderline cases)
                sorted_age_indices = np.argsort(agePreds[0])[::-1]
                second_age_idx = sorted_age_indices[1] if len(sorted_age_indices) > 1 else age_idx
                second_age = ageList[second_age_idx]
                second_age_confidence = float(agePreds[0][second_age_idx]) * 100
                
                # Create the single result
                result = {
                    "gender": gender,
                    "age": age,
                    "gender_confidence": f"{gender_confidence:.2f}%",
                    "age_confidence": f"{age_confidence:.2f}%",
                    "second_prediction": {
                        "age": second_age,
                        "confidence": f"{second_age_confidence:.2f}%"
                    },
                    "all_age_confidences": all_age_confidences,
                    "face_confidence": f"{confidence * 100:.2f}%",
                    "face_box": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    }
                }
                
                logger.info("Successfully detected primary face")
                return JsonResponse({"detections": [result]})  # Return as array for frontend compatibility
            else:
                logger.warning("No faces detected in the image")
                return JsonResponse({"error": "No faces detected"})
                
        except Exception as e:
            logger.error(f"Error during image processing: {str(e)}")
            return JsonResponse({"error": f"Processing error: {str(e)}"})

    return JsonResponse({"error": "Invalid request method"})
