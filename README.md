# Gender and Age Detection System

This Django-based application uses OpenCV and pre-trained deep learning models to detect faces in images and predict their gender and age.

## Features

- Real-time face detection through webcam
- Single-subject focus (detects the most prominent face)
- Gender classification (Male/Female)
- Enhanced age group prediction with confidence metrics
- Detailed confidence scores for all predictions
- Visual confidence indicators
- Secondary age predictions for borderline cases
- User-friendly interface with real-time feedback
- Webcam controls (start/stop) for better privacy

## Requirements

- Python 3.8+
- Django 5+
- OpenCV 4.5+
- NumPy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/patildhrup/gender_age_detection.git
cd gender_age_detection
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the dependencies:
```bash
pip install django opencv-python numpy
```

4. Run the development server:
```bash
python manage.py runserver
```

5. Open your browser and navigate to http://127.0.0.1:8000/

## Usage

1. Click the "Start Webcam" button and allow browser access to your webcam when prompted
2. Position your face in the camera view (make sure your face is the most prominent in the frame)
3. Click the "Analyze Face" button to detect and analyze your face
4. View the detection results on the right side panel
   - You'll see the predicted gender and age group
   - Confidence bars indicate how confident the model is in its prediction
   - Click "Show age prediction details" to see all possible age groups and their confidence scores
5. When finished, click "Stop Webcam" to disable the camera for privacy

## How Face Detection Works

The application is designed to focus on a single individual:

- The system scans the image and identifies all faces in the frame
- Only the face with the highest detection confidence is processed
- This ensures consistent results when analyzing one person at a time
- For best results, make sure your face is well-lit and centered in the frame

## Age Prediction Details

The application provides detailed age prediction information:

- **Primary age prediction**: The age group with highest confidence
- **Secondary prediction**: Alternative age range for borderline cases
- **Confidence metrics**: Visual bars showing prediction confidence levels
- **Full prediction breakdown**: All age groups with their confidence percentages
- **Combined predictions**: For uncertain cases, the app will suggest multiple possible age groups

## Model Information

This application uses pre-trained models from OpenCV's DNN module:

- **Face Detection**: OpenCV's face detector (SSD framework with ResNet base network)
- **Gender Classification**: Trained on Adience dataset
- **Age Classification**: Trained on Adience dataset with the following age groups:
  - (0-2)
  - (4-6)
  - (8-12)
  - (15-20)
  - (25-32)
  - (38-43)
  - (48-53)
  - (60-100)

## Face Processing Improvements

The application includes several enhancements to improve age prediction accuracy:

- **Face padding**: Extracts faces with additional padding around the detected region
- **Image preprocessing**: Applies histogram equalization to improve contrast
- **Separate processing pipelines**: Uses different preprocessing for age and gender prediction
- **Confidence-based combined predictions**: Presents multiple age ranges when confidence is low

## Troubleshooting

- If no faces are detected, try adjusting your position, lighting, or camera angle
- Ensure your browser has permission to access your webcam
- For better accuracy, make sure your face is well-lit and directly facing the camera
- If the webcam fails to start, try refreshing the page or checking your browser's settings
- Age prediction may be less accurate for certain age groups or in poor lighting conditions


## Acknowledgments

- Model weights and architecture from OpenCV's open model zoo 