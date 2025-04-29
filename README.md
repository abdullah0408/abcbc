🧠 Image Classification API with FastAPI & TensorFlow
This project implements an image classification pipeline using a transfer learning model (MobileNetV2) with TensorFlow and serves predictions via a FastAPI REST API.

🚀 Features
Transfer learning using MobileNetV2

Image preprocessing and augmentation

Train/validation dataset pipeline using tf.data

FastAPI-based endpoint for image prediction

Model checkpointing and early stopping

📁 Project Structure
bash
Copy
Edit
.
├── app.py                  # FastAPI app with prediction endpoint
├── train.py                # Training script
├── models/
│   └── best_model.h5       # Saved Keras model
├── data/
│   ├── metadata.csv        # CSV with image paths and labels
│   └── images/             # Folder containing image files
├── src/
│   ├── data_loader.py      # Data loading and preprocessing
│   └── model_tl.py         # Model architecture (Transfer Learning)
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
🧪 Setup & Installation
Clone the repository

bash
Copy
Edit
git clone https://github.com/yourusername/image-classifier-api.git
cd image-classifier-api
Create a virtual environment and install dependencies

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
Prepare the dataset

Place your images in data/images/

Ensure metadata.csv contains:

csv
Copy
Edit
image_id,label
img001,0
img002,1
...
🏋️‍♂️ Train the Model
Run the training script:

bash
Copy
Edit
python train.py
The model will be saved to models/best_model.h5.

🔍 Run the Prediction API
bash
Copy
Edit
uvicorn app:app --reload
Endpoint
POST /predict

Upload a .jpg image file.

Example (using curl):
bash
Copy
Edit
curl -X POST "http://localhost:8000/predict" \
  -H  "accept: application/json" \
  -H  "Content-Type: multipart/form-data" \
  -F "file=@example.jpg"
Response:

json
Copy
Edit
{
  "prediction": 1
}
📦 Requirements
See requirements.txt. Key dependencies:

fastapi

uvicorn

tensorflow

pillow

numpy

pandas

scikit-learn

📌 Notes
The model expects input images of shape (224, 224, 3) normalized to [0, 1].

Make sure labels in metadata.csv are numerically encoded (0, 1, etc.)

📜 License
This project is licensed under the MIT License.