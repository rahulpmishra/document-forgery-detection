# Document Forgery Detection System

A smart AI-powered tool that can tell if document images have been edited or tampered with. Built using advanced computer vision techniques and deep learning models with an easy-to-use web interface.

## How It Works

### Error Level Analysis (ELA)
Think of ELA as a digital magnifying glass that reveals hidden editing marks. When you save a JPEG image, it gets compressed and loses some quality. If someone edits part of that image and saves it again, the edited areas will have different compression patterns than the original parts. ELA highlights these differences, making it easier to spot where changes were made.

### Deep Learning Model
We use a Convolutional Neural Network (CNN) - a type of AI that's really good at understanding images. Just like how you can quickly tell the difference between a cat and a dog in a photo, our CNN has been trained to recognize the patterns that indicate whether a document image is original or has been modified.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rahulpmishra/document-forgery-detection.git
   cd document-forgery-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

**Option 1: Streamlit Web Application (Recommended)**
```bash
streamlit run streamlit_app.py
```

Open your browser and navigate to `http://localhost:8501`

- Upload any PNG, JPG, or JPEG document image
- View the original document and ELA analysis side by side
- Get instant prediction results with confidence percentage

**Option 2: Batch Testing Tool (For Multiple Images)**
```bash
# Test entire folder
python batch_test.py /path/to/test/images

# Test single image
python batch_test.py image.jpg

# Save results to JSON file
python batch_test.py /path/to/images --output results.json

# Use custom model
python batch_test.py /path/to/images --model ./models/my_model.h5
```

**Batch Tool Features:**
- ✅ Process multiple images at once
- ✅ Detailed results for each image with confidence scores
- ✅ Summary statistics (total count, forged/authentic percentages)
- ✅ Export results to JSON format
- ✅ Progress tracking and error handling
- ✅ Support for entire folders or single images

## 📊 Dataset Information

### Training Dataset - CASIA v2.0
The model was trained using the CASIA dataset available on Kaggle:

**Dataset Details:**
- **Total Images**: 11,129
- **Authentic images**: 8,144
- **Forged images**: 2,985

### 🔗 Training Resources

**📓 Complete Training Pipeline - Kaggle Notebook:**
[Image Forgery Detection CNN Training](https://www.kaggle.com/code/rahulprakashmishra/image-forgery-detection-cnn-training)

This notebook contains:
- Complete data exploration and preprocessing
- Model architecture and training process
- Evaluation metrics and results
- Model export for deployment

### 🎯 Test Dataset for Demo

**📁 Ready-to-use Test Images:**
[Test Images Dataset](https://www.kaggle.com/datasets/rahulprakashmishra/test-images)

- Download this dataset to get sample images for testing
- Perfect for presentations and demonstrations
- No need to search for test images - everything is ready to use
- Simply download, extract, and upload images to the Streamlit app

## 📁 Project Structure

```
document-forgery-detection/
├── streamlit_app.py          # Main Streamlit web application
├── batch_test.py             # Command-line batch testing tool
├── models/
│   └── trained_model.h5      # Trained CNN model (you need to add this)
├── requirements.txt          # Python dependencies
├── review_presentations/     # Project presentations
└── README.md                 # This file
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Deep Learning**: Keras/TensorFlow
- **Image Processing**: PIL (Pillow)
- **Data Processing**: NumPy
- **Deployment**: Streamlit Cloud compatible

## 📈 Model Performance

The trained CNN model achieves high accuracy in detecting forged vs authentic document images. Detailed performance metrics and training history are available in the Kaggle notebook linked above.

## 🔧 Features

### Web Interface (Streamlit)
- **Real-time Detection**: Fast inference for immediate results
- **ELA Visualization**: Side-by-side comparison of original and ELA images
- **Confidence Scoring**: Percentage confidence for each prediction
- **User-friendly Interface**: Drag-and-drop file upload
- **No Setup Required**: Just upload and analyze

### Batch Testing Tool (Command Line)
- **Bulk Processing**: Test hundreds of images at once
- **Detailed Reports**: Individual results with confidence scores
- **Summary Statistics**: Total counts and percentages
- **JSON Export**: Save results for further analysis
- **Progress Tracking**: Real-time processing updates
- **Error Handling**: Graceful handling of corrupted files

## 🚀 Deployment

The application is ready for deployment on:
- Streamlit Cloud
- Heroku
- AWS/GCP/Azure
- Local server

## 📋 TODO / Future Enhancements

- [ ] Add batch processing for multiple images
- [ ] Include more forgery detection techniques
- [ ] Add detailed analysis reports
- [ ] Implement API endpoints
- [ ] Add support for different image formats

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is publicly available under the MIT License.  
You are free to use, modify, and distribute it.

## 👨‍💻 Author

**Rahul P Mishra**
- LinkedIn: [@rahulpmishra](https://www.linkedin.com/in/rahulpmishra/)
- Kaggle: [@rahulprakashmishra](https://www.kaggle.com/rahulprakashmishra)

## 🙏 Acknowledgments

- CASIA Institute for providing the dataset
- Kaggle for providing the platform for training and data hosting
- Streamlit for the amazing web framework
- Open source community for various tools and libraries

---

**Note**: This project is for educational and research purposes. The test dataset is provided for easy demonstration - simply download from the Kaggle link above and use with the Streamlit app!