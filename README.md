# Saikat AI Portfolio

## Overview

A comprehensive web-based AI portfolio application built with **Streamlit** that showcases multiple machine learning models for various AI tasks. This portfolio demonstrates expertise in computer vision, natural language processing, and machine learning implementations.

## Features

### 1. **Recognize Number** 🔢
- Handwritten digit recognition (0-9)
- Uses CNN-based deep learning model
- Trained on MNIST dataset
- Real-time prediction on uploaded images
- Large prediction button and clear button below image upload

### 2. **Recognize Clothes** 👕
- Fashion item classification
- Identifies 10 different clothing categories:
  - T-shirt/Top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle Boot
- CNN model trained on Fashion MNIST dataset
- Large prediction button and clear button below image upload

### 3. **Review Analysis** ⭐
- Sentiment analysis for product/movie reviews
- Classifies reviews as Positive or Negative
- Uses SVM classifier with TF-IDF vectorization
- Trained on clothing review dataset
- Large analysis button below review text area

### 4. **Spam Email Detector** 📧
- Email classification system
- Identifies spam vs. legitimate emails
- SVM-based classifier with TF-IDF vectorization
- Practical text classification implementation
- Large analysis button below email text area

## Project Structure

```
Saikat_AI_Portfolio/
├── Portfolio_App.py                    # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
└── Model/
    ├── 01_a_Clothing_sentiment_svm_model.joblib
    ├── 01_b_Clothing_vectorizer.joblib
    ├── 02_a_spam_classifier_model.joblib
    ├── 02_b_tfidf_vectorizer.joblib
    ├── 03_A_mnist_cnn_m1_trained_model.keras
    └── 04_A_mnist_cnn_m2_trained_model.keras
```

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Steps

1. **Clone or download the project**
   ```bash
   cd Saikat_AI_Portfolio
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure all model files are in the `Model/` directory**

## Usage

### Running the Application

```bash
streamlit run Portfolio_App.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using Each Feature

#### Recognize Number
1. Navigate to "Recognize Number" from the left menu
2. Upload an image of a handwritten digit (JPG, JPEG, or PNG)
3. View the uploaded image preview
4. Click the **🔮 PREDICT NUMBER** button to get prediction or **🔄 CLEAR** button to upload a new image
5. View the predicted digit result

#### Recognize Clothes
1. Navigate to "Recognize Clothes" from the left menu
2. Upload an image of clothing (JPG, JPEG, or PNG)
3. View the uploaded image preview
4. Click the **👗 PREDICT CLOTHING** button to get prediction or **🔄 CLEAR** button to upload a new image
5. View the identified clothing item result

#### Review Analysis
1. Navigate to "Review Analysis" from the left menu
2. Enter a product or movie review in the text area
3. Click the **⭐ ANALYZE SENTIMENT** button below the text area
4. Get sentiment classification (Positive/Negative)

#### Spam Email Detector
1. Navigate to "Spam Email Detector" from the left menu
2. Paste or type an email content in the text area
3. Click the **📧 ANALYZE EMAIL** button below the text area
4. Get classification (Spam/Legitimate)

## Dependencies

The project requires the following Python packages:

| Package | Purpose |
|---------|---------|
| **streamlit** | Web application framework |
| **tensorflow** | Deep learning framework (for CNN models) |
| **pillow** | Image processing library |
| **numpy** | Numerical computing library |
| **joblib** | Model serialization and loading |
| **scikit-learn** | (Implicit) Used in model training for SVM |

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Technical Details

### Models Used

| Feature | Model Type | Framework | Input Size |
|---------|-----------|-----------|-----------|
| Recognize Number | CNN | TensorFlow/Keras | 28×28 pixels |
| Recognize Clothes | CNN | TensorFlow/Keras | 28×28 pixels |
| Review Analysis | SVM | Scikit-learn | TF-IDF vectors |
| Spam Detector | SVM | Scikit-learn | TF-IDF vectors |

### Data Preprocessing

**Image Processing:**
- Resize to target dimensions (28×28 pixels)
- Convert to appropriate color space (Grayscale for digits, RGB for clothes)
- Normalize pixel values to [0, 1] range

**Text Processing:**
- TF-IDF vectorization for text data
- Feature extraction for SVM classification

## Performance Considerations

- Models are cached using `@st.cache_resource` decorator to improve performance
- Initial page load may take a few seconds while models are loaded
- Subsequent predictions are faster due to cached models
- Image processing is optimized for quick inference

## User Interface Features

- **Intuitive Navigation**: Left sidebar menu for easy project selection
- **Dual Action Buttons**: 
  - Image-based features (Recognize Number/Clothes) have PREDICT and CLEAR buttons side by side
  - PREDICT button to analyze the uploaded image
  - CLEAR button (🔄) to reset and upload a new image
  - Text-based features display analysis buttons after text entry
- **Real-time Feedback**: Instant visual feedback with success/error messages
- **Clear CTAs**: Emoji icons with uppercase labels (🔮 PREDICT, 👗 PREDICT, ⭐ ANALYZE, 📧 ANALYZE)
- **Responsive Layout**: Wide layout design for optimal viewing on different screen sizes
- **Visual Animations**: Balloons celebration effect for positive predictions
- **User-Friendly Design**: Side-by-side buttons for optimal usability

## Browser Compatibility

The application works best on:
- Chrome/Chromium
- Firefox
- Safari
- Edge

## Troubleshooting

### Common Issues

1. **Model Files Not Found**
   - Ensure all `.joblib` and `.keras` files are in the `Model/` directory
   - Check file paths use forward slashes (`/`) for cross-platform compatibility

2. **TensorFlow/CUDA Issues**
   - If using GPU, ensure proper CUDA installation
   - CPU mode will work but may be slower
   - Install CPU-only TensorFlow if needed: `pip install tensorflow-cpu`

3. **Memory Issues**
   - Large images may cause memory problems
   - Application caches models to reduce memory usage
   - Close other applications if experiencing slowness

## Future Enhancements

- [ ] Add batch processing for multiple images
- [ ] Implement real-time camera input for digit/clothing recognition
- [ ] Add model accuracy metrics display
- [ ] Implement multi-language support
- [ ] Add data visualization for model predictions
- [ ] Create model retraining capability

## Author

**Saikat Das**

Portfolio of AI/ML projects demonstrating proficiency in:
- Deep Learning (CNNs)
- Machine Learning (SVM, Classification)
- Natural Language Processing (Sentiment Analysis)
- Web Development with Streamlit
- Model Deployment

## License

This project is open source and available for educational purposes.

## Contact

For questions or collaboration opportunities, feel free to reach out.

---

**Last Updated:** 2026-07-03  
**Framework Version:** Streamlit  
**Python Version:** 3.7+
