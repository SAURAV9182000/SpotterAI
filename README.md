# AI-Powered Skin Cancer Detection Platform

A comprehensive medical AI application for skin cancer detection with fairness evaluation across different skin tones and clinical decision support.

## Features

- **Multi-Model AI Analysis**: ResNet-50, EfficientNet-B4, Vision Transformer, and Hybrid CNN-ViT
- **Demographic Fairness Evaluation**: Analysis across Fitzpatrick skin types I-VI
- **Clinical Decision Support**: Risk assessment, confidence scoring, and medical recommendations
- **Interactive Results Dashboard**: Ensemble predictions and model performance comparisons
- **Image Quality Assessment**: Automatic evaluation of uploaded dermoscopic images

## Live Demo

🚀 **[View Live Application](https://your-app-name.streamlit.app/)**

## Quick Start

### Prerequisites

- Python 3.8+
- Git

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-skin-cancer-detection.git
cd ai-skin-cancer-detection
```

2. Install dependencies:
```bash
pip install -r app_requirements.txt
```

3. Run the application:
```bash
streamlit run app_simple.py
```

## Deployment on Streamlit Community Cloud

### Step 1: Create GitHub Repository

1. Create a new repository on GitHub
2. Upload all project files to your repository

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set main file path: `app_simple.py`
6. Click "Deploy!"

Your app will be live at: `https://your-app-name.streamlit.app/`

## Project Structure

```
├── app_simple.py                 # Main Streamlit application
├── app_requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── components/
│   ├── fairness_dashboard.py    # Fairness analysis components
│   ├── results_dashboard.py     # Results visualization
│   └── upload_interface.py      # Image upload interface
├── models/
│   ├── architectures.py         # AI model architectures
│   └── model_loader.py          # Model management
├── utils/
│   ├── clinical_support.py      # Clinical decision support
│   ├── explainability.py        # AI explainability features
│   └── fairness_metrics.py      # Fairness evaluation metrics
└── data/
    └── dataset_info.py          # Dataset information and statistics
```

## Usage

1. **Upload Image**: Upload a dermoscopic image (JPG, JPEG, PNG)
2. **Configure Analysis**: Select AI models and set confidence thresholds
3. **Review Results**: Analyze predictions, confidence scores, and clinical recommendations
4. **Fairness Assessment**: Evaluate model performance across different skin tones
5. **Clinical Support**: Access detailed medical recommendations and risk assessments

## AI Models

- **ResNet-50**: Deep residual network for robust feature extraction
- **EfficientNet-B4**: Efficient convolutional neural network
- **Vision Transformer**: Transformer-based image analysis
- **Hybrid CNN-ViT**: Combined convolutional and transformer architecture

## Medical Disclaimer

This application is for educational and research purposes only. It is not intended for clinical diagnosis or treatment decisions. Always consult qualified medical professionals for accurate diagnosis and treatment.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For technical support or questions about deployment, please open an issue on GitHub.
