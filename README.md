# College Computer Vision 2025

A comprehensive collection of computer vision lab assignments and practical implementations for academic year 2025. This repository contains hands-on exercises covering fundamental to advanced topics in computer vision, deep learning, and neural networks.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Jobsheets](#jobsheets)
  - [Jobsheet 02: Image Classification](#jobsheet-02-image-classification)
  - [Jobsheet 03: Image Regression Techniques](#jobsheet-03-image-regression-techniques)
  - [Jobsheet 04: Pose Analysis and Body Geometry](#jobsheet-04-pose-analysis-and-body-geometry)
  - [Jobsheet 05: Object Detection with R-CNN Family](#jobsheet-05-object-detection-with-r-cnn-family)
- [Usage](#usage)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository serves as a practical learning resource for computer vision students and enthusiasts. Each jobsheet focuses on specific computer vision concepts and techniques, providing step-by-step implementations using modern deep learning frameworks.

The materials are designed to be compatible with Google Colab, making it easy to run experiments without local GPU requirements.

## Prerequisites

- Basic understanding of Python programming
- Fundamental knowledge of machine learning concepts
- Familiarity with NumPy and basic linear algebra
- Understanding of neural networks (recommended)

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/setyanoegraha/college-computer-vision-2025.git
cd college-computer-vision-2025
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Google Colab

Each jobsheet includes a Colab badge at the top of the notebook. Simply click the badge to open and run the notebook in Google Colab with all dependencies pre-installed.

## Repository Structure

```
college-computer-vision-2025/
│
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
│
├── jobsheet-02/              # Image Classification
│   ├── jobsheet2.ipynb
│   └── image/
│
├── jobsheet-03/              # Image Regression
│   └── jobsheet3.ipynb
│
├── jobsheet-04/              # Pose Analysis
│   ├── jobsheet4.ipynb
│   └── requirements.txt
│
└── jobsheet-05/              # Object Detection
    └── jobsheet5.ipynb
```

## Jobsheets

### Jobsheet 02: Image Classification

**Topics Covered:**
- Introduction to image classification
- Working with simple datasets
- Building and training classification models
- Evaluating model performance

**Key Skills:**
- Dataset preprocessing and augmentation
- Convolutional Neural Networks (CNNs)
- Model training and validation
- Performance metrics analysis

**Notebook:** `jobsheet-02/jobsheet2.ipynb`

### Jobsheet 03: Image Regression Techniques

**Topics Covered:**
- Regression from synthetic images
- Circle radius prediction
- Continuous value estimation from visual data
- Loss functions for regression tasks

**Key Skills:**
- Regression model architecture
- Dataset generation for regression tasks
- Training regression models on image data
- Evaluating regression performance

**Notebook:** `jobsheet-03/jobsheet3.ipynb`

### Jobsheet 04: Pose Analysis and Body Geometry

**Topics Covered:**
- Camera initialization and image acquisition
- Real-time pose detection
- Body angle analysis
- Geometric measurements from human pose

**Key Skills:**
- OpenCV camera interface
- Pose estimation algorithms
- Keypoint detection
- Angle calculation from skeletal data
- Real-time processing

**Notebook:** `jobsheet-04/jobsheet4.ipynb`

### Jobsheet 05: Object Detection with R-CNN Family

**Topics Covered:**
- R-CNN (Region-based Convolutional Neural Network)
- Fast R-CNN improvements
- Faster R-CNN architecture
- Object detection pipelines

**Key Skills:**
- Region proposal methods
- Two-stage object detection
- Transfer learning for detection
- Performance optimization techniques
- GPU acceleration with PyTorch

**Notebook:** `jobsheet-05/jobsheet5.ipynb`

## Usage

### Running Notebooks Locally

1. Navigate to the desired jobsheet directory:
```bash
cd jobsheet-02
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Open the corresponding `.ipynb` file and run the cells sequentially.

### Running in Google Colab

1. Open the notebook link in your browser
2. Click the "Open in Colab" badge at the top of the notebook
3. Run cells sequentially by pressing `Shift + Enter`
4. Enable GPU runtime if needed:
   - Go to Runtime > Change runtime type
   - Select GPU as Hardware accelerator

### Tips for Best Results

- **GPU Acceleration**: Use GPU runtime for faster training, especially for jobsheets 02, 03, and 05
- **Dependencies**: Ensure all dependencies are installed before running experiments
- **Dataset Storage**: Some notebooks may require dataset downloads; ensure adequate storage space
- **Runtime Limits**: Be aware of Colab's runtime limits for long-running experiments

## Technologies

This repository utilizes modern computer vision and deep learning technologies:

**Core Libraries:**
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization

**Deep Learning Frameworks:**
- **PyTorch**: Primary framework for neural networks
- **TensorFlow/Keras**: Alternative framework implementation
- **Torchvision**: Computer vision utilities

**Computer Vision:**
- **OpenCV**: Image processing and computer vision operations
- **scikit-image**: Image processing algorithms
- **Pillow**: Image manipulation

**Utilities:**
- **Albumentations**: Advanced image augmentation
- **imgaug**: Additional augmentation techniques
- **pycocotools**: COCO dataset utilities for object detection

**Visualization:**
- **TensorBoard**: Training visualization
- **Plotly**: Interactive plotting

## Contributing

Contributions to improve the learning materials are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is created for educational purposes. Please refer to the institution's guidelines for usage and distribution.

---

**Repository Maintainer:** Setya Noegraha  
**Academic Year:** 2025  
**Course:** Computer Vision

For questions or issues, please open an issue in the GitHub repository.
