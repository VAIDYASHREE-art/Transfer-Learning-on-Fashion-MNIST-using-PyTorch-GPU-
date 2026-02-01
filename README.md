Overview

-This is a project that shows the application of transfer learning with a pre-trained VGG16 convolutional neural network on the Fashion-MNIST dataset.
-The project focuses on the design of data pipelines, GPU training, and adapting models rather than developing new models.

Key Concepts Covered

-Transfer learning with pre-trained CNNs
-Freezing feature extractors and training custom classifier heads
-GPU-accelerated training using PyTorch
-Image preprocessing and normalization compatible with ImageNet models

Tech Stack
- Python, PyTorch, Torchvision, NumPy, Pandas, Matplotlib

Dataset

-Fashion-MNIST (CSV format)
-Grayscale images (28×28) converted to 3-channel RGB to match VGG16 input requirements.

Pipeline Overview

-Load Fashion-MNIST data from CSV
-Train-test split
-Convert grayscale images → RGB
-Apply ImageNet-compatible transforms:
-Resize → Center Crop → Normalize
-Load pretrained VGG16
-Freeze convolutional layers
-Replace classifier head
-Train on GPU
-Evaluate accuracy on train and test sets

Model Architecture

-Backbone: VGG16 (pretrained on ImageNet)

Classifier Head:
Linear → ReLU → Dropout
Linear → ReLU → Dropout
Linear (10 classes)

Results
-Successfully fine-tuned the classifier head on Fashion-MNIST
-Achieved meaningful classification accuracy while keeping the backbone frozen
-Demonstrated efficient reuse of pretrained representations

Learning Outcomes
-Practical understanding of model reuse vs. training from scratch
-Experience handling real-world input constraints (channel mismatch, normalization)
-Clear exposure to production-style deep learning workflows

Future Improvements
-Unfreeze upper convolutional layers for partial fine-tuning
