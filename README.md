# Fashion MNIST Classification with PyTorch

A deep learning project that classifies Fashion MNIST clothing images using a fully connected neural network built with **PyTorch**.

The project started as a small-scale experiment using approximately **6,000 Fashion MNIST images**, trained on a CPU. This initial model achieved around **81% accuracy**. To improve the model, the full Fashion MNIST dataset was downloaded using the **Kaggle API**, providing **60,000 training images** and 10,000 test images. The training process was then moved to a **GPU** to efficiently handle the larger dataset.

With the full training dataset, the model achieved an accuracy of **97.95625%**, demonstrating a significant improvement over the initial 81% result.

## 📌 Project Overview

This project demonstrates an end-to-end image classification workflow using PyTorch:

* Downloading the Fashion MNIST dataset using the Kaggle API
* Loading the dataset with Pandas
* Visualizing Fashion MNIST images
* Separating image pixels from class labels
* Normalizing pixel values
* Creating custom PyTorch `Dataset` and `DataLoader` objects
* Building a fully connected neural network
* Training the model using stochastic gradient descent
* Using GPU acceleration for training
* Evaluating the model on unseen test data
* Comparing model performance between a small and full dataset

## 📊 Dataset

The project uses the **Fashion MNIST** dataset.

Each image is a grayscale **28 × 28 pixel** image, resulting in **784 input features** when flattened.

The dataset contains:

* **60,000 training images**
* **10,000 test images**
* **10 clothing classes**
* **784 pixel features per image**

The first column contains the class label, while the remaining 784 columns represent the pixel values.

The 10 classes are:

| Label | Clothing Item |
| ----: | ------------- |
|     0 | T-shirt/top   |
|     1 | Trouser       |
|     2 | Pullover      |
|     3 | Dress         |
|     4 | Coat          |
|     5 | Sandal        |
|     6 | Shirt         |
|     7 | Sneaker       |
|     8 | Bag           |
|     9 | Ankle boot    |

## 🔄 Data Preprocessing

The image data is separated into features and labels:

```python
X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values

X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values
```

Pixel values originally range from **0 to 255** and are normalized to approximately **0 to 1**:

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

The 784 pixel values represent the flattened version of each 28 × 28 image.

## 🧠 Neural Network Architecture

The project uses a fully connected feed-forward neural network:

```text
784 Input Features
        ↓
Linear(784 → 128)
        ↓
ReLU
        ↓
Linear(128 → 64)
        ↓
ReLU
        ↓
Linear(64 → 10)
        ↓
10 Class Outputs
```

The model is implemented using PyTorch:

```python
class MyNN(nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)
```

The final layer contains 10 outputs, corresponding to the 10 Fashion MNIST classes.

## ⚙️ Training

The model uses:

* **Optimizer:** SGD
* **Loss Function:** Cross Entropy Loss
* **Learning Rate:** 0.1
* **Batch Size:** 32
* **Epochs:** 100
* **Random Seed:** 42
* **Hardware:** GPU for the full-dataset experiment

The training process performs a forward pass, calculates the loss, performs backpropagation, and updates the model parameters.

```python
outputs = model(batch_features)

loss = criterion(outputs, batch_labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 🚀 CPU vs GPU and Dataset Comparison

An important part of this project was comparing the initial small-scale experiment with the full-dataset experiment.

### Initial Experiment

The first version used approximately **6,000 images** and was trained on a **CPU**.

**Accuracy: ~81%**

### Full Dataset Experiment

The dataset was then downloaded through the **Kaggle API**, giving access to the full **60,000-image training set**. The model was moved to a **GPU** for training.

**Accuracy: 97.95625%**

| Experiment   | Training Data | Hardware |      Accuracy |
| ------------ | ------------: | -------- | ------------: |
| Initial      | ~6,000 images | CPU      |          ~81% |
| Full Dataset | 60,000 images | GPU      | **97.95625%** |

The results show a substantial improvement when training on the full dataset. The GPU primarily provides faster and more efficient computation, while the increase in training data gives the neural network significantly more examples from which to learn.

## 📈 Results

The final model achieved:

**97.95625% accuracy**

on the evaluation dataset.

Compared with the initial **~81% accuracy**, this represents an improvement of approximately **17 percentage points**.

This experiment demonstrates how dataset size can have a major impact on neural network performance. It also demonstrates the practical benefit of GPU acceleration when training on a larger dataset.

## 🛠️ Technologies Used

* **Python**
* **PyTorch**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Kaggle API**
* **CUDA / GPU acceleration**

## ▶️ How to Run

Install the required dependencies:

```bash
pip install torch pandas numpy scikit-learn matplotlib kagglehub
```

Download the dataset using Kaggle:

```python
import kagglehub

path = kagglehub.dataset_download(
    "zalando-research/fashionmnist"
)

print("Path to dataset files:", path)
```

Load the resulting CSV files with Pandas:

```python
import pandas as pd

train_df = pd.read_csv(path + "/fashion-mnist_train.csv")
test_df = pd.read_csv(path + "/fashion-mnist_test.csv")
```

Then run the notebook from beginning to end.

## 📌 Key Takeaways

This project demonstrates several important concepts in practical deep learning:

1. **Larger datasets can significantly improve model performance.**
2. **GPU acceleration makes training larger datasets much more practical.**
3. **Normalization improves the neural network training process.**
4. **PyTorch `Dataset` and `DataLoader` provide an efficient training pipeline.**
5. A relatively simple fully connected neural network can achieve **97.95625% accuracy** on Fashion MNIST when trained with sufficient data.

The project provided a practical comparison between a small CPU-based experiment and a full-scale GPU-based training run, showing how both **data scale and computational resources** affect machine learning experiments.
