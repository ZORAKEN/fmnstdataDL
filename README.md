# Fashion MNIST Classification with PyTorch

A deep learning project that classifies Fashion MNIST clothing images using fully connected neural networks built with **PyTorch**.

The project started as a small-scale experiment using approximately **6,000 Fashion MNIST images**, trained on a CPU. The initial model achieved around **81% accuracy**.

The project was then expanded by downloading the full Fashion MNIST dataset using the **Kaggle API**, providing **60,000 training images** and **10,000 test images**. Training was moved to a **GPU** to make working with the larger dataset more efficient.

With the full training dataset, the model achieved an accuracy of **97.95625%**, showing a significant improvement over the initial 81% result.

The project was then extended further by experimenting with **GPU optimization, Batch Normalization, Dropout, Weight Decay, and Optuna hyperparameter optimization**.

---

## 📌 Project Overview

This project demonstrates an end-to-end image classification workflow using PyTorch:

* Downloading the Fashion MNIST dataset using the Kaggle API
* Loading the dataset with Pandas
* Visualizing Fashion MNIST images
* Separating image pixels from class labels
* Normalizing pixel values
* Creating PyTorch `Dataset` and `DataLoader` objects
* Building fully connected neural networks
* Training the models using stochastic gradient descent and other optimizers
* Using GPU acceleration with CUDA
* Adding Batch Normalization and Dropout
* Experimenting with Weight Decay
* Performing hyperparameter optimization using Optuna
* Evaluating the model on unseen test data
* Comparing model performance across different approaches

---

## 📊 Dataset

The project uses the **Fashion MNIST** dataset.

Each image is a grayscale **28 × 28 pixel** image, resulting in **784 input features** when flattened.

The full dataset contains:

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

---

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

Each 28 × 28 image is flattened into a vector containing **784 values**.

The processed data is then converted into PyTorch datasets and loaded using `DataLoader`.

---

## 🧠 Neural Network Architecture

The main ANN used in the project is a fully connected feed-forward neural network.

The basic architecture is:

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

The final layer contains 10 outputs corresponding to the 10 Fashion MNIST classes.

---

## ⚙️ Training

The initial and GPU experiments use a combination of the following training settings:

* **Loss Function:** Cross Entropy Loss
* **Optimizer:** SGD
* **Learning Rate:** 0.1
* **Batch Size:** 32
* **Epochs:** 100
* **Random Seed:** 42
* **Hardware:** CPU for the initial experiment
* **Hardware:** GPU for the full-dataset experiment

The basic training process performs a forward pass, calculates the loss, performs backpropagation, and updates the model parameters.

```python
outputs = model(batch_features)

loss = criterion(outputs, batch_labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

# 🚀 GPU Acceleration

After the initial CPU experiment, the project was moved to GPU-based training.

The model and training batches are moved to the available device:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

The GPU implementation allows the model to take advantage of CUDA acceleration when an NVIDIA GPU is available.

The project also uses `pin_memory=True` in the data loading pipeline to help improve data transfer between CPU memory and GPU memory.

This became particularly useful when increasing the training dataset from approximately **6,000 images to the full 60,000 images**.

---

# 🔧 Model Optimization

After developing the basic GPU version, the neural network was further optimized.

The optimized model introduces:

* **Batch Normalization**
* **Dropout**
* **Weight Decay**
* GPU acceleration

The optimized architecture follows the general structure:

```text
784 Input Features
        ↓
Linear(784 → 128)
        ↓
Batch Normalization
        ↓
ReLU
        ↓
Dropout
        ↓
Linear(128 → 64)
        ↓
Batch Normalization
        ↓
ReLU
        ↓
Dropout
        ↓
Linear(64 → 10)
        ↓
10 Class Outputs
```

Dropout is used to reduce overfitting, while Batch Normalization helps stabilize the training process.

---

# 🤖 Optuna Hyperparameter Optimization

The final stage of the project introduces **Optuna** for automated hyperparameter optimization.

Instead of manually selecting the model configuration, Optuna can test different combinations of hyperparameters and identify configurations that produce better validation performance.

The optimization experiment explores parameters such as:

* Number of hidden layers
* Number of neurons
* Learning rate
* Dropout rate
* Batch size
* Optimizer
* Weight decay
* Number of training epochs

The network is dynamically constructed based on the hyperparameters selected for each Optuna trial.

The general structure is:

```text
Input
  ↓
Hidden Layer 1
  ↓
Batch Normalization
  ↓
ReLU
  ↓
Dropout
  ↓
Hidden Layer 2
  ↓
...
  ↓
Output Layer
```

The Optuna experiment is contained in:

`OPTUNA OPTIMISED CODE.ipynb`

---

# 📓 Project Notebooks

The repository contains four main notebooks.

### 1. Basic PyTorch Implementation

**`fashion_mnist_pytorch.ipynb`**

This notebook contains the initial Fashion MNIST implementation using PyTorch.

It covers:

* Dataset loading
* Data exploration
* Image visualization
* Data preprocessing
* PyTorch datasets
* DataLoaders
* Neural network construction
* Model training
* Model evaluation

[View Notebook](https://github.com/ZORAKEN/fmnstdataDL/blob/main/fashion_mnist_pytorch.ipynb)

---

### 2. GPU Implementation

**`ann_fashion_mnist_pytorch_gpu.ipynb`**

This notebook moves the ANN training process to the GPU.

The main purpose is to investigate how GPU acceleration can be used when training on the larger Fashion MNIST dataset.

[View Notebook](https://github.com/ZORAKEN/fmnstdataDL/blob/main/ann_fashion_mnist_pytorch_gpu.ipynb)

---

### 3. Optimized GPU Implementation

**`ann_fashion_mnist_pytorch_gpu_optimized.ipynb`**

This version builds on the GPU implementation by introducing additional optimization techniques.

These include:

* Batch Normalization
* Dropout
* Weight Decay
* GPU acceleration

[View Notebook](https://github.com/ZORAKEN/fmnstdataDL/blob/main/ann_fashion_mnist_pytorch_gpu_optimized.ipynb)

---

### 4. Optuna Optimization

**`OPTUNA OPTIMISED CODE.ipynb`**

This notebook uses Optuna to automatically search through different neural network architectures and training hyperparameters.

The goal is to reduce the amount of manual experimentation required when selecting the best model configuration.

[View Notebook](https://github.com/ZORAKEN/fmnstdataDL/blob/main/OPTUNA%20OPTIMISED%20CODE.ipynb)

---

# 📈 CPU vs GPU and Dataset Comparison

One of the main experiments in this project was comparing the initial small-scale model with the full-dataset GPU implementation.

### Initial Experiment

The first version used approximately **6,000 images** and was trained on a **CPU**.

**Accuracy: ~81%**

### Full Dataset Experiment

The project was then expanded to use the full **60,000-image training dataset**.

Training was moved to a **GPU**.

**Accuracy: 97.95625%**

| Experiment   | Training Data | Hardware |      Accuracy |
| ------------ | ------------: | -------- | ------------: |
| Initial      | ~6,000 images | CPU      |          ~81% |
| Full Dataset | 60,000 images | GPU      | **97.95625%** |

The results demonstrate a substantial improvement when moving from the smaller training set to the full Fashion MNIST dataset.

The increase in training data gives the model significantly more examples to learn from, while GPU acceleration makes training on the larger dataset more practical.

---

# 📊 Results

The final reported model achieved:

## **97.95625% Accuracy**

This compares with approximately:

## **81% Accuracy**

from the initial small-scale experiment.

That represents an improvement of approximately **17 percentage points**.

| Metric          | Initial Model | Full Dataset Model |
| --------------- | ------------: | -----------------: |
| Training Images |        ~6,000 |             60,000 |
| Hardware        |           CPU |                GPU |
| Accuracy        |          ~81% |      **97.95625%** |

The experiment demonstrates how both **training data size** and **computational resources** can have a significant impact on a deep learning workflow.

---

# 🛠️ Technologies Used

* **Python**
* **PyTorch**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Kaggle API / KaggleHub**
* **CUDA**
* **Optuna**
* **Jupyter Notebook**

---

# ▶️ How to Run

Clone the repository:

```bash
git clone https://github.com/ZORAKEN/fmnstdataDL.git

cd fmnstdataDL
```

Install the required dependencies:

```bash
pip install torch pandas numpy scikit-learn matplotlib kagglehub optuna jupyter
```

Download the dataset using KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download(
    "zalando-research/fashionmnist"
)

print("Path to dataset files:", path)
```

Load the CSV files:

```python
import pandas as pd

train_df = pd.read_csv(
    path + "/fashion-mnist_train.csv"
)

test_df = pd.read_csv(
    path + "/fashion-mnist_test.csv"
)
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then run the notebooks from beginning to end.

---

# 📁 Project Structure

```text
fmnstdataDL/
│
├── fashion_mnist_pytorch.ipynb
│
├── ann_fashion_mnist_pytorch_gpu.ipynb
│
├── ann_fashion_mnist_pytorch_gpu_optimized.ipynb
│
├── OPTUNA OPTIMISED CODE.ipynb
│
└── README.md
```

---

# 📌 Key Takeaways

This project demonstrates several important concepts in practical deep learning:

1. **Training data size can have a major impact on model accuracy.**
2. **GPU acceleration makes larger-scale neural network training more practical.**
3. **Normalizing pixel values helps prepare image data for neural network training.**
4. **PyTorch `Dataset` and `DataLoader` provide a structured training pipeline.**
5. **Batch Normalization and Dropout can be used to improve the training process and reduce overfitting.**
6. **Weight Decay provides an additional form of regularization.**
7. **Optuna can automate the search for effective hyperparameters.**
8. A relatively simple fully connected neural network achieved **97.95625% accuracy** on the Fashion MNIST experiment.

Overall, this project shows the progression from a basic **CPU-based ANN trained on a small dataset** to a more advanced **GPU-based and optimized deep learning workflow**, followed by **automated hyperparameter optimization with Optuna**.
---

## ⭐ Project Summary

**Small Dataset → CPU → ~81%**

↓

**Full Dataset → GPU → 97.95625%**

↓

**Batch Normalization + Dropout + Weight Decay**

↓

**Optuna Hyperparameter Optimization**

This project demonstrates how a simple Fashion MNIST classification problem can be progressively improved through **more training data, GPU acceleration, model optimization, and automated hyperparameter tuning**.
