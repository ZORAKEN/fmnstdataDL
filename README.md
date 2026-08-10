# fmnstdataDL
# Fashion MNIST Classification with PyTorch

A simple neural network project for classifying **Fashion MNIST** images using **PyTorch**.

The project loads a CSV version of the Fashion MNIST dataset, preprocesses the image features, builds a fully connected neural network, trains it using stochastic gradient descent, and evaluates its classification accuracy on a held-out test set.

## 📌 Project Overview

Fashion MNIST contains grayscale images of clothing items. Each image is **28 × 28 pixels**, giving **784 input features** per image.

This project demonstrates a basic end-to-end PyTorch workflow:

* Load a Fashion MNIST CSV dataset
* Visualize sample images
* Split the data into training and test sets
* Normalize pixel values
* Create a custom PyTorch `Dataset`
* Create `DataLoader`s for batching
* Build a feed-forward neural network
* Train the model using SGD
* Evaluate the model using classification accuracy

## 🗂️ Project Structure

```text
.
├── fashion_mnist_pytorch.ipynb
├── fmnist_small.csv
└── README.md
```

## 🛠️ Technologies Used

* Python
* PyTorch
* Pandas
* Scikit-learn
* Matplotlib

## 📊 Dataset

The notebook expects a CSV file named:

```text
fmnist_small.csv
```

The first column contains the **class labels**, while the remaining **784 columns** contain the pixel values.

Each image can be reshaped from the 784 pixel values into:

```text
28 × 28
```

The notebook also visualizes the first 16 images in a 4 × 4 grid.

## 🔄 Data Preprocessing

The dataset is separated into features and labels:

```python
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
```

The data is split into training and test sets using an **80/20 split**:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

Pixel values are normalized from the original `0–255` range to approximately `0–1`:

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

## 🧠 Neural Network Architecture

The model is a fully connected feed-forward neural network:

```text
Input: 784 features
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
10 class outputs
```

The model is implemented using `torch.nn.Sequential`:

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

The final layer produces **10 outputs**, corresponding to the 10 Fashion MNIST classes.

## ⚙️ Training Configuration

The notebook uses the following hyperparameters:

| Parameter        |              Value |
| ---------------- | -----------------: |
| Epochs           |                100 |
| Learning Rate    |                0.1 |
| Batch Size       |                 32 |
| Optimizer        |                SGD |
| Loss Function    | Cross Entropy Loss |
| Train/Test Split |              80/20 |
| Random Seed      |                 42 |

The loss function is:

```python
criterion = nn.CrossEntropyLoss()
```

and the optimizer is:

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1
)
```

## 🚂 Training

During each training iteration, the model:

1. Performs a forward pass
2. Calculates the cross-entropy loss
3. Clears previous gradients
4. Performs backpropagation
5. Updates model parameters using SGD

```python
outputs = model(batch_features)

loss = criterion(outputs, batch_labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The average loss for each epoch is printed during training.

## 📈 Evaluation

After training, the model is evaluated without calculating gradients:

```python
model.eval()

total = 0
correct = 0

with torch.no_grad():

    for batch_features, batch_labels in test_loader:

        outputs = model(batch_features)

        _, predicted = torch.max(outputs, 1)

        total += batch_labels.shape[0]
        correct += (predicted == batch_labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")
```

Accuracy is calculated as:

```text
Accuracy = Correct Predictions / Total Predictions
```

and displayed as a percentage.

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Install dependencies

```bash
pip install torch pandas scikit-learn matplotlib
```

### 3. Add the dataset

Place the following file in the project directory:

```text
fmnist_small.csv
```

### 4. Run the notebook

Open:

```text
fashion_mnist_pytorch.ipynb
```

and run the cells from top to bottom.

## 📌 Notes

* The model uses raw flattened 28 × 28 images rather than convolutional layers.
* Pixel values are normalized by dividing by `255`.
* A custom PyTorch `Dataset` is used to convert the NumPy arrays into PyTorch tensors.
* The training `DataLoader` shuffles the data, while the test `DataLoader` does not.
* `torch.manual_seed(42)` is used for reproducibility.



## 📄 License

This project is intended for educational and experimentation purposes.
