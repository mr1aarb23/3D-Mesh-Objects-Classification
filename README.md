# 3D-Mesh-Objects-Classification


## Introduction

Welcome to 3d mesh objects classification with diffusionNet, this project is designed for the classification of 3d objects using deep learning techniques. The script is divided into twelve steps, providing a comprehensive guide through the entire process.

## Steps

### Step 1: Mount Google Drive

The code initiates by mounting your Google Drive account using the `google.colab` library. This step ensures easy access to files stored on Google Drive within your Colab environment.

```python
drive.mount('/content/drive')
```

### Step 2: Clone a Git Repository

Clone a Git repository from the specified URL ([https://github.com/nmwsharp/diffusion-net.git](https://github.com/nmwsharp/diffusion-net.git)) using the `!git clone` command. The cloned repository contains essential source code for subsequent script execution.

```bash
!git clone https://github.com/nmwsharp/diffusion-net.git
```

### Step 3: Copy Source Code to Google Drive

Use the `!cp -r` command to copy the source code from the cloned repository (`/content/diffusion-net/src/diffusion_net/`) to your Google Drive (`/content/drive/MyDrive`). This step facilitates organized and convenient access to the source code from your Google Drive.

```bash
!cp -r /content/diffusion-net/src/diffusion_net/ /content/drive/MyDrive
```

### Step 4: Install Python Dependencies

Install necessary Python packages using `pip install`. These packages include `trimesh`, `plyfile`, `polyscope`, `potpourri3d`, `python-utils`, `robust-laplacian`, `threadpoolctl`, `tqdm`, `typing-extensions`, and `joblib`. They are crucial for various data processing and machine learning operations.

```bash
pip install trimesh plyfile polyscope potpourri3d python-utils robust-laplacian threadpoolctl tqdm typing-extensions joblib
```

### Step 5: Define the `Meca_Dataset` Class

Define a custom class named `Meca_Dataset` to load and prepare the dataset. This class takes the root directory of the dataset, a training indicator (`train`), and the number of eigenvalues (`k_eig`) as inputs. It organizes data based on classes, loads 3D models, and calculates geometric operators.

```python
# Example usage:
dataset = Meca_Dataset(root_dir, train=True, k_eig=10)
```

### Step 6: Load Training and Test Datasets

Utilize the `Meca_Dataset` class to load training and test datasets from the specified root directory (`root_dir`). Two instances, `train_dataset` and `test_dataset`, are created.

```python
# Example usage:
train_dataset = dataset.load_dataset(train=True)
test_dataset = dataset.load_dataset(train=False)
```

### Step 7: Define Model Training Parameters

This section of the code sets various parameters for model training, such as the number of epochs (`n_epoch`), learning rate (`lr`), decay rate of the learning rate (`decay_rate`), etc.

```python
n_epoch = 20
lr = 0.001
decay_every = 5
decay_rate = 0.9
```

### Step 8: Create the DiffusionNet Model

Create the DiffusionNet model using the `DiffusionNet` class from the `diffusion_net` library. Configure the model with parameters like the number of input features, hidden layer width, etc. Move the model to the GPU if available.

```python
# Example usage:
model = DiffusionNet(input_features=64, hidden_width=128)
model.to(device)
```

### Step 9: Train the Model

The training loop begins here. Train the model over multiple epochs, updating the model based on predictions and loss at each epoch. Training loss and accuracy are monitored and displayed at each epoch. The learning rate may be adjusted periodically based on `decay_every` and `decay_rate`.

```python
# Example usage:
train_model(model, train_dataset, n_epoch, lr, decay_every, decay_rate)
```

### Step 10: Visualize Performance

Generate graphs to visualize the model's performance over epochs. Two graphs are created: one for training accuracy and another for training loss. Plotly is used for visualization.

```python
# Example usage:
visualize_performance()
```

### Step 11: Test on a 3D Object

Test the trained model on a specific 3D object loaded from an OBJ file specified in `mesh_path`. Calculate necessary geometric operators for the model, predict the class for the 3D object, and visualize the prediction.

```python
# Example usage:
test_on_3d_object(model, mesh_path)
```

### Step 12: Save the Trained Model

Finally, save the trained model to a specified file using `torch.save`.

```python
# Example usage:
save_trained_model(model, model_path)
```

## Additional Information

- The code is implemented in Python using the PyTorch deep learning library.
- Tested on a dataset of 3D models containing classes like "Cone," "Cube," "Cylinder," "Pyramid," "Rectangular," and "Sphere."
- The dataset link : https://drive.google.com/file/d/1LPS91tolW-bFdTq66PRz5ojx39DGIrZi/view?usp=drive_link
- The model can be utilized for the classification of 3D objects with varying shapes and sizes.
