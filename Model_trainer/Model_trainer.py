# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import io
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler
# Load arrays
import numpy as np
loaded_data = np.load('/kaggle/input/my-array/my_arrays.npz')  # For multiple arrays saved with np.savez()
# Access individual arrays from the loaded data
X_array = loaded_data['X_array']
Y_array = loaded_data['Y_array']
age_data = loaded_data['age_data']
sex_data = loaded_data['sex_data']
real_labels_array = loaded_data['real_labels_array']
#label encoder

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the label encoder on the string array
Y_array = label_encoder.fit_transform(Y_array)

#Train test split
X_train,X_val,Y_train, Y_val,age_train,age_val,sex_train,sex_val,real_labels_array_train,real_labels_array_val, = train_test_split(
    X_array,Y_array,age_data,sex_data,real_labels_array, test_size=0.2, random_state=42
)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Assuming you have already loaded your data and labels into numpy arrays
# X_train: 2D numpy array of shape (num_samples, num_leads, num_data_points)
# y_train: 1D numpy array of labels

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(Y_train, dtype=torch.long)

##   Age and Sex
age_train_tensor = torch.tensor(age_train, dtype=torch.float32)
sex_train_tensor = torch.tensor(sex_train, dtype=torch.float32)


# Define a multi-input CNN model
class MultiInputECGClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MultiInputECGClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_leads, out_channels=16, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.fc1 = nn.Linear(16 * ((num_data_points - 4) // 4) + 2, 128)  # Adding 2 for age and sex
        self.dropout1 = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, ecg, age, sex):
        ecg_features = self.pool(torch.relu(self.conv1(ecg)))
        ecg_features = ecg_features.view(ecg_features.size(0), -1)  # Flatten the tensor
        age = age.unsqueeze(1)  # Expand dimensions to match ecg_features
        sex = sex.unsqueeze(1)  # Expand dimensions to match ecg_features
        combined_features = torch.cat((ecg_features, age, sex), dim=1)
        combined_features = torch.relu(self.fc1(combined_features))
        combined_features = self.dropout1(combined_features)
        output = self.fc2(combined_features)
        return output                                            
"""
class MultiInputECGClassifier(nn.Module):
    def _init_(self, num_classes):
        super(MultiInputECGClassifier, self)._init_()
        self.conv1 = nn.Conv1d(in_channels=num_leads, out_channels=16, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(16)  # Add BatchNorm after the first convolution
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.fc1 = nn.Linear(16 * ((num_data_points - 4) // 4) + 2, 128)  # Adding 2 for age and sex
        self.bn2 = nn.BatchNorm1d(128)  # Add BatchNorm after the first fully connected layer
        self.dropout1 = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, ecg, age, sex):
        ecg_features = self.pool(torch.relu(self.bn1(self.conv1(ecg))))  # Apply BatchNorm after conv1
        ecg_features = ecg_features.view(ecg_features.size(0), -1)  # Flatten the tensor
        age = age.unsqueeze(1)  # Expand dimensions to match ecg_features
        sex = sex.unsqueeze(1)  # Expand dimensions to match ecg_features
        combined_features = torch.cat((ecg_features, age, sex), dim=1)
        combined_features = torch.relu(self.bn2(self.fc1(combined_features)))  # Apply BatchNorm after fc1
        combined_features = self.dropout1(combined_features)
        output = self.fc2(combined_features)
        return output                    
"""
# Initialize multi-input model
num_classes = len(set(Y_array))
num_leads = X_train.shape[1]
num_data_points = X_train.shape[2]
model = MultiInputECGClassifier(num_classes)


"""# Define loss function and optimizer        Oshadha
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)"""
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

# Create a DataLoader for batch training
batch_size = 16
train_dataset = TensorDataset(X_train_tensor, age_train_tensor, sex_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

losses = []
validation_losses = []
validation_accuracy = []
train_accuracy = []

# Convert validation numpy arrays to PyTorch tensors
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
age_val_tensor = torch.tensor(age_val, dtype=torch.float32)
sex_val_tensor = torch.tensor(sex_val, dtype=torch.float32)
y_val_tensor = torch.tensor(Y_val, dtype=torch.long)

##################################################3
l1_lambda = 0.001
# Training loop
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for inputs_ecg, inputs_age, inputs_sex, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs_ecg, inputs_age, inputs_sex)
        loss = criterion(outputs, labels)

###############################################
        # L1 regularization
        l1_reg = torch.tensor(0.0)
        for param in model.parameters():
            l1_reg += torch.norm(param, p=1)
        loss += l1_lambda * l1_reg  # Add L1 regularization term to the loss

#################################################
        loss.backward()
        
        running_loss += loss.item()
        optimizer.step()

    # print(running_loss, X_train.shape[0])
    losses.append(running_loss / X_train.shape[0])

    with torch.no_grad():
        val_outputs = model(X_val_tensor, age_val_tensor, sex_val_tensor)
        loss = criterion(val_outputs, y_val_tensor)
        # print(loss.item(), X_val.shape[0])
        validation_losses.append(loss.item() / X_val.shape[0])

    print(".", end="")

    if epoch % 5 == 4:
        model.eval()
        with torch.no_grad():
            print("\nEpoch {}".format(epoch+1))
            train_outputs = model(X_train_tensor, age_train_tensor, sex_train_tensor)
            predicted_train_labels = train_outputs.argmax(dim=1).numpy()
            train_acc = accuracy_score(y_train_tensor, predicted_train_labels)
            print(f"Train Accuracy: {train_acc:.8f}")
            train_accuracy.append(train_acc)

            val_outputs = model(X_val_tensor, age_val_tensor, sex_val_tensor)
            predicted_labels = val_outputs.argmax(dim=1).numpy()
            accuracy = accuracy_score(Y_val, predicted_labels)
            print(f"Validation Accuracy: {accuracy:.8f}")
            validation_accuracy.append(accuracy)  # Append validation accuracy to the list#######################################################

    # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
# Assuming you have loaded your validation data, labels, age, and sex inputs
# X_val: 2D numpy array of ECG data for validation
# age_val: 1D numpy array of ages for validation
# sex_val: 1D numpy array of sexes (0 or 1) for validation
# y_val: 1D numpy array of labels for validation

# Set the model to evaluation mode
model.eval()

# Disable gradient calculations since you're not training
with torch.no_grad():
    # Forward pass on validation data
    val_outputs = model(X_val_tensor, age_val_tensor, sex_val_tensor)
    # Get predicted class labels
    predicted_labels = val_outputs.argmax(dim=1)

# Convert predicted labels to numpy array
predicted_labels_np = predicted_labels.numpy()

# Calculate F1 score
#f1 = f1_score(Y_val, predicted_labels_np)

#print(f"F1 Score: {f1:.4f}")


# Calculate accuracy
accuracy = accuracy_score(Y_val, predicted_labels_np)
print(f"Validation Accuracy: {accuracy:.4f}")

plt.plot(losses, label="train loss")
plt.legend()
plt.show()

plt.plot(validation_losses, label="validation_losses")
plt.legend()
plt.show()







# Create a list of epochs (x-axis values)
epochs = range(1, len(validation_accuracy) + 1)

# Plot validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs, validation_accuracy, label='Validation Accuracy', marker='o', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy vs. Epochs')
plt.legend()

# If you have training accuracies, you can also plot them
if train_accuracy:
    plt.plot(epochs, train_accuracy, label='Training Accuracy', marker='o', linestyle='-')
    plt.legend()

# Show the plot
plt.grid(True)
plt.show()
