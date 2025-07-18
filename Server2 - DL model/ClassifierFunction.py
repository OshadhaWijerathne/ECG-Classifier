import os
import sys
import numpy as np
from scipy import io
from scipy.signal import resample
import torch
import torch.nn as nn
import csv


def ClassifierFunction(path):

    input_age_data = np.empty(1)
    input_sex_data = np.empty(1)
    input_lable_data = np.empty(1, dtype='<U1000')


    found_initial_frequency = False

    # Insert the path to your folder containing X person's data.
    data_folder_path = path

    hea_files = [file for file in os.listdir(
        data_folder_path) if file.endswith('.hea')]

    if len(hea_files) == 1:
        hea_file_path = os.path.join(data_folder_path, hea_files[0])
        with open(hea_file_path, "r") as hea_file:
            hea_contents = hea_file.readlines()

            # This part reads the patient header file to extract - sampling frequency, age and sex.
            for line in hea_contents:
                line = line.strip()
                f_name = line

                if not (found_initial_frequency):
                    first_line = line.split()
                    initial_frequency = int(first_line[-2])
                    found_initial_frequency = True

                if line.startswith('# Age:'):
                    age_value = line.split(': ')[-1]

                    if age_value == 'NaN' or int(age_value) <= 0 or int(age_value) > 100:
                        # header_info['Age'] = None
                        input_age_data[0] = 0.5    # Mean is 50
                    else:
                        input_age_data[0] = (int(age_value)/100)

                elif line.startswith('# Sex:'):
                    sex_type = line.split(': ')[-1]
                    if sex_type == 'Male' or sex_type == 'M':
                        input_sex_data[0] = 1

                    elif sex_type == 'Female' or sex_type == 'F':
                        input_sex_data[0] = 0

                    else:
                        input_sex_data[0] = 1

                if line.startswith('# Dx:'):
                    dx_value = line.split(': ')[-1]
                    input_str = dx_value.split(',')[:]
                    # Print the result

                    input_lable_data = list(input_str)

    elif len(hea_files) == 0:
        print("Error: No .hea file found in the folder.")

    else:
        print("Error: Multiple .hea files found in the folder.")

    # The following part resamples the given ECG record of the Patient. A Numpy array with resampled data will be returned.

    # Function to process and resample data from a single .mat file from folder.

    def process_and_resample_mat_files_in_folder(folder_path):

        mat_files = [file for file in os.listdir(
            data_folder_path) if file.endswith('.mat')]

        # Check if there is exactly one .mat file in the folder
        if len(mat_files) == 1:
            # Construct the full path to the .mat file
            mat_file_name = mat_files[0]
            mat_file_path = os.path.join(folder_path, mat_file_name)

            # Load data from the .mat file
            data = io.loadmat(mat_file_path)

            # Assuming each ECG data is stored as a 2D array
            ecg_data = data['val']

            num_leads, num_samples = ecg_data.shape

            # Initialize an empty array to store the resampled data
            resampled_data = np.zeros((num_leads, int(
                num_samples * desired_frequency / initial_frequency)), dtype=np.float64)

            # Resample each lead's data from initial frequency to 257Hz
            for i in range(num_leads):
                resampled_data[i, :] = resample(ecg_data[i, :], int(
                    num_samples * desired_frequency / initial_frequency))

            return resampled_data

        elif len(mat_files) == 0:
            print("No .mat files found in the folder.")

        else:
            print(
                "Multiple .mat files found in the folder. Please specify the file to process.")

    # Desired sampling frequency in Hz
    desired_frequency = 257

    resampled_ECG_data = process_and_resample_mat_files_in_folder(
        data_folder_path)

    # This part resized and standardizes the above resampled numpy array. A numpy array of shape (12, 4096) will be returned.

    # Standardizing the data points

    def standardize_array(arr):
        mean = np.mean(arr)
        std = np.std(arr)
        standardized_arr = (arr - mean) / std
        return standardized_arr

    def load_and_resize(resampled_ECG_nparray):

        # If each row contains more than 4096 data points, shorten it.
        if resampled_ECG_nparray.shape[1] > 4096:
            reshaped_data = np.array([row[:4096]
                                     for row in resampled_ECG_nparray])

        # If each row has lesser no of data points than 4096, zero pad each row until each have 4096 data points.
        elif resampled_ECG_nparray.shape[0] < 4096:
            padding = 4096 - resampled_ECG_nparray.shape[1]
            reshaped_data = np.pad(resampled_ECG_nparray,
                                   ((0, 0), (0, padding)), mode='constant')

        # Normalize array
        final_data_array = standardize_array(reshaped_data)

        return final_data_array

    # finalized ECG data record - ( Resampled, resized and normalized )
    final_ECG_data_array = load_and_resize(resampled_ECG_data)

    # Here we have defined the model structure and loaded it.

    num_classes = 23
    num_leads = final_ECG_data_array.shape[0]
    num_data_points = final_ECG_data_array.shape[1]

    class MultiInputECGClassifier(nn.Module):
        def __init__(self, num_classes):
            super(MultiInputECGClassifier, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=num_leads,
                                   out_channels=16, kernel_size=3)
            self.pool = nn.MaxPool1d(kernel_size=4)
            # Adding 2 for age and sex
            self.fc1 = nn.Linear(16 * ((num_data_points - 4) // 4) + 2, 128)
            self.dropout1 = nn.Dropout(p=0.6)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, ecg, age, sex):
            ecg_features = self.pool(torch.relu(self.conv1(ecg)))
            ecg_features = ecg_features.view(
                ecg_features.size(0), -1)  # Flatten the tensor
            age = age.unsqueeze(1)  # Expand dimensions to match ecg_features
            sex = sex.unsqueeze(1)  # Expand dimensions to match ecg_features
            combined_features = torch.cat((ecg_features, age, sex), dim=1)
            combined_features = torch.relu(self.fc1(combined_features))
            combined_features = self.dropout1(combined_features)
            output = self.fc2(combined_features)
            return output

    model = MultiInputECGClassifier(num_classes)

    # Load the saved model's state dictionary
    model.load_state_dict(torch.load(
        "Trained_model.pth"))

    # Change dimensions of the final ECG data array before making it a tensor.
    Final_ECG_Data_Array = np.expand_dims(final_ECG_data_array, axis=0)

    # Convert validation data to PyTorch tensors (similar to what was done during training)
    X_val_tensor = torch.tensor(Final_ECG_Data_Array, dtype=torch.float32)
    age_val_tensor = torch.tensor(input_age_data, dtype=torch.float32)
    sex_val_tensor = torch.tensor(input_sex_data, dtype=torch.float32)

    # label_19_array = ['111975006', '164873001', '164889003', '164909002', '164934002', '17338001', '270492004', '39732003', '425623009',
    #                   '426177001', '426783006', '427084000', '427393009', '428750005', '445118002', '59118001', '67741000119109', '698252002', '713426002']

    label_23_array = ['426783006', '164865005', '164861001', '428750005', '59118001', '164889003', '39732003', '164934002', '164873001', '270492004', '164884008', '426177001',
                      '164867002', '429622005', '284470004', '164909002', '111975006', '427084000', '713426002', '67741000119109', '164951009', '164930006', '10370003', '17338001']

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculations
    with torch.no_grad():
        # Forward pass on validation data
        val_outputs = model(X_val_tensor, age_val_tensor, sex_val_tensor)

        # Get predicted class labels
        predicted_labels = val_outputs.argmax(dim=1)

    predicted_output = label_23_array[predicted_labels.item()]

    # Load the CSV data into a 2D list
    data_list = []
    with open('Dx_map.csv', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row
        for row in csvreader:
            data_list.append([row[0], row[1], row[2]])

    # Function to find the names and abbreviations based on a list of values
    def find_names_and_abbreviations(target_values):
        results = []
        for target_value in target_values:
            for row in data_list:
                if target_value in row:
                    results.append(row[0]+ " - "+ row[2])  # Append the Name and Abbreviation

        return results

    # Find the relevant names and abbreviations for both predicted output and real values
    predicted_outputs=[str(predicted_output)]
    predicted_output_info = find_names_and_abbreviations(predicted_outputs)[0]
    real_value_info = find_names_and_abbreviations(input_lable_data)
    print('predicted output :', predicted_output_info)
    print('real value       :', real_value_info)
    print(age_value)
    print(sex_type)
    print(final_ECG_data_array.shape)
   

    return predicted_output_info, real_value_info ,age_value ,sex_type ,final_ECG_data_array
