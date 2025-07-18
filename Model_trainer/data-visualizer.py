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



# Create an empty 2D NumPy array with a shape of (999, 3)
rows = 43099
cols = 3
col =1    
preproccesed_header_data = np.zeros((rows, cols))        #preprccessed_header_data = [Age,Male,Female]

lable_data = np.empty(rows, dtype='<U1000') 
age_data = np.empty(rows)
sex_data = np.empty(rows)

 

def parse_header(file_path):
    header_info = {'Age': None, 'Sex': None, 'Dx': None, 'Rx': None, 'Hx': None}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            f_name = line
            if line.startswith('# Age:'):
                age_value = line.split(': ')[-1]
                if age_value == 'NaN' or int(age_value) <= 0 or int(age_value) > 100 :
                    #header_info['Age'] = None
                    preproccesed_header_data[i][0] = mean_age/100
                    age_data[i] = mean_age/100              ########
                else :
                    preproccesed_header_data[i][0] = (int(age_value))/100
                    age_data[i]=(int(age_value)/100)
            elif line.startswith('# Sex:'):
                sex_type = line.split(': ')[-1]
                if sex_type == 'Male' or sex_type == 'M':
                    preproccesed_header_data[i][1]=1
                    sex_data[i] = 1

                elif sex_type == 'Female' or sex_type == 'F':
                    preproccesed_header_data[i][2]=1
                    sex_data[i] = 0
                else:    
                    preproccesed_header_data[i][1]=1
                    sex_data[i] = 1

            """elif line.startswith('# Dx:'):
                dx_value = line.split(': ')[-1]
                header_info['Dx'] = dx_value
                lable_data[i] = dx_value"""
              
    return header_info


#root_folder = 'C:/Users/rwkos/Desktop/DSE PROJECT/classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2/training/georgia'
#root_folder = 'C:/Users/rwkos/Desktop/MVP Code/Final data Georgia'
#base_folder = 'C:/Users/rwkos/Desktop/Final Data Preprocess/Final Data/'
base_folder =  '/kaggle/input/final-dataset-ecg/Final Data/'
"""root_folder = base_folder + 'georgia_final'     """
#not_label_19_array = ['11157007', '164884008', '164890007', '164917005', '164921003', '164930006', '164931005', '195042002', '195060002', '195080001', '195101003', '195126007', '233917008', '251120003', '251146004', '251266004', '251268003', '253339007', '253352002', '27885002', '284470004', '425419005', '426434006', '426648003', '426664006', '426761007', '426995002', '428417006', '429622005', '445211001', '47665007', '55930002', '59931005', '713422000', '713427006', '74390002', '89792004']
label_19_array = ['111975006', '164873001', '164889003', '164909002', '164934002', '17338001', '270492004', '39732003', '425623009', '426177001', '426783006', '427084000', '427393009', '428750005', '445118002', '59118001', '67741000119109', '698252002', '713426002']
# Iterate through .hea files and parse header information
data = []
"""for file_name in os.listdir(folder_path):
    if file_name.endswith('.hea'):
        file_path = os.path.join(folder_path, file_name)
        header_info = parse_header(file_path)
        data.append(header_info)"""
data_paths = []
ecg_array = []        
i=0

data_sets = ['cpsc_2018_extra_final','cpsc_2018_final','georgia_final','ptb_final','ptb-xl_final','st_petersburg_incart_final']
for item in data_sets:
    root_folder = base_folder + item
    for root, _, files in os.walk(root_folder):
        #print(type(root))
        parts = root.split("/")
        # Get the last element from the list of parts
        last_part = parts[-1]
        output_str = last_part.split('\\')[-1]
        print(last_part)
        
        if last_part == "cpsc_2018_final":
            mean_age = 64
        elif last_part == "cpsc_2018_extra_final":
            mean_age = 65
        elif last_part == "georgia_final":
            mean_age = 62
        elif last_part == "ptb_final":
            mean_age = 56.3
        elif last_part == "ptb-xl_final":
            mean_age = 61
        k=1
        input_matfile = True
        for file_name in files:
            k+=1
            if file_name.endswith('.hea'):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        f_name = line
                        if line.startswith('# Dx:'):
                            dx_value = line.split(': ')[-1]
                            #header_info['Dx'] = dx_value
                            
                            """
                            # String to check
                            string_to_check = dx_value

                            # Check if the string is NOT in the array
                            is_in_array = np.isin(string_to_check, label_19_array)

                            # Print the result
                            if is_in_array:
                                lable_data[i] = dx_value
                                header_info = parse_header(file_path) 
                                i=i+1
                                input_matfile = True   
                            else:
                                #continue
                                input_matfile = False       """ 
                            lable_data[i] = dx_value
                            header_info = parse_header(file_path) 
                            i=i+1
                            input_matfile = True
                #data.append(header_info)
            elif file_name.endswith('.mat') and input_matfile:
                #print(str(os.getcwd()))
                #data_paths.append(os.path.join(folder_path, file_name))
                #data_paths.append('C:/Users/rwkos/Desktop/CNN/cpsc_2018/g1/'+file_name)
                data = io.loadmat(root_folder +'/'+ output_str +'/'+ file_name)  # Load .mat file
                array = data['final_data']         # Replace 'array' with the actual variable name in your .mat file
                ecg_array.append(array)
                #print(ecg_array.shape)
                #print(file_name)
                data_paths.append(root_folder+file_name)
                
import os
import numpy as np

# Specify the folder where you want to save the arrays
output_folder = "/kaggle/working/ecg_array/"

# Create the folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate through the list of arrays and save them to separate files
for i, array in enumerate(ecg_array):
    # Generate a unique filename for each array (e.g., array_0.npy, array_1.npy)
    filename = f"array_{i}.npy"

    # Create the full file path by combining the folder and filename
    file_path = os.path.join(output_folder, filename)

    # Save the array to the specified file
    np.save(file_path, array)


import numpy as np

# Save the ndarray to the specified file
np.save("/kaggle/working/lable_data.npy", lable_data)
output_folder = "/kaggle/working/ecg_array/"

# Initialize an empty list to store the loaded arrays
loaded_ecg_array = []

# List the files in the output folder
for filename in os.listdir(output_folder):
    if filename.endswith(".npy"):  # Check if the file is a NumPy array file
        # Create the full file path by combining the folder and filename
        file_path = os.path.join(output_folder, filename)

        # Load the array from the file using np.load()
        loaded_array = np.load(file_path)

        # Append the loaded array to the list
        loaded_ecg_array.append(loaded_array)
import numpy as np

# Specify the file path where you saved the ndarray
file_path = "/kaggle/working/lable_data.npy"

# Load the ndarray from the saved file
loaded_lable_data = np.load(file_path)
X_array = np.stack(loaded_ecg_array, axis=0)
real_labels_array = loaded_lable_data.copy()
Y_array = np.array([item.split(',')[0] for item in lable_data]) 
