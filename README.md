AutoEncoder Anomaly Detection
This repository contains an implementation of an AutoEncoder model for anomaly detection using TensorFlow and Keras. The code is designed to handle data preprocessing, model training, and anomaly detection. It includes classes for data encoding/decoding and the autoencoder model itself.

Table of Contents
Installation
Usage
Data Class
AutoEncoder Class
Example
License
Installation
To get started, clone this repository and install the required dependencies.

bash
Copy code
git clone https://github.com/your-username/autoencoder-anomaly-detection.git
cd autoencoder-anomaly-detection
pip install -r requirements.txt
The requirements.txt file should include the following dependencies:

Copy code
pandas
numpy
tensorflow
matplotlib
Usage
Data Class
The Data class is used for data loading and encoding/decoding.

Initialization
python
Copy code
data = Data('path_to_csv_file.csv')
Optionally, you can specify the encoding:

python
Copy code
data = Data('path_to_csv_file.csv', 'encoding')
Methods
columns(): Returns the columns of the dataframe.
data_encode_dict(column_name, string): Creates dictionaries for encoding and decoding data.
data_encoder(column_name, str_to_number, string): Encodes data based on the provided dictionaries.
data_decoder(output_list, number_to_str): Decodes data based on the provided dictionaries.
AutoEncoder Class
The AutoEncoder class is used to define and train an autoencoder model, and to check for anomalies.

Initialization
python
Copy code
autoencoder = AutoEncoder(input_dims)
Methods
compilefit(X_train, epochs, batchsize): Compiles and fits the autoencoder model.
validation(X_validation): Validates the model on a given dataset.
easyAutoEncoder(*args): An easy setup method for the autoencoder.
CheckAnomaly(check): Checks if the given data is an anomaly.
Example
python
Copy code
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# Initialize the Data class
data = Data('path_to_csv_file.csv')

# Print columns
print(data.columns())

# Create encoding/decoding dictionaries
encode_dict, decode_dict = data.data_encode_dict('column_name', True)

# Encode data
encoded_data = data.data_encoder('column_name', encode_dict, True)

# Initialize AutoEncoder
autoencoder = AutoEncoder(input_dims=len(data.columns()))

# Train AutoEncoder
autoencoder.easyAutoEncoder('path_to_csv_file.csv', 50, 'column1', 'column2', 'column3')

# Check for anomalies
anomalies = autoencoder.CheckAnomaly(np.array("element of column1" ,"element of column2","element of column3" )  # Example input
print("Is anomaly:", anomalies)






