
# AutoEncoder and Data Preprocessing

This repository contains two classes, `Data` and `AutoEncoder`, implemented using TensorFlow and Keras for data preprocessing and anomaly detection using an autoencoder neural network.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Data Class](#data-class)
  - [AutoEncoder Class](#autoencoder-class)
- [Examples](#examples)
  - [Data Class Example](#data-class-example)
  - [AutoEncoder Class Example](#autoencoder-class-example)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Installation
1. Install the required dependencies:
    ```bash
    pip install tensorflow pandas numpy matplotlib
    ```

## Usage

### Data Class

The `Data` class is used for loading and encoding data from a CSV file.

#### Initialization
```python
data = Data('path/to/your/csvfile.csv', 'encoding')  # Optional encoding
```

#### Methods

- `columns()`
  - Returns the columns of the dataframe.
  
- `data_encode_dict(column_name, string)`
  - Returns a dictionary for encoding and decoding a specific column.
  
- `data_encoder(column_name, str_to_number, string)`
  - Encodes the data in a column using the provided dictionary.
  
- `data_decoder(output_list, number_to_str)`
  - Decodes the data using the provided dictionary.

### AutoEncoder Class

The `AutoEncoder` class is used to create, train, and validate an autoencoder neural network for anomaly detection.

#### Initialization
```python
autoencoder = AutoEncoder(input_dims)
```

#### Methods

- `compilefit(X_train, epochs, batchsize)`
  - Compiles and trains the autoencoder on the training data.
  
- `validation(X_validation)`
  - Validates the autoencoder on the validation data.
  
- `easyAutoEncoder(csv_file, epochs, column_names)`
  - A simplified method to encode data, train the autoencoder, and save the encoding dictionaries.
  
- `CheckAnomaly(values)`
  - Checks if the provided values are anomalies based on the trained model.

## Examples

### Data Class Example
```python
data = Data('data.csv')
print(data.columns())

string_to_number, number_to_string = data.data_encode_dict('column_name', True)
encoded_data = data.data_encoder('column_name', string_to_number, True)
decoded_data = data.data_decoder(encoded_data, number_to_string)
```

### AutoEncoder Class Example
```python
autoencoder = AutoEncoder(input_dims=10)
autoencoder.easyAutoEncoder('data.csv', 50, 'column1', 'column2', 'column3')
anomaly = autoencoder.CheckAnomaly('value1', 'value2', 'value3')
print("Is anomaly:", anomaly)
```

## Dependencies

- TensorFlow
- Pandas
- NumPy
- Matplotlib

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

Feel free to reach out if you have any questions or need further assistance.






