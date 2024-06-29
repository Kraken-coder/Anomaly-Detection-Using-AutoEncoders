import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

class Data:
    def __init__(self ,*args):
        if(len(args) == 2):
            self.dataframe = pd.read_csv(args[0], low_memory= False , encoding = args[1])
        else:
            self.dataframe = pd.read_csv(args[0] , low_memory= False)
    def columns(self):
        return self.dataframe.columns
    def data_encode_dict(self, column_name , string):
        if string == True:
            string_list = list(set(self.dataframe[column_name].astype(str).to_list()))
            string_to_number = {}
            for i in range(len(string_list)):
                string_to_number[string_list[i]] =i
            number_to_string = {}
            for i in range(len(string_list)):
                number_to_string[i] = string_list[i]
            return string_to_number, number_to_string
        else:
            number_list = list(set(self.dataframe[column_name].to_list()))
            number_to_dictnumber = {}
            for i in range(len(number_list)):
                number_to_dictnumber[number_list[i]] = i
            dictnumber_to_number = {}
            for i in range(len(number_list)):
                dictnumber_to_number[i] = number_list[i]
            return number_to_dictnumber , dictnumber_to_number # encode , decode
    def data_encoder(self,*args):
        column_name = args[0]
        str_to_number = args[1]
        string = args[2]
        if len(args) == 3:
            input_list = self.dataframe[column_name].astype(str).to_list()
            output_list = []
            for inputs in input_list:
                output_list.append(str_to_number[inputs])
            return output_list
        else:
            input_list = self.dataframe[column_name].to_list()
            output_list = []
            for inputs in input_list:
                output_list.append(str_to_number[inputs])
            return output_list
    def data_decoder(self,*args):
        output_list = args[0]
        number_to_str = args[1]
        Decoded_list = []
        for elements in output_list:
            Decoded_list.append(number_to_str[elements])
        return Decoded_list

class AutoEncoder:
    def __init__(self, input_dims):
        self.input_dims = input_dims
        self.anomaly_threshold = 0
        self.autoencoder = Sequential([
            Dense(64, input_dim= input_dims, activation='relu'),  # Increase neurons in the first hidden layer
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(2, activation='relu'),  # Bottleneck layer
            Dense(8, activation='relu'),
            Dense(16, activation='relu'),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(input_dims, activation='relu')  # Output layer must match input dimension
        ])
    def compilefit(self,X_train,epochs , batchsize):
        self.autoencoder.compile(optimizer = 'adam' , loss = 'mse')
        self.autoencoder.fit(X_train,X_train,epochs = epochs , batch_size=batchsize)
        modelpredictions = self.autoencoder.predict(X_train)
        reconstruction_error = np.mean(np.square(X_train - modelpredictions), axis=1)
        self.anomaly_threshold = np.percentile(reconstruction_error, 95)  # 95th percentile as threshold
        plt.figure(figsize=(8, 8))
        plt.hist(reconstruction_error, bins=50)
        plt.axvline(self.anomaly_threshold, color='r', linestyle='--', label='Threshold')
        plt.legend()
        plt.title('Reconstruction Error Distribution')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.show()
        self.autoencoder.save_weights('auto_encoder.weights.h5')

    def validation(self, X_validation):
        modelpredictions = self.autoencoder.predict(X_validation)
        plt.figure(figsize=(8, 8))
        plt.scatter(X_validation.flatten(), modelpredictions.flatten(), c=modelpredictions.flatten(), cmap='viridis')
        plt.colorbar(label='Label')
        plt.title('Multidimensional Data')
        plt.xlabel('Actual')
        plt.ylabel('Model Predictions')
        plt.show()

    def easyAutoEncoder(self,*args):
        df = Data(args[0])
        epochs = args[1]
        encode_dictstore = []
        decode_dictstore = []
        for i in range(2,len(args)):
            encode_dict , decode_dict = df.data_encode_dict(args[i],True)
            encode_dictstore.append(encode_dict)
            decode_dictstore.append(decode_dict)
        outputtens = []
        for k in range(2,len(args)):
            outputtens.append(df.data_encoder(args[k],encode_dictstore[k-2], True))
        untranspose = np.array(outputtens)
        final_input_tens = np.transpose(untranspose)
        self.compilefit(final_input_tens,epochs,32)

    def CheckAnomaly(self,check):
        reconstruction = self.autoencoder.predict(check)
        reconstruction_error = np.mean(np.square(check - reconstruction), axis=1)
        return reconstruction_error > self.anomaly_threshold

















