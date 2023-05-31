import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input
from keras.models import load_model
from keras.layers import concatenate
from keras.models import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate
from tensorflow.keras.models import Model, Sequential
import keras.backend as K
import matplotlib.pyplot as plt
import pickle
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
def rmse(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.sqrt(np.square(np.subtract(actual,pred)).mean())

class LSTMModel:
    def __init__(self,units, look_back, num_features_in,num_features_out):

        self.look_back = look_back
        self.num_features_in = num_features_in
        self.num_features_out= num_features_out
        # self.scaler = []
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.model = self._create_model(units)
        self.dataset_size=[]
        self.min_age=7
        self.max_age=68
    def _create_model(self,units):
        input_layer = Input(shape=(self.look_back, self.num_features_in))
        lstm1 = LSTM(units, return_sequences=True, input_shape=(self.look_back, self.num_features_in))(input_layer)
        lstm2 = LSTM(units,return_sequences=True)(lstm1)
        lstm3 = LSTM(int(units/2))(lstm2)
        additional_input = Input(shape=(1))
        concatenated = concatenate([lstm3, additional_input])

        output_layer = Dense(self.num_features_out)(concatenated)

        model = Model(inputs=[input_layer, additional_input], outputs=output_layer)
        model.compile(loss='mean_squared_error', optimizer='adam',metrics=[root_mean_squared_error, 'mean_squared_error'])

        return model

    def train(self, data, labels, epochs):
        self.model.fit(data, labels, epochs=epochs, batch_size=64, verbose=1)

    def summary(self):
        return self.model.summary()

    def reverse_normalize(self, data):
        # data=data*(self.max-self.min)+self.min
        # return data
        return self.scaler.inverse_transform(data)
    def predict(self, data):
        # data = self.prepare_data(data)
        # data = self._normalize_data(data)
        # print(data)
        return self.model.predict(data,verbose=0)

    def prepare_data(self, data):
        data = self._normalize_data(data)
        data,y = self.create_look_back(data)
        return data,y
    def pad_output(self,data):
        padding=np.zeros(data.shape[0])
        data=np.c_[padding,data]
        data=np.c_[data,padding]
        data=np.c_[data,padding]
        return data
    
    def create_insulin_full_model(units=256):
        inputs1 = tf.keras.Input(shape=(20,3))
        lstm0 = tf.keras.layers.LSTM(256, return_sequences=True, input_shape=(20, 3))(inputs1)
        dense1 = tf.keras.layers.Dense(4)(lstm0)
        lstm1 = tf.keras.layers.LSTM(256, return_sequences=True, input_shape=(20, 4))(dense1)
        lstm2 = tf.keras.layers.LSTM(256,return_sequences=True)(lstm1)
        lstm3 = tf.keras.layers.LSTM(int(128),return_sequences=True)(lstm2)
        additional_input = tf.keras.Input(shape=(20,1))
        concatenated = tf.keras.layers.concatenate([lstm3, additional_input])
        new_layer=tf.keras.layers.GRU(64, return_sequences=False)(concatenated)
        output_layer = tf.keras.layers.Dense(20,activation='linear')(new_layer)
        model2=tf.keras.Model(inputs=[inputs1,additional_input],outputs=output_layer)
        model2.layers[3].trainable = False
        model2.layers[4].trainable = False
        model2.layers[5].trainable = False
        return model2

    def reshape_output(self,data):
        re_data=[]
        for i in range(0,len(data),20):
            for j in range(0,len(data[i])):
                re_data.append(data[i,j])
        return np.array(re_data)

    def _normalize_data(self, data):
        # self.min=data.min()
        # self.max=data.max()
        # data=(data-data.min())/(data.max()-data.min())
        # print(data)
        # for i in range(data.shape[2]):
        #     scaler=MinMaxScaler(feature_range=(0, 1))
        #     data[:,:,i]=scaler.fit_transform(data[:,:,i])
        #     self.scaler.append(scaler)



        data=self.scaler.fit_transform(data)
        return data
        
        # if(data.std() == 0):
        #     print("Skipping normalization")
        # else:
        #     return self.scaler.fit_transform(data)

    def normalize_data(self, data):
        # data=(data-self.min)/(self.max-self.min)

        data=self.scaler.transform(data)
        # for i in range(data.shape[2]):
        #     scaler=self.scaler[i]
        #     data[:,:,i]=scaler.transform(data[:,:,i])
            

        return data
        # return self.scaler.transform(data)

    # def create_look_back(self, data):
    #     look_back_data = np.zeros((data.shape[0]-self.look_back, self.look_back, self.num_features))
    #     y = np.zeros((data.shape[0]-self.look_back, 1))
    #     for i in range(len(data)-self.look_back):
    #         look_back_data[i] = data[i:(i+self.look_back), :]
    #         y[i] = data[i+self.look_back]
        #     return look_back_data, y
    # def create_look_back(self, data, num_series):
    #         row_num=0
    #         for i in range(len(data)):
    #             row_num+=data[i].shape[0]-self.look_back

    #         look_back_data = np.zeros((row_num,self.look_back,self.num_features))
    #         y = np.zeros((row_num, self.num_features, 1))
    #         for j in range(num_series):
    #             for i in range(len(data)-self.look_back):
    #                 look_back_data[j][i] = data[j][i:(i+self.look_back), :, :]
    #                 y[j][i] = data[j][i+self.look_back, :, :]
    #         return look_back_data, y
    
    def create_look_back(self,data,ages=0):
        look_back_data,y=[],[]
        re_age_data=[]
        # print("Length of Data:",len(data))
        for i in range(len(data)):
            j_range=len(data[i])-self.look_back-self.num_features_out
            # print("J Range:",j_range)
            for j in range(0,j_range):
                # print(data[i,j:j+self.look_back])
                look_back_data.append(data[i,j:j+self.look_back,:])
                y.append(data[i,j+self.look_back:j+self.look_back+self.num_features_out,1])
                if(ages!=0):
                    re_age_data.append(ages)
                # re_age_data
        if (ages!=0):
            return np.array(look_back_data),np.array(y),np.array(re_age_data)
        else:
            return np.array(look_back_data),np.array(y)


    def save_model(self, filepath,filepath_scaler):
        with open(filepath_scaler, 'wb') as file:
            pickle.dump(self.scaler, file)
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.save(filepath)

    def load_model(self, model_filepath, scaler_filepath):
        self.model = load_model(model_filepath)
        with open(scaler_filepath, 'rb') as file:
            self.scaler = pickle.load(file)

    def load_scalar(self, scaler_filepath):
        with open(scaler_filepath, 'rb') as file:
            self.scaler = pickle.load(file)
    def load_weights(self,path):
        self.model.load_weights(path)

    def mix_patients(self,data):
        re_data=[]

        for i in range(data.shape[0]):
            self.dataset_size.append(data.shape[1])
            for j in range(data.shape[1]):
                re_data.append(np.array(data[i,j,:]))
        return np.array(re_data)

    def unmix_patients(self,data):
        re_data=[]
        start=0
        end=0
        for i in range(len(self.dataset_size)):
            end+=self.dataset_size[i]
            # for j in range(start,end):
            re_data.append(data[start:end,:])
            start+=self.dataset_size[i]

        return np.array(re_data)

    def predict_new_patient(self,data,age,title):
        print(data)
        age=(age-self.min_age)/(self.max_age-self.min_age)
        data=data.reshape(1,data.shape[0],data.shape[1])
        patient_test_,patient_test_res=self.create_look_back(data)
        print("patient_test_res_shape",patient_test_res.shape)
        print("patient_test_",patient_test_.shape)
        age=np.zeros(patient_test_.shape[0])+age
        print(patient_test_.shape,age.shape)

        # self.train([patient_test_,age],patient_test_res,10)
        

        result=self.predict([patient_test_, age])
        patient_test_res
        print("Normal Result Shape",patient_test_res.shape)
        print("Predicted Result Shape",result.shape)


        result=self.reshape_output(result)
        patient_test_res=self.reshape_output(patient_test_res)
        print("Normal Result Shape",patient_test_res.shape)
        print("Predicted Result Shape",result.shape)
        result=self.pad_output(result)
        patient_test_res=self.pad_output(patient_test_res)

        print(result)
        result=self.reverse_normalize(result)
        patient_test_res=self.reverse_normalize(patient_test_res)
        print("Normal Result Shape",patient_test_res.shape)
        print("Predicted Result Shape",result.shape)
        print(title,rmse(patient_test_res[:,1],result[:,1]))

        fig1,ax1=plt.subplots()
        fig1.suptitle(title,fontsize=20)
        ax1.plot(result[:,1],label="Predicted")
        ax1.plot(patient_test_res[:,1],label="Patient Data")
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Glucose')
        fig1.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        fig1.show()
        # plt.show()