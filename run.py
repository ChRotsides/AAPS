import json
import sys
import numpy as np
from machineLearn4 import LSTMModel
import pandas as pd
import numpy as np
import subprocess
def compute_reward(glucose):
    if glucose >= 90 and glucose<= 140: 
        reward = 1
    elif (glucose >= 70 and glucose < 90) or (glucose > 140 and glucose <= 180):
        reward = 0.1
    elif (180<glucose and glucose<=300):
        reward = -0.4-(glucose-180)/200
    elif (30<=glucose and glucose<70):
        reward = -0.6+(glucose-180)/200
    else:
        reward=-1
    return reward
def reverse_map_value(x):
    return x * 19 / 0.08
def train_glucose(file_name,epochs=50):
    ml=LSTMModel(256,20,4,20)
    patient_name="adolescent#001"
    ml.load_weights("GlucosePatientModels\Patient_"+patient_name+"_Dexcom_Insulet_patientDataweights20-20-age-3layers-256-time.h5")
    # ml.load_weights("PatientModels/Patient_adolescent#007_Dexcom_Insulet_patientDataweights20-20-age-3layers-256-time.h5")
    ml.load_scalar("scalar20-20-age-3layers-256-time")
    # K.set_value(ml.model.optimizer.learning_rate, 0.001)
    # ml.model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'],run_eagerly=True)
    file_name=file_name+".json"
    data=json.loads(open("PatientData/"+file_name).read())
    input_data=data['data']
    time=[]
    glucose=[]
    meal=[]
    action=[]
    for i in range(len(input_data)):
        time.append(float(input_data[i]['time']%864000))
        glucose.append(float(input_data[i]['glucose']))
        meal.append(float(input_data[i]['meal']))
        action.append(float(input_data[i]['insulin']))
    # {"glucose": 155.66417, "insulin": 0.029475, "meal": 0, "time": 1685247487000}

    input_data=np.array([time,glucose,meal,action]).T
    print("Data",input_data)
    input_data=ml.normalize_data(input_data).reshape(1,-1,4)
    print("Data after norm",input_data)
    input_data,y,ages=ml.create_look_back(input_data,float(data['age']))

    print("Y:",y)
    print("Ages",ages)
    # exit()
    ml.train([input_data,ages],y,epochs=epochs)
    save_file_name_glucose="GlucosePatientModels/Patient_"+file_name.replace(".json","")+"weights20-20-age-3layers-256-time.h5"
    ml.model.save_weights(save_file_name_glucose)
    # Insulin
        # Convert JSON data to DataFrame
    # file_name=file_name+".json"
    # data=json.loads(open("PatientData/"+file_name).read())
    input_data2=data['data']
    df = pd.DataFrame(input_data2)

    # Apply reward function to glucose level
    df['reward'] = df['glucose'].apply(compute_reward)

    # Propagate reward to next 20 timesteps
    df['q_value'] = df['reward'].rolling(window=20, min_periods=1).sum()

    # Now shift the propagated reward 20 steps into the future
    df['q_value'] = df['q_value'].shift(-20)

    # Drop the last 20 samples where the reward is NaN due to the shift
    df = df.dropna()

    # Filter data to include only those samples with positive rewards
    reward_indexes = df.index[df['q_value'] > 0].tolist()
    df = df[df['q_value'] > 0]
    print(reward_indexes)
    # Prepare the data for training
    X = df[['time','glucose', 'meal', 'insulin']]
    y = df['q_value']
    filtered_reward_indexes = []
    filtered_y = []  # Create an empty pandas Series

    for i, r in enumerate(reward_indexes):
        if r < len(input_data):
            filtered_reward_indexes.append(r)
            filtered_y.append(y.iloc[i])

    reward_indexes = filtered_reward_indexes
    y = filtered_y
    y=np.array(y)
    yss=[]
    for i in range(len(y)):
        print(round(reverse_map_value(action[reward_indexes[i]])))
        ys=np.zeros(20)-0.01
        ys[min(round(reverse_map_value(action[reward_indexes[i]])),19)]=y[i]/y.max()
        yss.append(ys)
    # print(yss)

    insulin_model=ml.create_insulin_full_model()
    insulin_model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])
    insulin_model.load_weights("InsulinPatientModels\Insulin_adolescent#001_model_20_action_weights_dqn_keras.h5")


    
    input_data=input_data[reward_indexes,:,:]
    ages=ages[reward_indexes]
    pred=ml.predict([input_data,ages])
    yss=np.array(yss)
    # print(pred.shape)
    # print(yss.shape)
    # print(yss)
    # print(input_data[:,:1:].shape)
    insulin_model.fit([input_data[:,:,1:],pred],yss,epochs=epochs)
    save_file_name_insulin="InsulinPatientModels\Insulin_"+file_name.replace(".json","")+"_model_20_action_weights_dqn_keras.h5"
    insulin_model.save_weights(save_file_name_insulin)
    return save_file_name_glucose,save_file_name_insulin

def train_insulin(file_name,glucose_model_location,epochs=50):
    pass


def run():
    patient_data_file_name=""
    epochs=50
    try:
        patient_data_file_name=sys.argv[1]
    except:
        print("Please provide patient data file name")
        print("defaulting to patient_data")
        patient_data_file_name="patient_data"
    try:
        epochs=int(sys.argv[2])
    except:
        print("Please provide number of epochs to train")
        exit()


    f=open("PatientData/"+patient_data_file_name+".json",'r')
    data=json.loads(f.read())
    f.close()
    split_data=[]
    temp_data=[]
    for i in range(0,len(data['data'])-1):
        dt=data['data']
        time_current=dt[i]['time']
        time_next=dt[i+1]['time']
        print(abs(time_next-time_current))
        if abs(time_next-time_current)<=180000:
            temp_data.append(dt[i])
        else:
            split_data.append(np.array(temp_data))
            temp_data=[]

    split_data=np.array(split_data,dtype=object)

    glucose_model_file_location,insulin_model_file_location=train_glucose(patient_data_file_name,epochs=epochs)

    # define the script path
    script_path = "tfliteGlucoseModelExporter.py"

    # define the arguments
    args = [glucose_model_file_location]

    # build the command
    command = ["python", script_path] + args

    # call the script
    subprocess.call(command)
    script_path = "tfliteInsulinModelExporter.py"

    # define the arguments
    args = [insulin_model_file_location]

    # build the command
    command = ["python", script_path] + args

    # call the script
    subprocess.call(command)


if __name__ == '__main__':
    run()