from machineLearn4 import LSTMModel
from generatePatientData import GenPatientData
import numpy as np
import json
from tensorflow.keras import backend as K
ml=LSTMModel(256,20,4,20)
ml.load_weights("OrginalModels/weights20-20-age-3layers-256-time.h5")
# ml.load_weights("PatientModels/Patient_adolescent#007_Dexcom_Insulet_patientDataweights20-20-age-3layers-256-time.h5")
ml.load_scalar("scalar20-20-age-3layers-256-time")
# K.set_value(ml.model.optimizer.learning_rate, 0.001)
# ml.model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'],run_eagerly=True)
file_name="adult#009_Dexcom_Insulet_patientData.json"
data=json.loads(open("PatientData/"+file_name).read())
data=np.array([data['time'],data['glucose'],data['meal'],data['action']]).T
print("Data",data)
data=ml.normalize_data(data).reshape(1,-1,4)
# f=open("PatientModels/"+file_name.split("_")[0]+"_"+file_name.split("_")[1]+"_"+file_name.split("_")[2]+"_patientData_normalized.json",'w')
# f.write(json.dumps(data.tolist())) 
# f.close()
# exit()
print("Data after norm",data)
data,y,ages=ml.create_look_back(data,GenPatientData(file_name.split("_")[0],file_name.split("_")[1],file_name.split("_")[2]).env.age)

print("Y:",y)
print("Ages",ages)
# exit()
ml.train([data,ages],y,epochs=30)
ml.model.save_weights("PatientModels/Patient_"+file_name.replace(".json","")+"weights20-20-age-3layers-256-time.h5")
# ml.create_look_back()