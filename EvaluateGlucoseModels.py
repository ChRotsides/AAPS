


import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
from machineLearn4 import LSTMModel
from GlucoseEnv import GlucoseEnvironment
import matplotlib.lines as mlines
env=GlucoseEnvironment()


def average_glucose_prediction(predictions):
    avgs_arr = np.zeros((predictions.shape[1] + predictions.shape[0] - 1))
    times = np.zeros((predictions.shape[1] + predictions.shape[0] - 1))

    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            avgs_arr[i + j] += predictions[i, j]
            times[i + j] += 1

    # Avoid division by zero error
    times[times == 0] = 1

    return avgs_arr / times
patients=[#"adolescent#001","adolescent#002","adolescent#007","adult#001","adult#002","adult#009","child#002","child#006",
          "child#008"]


class EvaluateGlucoseModels:
        def __init__(self, patient_name='adolescent#001',sensor="Dexcom",pump="Insulet"):
            self.patientData = {'time':[],'glucose':[],'action':[],'meal':[]}
            self.patient_name=patient_name
            self.sensor=sensor
            self.pump=pump
            self.env= GlucoseEnvironment()
            self.env.reset(patient_name=patient_name,sensor=sensor,pump=pump)
            self.glucoseModel=LSTMModel(256,20,4,20)
            self.glucoseModel.load_weights("GlucosePatientModels/Patient_"+patient_name+"_"+sensor+"_"+pump+"_patientDataweights20-20-age-3layers-256-time.h5")
            self.glucoseModel.load_scalar("scalar20-20-age-3layers-256-time")
            self.insulinModel=self.glucoseModel.create_insulin_full_model()
            self.insulinModel.load_weights("InsulinPatientModels\Insulin_child#002_model_20_action_weights_dqn_keras.h5")
            # self.glucoseModel.model.compile_model()

        def run(self,loops):
            predictions=[]
            for i in range(0,loops):
                #  print("Hello")
                action=0
                if len(self.patientData['time'])>20:
                    # patient_20data=self.patientData[-20:]
                    # print(patient_20data["time"])
                    # exit()
                    data=[self.patientData['time'][-20:],self.patientData['glucose'][-20:],self.patientData['meal'][-20:],self.patientData['action'][-20:]]
                    data=np.array(data).T
                    print(data.shape)
                    data=self.glucoseModel.normalize_data(data).reshape(1,-1,4)
                    
                    # print(input_data)
                    age=self.env.age.reshape(1,-1)
                    prediction=self.glucoseModel.predict([data,age])
                    predictions.append(prediction)
                    # action_probs = model_20_action([state.reshape(-1,20,3),extra_info.reshape(-1,20,1)], training=False)
                    state=data[:,:,1:].reshape(-1,20,3)
                    action_probs=self.insulinModel.predict([state,prediction.reshape(-1,20,1)])
                    action=np.argmax(action_probs[0])
                    if "child" in self.patient_name:
                        action=action*0.04/19
                    else:
                        action=action*0.08/19
                    active_insulin=self.calculateActiveInsulin(self.env.patient.insulin_hist[-20:],self.env.patient.time_hist[-20:],7200000,14400000)
                    if ~self.isInsulinDosageSafe(action,self.env.patient.CGM_hist[-1],self.env.CF,80,active_insulin):
                        action=self.calculateInsulinDosage(self.env.CF,self.env.CR,self.env.patient.CGM_hist[-1],80,self.env.patient.CHO_hist[-1])

                print("action: ",action)
                glucose=self.env.step(action)
                print("glucose: ",glucose)
                self.patientData['time'].append(int(self.env.patient.time_hist[-1].timestamp()*1000)%86400) 
                self.patientData['glucose'].append(glucose)
                self.patientData['action'].append(self.env.patient.insulin_hist[-1])
                self.patientData['meal'].append(self.env.patient.CHO_hist[-1])
                print(i/loops*100,"%")
            return predictions,self.patientData['glucose']
        def save_data(self):
            import json
            import os
            if not os.path.exists("PatientData"):
                os.makedirs("PatientData")
            
            with open("PatientData/"+self.patient_name+"_"+self.sensor+"_"+self.pump+'_patientData.json', 'w') as outfile:
                json.dump(self.patientData, outfile)
            

        def evaluate(self,loops):
            predictions,glucose=self.run(loops)
            # self.save_data()
            # plt.plot(glucose)
            # plt.plot(predictions)
            # plt.show()
            return predictions,glucose

        def get_loops(self,days=1,hours=0):
            sample_time=self.env.sensor.sample_time
            loops=int((days*24*60)/sample_time)+int((hours*60)/sample_time)
            return loops


        def reset(self,patient_name='adolescent#001',sensor="Dexcom",pump="Insulet"):
            self.env.reset(patient_name=patient_name,sensor=sensor,pump=pump)
        
        def isInsulinDosageSafe(self,predictedInsulinDosage, currentGlucose, correctionFactor, hypoglycemicThreshold, activeInsulin):
            totalInsulin = predictedInsulinDosage + activeInsulin
            predictedGlucose = currentGlucose - (totalInsulin * correctionFactor)
            return predictedGlucose >= hypoglycemicThreshold

        def calculateActiveInsulin(self,previousDoses, previousDoseTimes, peakTime, insulinDuration):
            activeInsulin = 0
            for i in range(0,len(previousDoses)):
                timeSinceDose = previousDoseTimes[i].timestamp() * 1000
                insulinActivity = 0

                if timeSinceDose < peakTime:
                    insulinActivity = (timeSinceDose * timeSinceDose) / (peakTime * peakTime)
                elif timeSinceDose < insulinDuration:
                    insulinActivity = 1 - ((timeSinceDose - peakTime) * (timeSinceDose - peakTime)) / ((insulinDuration - peakTime) * (insulinDuration - peakTime))

                activeInsulin += previousDoses[i] * insulinActivity
            return activeInsulin

        def calculateInsulinDosage(self,correctionFactor, carbRatio, currentGlucose, targetGlucose, carbohydrateIntake):
            correctionDose = (currentGlucose - targetGlucose) / correctionFactor
            carbDose = carbohydrateIntake / carbRatio
            totalInsulinDosage = max(correctionDose + carbDose, 0)
            return totalInsulinDosage

ml=LSTMModel(256,20,4,20)
ml.load_scalar("scalar20-20-age-3layers-256-time")
for patient in patients:
    evaluator=EvaluateGlucoseModels(patient)
    predictions,glucose=evaluator.run(evaluator.get_loops(days=7,hours=0))
    predictions=np.array(predictions)
    glucose=np.array(glucose)
    fig1,ax1=plt.subplots()
    fig1.suptitle(patient,fontsize=20)
    print(predictions.shape)
    print(glucose.shape)
    avg_pred=average_glucose_prediction(predictions.reshape(-1,20))

    padded=ml.pad_output(avg_pred)
    norm_rev=ml.reverse_normalize(padded)[:,1]

    print(norm_rev.shape)
    print(norm_rev)
    # exit()
    ax1.plot(norm_rev[:-20],label="Predicted")
    ax1.plot(glucose[20:],label="Patient Data")
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Glucose')
    rmse=(norm_rev[:-18]-glucose[20:])*(norm_rev[:-18]-glucose[20:])
    rmse=np.sqrt(np.sum(rmse)/len(rmse))
    print("RMSE: ",rmse)
    fig1.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    line = mlines.Line2D([], [], color='red', markersize=15, label='RMSE: '+str(rmse)+' mg/dL')
    ax1.legend(handles=[line])
    # fig1.show()
    fig1.savefig("PatientFigs/"+patient+"_"+evaluator.sensor+"_"+evaluator.pump+"_glucose.png")

# def eval():
#         print(data)
#         age=(age-self.min_age)/(self.max_age-self.min_age)
#         data=data.reshape(1,data.shape[0],data.shape[1])
#         patient_test_,patient_test_res=self.create_look_back(data)
#         print("patient_test_res_shape",patient_test_res.shape)
#         print("patient_test_",patient_test_.shape)
#         age=np.zeros(patient_test_.shape[0])+age
#         print(patient_test_.shape,age.shape)

#         # self.train([patient_test_,age],patient_test_res,10)
#         print(patient_test_.shape, age.shape)
#         exit()
#         result=self.predict([patient_test_, age])
#         patient_test_res
#         print("Normal Result Shape",patient_test_res.shape)
#         print("Predicted Result Shape",result.shape)


#         result=self.reshape_output(result)
#         patient_test_res=self.reshape_output(patient_test_res)
#         print("Normal Result Shape",patient_test_res.shape)
#         print("Predicted Result Shape",result.shape)
#         result=self.pad_output(result)
#         patient_test_res=self.pad_output(patient_test_res)

#         print(result)
#         result=self.reverse_normalize(result)
#         patient_test_res=self.reverse_normalize(patient_test_res)
#         print("Normal Result Shape",patient_test_res.shape)
#         print("Predicted Result Shape",result.shape)
#         print(title,rmse(patient_test_res[:,1],result[:,1]))

#         fig1,ax1=plt.subplots()
#         fig1.suptitle(title,fontsize=20)
#         ax1.plot(result[:,1],label="Predicted")
#         ax1.plot(patient_test_res[:,1],label="Patient Data")
#         ax1.set_xlabel('Time')
#         ax1.set_ylabel('Glucose')
#         fig1.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
#         fig1.show()