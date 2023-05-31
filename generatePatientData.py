from GlucoseEnv import GlucoseEnvironment
from machineLearn4 import LSTMModel
import numpy as np
class GenPatientData():
    def __init__(self, patient_name='adolescent#001',sensor="Dexcom",pump="Insulet"):
        self.patientData = {'time':[],'glucose':[],'action':[],'meal':[]}
        self.patient_name=patient_name
        self.sensor=sensor
        self.pump=pump
        self.env= GlucoseEnvironment()
        self.env.reset(patient_name=patient_name,sensor=sensor,pump=pump)
        self.glucoseModel=LSTMModel(256,20,4,20)
        self.glucoseModel.load_weights("OrginalModels/weights20-20-age-3layers-256-time.h5")
        self.glucoseModel.load_scalar("scalar20-20-age-3layers-256-time")
        self.insulinModel=self.glucoseModel.create_insulin_full_model()
        self.insulinModel.load_weights("OrginalModels\model_20_action_weights_dqn_keras.h5")
        # self.glucoseModel.model.compile_model()

    def run(self,loops):
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
    def save_data(self):
        import json
        import os
        if not os.path.exists("PatientData"):
            os.makedirs("PatientData")
        
        with open("PatientData/"+self.patient_name+"_"+self.sensor+"_"+self.pump+'_patientData.json', 'w') as outfile:
            json.dump(self.patientData, outfile)
        

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
