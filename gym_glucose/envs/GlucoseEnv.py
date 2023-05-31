import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import scipy.signal
import time
from machineLearn4 import LSTMModel
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
from datetime import timedelta
from datetime import datetime
from keras import regularizers
import pkg_resources
import math
import pandas as pd
import warnings
import struct
from collections import namedtuple
from keras.models import load_model
# tf.config.run_functions_eagerly(True)
Action = namedtuple("patient_action", ['CHO', 'insulin'])
action_ =namedtuple("action", ['basal', 'bolus'])
Observation = namedtuple("observation", ['Gsub'])
Observation_ = namedtuple('Observation', ['CGM'])

try:
    from rllab.envs.base import Step
except ImportError:
    _Step = namedtuple("Step", ["observation", "reward", "done", "info"])

    def Step(observation, reward, done, **kwargs):
        """
        Convenience method creating a namedtuple with the results of the
        environment.step method.
        Put extra diagnostic info in the kwargs
        """
        return _Step(observation, reward, done, kwargs)

PATIENT_QUEST_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/Quest.csv')
PATIENT_PARAM_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')
# patient_name='adolescent#001'

def risk_index(BG, horizon):
    # BG is in mg/dL
    # horizon in samples
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        BG_to_compute = BG[-horizon:]
        fBG = 1.509 * (np.log(BG_to_compute)**1.084 - 5.381)
        rl = 10 * fBG[fBG < 0]**2
        rh = 10 * fBG[fBG > 0]**2
        LBGI = np.nan_to_num(np.mean(rl))
        HBGI = np.nan_to_num(np.mean(rh))
        RI = LBGI + HBGI
    return (LBGI, HBGI, RI)

class PatientEnv(T1DSimEnv):
    def __init__(self, patient, sensor, pump, scenario):
        super(PatientEnv, self).__init__(patient, sensor, pump, scenario)
    def risk_diff(BG_last_hour):
        if len(BG_last_hour) < 2:
            return 0
        else:
            _, _, risk_current = risk_index([BG_last_hour[-1]], 1)
            _, _, risk_prev = risk_index([BG_last_hour[-2]], 1)
            return risk_prev - risk_current
    def mini_step(self, action,extra_carbs=0.0):
        # current action
        patient_action = self.scenario.get_action(self.time)
        basal = self.pump.basal(action.basal)
        bolus = self.pump.bolus(action.bolus)
        insulin = basal + bolus
        CHO = patient_action.meal+extra_carbs
        patient_mdl_act = Action(insulin=insulin, CHO=CHO)

        # State update
        # print("Patient mdl act:",patient_mdl_act)
        self.patient.step(patient_mdl_act)

        # next observation
        BG = self.patient.observation.Gsub
        CGM = self.sensor.measure(self.patient)

        return CHO, insulin, BG, CGM
    def step(self, action, reward_fun=risk_diff,extra_carbs=0.0):
        '''
        action is a namedtuple with keys: basal, bolus
        '''
        CHO = 0.0
        insulin = 0.0
        BG = 0.0
        CGM = 0.0

        for _ in range(int(self.sample_time)):
            # Compute moving average as the sample measurements
            tmp_CHO, tmp_insulin, tmp_BG, tmp_CGM = self.mini_step(action,extra_carbs=extra_carbs)
            CHO += tmp_CHO / self.sample_time
            insulin += tmp_insulin / self.sample_time
            BG += tmp_BG / self.sample_time
            CGM += tmp_CGM / self.sample_time

        # Compute risk index
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)

        # Record current action
        self.CHO_hist.append(CHO)
        self.insulin_hist.append(insulin)

        # Record next observation
        self.time_hist.append(self.time)
        self.BG_hist.append(BG)
        self.CGM_hist.append(CGM)
        self.risk_hist.append(risk)
        self.LBGI_hist.append(LBGI)
        self.HBGI_hist.append(HBGI)

        # Compute reward, and decide whether game is over
        window_size = int(60 / self.sample_time)
        BG_last_hour = self.CGM_hist[-window_size:]
        reward = reward_fun(BG_last_hour)
        done = BG < 70 or BG > 350
        obs = Observation_(CGM=CGM)

        return Step(observation=obs,
                    reward=reward,
                    done=done,
                    sample_time=self.sample_time,
                    patient_name=self.patient.name,
                    meal=CHO,
                    patient_state=self.patient.state,
                    time=self.time,
                    bg=BG,
                    lbgi=LBGI,
                    hbgi=HBGI,
                    risk=risk)
    


class GlucoseEnvironment:
    def __init__(self):
        pass

    def reset(self, patient_name='adolescent#001',sensor_name='Dexcom',pump_name='Insulet'):
        self.glucose_predictor=LSTMModel(256,20,4,20)
        # self.glucose_predictor.load_model("model20-20-age-3layers-256-time","scalar20-20-age-3layers-256-time")
        self.glucose_predictor.load_scalar("scalar20-20-age-3layers-256-time")
        self.glucose_predictor.load_weights("GlucosePatientModels\Patient_"+patient_name+"_Dexcom_Insulet_patientDataweights20-20-age-3layers-256-time.h5")
        self.patient_name=patient_name
        self.sensor_name=sensor_name
        self.pump_name=pump_name
        self.patient_ = T1DPatient.withName(patient_name)
        self.sensor = CGMSensor.withName(sensor_name, seed=int(datetime.now().timestamp()))
        self.pump = InsulinPump.withName(pump_name)
        self.start_time=datetime.now()
        self.scenario = RandomScenario(start_time=self.start_time, seed=int(datetime.now().timestamp())+1000)
        self.patient = PatientEnv(self.patient_, self.sensor,self.pump,self.scenario)
        other_params=pd.read_csv(PATIENT_QUEST_FILE)
        self.other_params=other_params.loc[ other_params.Name == self.patient_name]
        patient_params=pd.read_csv(PATIENT_PARAM_FILE)
        self.patient_params=patient_params.loc[ other_params.Name == patient_name]
        self.age=self.other_params['Age'].values[0]
        self.CR=self.other_params['CR'].values[0]
        self.CF=self.other_params['CF'].values[0]
        self.TDI=self.other_params['TDI'].values[0]
        self.BW=self.patient_params['BW'].values[0]
        self.model_input=[]
        for i in range(0,20):
            self.patient.step(action_(basal=0, bolus=0))
            # print(self.patient.time,self.patient.CGM_hist[-1],self.patient.CHO_hist[-1],self.patient.insulin_hist[-1],self.patient.patient._params['BW'],self.age)
            self.model_input.append(np.array([self.patient.time.timestamp()%86400,self.patient.CGM_hist[-1],self.patient.CHO_hist[-1],self.patient.insulin_hist[-1]]))
        self.model_input=self.glucose_predictor.normalize_data(self.model_input)
        self.model_input=np.array(self.model_input).reshape(1,-1,4)
        # print(self.model_input.shape)
        self.max_age=self.glucose_predictor.max_age
        self.min_age=self.glucose_predictor.min_age
        self.age=(self.age-self.min_age)/(self.max_age-self.min_age)
        self.age=self.age.reshape(1,-1)
        prediction=self.glucose_predictor.predict([self.model_input,self.age])
        # print(prediction)
        # print(self.model_input[0,:,1])
        next_state=np.array([self.model_input[0,:,1].transpose(),self.model_input[0,:,2].transpose(),self.model_input[0,:,3].transpose()]).reshape(3,20).transpose()
        extra_info=np.array([prediction[0,:].transpose()])
        # print("State shape: ",state.shape)
        # exit()
        # print(state)
        # exit()
        # state=np.array(state).reshape(-1,1,10)
        return next_state,extra_info

    def step(self, action):
        # next_state,reward,done,info = self.generate_state(action)
        next_state=[]
        
        # for i in range(0,20):
        # exit()

        extra_carbs=0
        # print("CGM last:",self.patient.CGM_hist[-1])
        small_epsilon=0.01
        if np.random.rand() <= small_epsilon:
            extra_carbs=25
            
        self.patient.step(action_(basal=action, bolus=0),extra_carbs=extra_carbs)
            # print(self.patient.time,self.patient.CGM_hist[-1],self.patient.CHO_hist[-1],self.patient.insulin_hist[-1],self.patient.patient._params['BW'],age)
        new_data=np.array([self.patient.time.timestamp()%86400,self.patient.CGM_hist[-1]+extra_carbs,self.patient.CHO_hist[-1],self.patient.insulin_hist[-1]])
        new_data=new_data.reshape(1,-1)
        new_data=self.glucose_predictor.normalize_data(new_data)
        self.model_input=np.append(self.model_input,new_data.reshape((1,1,4)), axis=1)
        self.model_input=np.delete(self.model_input,0,axis=1)

        # print(self.model_input.shape)
        

        prediction=self.glucose_predictor.predict([self.model_input,self.age])

        next_state=np.array([self.model_input[0,:,1].transpose(),self.model_input[0,:,2].transpose(),self.model_input[0,:,3].transpose()]).reshape(3,20).transpose()
        extra_info=np.array([prediction[0,:].transpose()])

        unnorm_data=self.glucose_predictor.reshape_output(prediction)
        unnorm_data=self.glucose_predictor.pad_output(unnorm_data)
        unnorm_data=self.glucose_predictor.reverse_normalize(unnorm_data)
        window_size = int(60 / self.patient.sensor._params['sample_time'])
        BG_last_hour = self.patient.CGM_hist[-window_size:]
        reward = self.calculate_reward(BG_last_hour)
        
        if self.patient.CGM_hist[-1]<40:
            done=True
        elif self.patient.CGM_hist[-1]>350:
            done=True
        else:
            done = False

        # print("Reward: ",reward)
        return next_state,extra_info, reward, done


    def calculate_reward(self, next_state):
        # Define your reward function here based on the next_state
        glucose=next_state[-1]
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
        # _, _, risk_current = risk_index(next_state, 5)
        # _, _, risk_prev = risk_index(next_state, 15)
        # reward= risk_prev - risk_current
        # (_, _, reward) = risk_index(next_state,horizon=15)
        # reward=-reward
        # print("Reward:",reward)
        return reward
    def reward_function(insulin_dose, simulator_output, t):
        if t < 15:
            return 0 # no reward until the effect of insulin is observed
        else:
            # calculate the average of the simulator output over the last 15 time steps
            delayed_reward = np.mean(simulator_output[t-15:t])
            # apply a discount factor to the delayed reward
            discount_factor = 0.9
            discounted_reward = discount_factor ** 15 * delayed_reward
            # penalize large doses of insulin to avoid overdosing
            if insulin_dose > 10:
                penalty = 0.1 * insulin_dose
            else:
                penalty = 0
            # return the discounted reward minus the insulin dose penalty
            return discounted_reward - penalty