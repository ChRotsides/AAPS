from flask import Flask, request, jsonify, send_file
# import my_python_module
import datetime
import numpy as np
app = Flask(__name__)
from GlucoseEnv import GlucoseEnvironment
from threading import Thread
import json
env = GlucoseEnvironment()
env.reset()
print(env)
env.reset(patient_name='adolescent#002')
# exit()
@app.route('/reset', methods=['POST','GET'])
def reset():
    print("Patient Name recieved:",request.args.get('patient_name'))
    try:
        patient_name = request.args.get('patient_name')
    except:
        patient_name = None
    if patient_name is None:
        glucose=env.reset()
    else:
        glucose=env.reset(patient_name)

    result = {'glucose': glucose}
    return jsonify(result)

@app.route('/get_params', methods=['GET','POST'])
def get_params():
    CR=env.CR
    CF=env.CF
    TDI=env.TDI
    result = {'CR': int(CR),'CF':float(CF),'TDI':float(TDI),'Age':float(env.age)}
    return jsonify(result)


@app.route('/step', methods=['POST'])
def index():
    data = request.get_json()
    action = float(data.get('action'))
    meal = float(data.get('meal'))
    print("action: ",action," meal: ",meal)
    # for i in range(0,20):
    glucose=env.step(action,meal)
    result = {'time':env.patient.time_hist[-1],'glucose': glucose,'action':env.patient.insulin_hist[-1],'meal':env.patient.CHO_hist[-1]}
    return result

@app.route('/get_cgm_hist', methods=['GET','POST'])
def get_cgm_hist():
    data = request.get_json()
    horizon=int(data.get('horizon'))
    print(horizon)
    hist=env.patient.CGM_hist[-horizon:]
    result = {'cgm_hist': hist}
    return result

@app.route('/get_cho_hist', methods=['GET','POST'])
def get_cho_hist():
    print('get_cho_hist')
    data = request.get_json()
    horizon=int(data.get('horizon'))
    print(horizon)
    hist=env.patient.CHO_hist[-horizon:]
    result = {'cho_hist': hist}
    return result

@app.route('/get_insulin_hist', methods=['GET','POST'])
def get_insulin_hist():
    print('get_insulin_hist')
    data = request.get_json()
    horizon=int(data.get('horizon'))
    print(horizon)
    hist=env.patient.insulin_hist[-horizon:]
    result = {'insulin_hist': hist}
    return result

@app.route('/get_time_hist', methods=['GET','POST'])
def get_time_hist():
    print('get_time_hist')
    data = request.get_json()
    horizon=int(data.get('horizon'))
    print(horizon)
    hist=env.patient.time_hist[-horizon:]
    for i in range(len(hist)):
        hist[i]=int(hist[i].timestamp()*1000)
    result = {'time_hist': hist}
    return result

@app.route('/get_age', methods=['GET'])
def get_age():
    print('get_age')
    age = env.age
    ages = []
    for i in range(0, 20):
        ages.append(int(age))  # Convert age to a native Python int
    result = {'age': ages}
    return jsonify(result)

@app.route('/get_past_data', methods=['POST'])
def get_past_data():
    print('get_past_data')
    data = request.get_json()
    t=Thread(target=save_data,args=(data,))
    t.start()
    result = {'DataRecieved': True}
    return jsonify(result)

@app.route('/get_model/<model_name>', methods=['GET'])
def get_model(model_name):
    return send_file(f'AndroidModels/{model_name}.tflite', as_attachment=True)
@app.route('/pre_trained_insulin_get_model/<patient_name>/<part>', methods=['GET'])
def get_insulin_model(patient_name,part):
    return send_file(f'SplitInsulinModels/Insulin_{patient_name}_model_20_action_weights_dqn_keras-part{part}.tflite', as_attachment=True)
@app.route('/pre_trained_glucose_get_model/<patient_name>/<part>', methods=['GET'])
def get_glucose_model(patient_name,part):
    return send_file(f'SplitGlucoseModels/{patient_name}_Dexcom_Insulet_patientDataweights20-20-age-3layers-256-time-part{part}.tflite', as_attachment=True)

def save_data(data):
    # print(data)
    with open('PatientData\patient_data.json', 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    import sys
    ip=''
    port=''
    if sys.argv[1] is None:
        ip='192.168.0.2'
        port='5000'
    else:
        ip=sys.argv[1]
        port=sys.argv[2]
    app.run(host=ip, port=port)
