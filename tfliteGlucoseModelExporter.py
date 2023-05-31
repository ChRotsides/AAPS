import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate
from tensorflow.keras.models import Model, Sequential
import numpy as np
import tensorflow as tf
import sys
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate,GRU
from tensorflow.keras.models import Model, Sequential


def create_split_model(units=256):
    input_layer1 = Input(shape=(20, 4))
    lstm1 = LSTM(units, return_sequences=True, input_shape=(20, 4))(input_layer1)
    lstm2 = LSTM(units,return_sequences=True)(lstm1)
    lstm3 = LSTM(int(units/2))(lstm2)
    lstm_model = Model(inputs=input_layer1, outputs=lstm3)


    result=lstm_model.predict(np.random.rand(1, 20, 4))
    print(result.shape)
    input_of_other_model = np.array([[1]])
    con=np.concatenate([result, input_of_other_model],axis=-1)
    print(con.shape)
    input_layer2 = Input(shape=con.shape[-1])
    output_layer = Dense(20)(input_layer2)

    second_model = Model(inputs=input_layer2, outputs=output_layer)
    return lstm_model, second_model

def create_full_model(units=256):
    # LSTMModel(256,20,4,20)
    input_layer = Input(shape=(20, 4))
    lstm1 = LSTM(units, return_sequences=True, input_shape=(20, 4))(input_layer)
    lstm2 = LSTM(units,return_sequences=True)(lstm1)
    lstm3 = LSTM(int(units/2))(lstm2)
    additional_input = Input(shape=(1))
    concatenated = concatenate([lstm3, additional_input])

    output_layer = Dense(20)(concatenated)

    model = Model(inputs=[input_layer, additional_input], outputs=output_layer)
    return model

def checkAndgetArgs():
        originalModelPath = None
        newModelPath1 = None
        newModelPath2 = None
        try:
            arg1 = sys.argv[1]
            originalModelPath = sys.argv[1]
        except IndexError:
            arg1 = None
            print("Please enter the Name of the patient")
            exit()
        # try:
        #     arg2 = sys.argv[2]
        #     newModelPath1 = sys.argv[2]
        # except IndexError:
        #     arg2 = None
        #     print("Please enter the path for the first part of the new model")
        #     exit()
        # try:
        #     arg3 = sys.argv[3]
        #     newModelPath2 = sys.argv[3]
        # except IndexError:
        #     arg2 = None
        #     print("Please enter the path for the second part of the new model")
        #     exit()
        return originalModelPath



def main():
    patient_name= checkAndgetArgs()
    originalModelPath=patient_name
    newModelPath1="SplitGlucoseModels/"+patient_name+"_Dexcom_Insulet_patientDataweights20-20-age-3layers-256-time-part1"
    newModelPath2="SplitGlucoseModels/"+patient_name+"_Dexcom_Insulet_patientDataweights20-20-age-3layers-256-time-part2"
    model_full=create_full_model()
    model_full.load_weights(originalModelPath)
    model_full.summary()
    model_p1, model_p2=create_split_model()
    model_p1.summary()
    model_p2.summary()

    # exit()
    model_p1.layers[1].set_weights(model_full.layers[1].get_weights())
    model_p1.layers[2].set_weights(model_full.layers[2].get_weights())
    model_p1.layers[3].set_weights(model_full.layers[3].get_weights())
    model_p2.layers[1].set_weights(model_full.layers[6].get_weights())


    BATCH_SIZE=1
    STEPS=20
    INPUT_SIZE=4
    NEW_INPUT_SIZE = 1

    run_model = tf.function(lambda x: model_p1(x))


    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model_p1.inputs[0].dtype)
    )

    MODEL_DIR = newModelPath1

    model_p1.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
    tflite_model = converter.convert()
    with open("AndroidModels\GlucoseModel1"+'.tflite', 'wb') as f:
        f.write(tflite_model)

        
    run_model = tf.function(lambda x: model_p2(x))


    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1,129], model_p2.inputs[0].dtype)
    )

    MODEL_DIR = newModelPath2

    model_p2.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
    tflite_model = converter.convert()
    with open("AndroidModels\GlucoseModel2"+'.tflite', 'wb') as f:
        f.write(tflite_model)

if __name__ == "__main__":
    main()