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
    input_layer1 = Input(shape=(20, 3))
    lstm0 = LSTM(256, return_sequences=True, input_shape=(20, 4))(input_layer1)
    dense=Dense(4)(lstm0)
    lstm1 = LSTM(units,return_sequences=True)(dense)
    lstm2 = LSTM(units,return_sequences=True)(lstm1)
    lstm3 = LSTM(int(128),return_sequences=True)(lstm2)
    first_model = Model(inputs=input_layer1, outputs=lstm3)


    result=first_model.predict(np.random.rand(1, 20, 3))
    print(result.shape)
    input_of_other_model = np.zeros([1,20,1])
    con=np.concatenate([result, input_of_other_model],axis=-1)
    print(con.shape)
    input_layer2 = Input(shape=con.shape[1:])
    new_layer = GRU(64,return_sequences=False)(input_layer2)
    output_layer = Dense(20,activation='linear')(new_layer)

    second_model = Model(inputs=input_layer2, outputs=output_layer)
    return first_model, second_model

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
    originalModelPath="InsulinPatientModels\Insulin_"+patient_name+"_model_20_action_weights_dqn_keras.h5"
    newModelPath1="SplitInsulinModels/Insulin_"+patient_name+"_model_20_action_weights_dqn_keras-part1"
    newModelPath2="SplitInsulinModels/Insulin_"+patient_name+"_model_20_action_weights_dqn_keras-part2"
    model_full=create_insulin_full_model()
    model_full.load_weights(originalModelPath)
    model_full.summary()
    model_p1, model_p2=create_split_model()
    model_p1.summary()
    model_p2.summary()
    # exit()

    # exit()
    model_p1.layers[1].set_weights(model_full.layers[1].get_weights())
    model_p1.layers[2].set_weights(model_full.layers[2].get_weights())
    model_p1.layers[3].set_weights(model_full.layers[3].get_weights())
    model_p1.layers[4].set_weights(model_full.layers[4].get_weights())
    model_p1.layers[5].set_weights(model_full.layers[5].get_weights())
    model_p2.layers[1].set_weights(model_full.layers[8].get_weights())
    model_p2.layers[2].set_weights(model_full.layers[9].get_weights())



    BATCH_SIZE=1
    STEPS=20
    INPUT_SIZE=3
    NEW_INPUT_SIZE_1 = 1
    NEW_INPUT_SIZE_2 = 20
    NEW_INPUT_SIZE_3 = 129
    run_model = tf.function(lambda x: model_p1(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model_p1.inputs[0].dtype)
    )
    MODEL_DIR = newModelPath1
    model_p1.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
    tflite_model = converter.convert()
    with open(newModelPath1+'.tflite', 'wb') as f:
        f.write(tflite_model)

        
    run_model = tf.function(lambda x: model_p2(x))


    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([NEW_INPUT_SIZE_1,NEW_INPUT_SIZE_2,NEW_INPUT_SIZE_3], model_p2.inputs[0].dtype)
    )

    MODEL_DIR = newModelPath2

    model_p2.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
    tflite_model = converter.convert()
    with open(newModelPath2+'.tflite', 'wb') as f:
        f.write(tflite_model)


if __name__ == "__main__":
    main()