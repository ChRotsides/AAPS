
from keras import layers,models,optimizers,utils
import numpy as np
# from GlucoseEnv import GlucoseEnvironment
import tensorflow as tf
import random
from keras.layers import Dense, LSTM, Input, concatenate
import scipy.stats as stats
import logging
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
# import gym
# import gym_glucose.envs.MyGym as MyGym
from gym_glucose.envs.GlucoseEnv import GlucoseEnvironment
import math

def create_model_20_action(patient_name):
    inputs1 = tf.keras.Input(shape=(20,4))
    lstm1 = tf.keras.layers.LSTM(256, return_sequences=True, input_shape=(20, 4))(inputs1)
    lstm2 = tf.keras.layers.LSTM(256,return_sequences=True)(lstm1)
    lstm3 = tf.keras.layers.LSTM(int(128))(lstm2)
    additional_input = tf.keras.Input(shape=(1))
    concatenated = tf.keras.layers.concatenate([lstm3, additional_input])
    output_layer = tf.keras.layers.Dense(20)(concatenated)
    model=tf.keras.Model(inputs=[inputs1,additional_input],outputs=output_layer)


    model.load_weights("GlucosePatientModels/Patient_"+patient_name+"_Dexcom_Insulet_patientDataweights20-20-age-3layers-256-time.h5")
    for layer in model.layers:
        layer.trainable = False
    # model.summary()

    inputs1 = tf.keras.Input(shape=(20,3))
    lstm0 = tf.keras.layers.LSTM(256, return_sequences=True, input_shape=(20, 3))(inputs1)
    dense1 = tf.keras.layers.Dense(4)(lstm0)
    lstm1 = tf.keras.layers.LSTM(256, return_sequences=True, input_shape=(20, 4),weights=model.layers[1].get_weights())(dense1)
    lstm2 = tf.keras.layers.LSTM(256,return_sequences=True,weights=model.layers[2].get_weights())(lstm1)
    lstm3 = tf.keras.layers.LSTM(int(128),return_sequences=True,weights=model.layers[3].get_weights())(lstm2)
    additional_input = tf.keras.Input(shape=(20,1))
    concatenated = tf.keras.layers.concatenate([lstm3, additional_input])
    new_layer=tf.keras.layers.GRU(64, return_sequences=False)(concatenated)
    output_layer = tf.keras.layers.Dense(20,activation='linear')(new_layer)
    model2=tf.keras.Model(inputs=[inputs1,additional_input],outputs=output_layer)
    model2.layers[3].trainable = False
    model2.layers[4].trainable = False
    model2.layers[5].trainable = False
    return model2


def get_correct_models_for_patient(patient_name,choice=0):
    model_20_action=create_model_20_action(patient_name)
    model_target_20_action=create_model_20_action(patient_name)


    print(model_20_action.summary())
    print(model_target_20_action.summary())

    model_20_action_for_setting_weights=create_model_20_action(patient_name)
    model_target_20_action_for_setting_weights=create_model_20_action(patient_name)
    if choice==0:
        model_20_action_for_setting_weights.load_weights("OrginalModels\model_20_action_weights_dqn_keras.h5")
        model_target_20_action_for_setting_weights.load_weights("OrginalModels\model_20_action_weights_dqn_keras.h5")
    else:
        try:
            model_20_action_for_setting_weights.load_weights("InsulinPatientModels\Insulin_"+patient_name+"_model_20_action_weights_dqn_keras.h5")
            model_target_20_action_for_setting_weights.load_weights("InsulinPatientModels\Insulin_"+patient_name+"_model_target_20_action_weights_dqn_keras.h5")
        except:
            model_20_action_for_setting_weights.load_weights("OrginalModels\model_20_action_weights_dqn_keras.h5")
            model_target_20_action_for_setting_weights.load_weights("OrginalModels\model_20_action_weights_dqn_keras.h5")

    model_20_action.layers[1].set_weights(model_20_action_for_setting_weights.layers[1].get_weights())
    model_20_action.layers[2].set_weights(model_20_action_for_setting_weights.layers[2].get_weights())
    model_20_action.layers[8].set_weights(model_20_action_for_setting_weights.layers[8].get_weights())
    model_20_action.layers[9].set_weights(model_20_action_for_setting_weights.layers[9].get_weights())

    model_target_20_action.layers[1].set_weights(model_target_20_action_for_setting_weights.layers[1].get_weights())
    model_target_20_action.layers[2].set_weights(model_target_20_action_for_setting_weights.layers[2].get_weights())
    model_target_20_action.layers[8].set_weights(model_target_20_action_for_setting_weights.layers[8].get_weights())
    model_target_20_action.layers[9].set_weights(model_target_20_action_for_setting_weights.layers[9].get_weights())

    return model_20_action,model_target_20_action

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)

patients=[#"adolescent#001","adolescent#002",
        #   "adolescent#007","adult#001",
           "adult#009",]
        #  "child#002",
         #"child#008"]
for patient in patients:
    env=GlucoseEnvironment()
    model_20_action,model_target_20_action=get_correct_models_for_patient(patient_name=patient,choice=1)
    model_target_20_action.summary()
    model_20_action.summary()
    # exit()
    memory={'actions':[],"action_index":[],'states':[],'rewards':[],'extra_info':[]}


    seed = 42
    gamma = 0.99  # Discount factor for past rewards
    epsilon = 1.0  # Epsilon greedy parameter
    epsilon_min = 0.1  # Minimum epsilon greedy parameter
    epsilon_max = 1.0  # Maximum epsilon greedy parameter
    epsilon_interval = (
        epsilon_max - epsilon_min
    )  # Rate at which to reduce chance of random action being taken
    batch_size = 32  # Size of batch taken from replay buffer
    max_steps_per_episode = 100000

    # Experience replay buffers
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    extra_info_next_history = []
    extra_info_history=[]
    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    frame_count = 0
    # Number of frames to take random action and observe output
    epsilon_random_frames = 50
    # Number of frames for exploration
    epsilon_greedy_frames = 1000.0
    # Maximum replay length
    # Note: The Deepmind paper suggests 1000000 however this causes memory issues
    max_memory_length = 100000
    # Train the model after 100 actions
    update_after_actions = 100
    # How often to update the target network
    update_target_network = 1000
    # Using huber loss for stability
    loss_function = tf.keras.losses.Huber()
    num_actions=20
    def map_value(x,BW):
        if "child" in patient:
            return x* 0.04 / 19
        return x *0.08 / 19

    for jk in range(0,1000):  # Run until solved
        state,extra_info = env.reset(patient_name=patient)
        episode_reward = 0
        steps=0
        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.
            frame_count += 1
            steps+=1
            # Use epsilon-greedy for exploration
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(num_actions)
            else:
                # Predict action Q-values
                # From environment state
                action_probs = model_20_action([state.reshape(-1,20,3),extra_info.reshape(-1,20,1)], training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()
                # print("Action:",action)
                # print("Glucose:",env.patient.CGM_hist[-1])

            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # Apply the sampled action in our environment
            state_next,extra_info_next, reward, done = env.step(map_value(action,env.BW)) 
            print("Action:",action)
            print("Glucose:",env.patient.CGM_hist[-1])
            print("Dosage:",map_value(action,env.BW))
            print("Reward:",reward)
            state_next = np.array(state_next)
            extra_info_next = np.array(extra_info_next)

            episode_reward += reward

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            extra_info_history.append(extra_info)
            state_next_history.append(state_next)
            extra_info_next_history.append(extra_info_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            # Update every fourth frame and once batch size is over 32
            if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices]).reshape(-1,20,3)
                extra_info_sample = np.array([extra_info_history[i] for i in indices]).reshape(-1,20,1)
                state_next_sample = np.array([state_next_history[i] for i in indices]).reshape(-1,20,3)
                extra_info_next_sample = np.array([extra_info_next_history[i] for i in indices]).reshape(-1,20,1)
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target_20_action.predict([state_next_sample,extra_info_next_sample])
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(
                    future_rewards, axis=1
                )

                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, num_actions)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model_20_action([state_sample,extra_info_sample])

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model_20_action.trainable_variables)
                optimizer.apply_gradients(zip(grads, model_20_action.trainable_variables))

            if frame_count % update_target_network == 0:
                # update the the target network with new weights
                model_target_20_action.set_weights(model_20_action.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                break

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)
        print("Episode reward: {}".format(episode_reward))
        print("Running reward: {}".format(running_reward))
        print("Episode count: {}".format(episode_count))
        print("Steps: {}".format(steps))
        episode_count += 1

        model_20_action.save_weights('InsulinPatientModels/Insulin_'+patient+'_model_20_action_weights_dqn_keras.h5')
        model_target_20_action.save_weights('InsulinPatientModels/Insulin_'+patient+'_'+'model_target_20_action_weights_dqn_keras.h5')
        if running_reward > 1000:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            break


