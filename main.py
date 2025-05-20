import numpy as np
import tensorflow as tf
import pandas as pd
import time as tm
from haversine import haversine
from random import sample
import math
from Environment import environment
from DoubleDQN import Qnetwork

class ExperienceReplayBuffer:
    def __init__(self, size=50000):
        self.buffer = []
        self.buffersave = []
        self.buffersize = size
        self.currents1 = []
        self.currents2 = []
        self.actions = []
        self.rewards = []
        self.nexts1 = []
        self.nexts2 = []
        self.ds = []

    def append(self, exp):
        if len(self.buffer) + len(exp) >= self.buffersize:
            self.buffer[:len(self.buffer) + len(exp) - self.buffersize] = []
        if len(self.buffersave) + len(exp) >= self.buffersize:
            self.buffersave[:len(self.buffersave) + len(exp) - self.buffersize] = []
        if len(self.currents1) + len(exp) >= self.buffersize:
            self.currents1[:len(self.currents1) + len(exp) - self.buffersize] = []
        if len(self.currents2) + len(exp) >= self.buffersize:
            self.currents2[:len(self.currents2) + len(exp) - self.buffersize] = []
        if len(self.actions) + len(exp) >= self.buffersize:
            self.actions[:len(self.actions) + len(exp) - self.buffersize] = []
        if len(self.rewards) + len(exp) >= self.buffersize:
            self.rewards[:len(self.rewards) + len(exp) - self.buffersize] = []
        if len(self.nexts1) + len(exp) >= self.buffersize:
            self.nexts1[:len(self.nexts1) + len(exp) - self.buffersize] = []
        if len(self.nexts2) + len(exp) >= self.buffersize:
            self.nexts2[:len(self.nexts2) + len(exp) - self.buffersize] = []
        if len(self.ds) + len(exp) >= self.buffersize:
            self.ds[:len(self.ds) + len(exp) - self.buffersize] = []
        self.buffer.extend(exp)
        self.buffersave.append(exp)
        self.currents1.append(exp[0][0][0])
        self.currents2.append(exp[0][0][1])
        self.actions.append(exp[0][1])
        self.rewards.append(exp[0][2])
        self.nexts1.append(exp[0][3][0])
        self.nexts2.append(exp[0][3][1])
        self.ds.append(exp[0][4])

    def batch(self, num):
        # return np.reshape(np.array(sample(self.buffer, num)), [num, 5])
        # 直接返回采样到的 list，保持原始结构
        return sample(self.buffer, num)

def update_net(target_model, q_model):
    target_model.set_weights(q_model.get_weights())

print("------------for tensorflow 2 --------------")
Qnet = Qnetwork(s_size=2, a_size=4)
Targetnet = Qnetwork(s_size=2, a_size=4)

print("------------for env & google info--------------")
env = environment('40.468254,-86.980963', '40.445283,-86.948429')
step_rewardG, chargenumG, SOCG, timeG = env.origine_map_reward()  # The info of google map route
env.battery_charge()
step_length = 1000  # meter
env.length = 1000 / step_length
print("stride length: ", env.length)
learning_rate = 0.0001

print("------------for map --------------")
s = env.start_position
print("map bound: ", env.map_bound)
north = env.map_bound['north']
east = env.map_bound['east']
west = env.map_bound['west']
south = env.map_bound['south']
upper_left = (north, west)
upper_right = (north, east)
lower_left = (south, west)
lower_right = (south, east)
map_height = haversine(upper_left, lower_left)
map_wide = haversine(upper_left, upper_right) if haversine(upper_left, upper_right) > haversine(lower_left, lower_right) else haversine(lower_left, lower_right)
wide_grid_num = map_wide / (step_length/1000)
height_grid_num = map_height / (step_length/1000)
total_point = int(math.ceil(wide_grid_num) * math.ceil(height_grid_num))
print("total grid point: ", total_point)
max_train_step = 4 * math.ceil(wide_grid_num) * math.ceil(height_grid_num)
print("Max trainning steps: ", max_train_step)
pre_train_step = max_train_step * 5
print("Pre train step: ", pre_train_step)
s_list = list(s)
print("start position: ", s_list)
print("end position: ", env.end_position)
replay_buffer = ExperienceReplayBuffer()
print("------------Parameter --------------")
path = "./ev/model"
pre_train = pre_train_step  # don't update and train the model within these steps
train_num = 300   # total episode num
max_step = max_train_step
updata_f = 5   # frequency of copy weights from Qnet to Targetnet
batch_num = 32
gamma = 0.9 # discount factor
high_prob = 1
low_prob = 0.1
slope = (high_prob - low_prob) / 20000
pathload = "./ev/Result/47_proceed46/model"
load_model = False
modelnum = 27
sleep = False

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

if load_model:
    high_prob = 0.1

print("load buffer success............................")
print(replay_buffer.buffer)
print("------------Save training parameters--------------")
parameter = [env.map_bound, map_height, map_wide, step_length, wide_grid_num, height_grid_num, total_point, max_train_step, pre_train_step, learning_rate, step_rewardG, chargenumG, SOCG, timeG]
df_1 = pd.DataFrame([parameter])
df_1.to_csv("./ev/train_para.csv", header=["google map_boundary", "map_height(km)", "map_wide(km)", "step_length(m)", "wide_grid_num", "height_grid_num", "total_point", "max_train_step in episode", "pre_train_step", "learning_rate", "Google r", "Google charge num", "Google SOC", "Google time"])

print("sleep 5 min")
tm.sleep(10)
print("------------Start training --------------")

battery = []
total_step = 0
episode_num = 1
reward_history = []
e = high_prob
total_charge_num = 0

if load_model:
    print("Loading Model....")
    Qnet.load_weights(pathload + "/model-" + str(modelnum) + ".h5")
    print("Model restored.")

for episode in range(train_num):  # num of episode
    s = env.start_position
    s_list = list(s)
    # env.battery_charge()
    env.reset()
    ss, nn, total_nn = env.battery_condition()
    print("Current Episode: ", episode_num)
    # print("current position: ", env.current_position)
    episode_num = episode_num + 1
    in_ep_step = 0
    step_buffer = []
    episode_reward = 0
    random_a = 0
    network_a = 0
    avg_loss = 0
    overQ_num = 0 # OVER_QUERY_LIMIT
    unreach_step_history = []
    loss_history = []
    overQ_num_roll = 0

    while (in_ep_step <= max_step):
        if np.random.rand(1) < e or (total_step < pre_train and not load_model):
            action = np.random.randint(0, 4)
            test = 1
        else:
            q_values = Qnet(np.expand_dims(s_list, axis=0))
            action = tf.argmax(q_values[0]).numpy()
            test = 2

        if test == 1:
            random_a += 1
        else:
            network_a += 1

        s1, r, d, charge_num, SOC = env.step(action)
        # print(f"Step {in_ep_step}: current_position = {env.current_position}")  # <--- 新增这一行
        s = list(s1)
        episode_reward += r

        if env.status_dir_check != 'OVER_QUERY_LIMIT':
            step_buffer.append([s_list, action, r, s, d])
            # replay_buffer.append(np.reshape(np.array([s_list, action, r, s, d]), [1, 5]))
            replay_buffer.append([[s_list, action, r, s, d]])
            in_ep_step += 1
            total_step += 1
            s_list = s
        else:
            overQ_num += 1
            overQ_num_roll += 1

        if ((total_step > pre_train and env.status_dir_check != 'OVER_QUERY_LIMIT' and len(replay_buffer.buffer) > batch_num)
            or (load_model and len(replay_buffer.buffer) > batch_num)):
            if e > low_prob:
                e -= slope
            ex_batch = replay_buffer.batch(batch_num)
            # states = np.vstack(ex_batch[:, 0])
            # actions = ex_batch[:, 1].astype(np.int32)
            # rewards = ex_batch[:, 2].astype(np.float32)
            # next_states = np.vstack(ex_batch[:, 3])
            # dones = ex_batch[:, 4].astype(np.float32)
            states = np.array([exp[0] for exp in ex_batch], dtype=np.float32)
            actions = np.array([exp[1] for exp in ex_batch], dtype=np.int32)
            rewards = np.array([exp[2] for exp in ex_batch], dtype=np.float32)
            next_states = np.array([exp[3] for exp in ex_batch], dtype=np.float32)
            dones = np.array([exp[4] for exp in ex_batch], dtype=np.float32)

            # Double DQN target calculation
            next_q_values = Qnet(next_states)
            next_actions = tf.argmax(next_q_values, axis=1)
            target_q_values = Targetnet(next_states)
            target_q = tf.gather_nd(target_q_values, np.stack([np.arange(batch_num), next_actions.numpy()], axis=1))
            y = rewards + (1 - dones) * gamma * target_q.numpy()

            with tf.GradientTape() as tape:
                q_pred = Qnet(states)
                indices = np.stack([np.arange(batch_num), actions], axis=1)
                q_pred_a = tf.gather_nd(q_pred, indices)
                loss = tf.reduce_mean(tf.square(y - q_pred_a))
            grads = tape.gradient(loss, Qnet.trainable_variables)
            optimizer.apply_gradients(zip(grads, Qnet.trainable_variables))
            loss_history.append(loss.numpy())
            avg_loss += loss.numpy()

        if total_step % updata_f == 0:
            update_net(Targetnet, Qnet)

        if d:
            print("Success")
            print(f"  Steps to reach end: {in_ep_step}")
            print(f"  Total time: {env.time}")
            print(f"  Failed steps: {env.unreach_position_num}")
            print(f"  Final SOC: {ss}, Charge num (episode): {nn}, Total charge num: {total_nn}")
            print(f"  Last position: {env.current_position}")
            battery.append([charge_num, SOC])
            time = env.time
            if in_ep_step > 0:
                avg_loss = avg_loss / in_ep_step
            real_reward = episode_reward + 0.1 * (in_ep_step - env.unreach_position_num) - 1
            real_r_nofail = real_reward + env.unreach_position_num
            history = [episode+1, in_ep_step, time, episode_reward, real_reward, real_r_nofail, charge_num, SOC, env.unreach_position_num, d, random_a/(random_a+network_a), avg_loss, overQ_num, loss_history, step_buffer]
            if episode == 0:
                df = pd.DataFrame([history])
                df.to_csv("./ev/result.csv", header=["episode", "step", "time", "reward", "reward_notrain", "reward_nofail", "charge_num", "SOC", "unreach_position", "Reach", "Random_a", "Avg_loss", "overQuery_num", "Loss history", "Step history"])
            else:
                with open('./ev/result.csv', 'a') as f:
                    df = pd.DataFrame([history])
                    df.to_csv(f, header=False)
            print("Total time: ", time)
            print("number of failed step: ", env.unreach_position_num)
            env.time = 0
            env.unreach_position_num = 0
            ss, nn, total_nn = env.battery_condition()
            print(f"Final SOC: {ss}, Final charge_num (episode): {nn}, Total charge_num (all episodes): {total_nn}")
            break  # 或 continue

        if not d and in_ep_step == max_step:
            if in_ep_step > 0:
                avg_loss = avg_loss / in_ep_step
            print("Failed")
            time = env.time
            real_reward = episode_reward + 0.1 * (in_ep_step - env.unreach_position_num)
            real_r_nofail = real_reward + env.unreach_position_num
            history = [episode+1, in_ep_step, time, episode_reward, real_reward, real_r_nofail, charge_num, SOC, env.unreach_position_num, d, random_a/(random_a+network_a), avg_loss, overQ_num, loss_history, step_buffer]
            if episode == 0:
                df = pd.DataFrame([history])
                df.to_csv("./ev/result.csv", header=["episode", "step", "time", "reward", "reward_notrain", "reward_nofail", "charge_num", "SOC", "unreach_position", "Reach", "Random_a", "Avg_loss", "overQuery_num", "Loss history", "Step history"])
            else:
                with open('./ev/result.csv', 'a') as f:
                    df = pd.DataFrame([history])
                    df.to_csv(f, header=False)
            env.time = 0
            env.unreach_position_num = 0
            ss, nn, total_nn = env.battery_condition()
            print(f"Final SOC: {ss}, Final charge_num (episode): {nn}, Total charge_num (all episodes): {total_nn}")
            break

        if overQ_num_roll > 50:
            overQ_num_roll = 0
            print("Sleeping within episode for 60 min")
            tm.sleep(3600)

    if d and in_ep_step < 60 and episode > 10 and episode % 1 == 0 or (load_model and d):
        j = episode + 1
        Qnet.save_weights(path + "/model-" + str(j) + ".h5")
        print("Saved model with step less than steps 60")

    print("Last position: ", env.current_position)
    print("Destination: ", env.end_position)
    print("env.stridebound a, b", env.stridebounda, env.strideboundb)
    # env.current_position = env.start_position
    # s = env.start_position
    # s_list = list(s)
    # ss, nn = env.battery_condition()
    # print("SOC, charge_number: ", ss, nn)
    # env.charge_num = 0
    # reward_history.append(episode_reward)
    # env.battery_charge()

    # Save the replay buffer
    with open("./ev/buffercurrents1.txt", "w") as ff:
        for s in replay_buffer.currents1:
            ff.write(str(s) + "\n")
    with open("./ev/buffercurrents2.txt", "w") as mm:
        for s in replay_buffer.currents2:
            mm.write(str(s) + "\n")
    with open("./ev/bufferactions.txt", "w") as gg:
        for s in replay_buffer.actions:
            gg.write(str(s) + "\n")
    with open("./ev/bufferrewards.txt", "w") as hh:
        for s in replay_buffer.rewards:
            hh.write(str(s) + "\n")
    with open("./ev/buffernexts1.txt", "w") as ii:
        for s in replay_buffer.nexts1:
            ii.write(str(s) + "\n")
    with open("./ev/buffernexts2.txt", "w") as kk:
        for s in replay_buffer.nexts2:
            kk.write(str(s) + "\n")
    with open("./ev/bufferds.txt", "w") as jj:
        for s in replay_buffer.ds:
            jj.write(str(s) + "\n")

    if (total_step > 450 and total_step % 120 == 0) or sleep:
        print("Sleeping now for 20 min")
        tm.sleep(1230)
    print("-------------------------------------------------------------------------------")

print("------------------End----------------")
