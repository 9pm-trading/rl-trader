import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler

import sys
import time
import mysql.connector 
#import datetime

mydb = mysql.connector.connect(
  host="localhost",
  user="luxeave",
  passwd="jakarta08",
  database="mt4"
)

print(mydb)

np.seterr(over='ignore')


# Let's use AAPL (Apple), MSI (Motorola), SBUX (Starbucks)
def get_data():
    # returns a T x 3 list of stock prices
    # each row is a different stock
    # 0 = AAPL
    # 1 = MSI
    # 2 = SBUX
    df = pd.read_csv('EUR.csv')
    #print( list(df.columns) )
    return df
    #return df.values

def get_scaler(env):
    # return scikit-learn scaler object to scale the states
    # Note: you could also populate the replay buffer here

    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler

def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



class LinearModel:
    """ A linear regression model """
    def __init__(self, input_dim, n_action):
        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
        self.b = np.zeros(n_action)

        # momentum terms
        self.vW = 0
        self.vb = 0

        self.losses = []

    def predict(self, X):
        # make sure X is N x D

        #print('X:',X, 'X.shape' , X.shape)
        assert(len(X.shape) == 2)

        prediction = X.dot(self.W) + self.b

        #print('prediction:', prediction)
        #print('prediction[0]', prediction[0])
        #print('prediction.shape', prediction.shape) # 1, 27
        #sys.exit()

        return prediction
        #return X.dot(self.W) + self.b

    def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):
        # make sure X is N x D
        assert(len(X.shape) == 2)

        # the loss values are 2-D
        # normally we would divide by N only
        # but now we divide by N x K

        #print('Y.shape', Y.shape)
        num_values = np.prod(Y.shape)

        #print('num_values', num_values)

        # do one step of gradient descent
        # we multiply by 2 to get the exact gradient
        # (not adjusting the learning rate)
        # i.e. d/dx (x^2) --> 2x

        #print(X.dtype)
        #sys.exit()

        #print('performing prediction using X')
        Yhat = self.predict(X)

        #print('predicted result Yhat', Yhat)

        #print('performing loss function')
        gW = 2 * X.T.dot(Yhat - Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values

        #print('updating momentum terms')
        # update momentum terms
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb

        #print('updating W and b')
        # update params
        self.W += self.vW
        self.b += self.vb

        mse = np.mean((Yhat - Y)**2, dtype='int64')

        #print('mse', mse)

        self.losses.append(mse)

        #sys.exit()

    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)

class MultiStockEnv:

    def __init__(self, data, initial_investment=10000):
        # data
        self.stock_price_history = data
        self.n_step, _ = self.stock_price_history.shape

        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None
        self.count_win = None 
        self.count_lose = None 
        self.count_total_trades = None 
        self.drawdown = None

        # internal attributes
        #self.stock_owned = None
        #self.stock_price = None
        self.lot_buy = None 
        self.lot_sell = None 
        self.op_price_buy = None 
        self.op_price_sell = None
        self.cash_in_hand = None

        # external attributes 
        self.delta_C_SMA50_bar1 = None 
        self.delta_C_SMA21_bar1 = None 
        self.delta_C_SMA50_bar2 = None 
        self.delta_C_SMA21_bar2 = None
        self.delta_C_SMA50_bar3 = None 
        self.delta_C_SMA21_bar3 = None
        self.price_O_bar1 = None 
        self.price_O_bar2 = None 
        self.price_O_bar3 = None 
        self.price_H_bar1 = None 
        self.price_H_bar2 = None 
        self.price_H_bar3 = None 
        self.price_L_bar1 = None 
        self.price_L_bar2 = None 
        self.price_L_bar3 = None 
        self.price_C_bar1 = None 
        self.price_C_bar2 = None 
        self.price_C_bar3 = None

        #self.action_space = np.arange(3**self.n_stock)
        self.action_space = np.array([0,1,2,3,4])

        # action permutations
        # returns a nested list with elements like:
        # [0,0,0]
        # [0,0,1]
        # [0,0,2]
        # [0,1,0]
        # [0,1,1]
        # etc.
        # 0 = sell
        # 1 = hold
        # 2 = buy
        #self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))
        self.action_list = np.array([0,1,2,3,4])

        # calculate size of state
        #self.state_dim = self.n_stock * 2 + 1
        self.state_dim = 23

        self.reset()


    def reset(self):
        self.cur_step = 0
        self.count_win = 0 
        self.count_lose = 0 
        self.count_total_trades = 0 
        self.drawdown = 0
        #self.stock_owned = np.zeros(self.n_stock)
        #self.stock_price = self.stock_price_history[self.cur_step]
        self.lot_buy = 0 
        self.lot_sell = 0 
        self.op_price_buy = 0 
        self.op_price_sell = 0

        #print('self.cur_step->',self.cur_step)
        #print('self.stock_price_history.shape', self.stock_price_history.shape)
        #print('columns:', self.stock_price_history.columns)
        # assign prices before reformatted as obs
        self.delta_C_SMA50_bar1 = self.stock_price_history['D50-1'][self.cur_step] 

        #print('self.delta_C_SMA50_bar1 ->',self.delta_C_SMA50_bar1)

        self.delta_C_SMA21_bar1 = self.stock_price_history['D21-1'][self.cur_step] 
        self.delta_C_SMA50_bar2 = self.stock_price_history['D50-2'][self.cur_step] 
        self.delta_C_SMA21_bar2 = self.stock_price_history['D21-2'][self.cur_step]
        self.delta_C_SMA50_bar3 = self.stock_price_history['D50-3'][self.cur_step] 
        self.delta_C_SMA21_bar3 = self.stock_price_history['D21-3'][self.cur_step]
        self.price_O_bar1 = self.stock_price_history['OPEN_1'][self.cur_step]  
        self.price_O_bar2 = self.stock_price_history['OPEN_2'][self.cur_step]  
        self.price_O_bar3 = self.stock_price_history['OPEN_3'][self.cur_step]   
        self.price_H_bar1 = self.stock_price_history['HIGH_1'][self.cur_step]   
        self.price_H_bar2 = self.stock_price_history['HIGH_2'][self.cur_step]  
        self.price_H_bar3 = self.stock_price_history['HIGH_3'][self.cur_step]  
        self.price_L_bar1 = self.stock_price_history['LOW_1'][self.cur_step]  
        self.price_L_bar2 = self.stock_price_history['LOW_2'][self.cur_step]   
        self.price_L_bar3 = self.stock_price_history['LOW_3'][self.cur_step]   
        self.price_C_bar1 = self.stock_price_history['CLOSE_1'][self.cur_step]   
        self.price_C_bar2 = self.stock_price_history['CLOSE_2'][self.cur_step]   
        self.price_C_bar3 = self.stock_price_history['CLOSE_3'][self.cur_step]  

        self.cash_in_hand = self.initial_investment
        return self._get_obs()

    def step(self, action):
        assert action in self.action_space

        # get current value before performing the action
        prev_val = self._get_val()

        # update price, i.e. go to the next day
        self.cur_step += 1

        #print('cur_step:', self.cur_step)

        self.delta_C_SMA50_bar1 = self.stock_price_history['D50-1'][self.cur_step] 
        self.delta_C_SMA21_bar1 = self.stock_price_history['D21-1'][self.cur_step] 
        self.delta_C_SMA50_bar2 = self.stock_price_history['D50-2'][self.cur_step] 
        self.delta_C_SMA21_bar2 = self.stock_price_history['D21-2'][self.cur_step]
        self.delta_C_SMA50_bar3 = self.stock_price_history['D50-3'][self.cur_step] 
        self.delta_C_SMA21_bar3 = self.stock_price_history['D21-3'][self.cur_step]
        self.price_O_bar1 = self.stock_price_history['OPEN_1'][self.cur_step]  
        self.price_O_bar2 = self.stock_price_history['OPEN_2'][self.cur_step]  
        self.price_O_bar3 = self.stock_price_history['OPEN_3'][self.cur_step]   
        self.price_H_bar1 = self.stock_price_history['HIGH_1'][self.cur_step]   
        self.price_H_bar2 = self.stock_price_history['HIGH_2'][self.cur_step]  
        self.price_H_bar3 = self.stock_price_history['HIGH_3'][self.cur_step]  
        self.price_L_bar1 = self.stock_price_history['LOW_1'][self.cur_step]  
        self.price_L_bar2 = self.stock_price_history['LOW_2'][self.cur_step]   
        self.price_L_bar3 = self.stock_price_history['LOW_3'][self.cur_step]   
        self.price_C_bar1 = self.stock_price_history['CLOSE_1'][self.cur_step]   
        self.price_C_bar2 = self.stock_price_history['CLOSE_2'][self.cur_step]   
        self.price_C_bar3 = self.stock_price_history['CLOSE_3'][self.cur_step] 

        # perform the trade
        self._trade(action)

        # get the new value after taking the action
        cur_val = self._get_val()

        # reward is the increase in porfolio value
        reward = cur_val - prev_val

        done = 0

        if self.cur_step == self.n_step - 1 or cur_val <= 500:
            done = 1

        # done if we have run out of data
        #done = self.cur_step == self.n_step - 1

        # if margin called
        #done = cur_val <= 500

        # store the current value of the portfolio here
        info = {'cur_val': cur_val, 'win' : self.count_win, 'lose' : self.count_lose, 'total': self.count_total_trades , 'dd': self.drawdown }

        # conform to the Gym API
        return self._get_obs(), reward, done, info


    def _get_obs(self):
        obs = np.empty(self.state_dim, dtype='int64')

        obs[0] = self.lot_buy 
        obs[1] = self.lot_sell 
        obs[2] = self.op_price_buy 
        obs[3] = self.op_price_sell

        obs[4] = self.price_O_bar1
        obs[5] = self.price_O_bar2
        obs[6] = self.price_O_bar3

        obs[7] = self.price_H_bar1
        obs[8] = self.price_H_bar2
        obs[9] = self.price_H_bar3

        obs[10] = self.price_L_bar1
        obs[11] = self.price_L_bar2
        obs[12] = self.price_L_bar3

        obs[13] = self.price_C_bar1
        obs[14] = self.price_C_bar2
        obs[15] = self.price_C_bar3

        obs[16] = self.delta_C_SMA50_bar1
        obs[17] = self.delta_C_SMA50_bar2
        obs[18] = self.delta_C_SMA50_bar3

        obs[19] = self.delta_C_SMA21_bar1
        obs[20] = self.delta_C_SMA21_bar2
        obs[21] = self.delta_C_SMA21_bar3

        obs[-1] = self.cash_in_hand
        return obs

    def _get_val(self):
        #return self.stock_owned.dot(self.stock_price) + self.cash_in_hand
        lot_x_leverage = 100_000 
        val = self.lot_buy * (self.price_C_bar1-self.op_price_buy) * lot_x_leverage
        val += self.lot_sell * (self.op_price_sell-self.price_C_bar1) * lot_x_leverage
        val += self.cash_in_hand

        return val 

    def _trade(self, action):
        action_vec = self.action_list[action]
        lot_x_leverage = 100_000

        val_sell = 0
        val_buy = 0
        if self.lot_sell>0:
            val_sell = self.lot_sell * (self.op_price_sell-self.price_H_bar1) * lot_x_leverage
        if self.lot_buy>0:
            val_buy = self.lot_buy * (self.price_L_bar1-self.op_price_buy) * lot_x_leverage
        val = val_buy + val_sell
        if val<0:
            if self.drawdown==0:
                self.drawdown = val 
            elif val<self.drawdown:
                self.drawdown = val

        #for _, a in enumerate(action_vec):
        if action_vec==0: # sell open 
            if self.lot_sell==0 and self.cash_in_hand>=1000:
                self.lot_sell += 1
                self.op_price_sell = self.price_C_bar1
                self.count_total_trades += 1
        elif action_vec==2 and self.cash_in_hand>=1000: # buy open 
            if self.lot_buy==0:
                self.lot_buy += 1
                self.op_price_buy = self.price_C_bar1
                self.count_total_trades += 1
        elif action_vec==3: # sell close 
            if self.lot_sell>0:
                val = self.lot_sell * (self.op_price_sell-self.price_C_bar1) * lot_x_leverage
                if val>0:
                    self.count_win += 1
                else:
                    self.count_lose += 1
                self.cash_in_hand += val
                self.lot_sell = 0
                self.op_price_sell = 0
        elif action_vec==4: # buy close
            if self.lot_buy>0:
                val = self.lot_buy * (self.price_C_bar1-self.op_price_buy) * lot_x_leverage
                if val>0:
                    self.count_win += 1
                else:
                    self.count_lose += 1  
                self.cash_in_hand += val
                self.lot_buy = 0
                self.op_price_buy = 0

class DQNAgent(object):
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.gamma = 0.95  # discount rate
    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.model = LinearModel(state_size, action_size)

  def act(self, state):
    # random act must only be introduced on train - not - test
    if np.random.rand() <= self.epsilon:
      return np.random.choice(self.action_size)

    act_values = self.model.predict(state)
    #print('act_values[0]', act_values[0])
    return np.argmax(act_values[0])  # returns action

  def act_real(self, state):
    # random act must only be introduced on train - not - test
    #if np.random.rand() <= self.epsilon:
    #  return np.random.choice(self.action_size)

    act_values = self.model.predict(state)
    #print('act_values[0]', act_values[0])
    return np.argmax(act_values[0])  # returns action      

  def train(self, state, action, reward, next_state, done):
    if done:
      target = reward
    else:
      target = reward + self.gamma * np.amax(self.model.predict(next_state), axis=1)

    target_full = self.model.predict(state)
    target_full[0, action] = target

    # Run one training step
    self.model.sgd(state, target_full)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay


  def load(self, name):
    self.model.load_weights(name)


  def save(self, name):
    self.model.save_weights(name)


def play_one_episode(agent, env, is_train):

    #print('running play_one_episode')

    # note: after transforming states are already 1xD
    state = env.reset()
    state = scaler.transform([state])
    done = False

    while not done:
        #print('traversing states')
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])

        if is_train == 'train':
            agent.train(state, action, reward, next_state, done)
        state = next_state

    #return info['cur_val']
    return info

def obs_convert(ohlc, orders, state_dim):
    obs = np.empty(state_dim, dtype='int64')

    obs[0] = float(orders[0]['lot'])
    obs[1] = float(orders[1]['lot'])
    obs[2] = float(orders[0]['op'])
    obs[3] = float(orders[1]['op'])

    obs[4] = float(ohlc['open_1'])
    obs[5] = float(ohlc['open_2'])
    obs[6] = float(ohlc['open_3'])

    obs[7] = float(ohlc['high_1'])
    obs[8] = float(ohlc['high_2'])
    obs[9] = float(ohlc['high_3'])

    obs[10] = float(ohlc['low_1'])
    obs[11] = float(ohlc['low_2'])
    obs[12] = float(ohlc['low_3'])

    obs[13] = float(ohlc['close_1'])
    obs[14] = float(ohlc['close_2'])
    obs[15] = float(ohlc['close_3'])

    obs[16] = float(ohlc['D50_1'])
    obs[17] = float(ohlc['D50_2'])
    obs[18] = float(ohlc['D50_3'])

    obs[19] = float(ohlc['D21_1'])
    obs[20] = float(ohlc['D21_2'])
    obs[21] = float(ohlc['D21_3'])

    obs[-1] = float(ohlc['balance'])
    return obs

if __name__ == '__main__':

    SHOW_EVERY = 90

    # config
    models_folder = 'linear_fx_trader_models'
    rewards_folder = 'linear_fx_trader_rewards'
    num_episodes = 2000
    #batch_size = 32
    initial_investment = 10000


    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='either "train" or "test"')
    args = parser.parse_args()

    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)

    data = get_data()
    n_timesteps, _ = data.shape

    n_train = n_timesteps // 2

    train_data = data[:n_train]
    test_data = data[n_train:]
    test_data = test_data.reset_index(drop=True,inplace=False)

    env = MultiStockEnv(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)

    # store the final value of the portfolio (end of episode)
    portfolio_value = []
   
    if args.mode == 'train':
        # play the game num_episodes times
        for e in range(num_episodes):
            t0 = datetime.now()
            info = play_one_episode(agent, env, args.mode)
            dt = datetime.now() - t0
            print(f"episode: {e + 1}/{num_episodes}, episode end value: {info['cur_val']:.2f}, duration: {dt} win: {info['win']}, lose: {info['lose']}, total: {info['total']}, dd: {info['dd']}")
            portfolio_value.append(info['cur_val']) # append episode end portfolio value
        
        # save the weights when we are done
        # save the DQN
        agent.save(f'{models_folder}/linear.npz')
        # save the scaler
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        # plot losses
        #plt.plot(agent.model.losses)
        #plt.show()
        # save portfolio value for each episode
        np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)

    if args.mode == 'test':
        # then load the previous scaler
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # remake the env with test data
        env = MultiStockEnv(test_data, initial_investment)

        # make sure epsilon is not 1!
        # no need to run multiple episodes if epsilon = 0, it's deterministic
        agent.epsilon = 0.01

        # load trained weights
        agent.load(f'{models_folder}/linear.npz')

        t0 = datetime.now()
        info = play_one_episode(agent, env, args.mode)
        dt = datetime.now() - t0
        print(f"episode end value: {info['cur_val']:.2f}, duration: {dt} win: {info['win']}, lose: {info['lose']}, total: {info['total']}, dd: {info['dd']}") 

    local_timestamp = 0
    loop_counts = 0

    if args.mode == 'real': # REAL TIME RL HERE
        import db
        while True:
            now = datetime.now()
            if now.minute>=0 and now.minute<60:
                # QUERY TIMESTAMP
                db_time = db.get_timestamp()
                #print(type(now))
                if loop_counts % SHOW_EVERY == 0:
                    #print(f'now->{now} db_time->{db_time}') 
                    print('db_time',db_time,'local_timestamp',local_timestamp,'now.minute', now.minute)
                if db_time>local_timestamp: # TRAVERSE NEW STEP EVERYTIME NEW INFO COMES IN 
                                      
                    print('db_time',db_time,'pre local_timestamp',local_timestamp)    

                    local_timestamp = db_time

                    print('db_time',db_time,'post local_timestamp',local_timestamp)

                    '''    
                    # GET LATEST BAR INFO
                    recent_data = db.get_data() # DICTIONARY - OK

                    # GET ORDERS INFO
                    orders = db.get_orders('orders') # LIST OF DICT - OK

                    # CONVERT INTO STATES
                    state = obs_convert(recent_data, orders, state_size)
                    state = scaler.transform([state])

                    # PREDICT
                    action = agent.act_real(state)
                    print('action', action)
                    

                    # ACT BASED ON PREDICTION
                    if action==0:
                        if orders[1]['status']==0:                       
                            db.update_order('orders','status','1','OP_SELL') # SET STATUS -> 1
                    elif action==2:
                        if orders[0]['status']==0:
                            db.update_order('orders','status','1','OP_BUY')
                    elif action==3:
                        if orders[1]['status']==2:
                            db.update_order('orders','status','3','OP_SELL') # SET STATUS -> 3
                    elif action==4:
                        if orders[0]['status']==2:
                            db.update_order('orders','status','3','OP_BUY') # SET STATUS -> 3
                    '''
                    #sys.exit()
            loop_counts += 1
            time.sleep(10)
        
  
