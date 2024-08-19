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

def get_data():
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
        state, reward, done, info = env.step_scale(action)
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

        mse = np.mean((Yhat - Y)**2)

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

    def __init__(self, data, initial_investment=10000, target=50, dd=20, step_pt=0.002):
        # data
        self.stock_price_history = data
        self.n_step, _ = self.stock_price_history.shape

        # instance attributes
        self.initial_investment = initial_investment
        self.target = target
        self.dd_percent = dd
        self.step_pt = step_pt

        self.cur_step = None
        self.count_win = None 
        self.count_lose = None 
        self.count_total_trades = None 
        self.drawdown = None
        
        self.cash_in_hand = None

        # internal cycle attributes
        self.lot_buy_1 = None
        self.op_price_buy_1 = None
        self.sl_price_buy_1 = None 

        self.lot_buy_2 = None
        self.op_price_buy_2 = None
        self.sl_price_buy_2 = None 

        self.lot_buy_3 = None
        self.op_price_buy_3 = None
        self.sl_price_buy_3 = None 

        self.lot_sell_1 = None      
        self.op_price_sell_1 = None     
        self.sl_price_sell_1 = None

        self.lot_sell_2 = None      
        self.op_price_sell_2 = None     
        self.sl_price_sell_2 = None

        self.lot_sell_3 = None      
        self.op_price_sell_3 = None     
        self.sl_price_sell_3 = None

        self.accumulated_loss = None 
        self.max_drawdown = None
        
        self.mid_price = None
        self.tp_price_buy = None 
        self.tp_price_sell = None

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
       
        self.action_space = np.array([0,1,2]) # 0-DO_NOTHING, 1-START_SELL, 2-START_BUY

        # action permutations
        self.action_list = np.array([0,1,2])

        # calculate size of state
        self.state_dim = 18

        self.reset()

    def reset_cycle(self):
        self.lot_buy_1 = 0
        self.op_price_buy_1 = 0
        self.sl_price_buy_1 = 0 

        self.lot_buy_2 = 0
        self.op_price_buy_2 = 0
        self.sl_price_buy_2 = 0 

        self.lot_buy_3 = 0
        self.op_price_buy_3 = 0
        self.sl_price_buy_3 = 0 

        self.lot_sell_1 = 0      
        self.op_price_sell_1 = 0     
        self.sl_price_sell_1 = 0

        self.lot_sell_2 = 0      
        self.op_price_sell_2 = 0     
        self.sl_price_sell_2 = 0

        self.lot_sell_3 = 0      
        self.op_price_sell_3 = 0     
        self.sl_price_sell_3 = 0

        self.accumulated_loss = 0
        self.max_drawdown = 0     

        self.mid_price = 0  
        self.tp_price_buy = 0
        self.tp_price_sell = 0

    def reset(self):
        # internal attributes
        self.cur_step = 0
        self.count_win = 0 
        self.count_lose = 0 
        self.count_total_trades = 0 
        self.drawdown = 0

        self.cash_in_hand = self.initial_investment

        self.reset_cycle()

        # assign prices before reformatted as obs
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

        return self._get_obs()

    def step(self, action): 
        # IF NOT TAKING ANY POS-> MOVE 1 STEP-BAR INTO THE FUTURE
        # IF STARTING CYCLE -> COMPLETE CURRENT CYCLE N-BAR INTO THE FUTURE
        assert action in self.action_space
        
        # get current value before performing the action
        prev_val = self._get_val()

        # perform the trade
        self._trade(action) # 1 CYCLE - MULTIPLE BARS

        # get the new value after taking the action
        cur_val = self._get_val()

        # reward is the increase in porfolio value
        reward = cur_val - prev_val

        done = 0

        # END OF EPISODE
        if self.cur_step == self.n_step - 1 or cur_val <= 500:
            done = 1

        # store the current value of the portfolio here
        info = {'cur_val': cur_val, 'win' : self.count_win, 'lose' : self.count_lose, 'total': self.count_total_trades , 'dd': self.drawdown }

        #sys.exit()

        # conform to the Gym API
        return self._get_obs(), reward, done, info

    def step_scale(self, action): 
        # IF NOT TAKING ANY POS-> MOVE 1 STEP-BAR INTO THE FUTURE
        # IF STARTING CYCLE -> COMPLETE CURRENT CYCLE N-BAR INTO THE FUTURE
        assert action in self.action_space
        
        # get current value before performing the action
        prev_val = self._get_val()
        print('prev_val', prev_val)

        # perform the trade
        self._trade(action) # 1 CYCLE - MULTIPLE BARS

        # get the new value after taking the action
        cur_val = self._get_val()
        print('cur_val', cur_val)

        # reward is the increase in porfolio value
        reward = cur_val - prev_val
        print('reward', reward)

        # sys.exit()

        done = 0

        # END OF EPISODE
        if self.cur_step == self.n_step - 1 or cur_val <= 500:
            done = 1

        # store the current value of the portfolio here
        info = {'cur_val': cur_val, 'win' : self.count_win, 'lose' : self.count_lose, 'total': self.count_total_trades , 'dd': self.drawdown }

        # conform to the Gym API
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        obs = np.empty(self.state_dim, dtype='double')

        obs[0] = self.price_O_bar1
        obs[1] = self.price_O_bar2
        obs[2] = self.price_O_bar3

        obs[3] = self.price_H_bar1
        obs[4] = self.price_H_bar2
        obs[5] = self.price_H_bar3

        obs[6] = self.price_L_bar1
        obs[7] = self.price_L_bar2
        obs[8] = self.price_L_bar3

        obs[9] = self.price_C_bar1
        obs[10] = self.price_C_bar2
        obs[11] = self.price_C_bar3

        obs[12] = self.delta_C_SMA50_bar1
        obs[13] = self.delta_C_SMA50_bar2
        obs[14] = self.delta_C_SMA50_bar3

        obs[15] = self.delta_C_SMA21_bar1
        obs[16] = self.delta_C_SMA21_bar2
        obs[17] = self.delta_C_SMA21_bar3

        obs[-1] = self.cash_in_hand
        return obs

    def _get_val(self):
        return self.cash_in_hand 

    def one_step_bar(self):
        self.cur_step += 1

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

    def calculate_lot_buy(self, target_usd, level):
        total_buy_slot = 0
        total_buy_lot = 0

        total_running_buy = 0

        if self.lot_buy_1==0:
            total_buy_slot += self.step_pt * 3 * 100_000 * 0.5
        elif self.lot_buy_1>0:
            total_running_buy += (self.tp_price_buy-self.op_price_buy_1) * 100_000 * self.lot_buy_1

        if self.lot_buy_2==0:    
            total_buy_slot += self.step_pt * 2 * 100_000 * 0.3
        elif self.lot_buy_2>0:
            total_running_buy += (self.tp_price_buy-self.op_price_buy_2) * 100_000 * self.lot_buy_2

        if self.lot_buy_3==0:
            total_buy_slot += self.step_pt * 100_000 * 0.2
        
        if total_buy_slot>0:
            total_buy_lot = ( target_usd - total_running_buy ) / total_buy_slot 

        sb1_lot = 0.5 * total_buy_lot
        sb2_lot = 0.3 * total_buy_lot 
        sb3_lot = 0.2 * total_buy_lot 

        print(f'SB1:{self.op_price_buy_1:.5f} SB2:{self.op_price_buy_2:.5f} SB3:{self.op_price_buy_3:.5f} SBTP:{self.tp_price_buy:.5f} - SS1:{self.op_price_sell_1:.5f} SS2:{self.op_price_sell_2:.5f} SS3:{self.op_price_sell_3:.5f} SSTP:{self.tp_price_sell:.5f}') 
        print(f'BUY target:{target_usd} running:{total_running_buy} calc lot SB1:{sb1_lot:.2f} SB2:{sb2_lot:.2f} SB3:{sb3_lot:.2f}')

        sb1_pt = 3 * self.step_pt
        sb2_pt = 2 * self.step_pt
        sb1_pt_actual = self.tp_price_buy - self.op_price_buy_1 
        sb2_pt_actual = self.tp_price_buy - self.op_price_buy_2 

        if level==1:
            sb1_lot = sb1_lot * ( sb1_pt / sb1_pt_actual )
            return sb1_lot 
        elif level==2:
            sb2_lot = sb2_lot * ( sb2_pt / sb2_pt_actual )
            return sb2_lot 
        
        return sb3_lot

    def calculate_loss(self, op_price, sl_price, lot):
        val = (op_price - sl_price) * 100_000 * lot
        return abs(val)

    def calculate_lot_sell(self, target_usd, level):
        total_sell_slot = 0
        total_sell_lot = 0
        total_running_sell = 0

        step_pips = self.step_pt * 100_000
        dollar_per_pip = 1

        if self.lot_sell_1==0:
            total_sell_slot += step_pips * 3 * dollar_per_pip * 0.5
        elif self.lot_sell_1>0:
            total_running_sell += (self.op_price_sell_1-self.tp_price_sell) * 100_000 * self.lot_sell_1

        if self.lot_sell_2==0:    
            total_sell_slot += step_pips * 2 * dollar_per_pip * 0.3
        elif self.lot_sell_2>0:
            total_running_sell += (self.op_price_sell_2-self.tp_price_sell) * 100_000 * self.lot_sell_2

        if self.lot_sell_3==0:
            total_sell_slot += step_pips * dollar_per_pip * 0.2
        
        if total_sell_slot>0:
            total_sell_lot = ( target_usd - total_running_sell ) / total_sell_slot 

        ss1_lot = 0.5 * total_sell_lot
        ss2_lot = 0.3 * total_sell_lot 
        ss3_lot = 0.2 * total_sell_lot 

        print(f'SB1:{self.op_price_buy_1:.5f} SB2:{self.op_price_buy_2:.5f} SB3:{self.op_price_buy_3:.5f} SBTP:{self.tp_price_buy:.5f} - SS1:{self.op_price_sell_1:.5f} SS2:{self.op_price_sell_2:.5f} SS3:{self.op_price_sell_3:.5f} SSTP:{self.tp_price_sell:.5f}') 
        print(f'SELL target:{target_usd} running:{total_running_sell} calc lot SS1:{ss1_lot:.2f} SS2:{ss2_lot:.2f} SS3:{ss3_lot:.2f}')

        ss1_pt = 3*self.step_pt
        ss2_pt = 2*self.step_pt  
        ss1_pt_actual = ( self.op_price_sell_1 - self.tp_price_sell ) 
        ss2_pt_actual = ( self.op_price_sell_2 - self.tp_price_sell )

        if level==1:
            ss1_lot = ss1_lot * ( ss1_pt / ss1_pt_actual )
            return ss1_lot 
        elif level==2:
            ss2_lot = ss2_lot * ( ss2_pt / ss2_pt_actual )
            return ss2_lot 
        
        return ss3_lot    

    def delta(self, price1, price2):
        val = price1-price2
        return abs(val)

    def set_first_buy(self, ask):
        # BUY GRID
        self.op_price_buy_1 = ask
        self.sl_price_buy_1 = self.op_price_buy_1 - self.step_pt

        self.op_price_buy_2 = self.op_price_buy_1 + self.step_pt
        self.sl_price_buy_2 = self.op_price_buy_2 - self.step_pt

        self.op_price_buy_3 = self.op_price_buy_2 + self.step_pt
        self.sl_price_buy_3 = self.op_price_buy_3 - self.step_pt

        self.tp_price_buy = self.op_price_buy_3 + self.step_pt
        # CENTER
        self.mid_price = self.op_price_buy_1 - self.step_pt
        # SELL GRID
        self.op_price_sell_1 = self.mid_price - self.step_pt
        self.sl_price_sell_1 = self.op_price_sell_1 + self.step_pt

        self.op_price_sell_2 = self.op_price_sell_1 - self.step_pt
        self.sl_price_sell_2 = self.op_price_sell_2 + self.step_pt

        self.op_price_sell_3 = self.op_price_sell_2 - self.step_pt
        self.sl_price_sell_3 = self.op_price_sell_3 + self.step_pt 

        self.tp_price_sell = self.op_price_sell_3 - self.step_pt              
        # DD
        self.max_drawdown = self.cash_in_hand * self.dd_percent * 0.01   

        print(f'SET CYCLE SB1:{self.op_price_buy_1:.5f} SB2:{self.op_price_buy_2:.5f} SB3:{self.op_price_buy_3:.5f} - SS1:{self.op_price_sell_1:.5f} SS2:{self.op_price_sell_2:.5f} SS3:{self.op_price_sell_3:.5f} maxdd:{self.max_drawdown:.2f}') 

        # ACT NOW - SET FIRST LV1->BUY, AND OTHER LEVELS
        self.lot_buy_1 = self.calculate_lot_buy(self.target, 1)

        print(f'OP SB1@{ask} {self.lot_buy_1:.2f} lot')

    def set_first_sell(self, bid):
        # SELL GRID 
        self.op_price_sell_1 = bid 
        self.sl_price_sell_1 = self.op_price_sell_1 + self.step_pt

        self.op_price_sell_2 = self.op_price_sell_1 - self.step_pt
        self.sl_price_sell_2 = self.op_price_sell_2 + self.step_pt

        self.op_price_sell_3 = self.op_price_sell_2 - self.step_pt
        self.sl_price_sell_3 = self.op_price_sell_3 + self.step_pt 

        self.tp_price_sell = self.op_price_sell_3 - self.step_pt 
        
        # CENTER
        self.mid_price = self.op_price_sell_1 + self.step_pt
        # BUY GRID 
        self.op_price_buy_1 = self.mid_price + self.step_pt  
        self.sl_price_buy_1 = self.op_price_buy_1 - self.step_pt

        self.op_price_buy_2 = self.op_price_buy_1 + self.step_pt
        self.sl_price_buy_2 = self.op_price_buy_2 - self.step_pt

        self.op_price_buy_3 = self.op_price_buy_2 + self.step_pt
        self.sl_price_buy_3 = self.op_price_buy_3 - self.step_pt

        self.tp_price_buy = self.op_price_buy_3 + self.step_pt  
        # DD
        self.max_drawdown = self.cash_in_hand * self.dd_percent * 0.01 

        print(f'SET CYCLE SB1:{self.op_price_buy_1:.5f} SB2:{self.op_price_buy_2:.5f} SB3:{self.op_price_buy_3:.5f} - SS1:{self.op_price_sell_1:.5f} SS2:{self.op_price_sell_2:.5f} SS3:{self.op_price_sell_3:.5f} maxdd:{self.max_drawdown}')

        self.lot_sell_1 = self.calculate_lot_sell(self.target, 1) 
        print(f'OP SS1@{bid} {self.lot_sell_1:.2f} lot')

    def grid_sell_up(self, new_mid): # TRAIL GRID SELL UP       
        self.mid_price = new_mid 
        self.op_price_sell_1 = self.mid_price - self.step_pt
        self.op_price_sell_2 = self.op_price_sell_1 - self.step_pt 
        self.op_price_sell_3 = self.op_price_sell_2 - self.step_pt 
        self.tp_price_sell = self.op_price_sell_3 - self.step_pt

        print(f'GRID SELL UP - SS1:{self.op_price_sell_1:.5f} SS2:{self.op_price_sell_2:.5f} SS3:{self.op_price_sell_3:.5f} maxdd:{self.max_drawdown}')

    def grid_buy_down(self, new_mid):
        self.mid_price = new_mid 
        self.op_price_buy_1 = self.mid_price + self.step_pt
        self.op_price_buy_2 = self.op_price_buy_1 + self.step_pt 
        self.op_price_buy_3 = self.op_price_buy_2 + self.step_pt 
        self.tp_price_buy = self.op_price_buy_3 + self.step_pt

        print(f'GRID BUY DOWN - SB1:{self.op_price_buy_1:.5f} SB2:{self.op_price_buy_2:.5f} SB3:{self.op_price_buy_3:.5f} maxdd:{self.max_drawdown}')

    def normalize_SB1_SB2(self):
        # NORMALIZE SB1
        if self.lot_buy_1==0 and self.delta(self.op_price_buy_1, self.op_price_buy_3)<0.5*self.step_pt:                                    
            if self.price_L_bar1 < self.op_price_buy_3-(3*self.step_pt):
                print(f'PRE-NORM SB1:{self.op_price_buy_1}')
                self.op_price_buy_1 = self.op_price_buy_3-(2*self.step_pt)
                print(f'POST-NORM SB1:{self.op_price_buy_1} 2 STEPS')
            elif self.price_L_bar1 < self.op_price_buy_3-(2*self.step_pt):
                print(f'PRE-NORM SB1:{self.op_price_buy_1}')
                self.op_price_buy_1 = self.op_price_buy_3-(1*self.step_pt) 
                print(f'POST-NORM SB1:{self.op_price_buy_1} 1 STEP')                  
        # NORMALIZE SB2
        if self.lot_buy_2==0 and self.delta(self.op_price_buy_2, self.op_price_buy_3)<0.5*self.step_pt: 
            if self.price_L_bar2 < self.op_price_buy_3-(2*self.step_pt):
                print(f'PRE-NORM SB2:{self.op_price_buy_2}')
                self.op_price_buy_2 = self.op_price_buy_3-(1*self.step_pt)
                print(f'POST-NORM SB2:{self.op_price_buy_2} 1 STEP')       

    def normalize_SS1_SS2(self):
        # NORMALIZE SS1
        if self.lot_sell_1==0 and self.delta(self.op_price_sell_1,self.op_price_sell_3)<0.5*self.step_pt: # LV1 & LV3 SELL MERGED
            if self.price_H_bar1 > self.op_price_sell_3 + ( 3 * self.step_pt ): # NORMALIZE SS1 - TO LV2
                print(f'PRE-NORM SS1:{self.op_price_sell_1}')
                self.op_price_sell_1 = self.op_price_sell_3 + ( 2 * self.step_pt )
                print(f'POST-NORM SS1:{self.op_price_sell_1} 2 STEPS')
            elif self.price_H_bar1 > self.op_price_sell_3 + ( 2 * self.step_pt ):  # NORMALIZE SS1 - TO LV1
                print(f'PRE-NORM SS1:{self.op_price_sell_1}')
                self.op_price_sell_1 = self.op_price_sell_3 + ( 1 * self.step_pt )
                print(f'POST-NORM SS1:{self.op_price_sell_1} 1 STEP')
        # NORMALIZE SS2
        if self.lot_sell_2==0 and self.delta(self.op_price_sell_2,self.op_price_sell_3)<0.5*self.step_pt: # LV2 & LV3 SELL MERGED 
            if self.price_H_bar1 > self.op_price_sell_3 + ( 2 * self.step_pt ):  # NORMALIZE LV1 - 1 STEP UP
                print(f'PRE-NORM SS2:{self.op_price_sell_2}')
                self.op_price_sell_2 = self.op_price_sell_3 + ( 1 * self.step_pt )
                print(f'POST-NORM SS2:{self.op_price_sell_2} 1 STEP')

    def buy_tp_hit(self):
        # BREAK ON CYCLE END - PROFIT OR DD HIT
        profit = 0
        if self.lot_buy_3>0 and self.price_H_bar1 >= self.tp_price_buy: # SB3 HIT TP
            profit_SB3 = self.step_pt * 100_000 * self.lot_buy_3
            profit += profit_SB3
            print(f'TP SB3:{profit_SB3}')
        if self.lot_buy_2>0 and self.price_H_bar1 >= self.tp_price_buy:
            profit_SB2 = self.delta(self.op_price_buy_2,self.tp_price_buy) * 100_000 * self.lot_buy_2
            profit += profit_SB2
            print(f'TP SB2:{profit_SB2}')
        if self.lot_buy_1>0 and self.price_H_bar1 >= self.tp_price_buy:
            profit_SB1 = self.delta(self.op_price_buy_1,self.tp_price_buy) * 100_000 * self.lot_buy_1
            profit += profit_SB1
            print(f'TP SB1:{profit_SB1}')
        if profit>0:
            self.cash_in_hand += profit - self.accumulated_loss
            print(f'SB profit:{profit} loss:{self.accumulated_loss} balance:{self.cash_in_hand}')
            return True # TP ACHIEVED
        return False

    def sell_tp_hit(self):
        # BREAK ON CYCLE END - PROFIT OR DD HIT
        profit = 0
        if self.lot_sell_3>0 and self.price_L_bar1 <= self.tp_price_sell: # SB3 HIT TP
            profit_SS3 = self.step_pt * 100_000 * self.lot_sell_3
            profit += profit_SS3
            print(f'TP SS3:{profit_SS3}')
        if self.lot_sell_2>0 and self.price_L_bar1 <= self.tp_price_sell:
            profit_SS2 = self.delta(self.op_price_sell_2,self.tp_price_sell) * 100_000 * self.lot_sell_2
            profit += profit_SS2
            print(f'TP SS2:{profit_SS2}')
        if self.lot_sell_1>0 and self.price_L_bar1 <= self.tp_price_sell:
            profit_SS1 = self.delta(self.op_price_sell_1,self.tp_price_sell) * 100_000 * self.lot_sell_1
            profit += profit_SS1
            print(f'TP SS1:{profit_SS1}')

        if profit>0:
            self.cash_in_hand += profit - self.accumulated_loss
            print(f'SS profit:{profit} loss:{self.accumulated_loss} balance:{self.cash_in_hand}')
            return True # TP ACHIEVED
        return False       

    def max_dd_hit(self):
        # MAX DD EVAL - DEBATABLE TO PUT THIS ON EVERY LV SL HIT
        if self.accumulated_loss > self.max_drawdown:
            self.cash_in_hand -= self.accumulated_loss
            print(f'MAX DD HIT loss:{self.accumulated_loss} balance:{self.cash_in_hand}')
            return True 
        return False

    def SS1_SS2_SS3_sl_monitor(self):
        # SCAN FOR SELL SL HIT
        if self.lot_sell_1>0 and self.price_H_bar1>=self.sl_price_sell_1:
            loss_SS1 = self.calculate_loss(self.op_price_sell_1,self.sl_price_sell_1,self.lot_sell_1)
            self.accumulated_loss += loss_SS1
            print(f'SS1 SL HIT loss:{loss_SS1:.2f}')
            self.lot_sell_1 = 0
            # BUMP LV1
            if self.delta(self.op_price_sell_1, self.sl_price_sell_1) < 0.5*self.step_pt:
                self.op_price_sell_1 -= self.step_pt
                self.sl_price_sell_1 = self.op_price_sell_1 + self.step_pt
                print(f'SS1 BEP? BUMPED LOWER SS1:{self.op_price_sell_1:.5f}')
        if self.lot_sell_2>0 and self.price_H_bar2>=self.sl_price_sell_2:
            loss_SS2 = self.calculate_loss(self.op_price_sell_2,self.sl_price_sell_2,self.lot_sell_2)
            self.accumulated_loss += loss_SS2
            self.lot_sell_2 = 0
            print(f'SS2 SL HIT loss:{loss_SS2:.2f}')
            # BUMP LV2
            if self.delta(self.op_price_sell_2, self.sl_price_sell_2) < 0.5*self.step_pt:
                self.op_price_sell_2 -= self.step_pt
                self.sl_price_sell_2 = self.op_price_sell_2 + self.step_pt
                print(f'SS2 BEP? BUMPED LOWER SS2:{self.op_price_sell_2:.5f}')
        if self.lot_sell_3>0 and self.price_H_bar3>=self.sl_price_sell_3:
            loss_SS3 = self.calculate_loss(self.op_price_sell_3,self.sl_price_sell_3,self.lot_sell_3)
            self.accumulated_loss += loss_SS3
            self.lot_sell_3 = 0
            print(f'SS3 SL HIT loss:{loss_SS3:.2f}')

    def SB1_SB2_SB3_sl_monitor(self):
        # SCAN FOR BUY SL HIT
        if self.lot_buy_1>0 and self.price_L_bar1<=self.sl_price_buy_1:
            loss_SB1 = self.calculate_loss(self.op_price_buy_1,self.sl_price_buy_1,self.lot_buy_1)
            self.accumulated_loss += loss_SB1
            self.lot_buy_1 = 0
            print(f'SB1 SL HIT loss:{loss_SB1:.2f}')
            # BUMP LV1 BUY IF BEP
            if self.delta(self.op_price_buy_1, self.sl_price_buy_1) < 0.5*self.step_pt:
                self.op_price_buy_1 += self.step_pt
                self.sl_price_buy_1 = self.op_price_buy_1 - self.step_pt
                print(f'SB1 BEP? BUMPED HIGHER SB1:{self.op_price_buy_1:.5f}')
        if self.lot_buy_2>0 and self.price_L_bar1<=self.sl_price_buy_2:
            loss_SB2 = self.calculate_loss(self.op_price_buy_2, self.sl_price_buy_2, self.lot_buy_2)
            self.accumulated_loss += loss_SB2
            self.lot_buy_2 = 0
            print(f'SB2 SL HIT loss:{loss_SB2:.2f}')
            # BUMP LV2 BUY IF BEP 
            if self.delta(self.op_price_buy_2, self.sl_price_buy_2) < 0.5*self.step_pt:
                self.op_price_buy_2 += self.step_pt 
                self.sl_price_buy_2 = self.op_price_buy_2 - self.step_pt
                print(f'SB2 BEP? BUMPED HIGHER SB2:{self.op_price_buy_2:.5f}') 
        if self.lot_buy_3>0 and self.price_L_bar1<=self.sl_price_buy_3:   
            loss_SB3 = self.calculate_loss(self.op_price_buy_3, self.sl_price_buy_3, self.lot_buy_3)
            self.accumulated_loss += loss_SB3
            self.lot_buy3 = 0
            print(f'SB3 SL HIT loss:{loss_SB3:.2f}')

    def SB1_SB2_SB3_op_monitor(self):
        # ESTABLISH ENTRY FOR BUY LV1, LV2, LV3
        total_target = self.target + self.accumulated_loss
        # SB1
        if self.lot_buy_1==0 and self.price_H_bar1>=self.op_price_buy_1:
            self.lot_buy_1 = self.calculate_lot_buy(total_target, 1)
            self.sl_price_buy_1 = self.op_price_buy_1 - self.step_pt
            print(f'SB1 OP:{self.op_price_buy_1:.5f} {self.lot_buy_1:.2f} lot')
        # SB2
        if self.lot_buy_2==0 and self.price_H_bar1>=self.op_price_buy_2:                            
            self.lot_buy_2 = self.calculate_lot_buy(total_target, 2)
            self.sl_price_buy_2 = self.op_price_buy_2 - self.step_pt
            print(f'SB2 OP:{self.op_price_buy_2:.5f} {self.lot_buy_2:.2f} lot')
            # ESTABLISH TRAIL BEP FOR LV1 & LV2 - IF 1 STEP BEHIND
            if self.delta(self.sl_price_buy_2,self.sl_price_buy_1)>0.5*self.step_pt:
                self.sl_price_buy_1 = self.op_price_buy_1
                print(f'SB1 TRAILED BEP')
            # TRAIL GRID SELL UP
            self.grid_sell_up(self.op_price_buy_1)
        # SB3
        if self.lot_buy_3==0 and self.price_H_bar1>=self.op_price_buy_3:
            self.lot_buy_3 = self.calculate_lot_buy(total_target, 3)
            self.sl_price_buy_3 = self.op_price_buy_3 - self.step_pt
            print(f'SB3 OP:{self.op_price_buy_3:.5f} {self.lot_buy_3:.2f} lot')
            # ESTABLISH TRAIL BEP FOR LV1 & LV2 IF 1 STEP BEHIND
            if self.delta(self.sl_price_buy_3,self.sl_price_buy_2)>0.5*self.step_pt:
                self.sl_price_buy_2 = self.op_price_buy_2
                print(f'SB2 TRAILED BEP')
            if self.delta(self.sl_price_buy_3,self.sl_price_buy_1)>0.5*self.step_pt:
                self.sl_price_buy_1 = self.op_price_buy_1
                print(f'SB1 TRAILED BEP')
            # TRAIL GRID SELL UP
            self.grid_sell_up(self.op_price_buy_2)

    def SS1_SS2_SS3_op_monitor(self):
        # ESTABLISH ENTRY FOR SELL LV1, LV2, LV3
        total_target = self.target + self.accumulated_loss
        # SS1
        if self.lot_sell_1==0 and self.price_L_bar1<=self.op_price_sell_1:
            self.lot_sell_1 = self.calculate_lot_sell(total_target, 1)
            self.sl_price_sell_1 = self.op_price_sell_1 + self.step_pt 
            print(f'SS1 OP:{self.op_price_sell_1:.5f} {self.lot_sell_1:.2f} lot')
        # SS2
        if self.lot_sell_2==0 and self.price_L_bar1<=self.op_price_sell_2:
            self.lot_sell_2 = self.calculate_lot_sell(total_target, 2)
            self.sl_price_sell_2 = self.op_price_sell_2 + self.step_pt
            print(f'SS2 OP:{self.op_price_sell_2:.5f} {self.lot_sell_2:.2f} lot') 
            # ESTABLISH TRAIL BEP FOR LV1 & LV2 - BELUM TENTU - ONLY IF PRICE IS BEHIND 1 LV
            if self.delta(self.sl_price_sell_1,self.sl_price_sell_2)>0.5*self.step_pt:
                self.sl_price_sell_1 = self.op_price_sell_1
                print(f'SS1 TRAILED BEP')
            # TRAIL GRID BUY DOWN
            self.grid_buy_down(self.op_price_sell_1)
        # SS3
        if self.lot_sell_3==0 and self.price_L_bar1<=self.op_price_sell_3:
            self.lot_sell_3 = self.calculate_lot_sell(total_target, 3)
            self.sl_price_sell_3 = self.op_price_sell_3 + self.step_pt 
            print(f'SS3 OP:{self.op_price_sell_3:.5f} {self.lot_sell_3:.2f} lot')
            # ESTABLISH TRAIL BEP FOR LV1 & LV2 - IF A STEP BEHIND
            if self.delta(self.sl_price_sell_3,self.sl_price_sell_2)>0.5*self.step_pt:
                self.sl_price_sell_2 = self.op_price_sell_2
                print(f'SS2 TRAILED BEP')
            if self.delta(self.sl_price_sell_3,self.sl_price_sell_1)>0.5*self.step_pt:
                self.sl_price_sell_1 = self.op_price_sell_1
                print(f'SS1 TRAILED BEP')
            # TRAIL GRID BUY DOWN
            self.grid_buy_down(self.op_price_sell_2)

    def _trade(self, action):
        import random

        action_vec = self.action_list[action]

        if self.cash_in_hand>=1000:
            print(f'action_vec->{action_vec}')
            if action_vec==0: # NO TRADE 
                self.one_step_bar()
            elif action_vec==1: # SELL FIRST 
                self.reset_cycle()
                self.set_first_buy(self.price_C_bar1)      
            elif action_vec==2: # BUY FIRST 
                self.reset_cycle()
                self.set_first_sell(self.price_C_bar1)
            
            if action_vec==1 or action_vec==2:
                # LOOP UNTIL PROFIT OR LOSS    
                while True:
                    # TRAVERSE 1 STEP BAR
                    self.one_step_bar()
                    # EVALUATE WHAT HAPPENED
                    flip = random.random()
                    if flip>0.5: 
                        # EVALUATING HIGH FIRST
                        self.SS1_SS2_SS3_sl_monitor()
                        # MAX DD EVAL - DEBATABLE TO PUT THIS ON EVERY LV SL HIT
                        if self.max_dd_hit()==True:
                            break 
                        self.SB1_SB2_SB3_op_monitor()                     
                        self.normalize_SS1_SS2()
                        if self.buy_tp_hit()==True: # SB TP
                            break
                        
                        # AND THEN EVALUATING LOW 
                        self.SB1_SB2_SB3_sl_monitor()
                        if self.max_dd_hit()==True:
                            break
                        self.SS1_SS2_SS3_op_monitor()
                        self.normalize_SB1_SB2()                   
                        if self.sell_tp_hit()==True: # SS TP
                            break

                    else: # LOW FIRST
                        # EVALUATING LOW FIRST
                        self.SB1_SB2_SB3_sl_monitor()
                        if self.max_dd_hit()==True:
                            break
                        self.SS1_SS2_SS3_op_monitor()
                        self.normalize_SB1_SB2()                    
                        if self.sell_tp_hit()==True: # SS TP
                            break
                        # AND THEN EVALUATING HIGH
                        self.SS1_SS2_SS3_sl_monitor()
                        # MAX DD EVAL - DEBATABLE TO PUT THIS ON EVERY LV SL HIT
                        if self.max_dd_hit()==True:
                            break 
                        self.SB1_SB2_SB3_op_monitor()                     
                        self.normalize_SS1_SS2()
                        if self.buy_tp_hit()==True: # SB TP
                            break
                

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

    # note: after transforming states are already 1xD
    state = env.reset()

    print('state pre scale', state)
    state = scaler.transform([state])
    print('state post scale', state)
    done = False

    while not done:
        #print('traversing states')
        action = agent.act(state)
        # IF NOT TAKING ANY POS-> MOVE 1 STEP-BAR INTO THE FUTURE
        # IF STARTING CYCLE -> COMPLETE CURRENT CYCLE N-BAR INTO THE FUTURE
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])

        if is_train == 'train':
            agent.train(state, action, reward, next_state, done)
        state = next_state

    #return info['cur_val']
    return info

def obs_convert(ohlc, orders, state_dim):
    obs = np.empty(state_dim, dtype='double')

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

    # config
    models_folder = 'linear_fx_trader_models'
    rewards_folder = 'linear_fx_trader_rewards'
    num_episodes = 2000
    #batch_size = 32
    initial_investment = 10000

    SHOW_EVERY = 500

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

    ep_rewards = [] 
    aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': [] }

    if args.mode == 'train':
        # play the game num_episodes times
        for e in range(num_episodes):
            t0 = datetime.now()
            info = play_one_episode(agent, env, args.mode)
            dt = datetime.now() - t0
            #print(f"episode: {e + 1}/{num_episodes}, episode end value: {info['cur_val']:.2f}, duration: {dt} win: {info['win']}, lose: {info['lose']}, total: {info['total']}, dd: {info['dd']}")
            portfolio_value.append(info['cur_val']) # append episode end portfolio value
            # EXTRA
            ep_rewards.append(info['cur_val'])

            if not e & SHOW_EVERY == 0: # EVERY SHOW_EVERY 
                average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:]) # 500 FROM BEHIND
                aggr_ep_rewards['ep'].append(e)
                aggr_ep_rewards['avg'].append(average_reward)
                aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:])) # 500 FROM BEHIND->RECENT
                aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:])) # 500 FROM BEHIND->RECENT

                print(f'Episode: {e} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}')
    
        # save the weights when we are done
        # save the DQN
        agent.save(f'{models_folder}/linear.npz')
        # save the scaler
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

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

    if args.mode == 'real': # REAL TIME RL HERE
        import db as db
        while True:
            now = datetime.now()
            if now.minute>=0 and now.minute<11:
                
                # QUERY TIMESTAMP
                db_time = db.get_timestamp()
                
                if db_time>local_timestamp: # TRAVERSE NEW STEP EVERYTIME NEW INFO COMES IN
                    # GET LATEST BAR INFO
                    recent_data = db.get_data() # DICTIONARY - OK

                    # CHECK OUTPUT OF RECENT_DATA HERE
                    print('recent_data', recent_data)

                    local_timestamp = db_time

                    # GET ORDERS INFO
                    orders = db.get_orders('orders') # LIST OF DICT - OK

                    # CHECK OUTPUT OF ORDERS_INFO HERE
                    print('orders',orders)

                    # CONVERT INTO STATES
                    state = obs_convert(recent_data, orders, state_size)

                    # PRINT OUT STATE MATRIX
                    print('state', state)

                    with open(f'{models_folder}/scaler.pkl', 'rb') as f:
                        scaler = pickle.load(f)

                    state = scaler.transform([state])

                    # PRINT OUT SCALED MATRIX
                    print('scaled state', state)

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

                    #sys.exit()

            time.sleep(10)
            # END WHILE
        
        # END REAL

    if args.mode=='train':
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
        plt.legend(loc=4) # 4-LOWER RIGHT
        plt.show()
        
        
  
