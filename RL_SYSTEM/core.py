import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import sys

def get_mode():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='either "train" or "test"')
    args = parser.parse_args()
    return args.mode

def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def prep_folder(input):
    maybe_make_dir(input['models_folder'])
    maybe_make_dir(input['rewards_folder'])

def get_data_H1():
    df = pd.read_csv('eurusd_h1_2013_2020.csv')
    return df

def get_data_m1():
    df = pd.read_csv('eurusd_m1_2013_2020.csv')
    return df

def get_scaler(env):
    import random 
    states = []
    for _ in range(env.H1_bars-1):
        action = np.random.choice(env.action_space)
        next_state = env.step_scale(action)
        states.append(next_state)
        if next_state[7]<0:
            break 
    scaler = StandardScaler()
    scaler.fit(states)
    print('scaler fitted')

    return scaler 

class PairEnv:

    def __init__(self, data_H1, data_m1, settings_dic):
        # DF
        self.data_H1 = data_H1 
        self.H1_bars, _ = self.data_H1.shape 
        # DF
        self.data_m1 = data_m1 
        self.m1_bars, _ = self.data_m1.shape 
        # SETTINGS - CONSTANTS
        self.initial_investment = settings_dic['initial_investment']
        self.target = settings_dic['target_usd']
        self.step_pt = settings_dic['step']
        self.dd = settings_dic['dd_percent_max']
        self.max_drawdown = self.initial_investment * self.dd * 0.01
        self.ideal_drawdown = self.initial_investment * settings_dic['dd_percent_ideal'] * 0.01
        self.kb_ratio = settings_dic['kickback_ratio']
        self.dd_weight = settings_dic['dd_weight']

        self.early_exit_level = 2.5 * self.target # 250
        self.early_exit_usd = 0.25 * self.target # 25
        self.begin_hour = 7
        self.end_hour = 17

        # TRACKER
        self.index_H1 = None 

        # CYCLE ATTRIBUTES
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

        self.mid_price = None
        self.tp_price_buy = None 
        self.tp_price_sell = None

        # CYCLE INFOS
        #self.count_win = None 
        #self.count_lose = None 
        #self.count_total_trades = None 
        self.accumulated_loss = None
        self.log = None

        # EPS INFO
        self.drawdown = None

        # DECISION VARS - STATES EXTERNAL
        self.close_change_pct_H1 = None 
        self.ema50_delta_pct_H1 = None 
        self.ema21_delta_pct_H1 = None

        self.price_open_H1_1 = None
        self.price_high_H1_1 = None 
        self.price_low_H1_1 = None 
        self.price_close_H1_1 = None 

        self.price_open_H1_2 = None
        self.price_high_H1_2 = None 
        self.price_low_H1_2 = None 
        self.price_close_H1_2 = None 

        self.price_open_H1_3 = None
        self.price_high_H1_3 = None 
        self.price_low_H1_3 = None 
        self.price_close_H1_3 = None 

        self.price_ema50_H1_1 = None 
        self.price_ema50_H1_2 = None
        self.price_ema50_H1_3 = None

        self.price_ema21_H1_1 = None 
        self.price_ema21_H1_2 = None
        self.price_ema21_H1_3 = None

        self.day_H1 = None
        self.hour_H1 = None 
        
        # SIMULATION VARS
        self.price_open_m1 = None
        self.price_high_m1 = None 
        self.price_low_m1 = None 
        self.price_close_m1 = None     

        # Q S A - STATES INTERNAL
        self.balance = None 
        self.ep_reward = None

        self.action_space = np.array([0,1,2])

        self.state_dim = 24

        # FILTER
        

        # RESET EPISODE
        self.reset_eps()

    def _get_obs_H1(self):
        obs = np.empty(self.state_dim, dtype='float64') 

        obs[0] = self.close_change_pct_H1
        obs[1] = self.ema50_delta_pct_H1
        obs[2] = self.ema21_delta_pct_H1

        obs[3] = self.price_open_H1_1
        obs[4] = self.price_high_H1_1
        obs[5] = self.price_low_H1_1
        obs[6] = self.price_close_H1_1

        obs[7] = self.price_open_H1_2
        obs[8] = self.price_high_H1_2
        obs[9] = self.price_low_H1_2
        obs[10] = self.price_close_H1_2

        obs[11] = self.price_open_H1_3
        obs[12] = self.price_high_H1_3
        obs[13] = self.price_low_H1_3
        obs[14] = self.price_close_H1_3

        obs[15] = self.price_ema50_H1_1
        obs[16] = self.price_ema50_H1_2
        obs[17] = self.price_ema50_H1_3

        obs[18] = self.price_ema21_H1_1
        obs[19] = self.price_ema21_H1_2
        obs[20] = self.price_ema21_H1_3

        obs[21] = self.day_H1
        obs[22] = self.hour_H1

        obs[-1] = self.balance

        return obs 

    def reset_eps(self): # INITIAL STATE - STATE 0
        self.index_H1 = 0
        self.update_states_H1()

        self.balance = self.initial_investment
        self.drawdown = 0
        self.ep_reward = 0
        
        return self._get_obs_H1()

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

        self.mid_price = 0  
        self.tp_price_buy = 0
        self.tp_price_sell = 0

        # IMPORTANT STATES
        self.accumulated_loss = 0
        self.log = []

    def _get_val(self):
        return self.balance 

    def skip_one_bar(self):
        self.index_H1 += 1
        self.update_states_H1()

    def update_states_H1(self):

        self.price_open_H1_1 = self.data_H1['open_1'][self.index_H1]
        self.price_high_H1_1 = self.data_H1['high_1'][self.index_H1] 
        self.price_low_H1_1 = self.data_H1['low_1'][self.index_H1] 
        self.price_close_H1_1 = self.data_H1['close_1'][self.index_H1]

        self.price_open_H1_2 = self.data_H1['open_2'][self.index_H1]
        self.price_high_H1_2 = self.data_H1['high_2'][self.index_H1] 
        self.price_low_H1_2 = self.data_H1['low_2'][self.index_H1] 
        self.price_close_H1_2 = self.data_H1['close_2'][self.index_H1]

        self.price_open_H1_3 = self.data_H1['open_3'][self.index_H1]
        self.price_high_H1_3 = self.data_H1['high_3'][self.index_H1] 
        self.price_low_H1_3 = self.data_H1['low_3'][self.index_H1] 
        self.price_close_H1_3 = self.data_H1['close_3'][self.index_H1]        

        self.price_ema50_H1_1 = self.data_H1['EMA50_1'][self.index_H1] 
        self.price_ema50_H1_2 = self.data_H1['EMA50_2'][self.index_H1]
        self.price_ema50_H1_3 = self.data_H1['EMA50_3'][self.index_H1]

        self.price_ema21_H1_1 = self.data_H1['EMA21_1'][self.index_H1] 
        self.price_ema21_H1_2 = self.data_H1['EMA21_2'][self.index_H1]
        self.price_ema21_H1_3 = self.data_H1['EMA21_3'][self.index_H1]

        close_change = (self.price_close_H1_1 - self.price_close_H1_2)/self.price_close_H1_2
        ema50_close_delta = (self.price_close_H1_1-self.price_ema50_H1_1)/self.price_ema50_H1_1
        ema21_close_delta = (self.price_close_H1_1-self.price_ema21_H1_1)/self.price_ema21_H1_1
        self.close_change_pct_H1 = close_change * 100 
        self.ema50_delta_pct_H1 = ema50_close_delta * 100
        self.ema21_delta_pct_H1 = ema21_close_delta * 100

        self.day_H1 = self.data_H1['day'][self.index_H1]
        self.hour_H1 = self.data_H1['hour'][self.index_H1]

    def step_scale(self, action):
        assert action in self.action_space

        self.index_H1 += 1
        self.update_states_H1()
        
        contract_size = 10000 # 0.1 LOT
        
        self.update_states_H1()
        open_price = self.price_open_H1_1
        close_price = self.price_close_H1_1

        if action==1: # SELL             
            profit = ( open_price - close_price ) * contract_size             
            self.balance += profit # obs[7]
        elif action==2: # BUY
            profit = ( close_price - open_price ) * contract_size
            self.balance += profit # obs[7] 

        return self._get_obs_H1() # RETURN NEXT STATE

    def calc_balance_change(self, profit):      
        balance_change = ( profit / self.balance ) * 100
        return balance_change
    
    def calculate_lot_buy(self, target_usd, level):
        total_buy_slot = 0
        total_buy_lot = 0

        total_running_buy = 0

        if self.lot_buy_1==0:
            total_buy_slot += (self.step_pt/self.kb_ratio) * 3 * 100_000 * 0.5
        elif self.lot_buy_1>0:
            total_running_buy += ((self.tp_price_buy-self.op_price_buy_1)/self.kb_ratio) * 100_000 * self.lot_buy_1

        if self.lot_buy_2==0:    
            total_buy_slot += (self.step_pt/self.kb_ratio) * 2 * 100_000 * 0.3
        elif self.lot_buy_2>0:
            total_running_buy += ((self.tp_price_buy-self.op_price_buy_2)/self.kb_ratio) * 100_000 * self.lot_buy_2

        if self.lot_buy_3==0:
            total_buy_slot += (self.step_pt/self.kb_ratio) * 100_000 * 0.2
        
        if total_buy_slot>0:
            total_buy_lot = ( target_usd - total_running_buy ) / total_buy_slot 

        sb1_lot = 0.5 * total_buy_lot
        sb2_lot = 0.3 * total_buy_lot 
        sb3_lot = 0.2 * total_buy_lot 

        #print(f'SB1:{self.op_price_buy_1:.5f} SB2:{self.op_price_buy_2:.5f} SB3:{self.op_price_buy_3:.5f} SBTP:{self.tp_price_buy:.5f} - SS1:{self.op_price_sell_1:.5f} SS2:{self.op_price_sell_2:.5f} SS3:{self.op_price_sell_3:.5f} SSTP:{self.tp_price_sell:.5f}') 
        #print(f'BUY target:{target_usd} running:{total_running_buy} calc lot SB1:{sb1_lot:.2f} SB2:{sb2_lot:.2f} SB3:{sb3_lot:.2f}')

        sb1_pt = 3 * self.step_pt
        sb2_pt = 2 * self.step_pt
        sb1_pt_actual = self.tp_price_buy - self.op_price_buy_1 
        sb2_pt_actual = self.tp_price_buy - self.op_price_buy_2 

        if level==1:
            if sb1_pt_actual>0:
                sb1_lot = sb1_lot * ( sb1_pt / sb1_pt_actual )
            return sb1_lot 
        elif level==2:
            if sb2_pt_actual>0:
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
            total_sell_slot += (step_pips/self.kb_ratio) * 3 * dollar_per_pip * 0.5
        elif self.lot_sell_1>0:
            total_running_sell += ((self.op_price_sell_1-self.tp_price_sell)/self.kb_ratio) * 100_000 * self.lot_sell_1

        if self.lot_sell_2==0:    
            total_sell_slot += (step_pips/self.kb_ratio) * 2 * dollar_per_pip * 0.3
        elif self.lot_sell_2>0:
            total_running_sell += ((self.op_price_sell_2-self.tp_price_sell)/self.kb_ratio) * 100_000 * self.lot_sell_2

        if self.lot_sell_3==0:
            total_sell_slot += (step_pips/self.kb_ratio) * dollar_per_pip * 0.2
        
        if total_sell_slot>0:
            total_sell_lot = ( target_usd - total_running_sell ) / total_sell_slot 

        ss1_lot = 0.5 * total_sell_lot
        ss2_lot = 0.3 * total_sell_lot 
        ss3_lot = 0.2 * total_sell_lot 

        #print(f'SB1:{self.op_price_buy_1:.5f} SB2:{self.op_price_buy_2:.5f} SB3:{self.op_price_buy_3:.5f} SBTP:{self.tp_price_buy:.5f} - SS1:{self.op_price_sell_1:.5f} SS2:{self.op_price_sell_2:.5f} SS3:{self.op_price_sell_3:.5f} SSTP:{self.tp_price_sell:.5f}') 
        #print(f'SELL target:{target_usd} running:{total_running_sell} calc lot SS1:{ss1_lot:.2f} SS2:{ss2_lot:.2f} SS3:{ss3_lot:.2f}')

        ss1_pt = 3*self.step_pt
        ss2_pt = 2*self.step_pt  
        ss1_pt_actual = ( self.op_price_sell_1 - self.tp_price_sell ) 
        ss2_pt_actual = ( self.op_price_sell_2 - self.tp_price_sell )

        if level==1:
            if ss1_pt_actual>0:
                ss1_lot = ss1_lot * ( ss1_pt / ss1_pt_actual )
            return ss1_lot 
        elif level==2:
            if ss2_pt_actual>0:
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

        #print(f'SET CYCLE SB1:{self.op_price_buy_1:.5f} SB2:{self.op_price_buy_2:.5f} SB3:{self.op_price_buy_3:.5f} - SS1:{self.op_price_sell_1:.5f} SS2:{self.op_price_sell_2:.5f} SS3:{self.op_price_sell_3:.5f} maxdd:{self.max_drawdown:.2f}') 
        self.log.append(f'SET CYCLE BUY balance:{self.balance} SB1:{self.op_price_buy_1:.5f} SB2:{self.op_price_buy_2:.5f} \
            SB3:{self.op_price_buy_3:.5f} - SS1:{self.op_price_sell_1:.5f} SS2:{self.op_price_sell_2:.5f} SS3:{self.op_price_sell_3:.5f} \
                maxdd:{self.max_drawdown:.2f} \n')
        
        # ACT NOW - SET FIRST LV1->BUY, AND OTHER LEVELS
        self.lot_buy_1 = self.calculate_lot_buy(self.target, 1)

        #print(f'OP SB1@{ask} {self.lot_buy_1:.2f} lot')

        self.log.append(f'OP SB1@{ask} {self.lot_buy_1:.2f} lot \n')

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
        
       
        #print(f'SET CYCLE SB1:{self.op_price_buy_1:.5f} SB2:{self.op_price_buy_2:.5f} SB3:{self.op_price_buy_3:.5f} - SS1:{self.op_price_sell_1:.5f} SS2:{self.op_price_sell_2:.5f} SS3:{self.op_price_sell_3:.5f} maxdd:{self.max_drawdown}')

        self.log.append(f'SET CYCLE SELL balance:{self.balance} SB1:{self.op_price_buy_1:.5f} SB2:{self.op_price_buy_2:.5f} \
            SB3:{self.op_price_buy_3:.5f} - SS1:{self.op_price_sell_1:.5f} SS2:{self.op_price_sell_2:.5f} \
                SS3:{self.op_price_sell_3:.5f} maxdd:{self.max_drawdown} \n')

        self.lot_sell_1 = self.calculate_lot_sell(self.target, 1) 
        #print(f'OP SS1@{bid} {self.lot_sell_1:.2f} lot')

        self.log.append(f'OP SS1@{bid} {self.lot_sell_1:.2f} lot \n')

    def grid_sell_up(self, new_mid): # TRAIL GRID SELL UP 

        if self.mid_price<new_mid:      
            self.mid_price = new_mid 
            self.op_price_sell_1 = self.mid_price - self.step_pt
            self.op_price_sell_2 = self.op_price_sell_1 - self.step_pt 
            self.op_price_sell_3 = self.op_price_sell_2 - self.step_pt 
            self.tp_price_sell = self.op_price_sell_3 - self.step_pt

            #print(f'GRID SELL UP - SS1:{self.op_price_sell_1:.5f} SS2:{self.op_price_sell_2:.5f} SS3:{self.op_price_sell_3:.5f} maxdd:{self.max_drawdown}')

            self.log.append(f'GRID SELL UP - SS1:{self.op_price_sell_1:.5f} SS2:{self.op_price_sell_2:.5f} SS3:{self.op_price_sell_3:.5f} \
                maxdd:{self.max_drawdown} \n')

    def grid_buy_down(self, new_mid):

        if self.mid_price>new_mid:
            self.mid_price = new_mid 
            self.op_price_buy_1 = self.mid_price + self.step_pt
            self.op_price_buy_2 = self.op_price_buy_1 + self.step_pt 
            self.op_price_buy_3 = self.op_price_buy_2 + self.step_pt 
            self.tp_price_buy = self.op_price_buy_3 + self.step_pt

            #print(f'GRID BUY DOWN - SB1:{self.op_price_buy_1:.5f} SB2:{self.op_price_buy_2:.5f} SB3:{self.op_price_buy_3:.5f} maxdd:{self.max_drawdown}')

            self.log.append(f'GRID BUY DOWN - SB1:{self.op_price_buy_1:.5f} SB2:{self.op_price_buy_2:.5f} SB3:{self.op_price_buy_3:.5f} \
                maxdd:{self.max_drawdown} \n')

    def normalize_SB1_SB2(self, askbid_m1):
        # NORMALIZE SB1 ON LV3
        if self.lot_buy_1==0 and self.delta(self.op_price_buy_1, self.op_price_buy_3)<0.5*self.step_pt:                                    
            if askbid_m1 < self.op_price_buy_3-(3*self.step_pt):
                #print(f'PRE-NORM SB1:{self.op_price_buy_1}')
                self.op_price_buy_1 = self.op_price_buy_3-(2*self.step_pt) # KE LEVEL 1
                #print(f'POST-NORM SB1:{self.op_price_buy_1} 2 STEPS')

                self.log.append(f'POST-NORM SB1:{self.op_price_buy_1} 2 STEPS \n')

            elif askbid_m1 < self.op_price_buy_3-(2*self.step_pt):
                #print(f'PRE-NORM SB1:{self.op_price_buy_1}')
                self.op_price_buy_1 = self.op_price_buy_3-(1*self.step_pt) # KE LEVEL 2
                #print(f'POST-NORM SB1:{self.op_price_buy_1} 1 STEP')  

                self.log.append(f'POST-NORM SB1:{self.op_price_buy_1} 1 STEP \n')
                                
        # NORMALIZE SB2 ON LV3
        if self.lot_buy_2==0 and self.delta(self.op_price_buy_2, self.op_price_buy_3)<0.5*self.step_pt: 
            if askbid_m1 < self.op_price_buy_3-(2*self.step_pt):
                #print(f'PRE-NORM SB2:{self.op_price_buy_2}')
                self.op_price_buy_2 = self.op_price_buy_3-(1*self.step_pt) # KE LEVEL 2
                #print(f'POST-NORM SB2:{self.op_price_buy_2} 1 STEP')   

                self.log.append(f'POST-NORM SB2:{self.op_price_buy_2} 1 STEP \n')

        # NORMALIZE SB1 ON LV2 
        if self.lot_buy_1==0 and self.delta(self.op_price_buy_1, self.op_price_buy_2)<0.5*self.step_pt:
            if askbid_m1 < self.op_price_buy_2-(2*self.step_pt):
                #print(f'PRE-NORM SB1:{self.op_price_buy_1}')
                self.op_price_buy_1 = self.op_price_buy_2-(1*self.step_pt) # KE LEVEL 1
                #print(f'POST-NORM SB1:{self.op_price_buy_1}')

                self.log.append(f'POST-NORM SB1:{self.op_price_buy_1} \n')

    def normalize_SS1_SS2(self, askbid_m1):
        # NORMALIZE SS1 ON SS3
        if self.lot_sell_1==0 and self.delta(self.op_price_sell_1,self.op_price_sell_3)<0.5*self.step_pt: # LV1 & LV3 SELL MERGED
            if askbid_m1 > self.op_price_sell_3 + ( 3 * self.step_pt ): 
                #print(f'PRE-NORM SS1:{self.op_price_sell_1}')
                self.op_price_sell_1 = self.op_price_sell_3 + ( 2 * self.step_pt ) # NORMALIZE SS1 - TO LV1
                #print(f'POST-NORM SS1:{self.op_price_sell_1} 2 STEPS')

                self.log.append(f'POST-NORM SS1:{self.op_price_sell_1} 2 STEPS \n')

            elif askbid_m1 > self.op_price_sell_3 + ( 2 * self.step_pt ):  
                #print(f'PRE-NORM SS1:{self.op_price_sell_1}')
                self.op_price_sell_1 = self.op_price_sell_3 + ( 1 * self.step_pt ) # NORMALIZE SS1 - TO LV2
                #print(f'POST-NORM SS1:{self.op_price_sell_1} 1 STEP')

                self.log.append(f'POST-NORM SS1:{self.op_price_sell_1} 1 STEP \n')

        # NORMALIZE SS2 ON SS3
        if self.lot_sell_2==0 and self.delta(self.op_price_sell_2,self.op_price_sell_3)<0.5*self.step_pt: # LV2 & LV3 SELL MERGED 
            if askbid_m1 > self.op_price_sell_3 + ( 2 * self.step_pt ): 
                #print(f'PRE-NORM SS2:{self.op_price_sell_2}')
                self.op_price_sell_2 = self.op_price_sell_3 + ( 1 * self.step_pt )  # NORMALIZE SS2 - TO LV2
                #print(f'POST-NORM SS2:{self.op_price_sell_2} 1 STEP')

                self.log.append(f'POST-NORM SS2:{self.op_price_sell_2} 1 STEP \n')

        # NORMALIZE SS1 ON SS2
        if self.lot_sell_1==0 and self.delta(self.op_price_sell_1,self.op_price_sell_2)<0.5*self.step_pt: # LV1 & LV2 SELL MERGED
            if askbid_m1 > self.op_price_sell_2 + ( 2 * self.step_pt ):
                #print(f'PRE-NORM SS1:{self.op_price_sell_1}')
                self.op_price_sell_1 = self.op_price_sell_2 + ( 1 * self.step_pt ) # NORMALIZE SS1 - TO LV1
                #print(f'POST-NORM SS1:{self.op_price_sell_1} 1 STEP')

                self.log.append(f'POST-NORM SS1:{self.op_price_sell_1} 1 STEP \n')

    def buy_tp_hit(self, askbid_m1):
        # BREAK ON CYCLE END - PROFIT OR DD HIT
        profit = 0
        if self.lot_buy_3>0 and askbid_m1 >= self.tp_price_buy: # SB3 HIT TP
            profit_SB3 = self.step_pt * 100_000 * self.lot_buy_3
            profit += profit_SB3
            #print(f'TP SB3:{profit_SB3}')
            self.lot_buy_3 = 0
            self.log.append(f'TP SB3:{profit_SB3} \n')

        if self.lot_buy_2>0 and askbid_m1 >= self.tp_price_buy:
            profit_SB2 = self.delta(self.op_price_buy_2,self.tp_price_buy) * 100_000 * self.lot_buy_2
            profit += profit_SB2
            #print(f'TP SB2:{profit_SB2}')
            self.lot_buy_2 = 0
            self.log.append(f'TP SB2:{profit_SB2} \n')

        if self.lot_buy_1>0 and askbid_m1 >= self.tp_price_buy:
            profit_SB1 = self.delta(self.op_price_buy_1,self.tp_price_buy) * 100_000 * self.lot_buy_1
            profit += profit_SB1
            #print(f'TP SB1:{profit_SB1}')
            self.lot_buy_1 = 0
            self.log.append(f'TP SB1:{profit_SB1} \n')

        if profit>0:
            self.balance += profit - self.accumulated_loss
            #print(f'SB profit:{profit} loss:{self.accumulated_loss} balance:{self.balance}')

            self.log.append(f'SB profit:{profit} loss:{self.accumulated_loss} balance:{self.balance} \n')

            return True # TP ACHIEVED
        return False

    def sell_tp_hit(self, askbid_m1):
        # BREAK ON CYCLE END - PROFIT OR DD HIT
        profit = 0
        if self.lot_sell_3>0 and askbid_m1 <= self.tp_price_sell: # SB3 HIT TP
            profit_SS3 = self.step_pt * 100_000 * self.lot_sell_3
            profit += profit_SS3
            self.lot_sell_3 = 0
            #print(f'TP SS3:{profit_SS3}')

            self.log.append(f'TP SS3:{profit_SS3} \n')

        if self.lot_sell_2>0 and askbid_m1 <= self.tp_price_sell:
            profit_SS2 = self.delta(self.op_price_sell_2,self.tp_price_sell) * 100_000 * self.lot_sell_2           
            profit += profit_SS2
            self.lot_sell_2 = 0
            #print(f'TP SS2:{profit_SS2}')

            self.log.append(f'TP SS2:{profit_SS2} \n')

        if self.lot_sell_1>0 and askbid_m1 <= self.tp_price_sell:
            profit_SS1 = self.delta(self.op_price_sell_1,self.tp_price_sell) * 100_000 * self.lot_sell_1
            profit += profit_SS1
            self.lot_sell_1 = 0
            #print(f'TP SS1:{profit_SS1}')

            self.log.append(f'TP SS1:{profit_SS1} \n')

        if profit>0:
            self.balance += profit - self.accumulated_loss
            #print(f'SS profit:{profit} loss:{self.accumulated_loss} balance:{self.balance}')

            self.log.append(f'SS profit:{profit} loss:{self.accumulated_loss} balance:{self.balance} \n')

            return True # TP ACHIEVED
        return False  

    def early_exit_hit(self, askbid_m1):
        if self.accumulated_loss >= self.early_exit_level:
            cur_value_usd = self.floating_usd(askbid_m1)
            if cur_value_usd >= self.early_exit_usd:
                self.balance +=  cur_value_usd - self.accumulated_loss

                self.log.append(f'Early Exit Hit:{cur_value_usd} loss:{self.accumulated_loss} balance:{self.balance} \n')
                return True # EARLY EXIT HIT
        return False

    def max_dd_hit(self):
        # MAX DD EVAL - DEBATABLE TO PUT THIS ON EVERY LV SL HIT
        if self.accumulated_loss > self.max_drawdown:
            self.balance -= self.accumulated_loss
            #print(f'MAX DD HIT loss:{self.accumulated_loss} balance:{self.balance}')

            self.log.append(f'MAX DD HIT loss:{self.accumulated_loss} balance:{self.balance} \n')

            return True 
        return False

    def SS1_SS2_SS3_sl_monitor(self, askbid_m1):
        # SCAN FOR SELL SL HIT
        if self.lot_sell_1>0 and askbid_m1>=self.sl_price_sell_1:
            loss_SS1 = self.calculate_loss(self.op_price_sell_1,self.sl_price_sell_1,self.lot_sell_1)
            self.accumulated_loss += loss_SS1
            self.lot_sell_1 = 0
            #self.cycle_drawdown += loss_SS1
            #print(f'SS1 SL HIT loss:{loss_SS1:.2f}')

            self.log.append(f'SS1 SL HIT loss:{loss_SS1:.2f} \n')
            
            # BUMP LV1
            if self.delta(self.op_price_sell_1, self.sl_price_sell_1) < 0.5*self.step_pt:
                self.op_price_sell_1 -= self.step_pt
                self.sl_price_sell_1 = self.op_price_sell_1 + self.step_pt
                #print(f'SS1 BEP? BUMPED LOWER SS1:{self.op_price_sell_1:.5f}')

                self.log.append(f'SS1 BEP? BUMPED LOWER SS1:{self.op_price_sell_1:.5f} \n')

        if self.lot_sell_2>0 and askbid_m1>=self.sl_price_sell_2:
            loss_SS2 = self.calculate_loss(self.op_price_sell_2,self.sl_price_sell_2,self.lot_sell_2)
            self.accumulated_loss += loss_SS2
            self.lot_sell_2 = 0
            #self.cycle_drawdown += loss_SS2
            
            #print(f'SS2 SL HIT loss:{loss_SS2:.2f}')

            self.log.append(f'SS2 SL HIT loss:{loss_SS2:.2f} \n')

            # BUMP LV2
            if self.delta(self.op_price_sell_2, self.sl_price_sell_2) < 0.5*self.step_pt:
                self.op_price_sell_2 -= self.step_pt
                self.sl_price_sell_2 = self.op_price_sell_2 + self.step_pt
                #print(f'SS2 BEP? BUMPED LOWER SS2:{self.op_price_sell_2:.5f}')

                self.log.append(f'SS2 BEP? BUMPED LOWER SS2:{self.op_price_sell_2:.5f} \n')

        if self.lot_sell_3>0 and askbid_m1>=self.sl_price_sell_3:
            loss_SS3 = self.calculate_loss(self.op_price_sell_3,self.sl_price_sell_3,self.lot_sell_3)
            self.accumulated_loss += loss_SS3
            self.lot_sell_3 = 0
            #self.cycle_drawdown += loss_SS3
            
            #print(f'SS3 SL HIT loss:{loss_SS3:.2f}')
            self.log.append(f'SS3 SL HIT loss:{loss_SS3:.2f} \n')

    def SB1_SB2_SB3_sl_monitor(self, askbid_m1):
        # SCAN FOR BUY SL HIT
        if self.lot_buy_1>0 and askbid_m1<=self.sl_price_buy_1:
            loss_SB1 = self.calculate_loss(self.op_price_buy_1,self.sl_price_buy_1,self.lot_buy_1)
            self.accumulated_loss += loss_SB1
            self.lot_buy_1 = 0
            #self.cycle_drawdown += loss_SB1
            
            #print(f'SB1 SL HIT loss:{loss_SB1:.2f}')

            self.log.append(f'SB1 SL HIT loss:{loss_SB1:.2f} \n')

            # BUMP LV1 BUY IF BEP
            if self.delta(self.op_price_buy_1, self.sl_price_buy_1) < 0.5*self.step_pt:
                self.op_price_buy_1 += self.step_pt
                self.sl_price_buy_1 = self.op_price_buy_1 - self.step_pt
                #print(f'SB1 BEP? BUMPED HIGHER SB1:{self.op_price_buy_1:.5f}')
                self.log.append(f'SB1 BEP? BUMPED HIGHER SB1:{self.op_price_buy_1:.5f} \n')

        if self.lot_buy_2>0 and askbid_m1<=self.sl_price_buy_2:
            loss_SB2 = self.calculate_loss(self.op_price_buy_2, self.sl_price_buy_2, self.lot_buy_2)
            self.accumulated_loss += loss_SB2
            #self.cycle_drawdown += loss_SB2
            self.lot_buy_2 = 0
            #print(f'SB2 SL HIT loss:{loss_SB2:.2f}')
            # BUMP LV2 BUY IF BEP 
            if self.delta(self.op_price_buy_2, self.sl_price_buy_2) < 0.5*self.step_pt:
                self.op_price_buy_2 += self.step_pt 
                self.sl_price_buy_2 = self.op_price_buy_2 - self.step_pt
                #print(f'SB2 BEP? BUMPED HIGHER SB2:{self.op_price_buy_2:.5f}') 
                self.log.append(f'SB2 BEP? BUMPED HIGHER SB2:{self.op_price_buy_2:.5f} \n')

        if self.lot_buy_3>0 and askbid_m1<=self.sl_price_buy_3:   
            loss_SB3 = self.calculate_loss(self.op_price_buy_3, self.sl_price_buy_3, self.lot_buy_3)
            self.accumulated_loss += loss_SB3
            self.lot_buy_3 = 0
            #self.cycle_drawdown += loss_SB3
            
            #print(f'SB3 SL HIT loss:{loss_SB3:.2f}')
            self.log.append(f'SB3 SL HIT loss:{loss_SB3:.2f} \n')

    def SB1_SB2_SB3_op_monitor(self, askbid_m1):
        # ESTABLISH ENTRY FOR BUY LV1, LV2, LV3
        total_target = self.target + self.accumulated_loss
        # SB1
        if self.lot_buy_1==0 and askbid_m1>=self.op_price_buy_1:
            self.lot_buy_1 = self.calculate_lot_buy(total_target, 1)
            self.sl_price_buy_1 = self.op_price_buy_1 - self.step_pt
            #print(f'SB1 OP:{self.op_price_buy_1:.5f} {self.lot_buy_1:.2f} lot')

            self.log.append(f'SB1 OP:{self.op_price_buy_1:.5f} {self.lot_buy_1:.2f} lot \n')

        # SB2
        if self.lot_buy_2==0 and askbid_m1>=self.op_price_buy_2:                            
            self.lot_buy_2 = self.calculate_lot_buy(total_target, 2)
            self.sl_price_buy_2 = self.op_price_buy_2 - self.step_pt
            #print(f'SB2 OP:{self.op_price_buy_2:.5f} {self.lot_buy_2:.2f} lot')

            self.log.append(f'SB2 OP:{self.op_price_buy_2:.5f} {self.lot_buy_2:.2f} lot \n')

            # ESTABLISH TRAIL BEP FOR LV1 & LV2 - IF 1 STEP BEHIND
            if self.delta(self.sl_price_buy_2,self.sl_price_buy_1)>0.5*self.step_pt:
                self.sl_price_buy_1 = self.op_price_buy_1
                #print(f'SB1 TRAILED BEP')

                self.log.append('SB1 TRAILED BEP \n')

            # TRAIL GRID SELL UP
            self.grid_sell_up(self.op_price_buy_1)
        # SB3
        if self.lot_buy_3==0 and askbid_m1>=self.op_price_buy_3:
            self.lot_buy_3 = self.calculate_lot_buy(total_target, 3)
            self.sl_price_buy_3 = self.op_price_buy_3 - self.step_pt
            #print(f'SB3 OP:{self.op_price_buy_3:.5f} {self.lot_buy_3:.2f} lot')

            self.log.append(f'SB3 OP:{self.op_price_buy_3:.5f} {self.lot_buy_3:.2f} lot \n')

            # ESTABLISH TRAIL BEP FOR LV1 & LV2 IF 1 STEP BEHIND
            if self.delta(self.sl_price_buy_3,self.sl_price_buy_2)>0.5*self.step_pt:
                self.sl_price_buy_2 = self.op_price_buy_2
                #print(f'SB2 TRAILED BEP')

                self.log.append('SB2 TRAILED BEP \n')

            #if self.delta(self.sl_price_buy_3,self.sl_price_buy_1)>0.5*self.step_pt:
            #    self.sl_price_buy_1 = self.op_price_buy_1
                #print(f'SB1 TRAILED BEP')

            #    self.log.append('SB1 TRAILED BEP \n')

            # TRAIL GRID SELL UP
            self.grid_sell_up(self.op_price_buy_2)

    def SS1_SS2_SS3_op_monitor(self, askbid_m1):
        # ESTABLISH ENTRY FOR SELL LV1, LV2, LV3
        total_target = self.target + self.accumulated_loss
        # SS1
        if self.lot_sell_1==0 and askbid_m1<=self.op_price_sell_1:
            self.lot_sell_1 = self.calculate_lot_sell(total_target, 1)
            self.sl_price_sell_1 = self.op_price_sell_1 + self.step_pt 
            #print(f'SS1 OP:{self.op_price_sell_1:.5f} {self.lot_sell_1:.2f} lot')

            self.log.append(f'SS1 OP:{self.op_price_sell_1:.5f} {self.lot_sell_1:.2f} lot \n')

        # SS2
        if self.lot_sell_2==0 and askbid_m1<=self.op_price_sell_2:
            self.lot_sell_2 = self.calculate_lot_sell(total_target, 2)
            self.sl_price_sell_2 = self.op_price_sell_2 + self.step_pt
            #print(f'SS2 OP:{self.op_price_sell_2:.5f} {self.lot_sell_2:.2f} lot') 

            self.log.append(f'SS2 OP:{self.op_price_sell_2:.5f} {self.lot_sell_2:.2f} lot \n')

            # ESTABLISH TRAIL BEP FOR LV1 & LV2 - BELUM TENTU - ONLY IF PRICE IS BEHIND 1 LV
            if self.delta(self.sl_price_sell_1,self.sl_price_sell_2)>0.5*self.step_pt:
                self.sl_price_sell_1 = self.op_price_sell_1
                #print(f'SS1 TRAILED BEP')
                self.log.append('SS1 TRAILED BEP \n')
            # TRAIL GRID BUY DOWN
            self.grid_buy_down(self.op_price_sell_1)
        # SS3
        if self.lot_sell_3==0 and askbid_m1<=self.op_price_sell_3:
            self.lot_sell_3 = self.calculate_lot_sell(total_target, 3)
            self.sl_price_sell_3 = self.op_price_sell_3 + self.step_pt 
            
            #print(f'SS3 OP:{self.op_price_sell_3:.5f} {self.lot_sell_3:.2f} lot')
            self.log.append(f'SS3 OP:{self.op_price_sell_3:.5f} {self.lot_sell_3:.2f} lot \n')

            # ESTABLISH TRAIL BEP FOR LV1 & LV2 - IF A STEP BEHIND
            if self.delta(self.sl_price_sell_3,self.sl_price_sell_2)>0.5*self.step_pt:
                self.sl_price_sell_2 = self.op_price_sell_2
                #print(f'SS2 TRAILED BEP')
                self.log.append('SS2 TRAILED BEP \n')
            #if self.delta(self.sl_price_sell_3,self.sl_price_sell_1)>0.5*self.step_pt:
            #    self.sl_price_sell_1 = self.op_price_sell_1
                #print(f'SS1 TRAILED BEP')
            #    self.log.append('SS1 TRAILED BEP \n')
            # TRAIL GRID BUY DOWN
            self.grid_buy_down(self.op_price_sell_2)

    def update_bar_m1(self, i):
        self.price_open_m1 = self.data_m1['open'][i]
        self.price_high_m1 = self.data_m1['high'][i] 
        self.price_low_m1 = self.data_m1['low'][i] 
        self.price_close_m1 = self.data_m1['close'][i]

    def buy_value_usd(self, op, bid, lot):
        value_usd = (bid - op) * 100_000 * lot
        return value_usd

    def sell_value_usd(self, op, ask, lot):
        value_usd = (op - ask) * 100_000 * lot 
        return value_usd

    def floating_usd(self, cur_price):
        value_usd = 0 
        if self.lot_buy_1>0:
            value_usd += self.buy_value_usd(self.op_price_buy_1,cur_price,self.lot_buy_1)
        if self.lot_buy_2>0:
            value_usd += self.buy_value_usd(self.op_price_buy_2,cur_price,self.lot_buy_2)
        if self.lot_buy_3>0:
            value_usd += self.buy_value_usd(self.op_price_buy_3,cur_price,self.lot_buy_3)

        if self.lot_sell_1>0:
            value_usd += self.sell_value_usd(self.op_price_sell_1,cur_price,self.lot_sell_1)
        if self.lot_sell_2>0:
            value_usd += self.sell_value_usd(self.op_price_sell_2,cur_price,self.lot_sell_2)
        if self.lot_sell_3>0:
            value_usd += self.sell_value_usd(self.op_price_sell_3,cur_price,self.lot_sell_3)

        return value_usd

    def step(self, action):
        import random

        assert action in self.action_space  

        print(f'action->{action}')

        # EVAL CYCLE ON M1
        if action>0: #and self.balance>=1000:

            # RECORD CUR STATS FOR REWARD CALC
            pre_val = self.balance 

            # GET NECESSARY INFO FROM H1 TO FIND M1 INDEX
            filt = ( self.data_m1['date']==self.data_H1['date'][self.index_H1+1] ) & ( self.data_m1['hour']==self.data_H1['hour'][self.index_H1+1] )
            index_m1 = self.data_m1.loc[filt].index.tolist()[0] 
            # MUST ANTICIPATE IF NOT FOUND - index_m1

            # PERFORM SIMULATION ON M1 - TRADE
            self.update_bar_m1(index_m1)

            # THIS IS A SINGLE CYCLE
            self.reset_cycle()
          
            if action==1: # SELL FIRST 
                self.set_first_buy(self.price_open_m1)      
            elif action==2: # BUY FIRST 
                self.set_first_sell(self.price_open_m1)

            done_ep = 0
            done_cycle = 0

            if action==1 or action==2:
                # 1 CYCLE - LOOP UNTIL PROFIT OR LOSS    
                while True:

                    flip = random.random()
                    flop = random.random()
                    firsthalf = int(flip * 60)
                    secondhalf = 60 - firsthalf
                    thirdquarter = int(flop * secondhalf)
                    finalquarter = secondhalf - thirdquarter
                    askbid = np.array([])

                    # INJECT/CALC ASKBID SERIES HERE
                    if self.price_close_m1>self.price_open_m1: # BULL
                        swing_low = np.linspace(self.price_open_m1, self.price_low_m1, firsthalf)                        
                        swing_high = np.linspace(self.price_low_m1, self.price_high_m1, thirdquarter)
                        swing_close = np.linspace(self.price_high_m1, self.price_close_m1, finalquarter)
                        askbid = np.append(askbid, swing_low)
                        askbid = np.append(askbid, swing_high)
                        askbid = np.append(askbid, swing_close)
                    else: # BEAR
                        swing_high = np.linspace(self.price_open_m1, self.price_high_m1, firsthalf)
                        swing_low = np.linspace(self.price_high_m1, self.price_low_m1, thirdquarter)
                        swing_close = np.linspace(self.price_low_m1, self.price_close_m1, finalquarter)
                        askbid = np.append(askbid, swing_low)
                        askbid = np.append(askbid, swing_high)
                        askbid = np.append(askbid, swing_close)
                        
                    # EVALUATE WHAT HAPPENED                    
                    for j in range(askbid.shape[0]):

                        askbid_m1 = askbid[j]

                        #self.log.append(f'new tick: {askbid_m1}\n')

                        if self.buy_tp_hit(askbid_m1) or self.sell_tp_hit(askbid_m1) or self.early_exit_hit(askbid_m1): # SB TP
                            done_cycle = 1
                            break # EXIT FOR

                        # EVALUATING HIGH FIRST
                        self.SS1_SS2_SS3_sl_monitor(askbid_m1)
                        # MAX DD EVAL - DEBATABLE TO PUT THIS ON EVERY LV SL HIT
                        if self.max_dd_hit()==True:
                            done_ep = 1
                            break # EXIT FOR
                        self.SB1_SB2_SB3_op_monitor(askbid_m1)                     
                        self.normalize_SS1_SS2(askbid_m1)
                        
                        
                        # AND THEN EVALUATING LOW 
                        self.SB1_SB2_SB3_sl_monitor(askbid_m1)
                        if self.max_dd_hit()==True:
                            done_ep = 1
                            break # EXIT FOR
                        self.SS1_SS2_SS3_op_monitor(askbid_m1)
                        self.normalize_SB1_SB2(askbid_m1)                   
                        
                        #if self.sell_tp_hit(askbid_m1)==True: # SS TP
                        #    done_cycle = 1
                        #    break # EXIT FOR
                    
                    # DONE EP -> DEFINITELY OUT OF CYCLE
                    if done_ep==0 and index_m1==self.m1_bars-1:
                        done_ep = 1
                        break # EXIT WHILE
                    elif done_ep==1:
                        break # EXIT WHILE

                    # DONE CYCLE -> IN PROFIT
                    if done_cycle==1:
                        break

                    # STEP+1 INDEX FOR NEXT WHILE ITERATION
                    index_m1 += 1
                    self.update_bar_m1(index_m1)
                # END OF CYCLE - TRAVERSING ON M1

                # get the new value after taking the action
                post_val = self.balance

                # reward is the increase in porfolio value
                reward = 0
                if post_val>pre_val:
                    reward += 1
                if self.accumulated_loss >= self.ideal_drawdown:
                    # 0.5 IS CONSTANT TO WEIGHT HOW BAD THE DD
                    # 2X IDEAL DD * 0.5 = 1 UNIT SUBTRACT - VERY BAD
                    reward -= self.dd_weight * ( self.accumulated_loss / self.ideal_drawdown )  
                    # THIS IS OPEN FOR PONDERING AND DISCUSSION
                    # QUANTITATIVELY HOW TO MEASURE HOW GOOD A SMALL DRAWDOWN IS
                    # DD < 250 -> REWARDED 0.5 -> 2x TARGET
                    # 250 > DD > 500 REWARDED 0.25
                    # 500 > DD > 750 REWARDED 0
                    # 750 > DD > 1000 REWARDED -0.25
                    # AND SO ON              
                
                # EXIT ON END OF EPS / OUT OF BARS ON M1 OR H1
                if done_ep==0:
                    if self.index_H1==self.H1_bars-1 or index_m1==self.m1_bars-1:
                        done_ep = 1
                        reward = 0
                    else:
                        index_m1 += 1
                        # GET CORRESPONDING H1 INDEX
                        filt = ( self.data_H1['date']==self.data_m1['date'][index_m1] ) & ( self.data_H1['hour']==self.data_m1['hour'][index_m1] )
                        matched_index_H1 = self.data_H1.loc[filt].index.tolist()[0]
                        self.index_H1 = matched_index_H1 + 1
                        self.update_states_H1()
                elif done_ep==1:
                    reward = 0
                    #file1 = open("log.txt","a") 
                    #file1.writelines(self.log) 
                    #file1.close() 
                    #print('log printed out')

                self.ep_reward += reward

                # UPDATE HOW DEEP IS THE DD
                if self.accumulated_loss>self.drawdown:
                    self.drawdown = self.accumulated_loss

                # LATEST CYCLE STATS
                info = {'post_val': post_val, 'dd': self.drawdown, 'ep_reward': self.ep_reward }

                # conform to the Gym API
                return self._get_obs_H1(), reward, done_ep, info

                # END STEP / EPISODE
        elif action==0:

            #print('action==0')

            #self.index_H1 += 1
            #self.update_states_H1()
            self.one_step_bar_H1()

            reward = 0
            # END OF EPISODE
            done = 0
            if self.index_H1 == self.H1_bars - 1:
                done = 1
            #info = {'post_val': self.balance, 'win' : 0, 'lose' : 0, 'total': 0 , 'dd': 0 }
            info = {'post_val': self.balance, 'dd': self.drawdown, 'ep_reward': self.ep_reward }
            return self._get_obs_H1(), reward, done, info

    def one_step_bar_H1(self):
        self.index_H1 += 1
        self.update_states_H1()

    def get_obs(self):
        return self._get_obs_H1()

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

        #print('X:',X, 'X.shape' , X.shape, 'len(X.shape)->', len(X.shape))
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

    return np.argmax(act_values[0])  # returns action

  def act_real(self, state):
    act_values = self.model.predict(state)
   
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

def filter_H1(obs):

    # TIME FILTER
    day_1 = obs[21]
    hour_1 = obs[22]
    if hour_1>=9 and hour_1<=16 and day_1<4:
        open_1 = obs[3]
        close_1 = obs[6]
        ema21_1 = obs[18]
        if open_1<ema21_1 and close_1>ema21_1:
            return True 
        elif open_1>ema21_1 and close_1<ema21_1:
            #print('filtered->ok')
            return True
    #print('filtered->skip')
    return False     

def play_one_episode(agent, env, scaler):
    state_raw = env.reset_eps() # RESET ENV

    #print('play_one_episode') EVALD

    state_scaled = scaler.transform([state_raw]) # SCALE STATE

    done = False 

    # EP INFO INIT
    info = {'post_val': env.balance, 'dd': env.drawdown, 'ep_reward': env.ep_reward }

    # TRAVERSE DECISION BARS H1
    while not done:

        if filter_H1(state_raw):
            # REACH NEXT_STATE, GET REWARD
            action = agent.act(state_scaled) # PREDICT ACTION ON SCALED STATE

            # STEP
            next_state_raw, reward, done, info = env.step(action) # STEP/ACT
            # INFO RECEIVED IS 1 CYCLE ONLY

            if np.all(np.isfinite(next_state_raw)):
                next_state_scaled = scaler.transform([next_state_raw])

                # TRAIN DL MODEL
                agent.train(state_scaled, action, reward, next_state_scaled, done)

                # TRANSITION TO NEXT_STATE
                state_raw = next_state_raw
                state_scaled = next_state_scaled
            else:
                env.one_step_bar_H1()
                next_state_raw = env.get_obs()
                # TRANSITION TO NEXT_STATE - ALREADY SCALED
                state_raw = next_state_raw
                state_scaled = scaler.transform([state_raw])
                
        else: # SKIP TO NEXT BAR
            env.one_step_bar_H1()
            next_state_raw = env.get_obs()
            # TRANSITION TO NEXT_STATE - ALREADY SCALED
            state_raw = next_state_raw
            state_scaled = scaler.transform([state_raw])

    #print(f'1 step ends')
    #sys.exit()
        

    # ONE EPISODE FINISHED
    # RETURN 1 EP - MULTI CYCLES - INFO
    return info

def obs_convert(ohlc, state_dim):
    obs = np.empty(state_dim, dtype='double')

    close_bar1 = float(ohlc['close_1'])
    close_bar2 = float(ohlc['close_2'])
    close_change_pct = ((close_bar1 - close_bar2)/close_bar2) * 100

    ema50 = float(ohlc['D50_1'])
    ema21 = float(ohlc['D21_1'])

    ema50_delta_pct = ((close_bar1-ema50)/close_bar1) * 100
    ema21_delta_pct = ((close_bar1-ema21)/close_bar1) * 100

    obs[0] = close_change_pct
    obs[1] = ema50_delta_pct
    obs[2] = ema21_delta_pct
    obs[3] = float(ohlc['open_1'])
    obs[4] = float(ohlc['high_1'])
    obs[5] = float(ohlc['low_1'])
    obs[6] = close_bar1

    obs[-1] = float(ohlc['balance'])

    return obs
        