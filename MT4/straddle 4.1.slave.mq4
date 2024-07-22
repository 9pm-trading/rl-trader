//+------------------------------------------------------------------+
//|                                                 straddle 3.0.mq4 |
//|                                  Copyright 2020, Stephen Antoni. |
//|                                          https://www.luxeave.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, Stephen Antoni."
#property link      "https://www.luxeave.com"
#property version   "3.0"
#property strict

//#include <MQLMySQL.mqh>

string host = "localhost";
string user = "root";
string pass = "";
string database = "mt4";
int port = 3306;

int DB;
string socket = "0";
int flag = 0;

enum TRADE_MODE {
   MODE_NEUTRAL = 0,
   MODE_SECONDARY = 1
};

enum TRANSITION_TYPE {
   TRANSITION_BUY = 0,
   TRANSITION_SELL = 1
};

enum SIGNAL_MODE {
   SIGNAL_MA = 0,
   SIGNAL_TIME = 1
};

enum FIRST_POS {
   FIRST_BUY = 0,
   FIRST_SELL = 1
};

enum THIS_CYCLE_STATE {
   NOPOS_MID = 0,  
   CYCLE_DONE = 1,
   CYCLE_STARTED = 2
};

struct order {
   bool execute;
   double lot;
   double tp_price;
   double sl_price;
   double sl_pt;
   int magic;
   string comment; 
   int level;  
};

class Cycle {

   public:
      // CONST SETTING
      //static const double step_pt; 
      static const double target_usd;
      static const double ratio_lv1;
      static const double ratio_lv2;
      static const double ratio_lv3;
      
      static const int magic_1;
      static const int magic_2;
      static const int magic_3;
      static const double kb_ratio;
      
      static const bool early_exit;
      static const double early_exit_lv;
      static const double early_exit_usd;
   
      // STATIC CLASS VARIABLES - PERSIST THROUGHOUT INSTANCES
      static int count;
   
      // STATES
      
      double step_pt; // SET BY AI
      
      THIS_CYCLE_STATE state;      
      double mid_price;
      double loss;
      //bool trail_status;
      bool early_exit_status;
   
      // OP PRICES
      double op_price_sb1;
      double op_price_sb2;
      double op_price_sb3;
      double op_price_ss1;
      double op_price_ss2;
      double op_price_ss3;
      // REAL OP PRICES 
      double real_price_sb1;
      double real_price_sb2;
      double real_price_sb3;
      double real_price_ss1;
      double real_price_ss2;
      double real_price_ss3;     
      // SL PRICES
      double sl_price_sb1;
      double sl_price_sb2;
      double sl_price_sb3;
      double sl_price_ss1;
      double sl_price_ss2;
      double sl_price_ss3;
      
      // BEP
      bool bep_sb1;
      bool bep_sb2;
      bool bep_ss1;
      bool bep_ss2;
      
      // TP PRICES
      double tp_price_buy;
      double tp_price_sell;
      // LOTS
      double lot_sb1;
      double lot_sb2;
      double lot_sb3;
      double lot_ss1;
      double lot_ss2;
      double lot_ss3;
      // TICKETS
      int ticket_sb1;
      int ticket_sb2;
      int ticket_sb3;
      int ticket_ss1;
      int ticket_ss2;
      int ticket_ss3;
      
      double profit_1;
      double profit_2;
      double profit_3;
      
   // CONSTRUCTOR
   Cycle(FIRST_POS first, double op_price, double prev_loss, double step) {  
      
      step_pt = step; // FIXED HERE
      
      if ( first==FIRST_BUY ) {                 
         mid_price = op_price - (step_pt*Point());
      } else if ( first==FIRST_SELL ) {
         mid_price = op_price + (step_pt*Point());
      }    

      state = CYCLE_STARTED;
      early_exit_status = false;
      
      loss = prev_loss;
      op_price_sb1 = mid_price + (step_pt*Point());
      op_price_sb2 = op_price_sb1 + (step_pt*Point());
      op_price_sb3 = op_price_sb2 + (step_pt*Point());
      op_price_ss1 = mid_price - (step_pt*Point());
      op_price_ss2 = op_price_ss1 - (step_pt*Point());
      op_price_ss3 = op_price_ss2 - (step_pt*Point());
      
      real_price_sb1 = 0;
      real_price_sb2 = 0;
      real_price_sb3 = 0;
      real_price_ss1 = 0;
      real_price_ss2 = 0;
      real_price_ss3 = 0;
      
      sl_price_sb1 = op_price_sb1 - (step_pt*Point());
      sl_price_sb2 = op_price_sb2 - (step_pt*Point());
      sl_price_sb3 = op_price_sb3 - (step_pt*Point());
      sl_price_ss1 = op_price_ss1 + (step_pt*Point());
      sl_price_ss2 = op_price_ss2 + (step_pt*Point());
      sl_price_ss3 = op_price_ss3 + (step_pt*Point());
      
      tp_price_buy = mid_price + (4*step_pt*Point());
      tp_price_sell = mid_price - (4*step_pt*Point());
      
      lot_sb1 = 0;
      lot_sb2 = 0;
      lot_sb3 = 0;
      lot_ss1 = 0;
      lot_ss2 = 0;
      lot_ss3 = 0; 
      
      ticket_sb1 = 0;
      ticket_sb2 = 0;
      ticket_sb3 = 0;
      ticket_ss1 = 0;
      ticket_ss2 = 0;
      ticket_ss3 = 0; 
      
      profit_1 = 0;
      profit_2 = 0;
      profit_3 = 0;
      
      bep_sb1 = false;
      bep_sb2 = false;
      bep_ss1 = false;
      bep_ss2 = false;
      
      count += 1;   
   }    
};

enum SIGNAL_STATE {
   SIGNAL_NEUTRAL = 0,
   STRETCHED_BULL = 1,
   STRETCHED_BEAR = 2
};

// INPUTS
string table = "eurusd_d1";

input double target = 10.0;
input double ratio1 = 0.5;
input double ratio2 = 0.3;
input double ratio3 = 0.2;
input double kickBackRatio = 1.2;
input int magic1 = 1111;
input int magic2 = 2222;
input int magic3 = 3333;
input int slip = 50;

input bool early_exit_enabled = false;
input double early_exit_dd = 200.0;
input double early_exit_dollar = 25.0;

input double resume_op_price_buy = 0.0;
input double resume_op_price_sell = 0.0;
input double resume_loss = 0.0;
input double resume_step = 0.0;

input string url_bar_up = "http://straddle.online/updatebar.php?";
input string url_action_up = "http://straddle.online/updateaction.php?";
input string url_action_get = "http://straddle.online/getaction.php";

// INITIALIZE CONSTS
const double Cycle::target_usd = target;
const double Cycle::ratio_lv1 = ratio1;
const double Cycle::ratio_lv2 = ratio2;
const double Cycle::ratio_lv3 = ratio3;

const double Cycle::kb_ratio = kickBackRatio;

const int Cycle::magic_1 = magic1;
const int Cycle::magic_2 = magic2;
const int Cycle::magic_3 = magic3;

const bool Cycle::early_exit = early_exit_enabled;
const double Cycle::early_exit_lv = early_exit_dd;
const double Cycle::early_exit_usd = early_exit_dollar;

// INITIALIZE STATIC
int Cycle::count = 0;

// ARRAY OF CYCLES
Cycle *cycle_array[];

order set_order_buy;
order set_order_sell;

order set_bep_buy;
order set_bep_sell;

bool close_all;

bool terminate;

bool expiry_terminate;

int beginTimeHour;
int beginTimeMinute;
int endTimeHour;
int endTimeMinute;

//int daily_attempt;
TRADE_MODE trade_mode;
TRANSITION_TYPE transition_signal;
double transition_loss;
int buy_ticket_prime;
int sell_ticket_prime;
bool buy_close_prime;
bool sell_close_prime;

double step_array[15];

int timer;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{  
   if ( resume_loss>0 ) {
      trade_mode = MODE_SECONDARY;
   } else if ( resume_loss==0 ) {
      trade_mode = MODE_NEUTRAL;
   } 
   
   transition_loss = 0;
   
   buy_close_prime = false;
   sell_close_prime = false;
   
   step_array[1] = 150.0; 
   step_array[2] = 175.0; 
   step_array[3] = 200.0; 
   step_array[4] = 225.0; 
   step_array[5] = 250.0; 
   
   step_array[6] = 275.0; 
   step_array[7] = 300.0; 
   step_array[8] = 150.0; 
   step_array[9] = 175.0; 
   step_array[10] = 200.0; 
   
   step_array[11] = 225.0; 
   step_array[12] = 250.0; 
   step_array[13] = 275.0; 
   step_array[14] = 300.0; 
   
   //--------------------------- RESUME MODULE ----------------------//
   
   if ( resume_op_price_buy>0 ) {
      BuyResume(resume_loss, resume_op_price_buy, resume_step);
   } else if ( resume_op_price_sell>0 ) {
      SellResume(resume_loss, resume_op_price_sell, resume_step);
   }
   
   if ( resume_op_price_buy>0 || resume_op_price_sell>0 ) {
   
      trade_mode = MODE_SECONDARY;
   
      // TRACKS STATE OF SB1, SB2, SB3
      int bm1_count = OrderCount(magic1,OP_BUY,Symbol());
      int bm2_count = OrderCount(magic2,OP_BUY,Symbol());
      int bm3_count = OrderCount(magic3,OP_BUY,Symbol());
      
      if ( bm1_count>0 ) {
         int ticket = GetTicket(OP_BUY,magic1);
                  
         double op = GetOpenPrice(ticket);
         double sl = GetSLPriceByTicket(ticket);
         
         cycle_array[Cycle::count-1].ticket_sb1 = ticket;
         cycle_array[Cycle::count-1].real_price_sb1 = op;
         cycle_array[Cycle::count-1].lot_sb1 = GetLotsize(ticket);
         cycle_array[Cycle::count-1].sl_price_sb1 = sl;
         
         if ( op==sl )
            cycle_array[Cycle::count-1].bep_sb1 = true;
      }
      
      if ( bm2_count>0 ) {
         int ticket = GetTicket(OP_BUY,magic2);         
         
         double op = GetOpenPrice(ticket);
         double sl = GetSLPriceByTicket(ticket);
         
         cycle_array[Cycle::count-1].ticket_sb2 = ticket;        
         cycle_array[Cycle::count-1].real_price_sb2 = op;
         cycle_array[Cycle::count-1].lot_sb2 = GetLotsize(ticket);
         cycle_array[Cycle::count-1].sl_price_sb2 = sl;
         
         if ( op==sl )
            cycle_array[Cycle::count-1].bep_sb2 = true;
      }
      
      if ( bm3_count>0 ) {
         int ticket = GetTicket(OP_BUY,magic3);
         cycle_array[Cycle::count-1].ticket_sb3 = ticket;
         cycle_array[Cycle::count-1].real_price_sb3 = GetOpenPrice(ticket);
         cycle_array[Cycle::count-1].lot_sb3 = GetLotsize(ticket);
         cycle_array[Cycle::count-1].sl_price_sb3 = GetSLPriceByTicket(ticket);
      }
      
      // TRACKS STATE OF SS1, SS2, SS3
      int sm1_count = OrderCount(magic1,OP_SELL,Symbol());
      int sm2_count = OrderCount(magic2,OP_SELL,Symbol());
      int sm3_count = OrderCount(magic3,OP_SELL,Symbol());
      
      if ( sm1_count>0 ) {
         int ticket = GetTicket(OP_SELL,magic1);        
         
         double op = GetOpenPrice(ticket);
         double sl = GetSLPriceByTicket(ticket);
         
         cycle_array[Cycle::count-1].ticket_ss1 = ticket;
         cycle_array[Cycle::count-1].real_price_ss1 = op;
         cycle_array[Cycle::count-1].lot_ss1 = GetLotsize(ticket);
         cycle_array[Cycle::count-1].sl_price_ss1 = sl;
         
         if ( op==sl )
            cycle_array[Cycle::count-1].bep_ss1 = true;
      }
      
      if ( sm2_count>0 ) {
         int ticket = GetTicket(OP_SELL,magic2);
         
         double op = GetOpenPrice(ticket);
         double sl = GetSLPriceByTicket(ticket);
         
         cycle_array[Cycle::count-1].ticket_ss2 = ticket;
         cycle_array[Cycle::count-1].real_price_ss2 = op;
         cycle_array[Cycle::count-1].lot_ss2 = GetLotsize(ticket);
         cycle_array[Cycle::count-1].sl_price_ss2 = sl;
         
         if ( op==sl )
            cycle_array[Cycle::count-1].bep_ss2 = true;
      }
      
      if ( sm3_count>0 ) {
         int ticket = GetTicket(OP_SELL,magic3);
         cycle_array[Cycle::count-1].ticket_ss3 = ticket;
         cycle_array[Cycle::count-1].real_price_ss3 = GetOpenPrice(ticket);
         cycle_array[Cycle::count-1].lot_ss3 = GetLotsize(ticket);
         cycle_array[Cycle::count-1].sl_price_ss3 = GetSLPriceByTicket(ticket);
      }
   
   }
   
   if ( trade_mode==MODE_NEUTRAL ) { 
      UpdateAction(url_action_up, 0); // MUST SET ACTION VECTOR TO 0 AS WELL
   }
   
   return(INIT_SUCCEEDED);
}
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
//---
   DestroyCycle(cycle_array);
}
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{  
   if ( close_all ) {
      if ( CloseAll(magic1,magic2,magic3,slip) ) {
         close_all = false;
         cycle_array[Cycle::count-1].state = CYCLE_DONE;
         trade_mode = MODE_NEUTRAL;         
         //UpdateAction(url_action_up, 0); // MUST SET ACTION VECTOR TO 0 AS WELL
      }
      return;
   }

   if ( Cycle::count>0 ) {
      int i = Cycle::count-1;
      Comment(
         "SBTP - " + DoubleToStr(cycle_array[i].tp_price_buy,Digits()) , "\n" ,
         "SB3 - " + DoubleToStr(cycle_array[i].op_price_sb3,Digits()) + " - " + DoubleToStr(cycle_array[i].lot_sb3,2) + " - TK: " + IntegerToString(cycle_array[i].ticket_sb3)  , "\n",       
         "SB2 - " + DoubleToStr(cycle_array[i].op_price_sb2,Digits()) + " - " + DoubleToStr(cycle_array[i].lot_sb2,2) + " - TK: " + IntegerToString(cycle_array[i].ticket_sb2)  , "\n",
         "SB1 - " + DoubleToStr(cycle_array[i].op_price_sb1,Digits()) + " - " + DoubleToStr(cycle_array[i].lot_sb1,2) + " - TK: " + IntegerToString(cycle_array[i].ticket_sb1)  , "\n",
         "MID - " + DoubleToStr(cycle_array[i].mid_price,Digits()) , "\n",
         "SS1 - " + DoubleToStr(cycle_array[i].op_price_ss1,Digits()) + " - " + DoubleToStr(cycle_array[i].lot_ss1,2) + " - TK: " + IntegerToString(cycle_array[i].ticket_ss1)  , "\n",
         "SS2 - " + DoubleToStr(cycle_array[i].op_price_ss2,Digits()) + " - " + DoubleToStr(cycle_array[i].lot_ss2,2) + " - TK: " + IntegerToString(cycle_array[i].ticket_ss2)  , "\n",
         "SS3 - " + DoubleToStr(cycle_array[i].op_price_ss3,Digits()) + " - " + DoubleToStr(cycle_array[i].lot_ss3,2) + " - TK: " + IntegerToString(cycle_array[i].ticket_ss3)  , "\n",
         "SSTP - " + DoubleToStr(cycle_array[i].tp_price_sell,Digits()) , "\n",
         
         "loss: " , DoubleToStr(cycle_array[i].loss,2) , "\n",
         "status: " , cycle_array[i].state , "\n"
      );
   }

   if ( NewBar() ) {  
      //UpdateBar(url_bar_up, table);

      /*if ( trade_mode==MODE_NEUTRAL ) { 
         UpdateAction(url_action_up, 0); // MUST SET ACTION VECTOR TO 0 AS WELL
      }*/
          
      timer = 300; // 300 SECONDS DELAYED BEFORE ACTION PREDICTION CHECKING
   } // END NEW BAR
   
   // TRIGGER
   if ( TimedExecution(timer) && trade_mode==MODE_NEUTRAL ) {
   
      int action = GetAction(url_action_get);
      
      Print("action->"+IntegerToString(action));
      if ( action>0 && action<8 ) {
         double step = step_array[action];// GET STEP
         trade_mode = MODE_SECONDARY;
         BuySequence(0, step);
      } else if ( action>=8 ) {
         double step = step_array[action];// GET STEP
         trade_mode = MODE_SECONDARY;
         SellSequence(0, step);
      }     
   } 
   // END TRIGGER 
   
   // BUY OPEN
   if ( set_order_buy.execute==true ) {
      int ticket;
      if ( OpenBuy(set_order_buy.lot, slip, set_order_buy.sl_pt, set_order_buy.tp_price, set_order_buy.comment, set_order_buy.magic, ticket)==true ) {
         set_order_buy.execute = false; 
         update_open_buy(cycle_array[Cycle::count-1],ticket,set_order_buy.level);  
         if ( set_order_buy.level==1 )
            cycle_array[Cycle::count-1].bep_sb1 = false;   
         else if ( set_order_buy.level==2 )       
            cycle_array[Cycle::count-1].bep_sb2 = false;    
      }
   
      return;
   }
   
   // SELL OPEN
   if ( set_order_sell.execute==true ) {
      int ticket;
      if ( OpenSell(set_order_sell.lot, slip, set_order_sell.sl_pt , set_order_sell.tp_price, set_order_sell.comment, set_order_sell.magic, ticket)==true ) {
         set_order_sell.execute = false;           
         update_open_sell(cycle_array[Cycle::count-1],ticket,set_order_sell.level); 
         if ( set_order_sell.level==1 )
            cycle_array[Cycle::count-1].bep_ss1 = false;   
         else if ( set_order_sell.level==2 )       
            cycle_array[Cycle::count-1].bep_ss2 = false;                          
      }
   
      return;
   }    

   // PER TICK
   if ( Cycle::count>0 && cycle_array[Cycle::count-1].state!=CYCLE_DONE ) {
      
      // EARLY EXIT
      if ( EarlyExit(cycle_array[Cycle::count-1])==true ) {
         close_all = true;
         return;
      }
      
      // SB1 BEP
      if ( cycle_array[Cycle::count-1].lot_sb1>0 && cycle_array[Cycle::count-1].bep_sb1==false ) {
         if ( Bid>=cycle_array[Cycle::count-1].real_price_sb1 + (cycle_array[Cycle::count-1].step_pt*Point()) ) {
            cycle_array[Cycle::count-1].bep_sb1 = true;
            ModifBEP(OP_BUY, 1);
         }
      }
      
      // SB2 BEP
      if ( cycle_array[Cycle::count-1].lot_sb2>0 && cycle_array[Cycle::count-1].bep_sb2==false ) {
         if ( Bid>=cycle_array[Cycle::count-1].real_price_sb2 + (cycle_array[Cycle::count-1].step_pt*Point()) ) {
            cycle_array[Cycle::count-1].bep_sb2 = true;
            ModifBEP(OP_BUY, 2);
         }
      }
      
      // SS1 BEP
      if ( cycle_array[Cycle::count-1].lot_ss1>0 && cycle_array[Cycle::count-1].bep_ss1==false ) {
         if ( Ask<=cycle_array[Cycle::count-1].real_price_ss1 - (cycle_array[Cycle::count-1].step_pt*Point()) ) {
            ModifBEP(OP_SELL, 1);
            cycle_array[Cycle::count-1].bep_ss1 = true;
         }
      }
      
      // SS2 BEP
      if ( cycle_array[Cycle::count-1].lot_ss2>0 && cycle_array[Cycle::count-1].bep_ss2==false ) {
         if ( Ask<=cycle_array[Cycle::count-1].real_price_ss2 - (cycle_array[Cycle::count-1].step_pt*Point()) ) {
            ModifBEP(OP_SELL, 2);
            cycle_array[Cycle::count-1].bep_ss2 = true;
         }
      }
   
      // BUY SL 
      SB1_SB2_SB3_sl_monitor(cycle_array[Cycle::count-1]);
      // BUY OP
      SB1_SB2_SB3_op_monitor(cycle_array[Cycle::count-1]);
      // NORMALIZE SB1 & SB2
      Normalize_SB1_SB2(cycle_array[Cycle::count-1]);
      // BUY TP
      if ( buy_tp_hit(cycle_array[Cycle::count-1])==true ) {
         cycle_array[Cycle::count-1].state = CYCLE_DONE;
         trade_mode = MODE_NEUTRAL;         
         //update_db_bridge(DB, "action_vector", 0); // MUST SET ACTION VECTOR TO 0 AS WELL
         //UpdateAction(url_action_up, 0);
         return;
      }
      
      // SELL SL
      SS1_SS2_SS3_sl_monitor(cycle_array[Cycle::count-1]);
      // SELL OP 
      SS1_SS2_SS3_op_monitor(cycle_array[Cycle::count-1]);
      // NORMALIZE SS1 & SS2
      Normalize_SS1_SS2(cycle_array[Cycle::count-1]);
      // SELL TP
      if ( sell_tp_hit(cycle_array[Cycle::count-1])==true ) {
         cycle_array[Cycle::count-1].state = CYCLE_DONE;
         trade_mode = MODE_NEUTRAL;         
         //update_db_bridge(DB, "action_vector", 0); // MUST SET ACTION VECTOR TO 0 AS WELL
         //UpdateAction(url_action_up, 0);
         return;
      }
   }

}
//+------------------------------------------------------------------+

double GetSLPrice(int type, int mn) {
   for ( int i=0; i<OrdersTotal(); i++ ) {
      if ( OrderSelect(i,SELECT_BY_POS)==true ) {
         if ( OrderSymbol()==Symbol() && OrderType()==type && OrderMagicNumber()==mn )
            return OrderStopLoss();
      }
   }
   
   return 0;
}

double GetSLPriceByTicket(int tk) {
   if ( OrderSelect(tk,SELECT_BY_TICKET)==true ) {
      return OrderStopLoss();
   }     
   return 0;
}
 
int GetTicket(int type, int mn) {
   for ( int i=0; i<OrdersTotal(); i++ ) {
      if ( OrderSelect(i,SELECT_BY_POS)==true ) {
         if ( OrderSymbol()==Symbol() && OrderType()==type && OrderMagicNumber()==mn )
            return OrderTicket();
      }
   }
   
   return 0;
}

bool TimedExecution(int& countdown){
   if ( countdown>0 ) {
      for ( int i=countdown; i>0; i-- ) {
         Sleep(1000);
      }
      countdown = 0;
      
      return true;  
   }
   
   return false;
}

void UpdateAction(string url, int action){
   string update_str = url + "action=" + IntegerToString(action);
   
   string headers;
   char post[], result[];
   int res;
   
   ResetLastError();
   
   res = WebRequest("GET", update_str , NULL, NULL, 5000, post, 0, result, headers);
   
   if ( res==-1 ) {
      Alert("ERROR: ",GetLastError());
   } else {
      string str_res = CharArrayToString(result);
      Alert(str_res);
   }
}

int GetAction(string url){
   string headers;
   char post[], result[];
   int res;
   
   ResetLastError();
   
   res = WebRequest("GET", url , NULL, NULL, 5000, post, 0, result, headers);
   
   if ( res==-1 ) {
      Alert("ERROR: ",GetLastError());
      
   } else {
     
      string str_res = CharArrayToString(result); // TESTED OK
            
      int action = StrToInteger( ParseValue(str_res, "[{\"action_vector\":\"") );
      return action;
   }

   return -1;
}

string ParseValue(string source, string key){
   
   int index_start = StringFind(source,key,0);
   
   if ( index_start>=0 ) {
      index_start += 19;
      int index_end = StringFind(source,"\"",index_start);
      int len = index_end - index_start;
      if ( len>0 ) {
         return StringSubstr(source, index_start, len);
      } 
   }
   
   return "-1";
}

//void UpdateDB(int db,string tb) {
void UpdateBar(string url, string tbl){
   // GET PRICE DATA HERE
   
   datetime time_1 = iTime(Symbol(),PERIOD_H1,0);
   
   string timestamp = IntegerToString( time_1 );
   string open_1 = DoubleToStr( Open[1] , Digits());
   string high_1 =  DoubleToStr( High[1] , Digits()) ;
   string low_1 =  DoubleToStr( Low[1] , Digits()) ;
   string close_1 =  DoubleToStr( Close[1] , Digits()) ;
   
   string open_2 =  DoubleToStr( Open[2] , Digits()) ;
   string high_2 =  DoubleToStr( High[2] , Digits()) ;
   string low_2 =  DoubleToStr( Low[2] , Digits()) ;
   string close_2 =  DoubleToStr( Close[2] , Digits()) ;
   
   string open_3 =  DoubleToStr( Open[3] , Digits()) ;
   string high_3 =  DoubleToStr( High[3] , Digits()) ;
   string low_3 =  DoubleToStr( Low[3] , Digits()) ;
   string close_3 =  DoubleToStr( Close[3] , Digits()) ;
   
   double ma50_1 = iMA(Symbol(),PERIOD_H1,50,0,MODE_SMA,PRICE_CLOSE,1);
   double ma50_2 = iMA(Symbol(),PERIOD_H1,50,0,MODE_SMA,PRICE_CLOSE,2);
   double ma50_3 = iMA(Symbol(),PERIOD_H1,50,0,MODE_SMA,PRICE_CLOSE,3);
   
   double ma21_1 = iMA(Symbol(),PERIOD_H1,21,0,MODE_SMA,PRICE_CLOSE,1);
   double ma21_2 = iMA(Symbol(),PERIOD_H1,21,0,MODE_SMA,PRICE_CLOSE,2);
   double ma21_3 = iMA(Symbol(),PERIOD_H1,21,0,MODE_SMA,PRICE_CLOSE,3);
   
   double delta_ma50_1 = Close[1] - ma50_1;
   double delta_ma50_2 = Close[2] - ma50_2;
   double delta_ma50_3 = Close[3] - ma50_3;
   
   double delta_ma21_1 = Close[1] - ma21_1;
   double delta_ma21_2 = Close[2] - ma21_2;
   double delta_ma21_3 = Close[3] - ma21_3;
   
   string D50_1 =  DoubleToStr( delta_ma50_1, Digits() ) ;
   string D50_2 =  DoubleToStr( delta_ma50_2, Digits() ) ;
   string D50_3 =  DoubleToStr( delta_ma50_3, Digits() ) ;
   
   string D21_1 =  DoubleToStr( delta_ma21_1, Digits() ) ;
   string D21_2 =  DoubleToStr( delta_ma21_2, Digits() ) ;
   string D21_3 =  DoubleToStr( delta_ma21_3, Digits() ) ;
   
   string day_str = IntegerToString( TimeDayOfWeek(time_1) );
   string hour_str = IntegerToString( TimeHour(time_1) );
   
   string bal =  DoubleToStr( AccountBalance(), 2 ) ;

   string update_str = url + "table=" + tbl + "&symbol=" + Symbol() + "&timestamp=" + timestamp + 
                       "&open1=" + open_1 + "&high1=" + high_1 + "&low1=" + low_1 + "&close1=" + close_1 + 
                       "&open2=" + open_2 + "&high2=" + high_2 + "&low2=" + low_2 + "&close2=" + close_2 +
                       "&open3=" + open_3 + "&high3=" + high_3 + "&low3=" + low_3 + "&close3=" + close_3 + 
                       "&D501=" + D50_1 + "&D502=" + D50_2 + "&D503=" + D50_3 + "&D211=" + D21_1 + "&D212=" + D21_2 + "&D213=" + D21_3 +
                       "&day=" + day_str + "&hour=" + hour_str + "&bal=" + bal; 
                       
   string headers;
   char post[], result[];
   int res;
   
   ResetLastError();
   
   res = WebRequest("GET", update_str , NULL, NULL, 5000, post, 0, result, headers);
   
   if ( res==-1 ) {
      Alert("ERROR: ",GetLastError());
      //MessageBox("Apa nih", "Box Title Kali Ya", MB_ICONINFORMATION);
   } else {
      //PrintFormat("Apa juga nih", ArraySize(result));
      //Alert("Bar Update Success");
      string str_res = CharArrayToString(result);
      Alert(str_res);
      //Alert(update_str);
   }
}

/*
void UpdateStatus(int db, int status) {
   
   string tb = "bridge";
   
   string query_select = "UPDATE `" + tb + "` SET cycle_status=" + IntegerToString(status)                                                                               
                                    + " WHERE id=0";

   if (MySqlExecute(db, query_select)) {    
      Print ("Succeeded: ", query_select);    
   } else {    
      Print ("Error: ", MySqlErrorDescription);
      Print ("Query: ", query_select);
   }     
}

void update_db_bridge(int db, string field, int value_int) {
   
   string tb = "bridge";
   
   string query_select = "UPDATE `" + tb + "` SET " + field + " =" + IntegerToString(value_int)                                                                               
                                    + " WHERE id=0";

   if (MySqlExecute(db, query_select)) {    
      Print ("Succeeded: ", query_select);    
   } else {    
      Print ("Error: ", MySqlErrorDescription);
      Print ("Query: ", query_select);
   }     
}
*/

/*int CountRow(string tbl) {
   //////////////////////////////////////////////////////////////////////////////////////////////
   ////////////////////////////////////////// CHECK TABLE EXISTENCE /////////////////////////////  
   if (DB == -1) {
      Print ("Connection failed! Error: "+MySqlErrorDescription); }
   else if ( DB>=0 ) {
   
      string query_select;
      
      query_select = "SELECT COUNT(*) FROM `" + tbl + "`";
      
      int Cursor,Rows;
      Cursor = MySqlCursorOpen(DB, query_select);
    
      if (Cursor >= 0) {
         Rows = MySqlCursorRows(Cursor);                  
            int count = 0;
         
            if (MySqlCursorFetchRow(Cursor)) {
               count = MySqlGetFieldAsInt(Cursor, 0); // id
            }
         
            MySqlCursorClose(Cursor); // NEVER FORGET TO CLOSE CURSOR !!!
            return count;
      } else {   
         
         Print ("Cursor opening failed. Error: ", MySqlErrorDescription);
      }
      
      MySqlCursorClose(Cursor); // NEVER FORGET TO CLOSE CURSOR !!!
      
      ////////////////////////////////////////// CHECK TABLE EXISTENCE /////////////////////////////
      //////////////////////////////////////////////////////////////////////////////////////////////
   }
   
   return -1;
}*/


//+------------------------- END MYSQL MODULE -----------------------+

bool CloseBuy(int tk, int slippage) {
   if ( OrderSelect(tk, SELECT_BY_TICKET) ) {
      if ( OrderClose(tk, OrderLots(), Bid, slippage, clrAliceBlue)==true ) {
         return true;
      } else {
         Alert("PRIME BUY CLOSE ERR: " + IntegerToString(GetLastError()) );
         
         if ( IsExit(tk)==true )
            return true;
      }
   }
   
   return false;
}

bool CloseSell(int tk, int slippage) {
   if ( OrderSelect(tk, SELECT_BY_TICKET) ) {
      if ( OrderClose(tk, OrderLots(), Ask, slippage, clrAliceBlue)==true ) {
         return true;
      } else {
         Alert("PRIME SELL CLOSE ERR: " + IntegerToString(GetLastError()) );
         
         if ( IsExit(tk)==true )
            return true;
      }
   }
   
   return false;
}

void BuySequence(double trans_loss, double step) {
   CreateCycle(cycle_array, FIRST_BUY, Ask, trans_loss, step);
   
   OrderFirstBuy(true, step);
}

void OrderFirstBuy(bool is_entry, double step){
   set_order_buy.execute = is_entry;
   set_order_buy.lot = calculate_lot_buy(1,cycle_array[Cycle::count-1]);
   set_order_buy.tp_price = cycle_array[Cycle::count-1].tp_price_buy; 
   set_order_buy.sl_pt = step;  
   set_order_buy.magic = magic1;
   set_order_buy.comment = "SB1";
   set_order_buy.level = 1;
   Print("set_order_buy.tp_price:" + DoubleToStr(set_order_buy.tp_price,Digits()) );
   Print("set_order_buy.lot:" + DoubleToStr(set_order_buy.lot,2));
}

void BuyResume(double trans_loss, double op_price, double step) {
   CreateCycle(cycle_array, FIRST_BUY, op_price, trans_loss, step);

   OrderFirstBuy(false, step);
} 

void SellSequence(double trans_loss, double step) {
   CreateCycle(cycle_array, FIRST_SELL, Bid, trans_loss, step);
   
   OrderFirstSell(true, step);
}

void SellResume(double trans_loss, double op_price, double step) {
   CreateCycle(cycle_array, FIRST_SELL, op_price, trans_loss, step);
  
   OrderFirstSell(false, step);
}

void OrderFirstSell(bool is_entry, double step){
   set_order_sell.execute = is_entry;
   set_order_sell.lot = calculate_lot_sell(1,cycle_array[Cycle::count-1]);
   set_order_sell.tp_price = cycle_array[Cycle::count-1].tp_price_sell;
   
   set_order_sell.sl_pt = step;
   
   set_order_sell.magic = magic1;
   set_order_sell.comment = "SS1";
   set_order_sell.level = 1;

   Print("set_order_sell.tp_price:" + DoubleToStr(set_order_sell.tp_price,Digits()) );
   Print("set_order_sell.lot:" + DoubleToStr(set_order_sell.lot,2));
}

bool NewDay()
{
   static datetime time_stamp_d1 = 0;
  
   if( time_stamp_d1 != iTime(Symbol(),PERIOD_D1,0) ) 
   {
      time_stamp_d1 = iTime(Symbol(),PERIOD_D1,0);
      return (true);
   }
      
   return (false);
}

bool IsTimeOk() {
   datetime curTime = TimeCurrent();
   int curHour = TimeHour(curTime);
   int curMinute = TimeMinute(curTime);   

   if ( curHour==beginTimeHour || ( curHour>=beginTimeHour && curHour<=endTimeHour ) ) {
      return true;
   }
   
   return false;
}

int TimeToHour(string timeFormat)
{
   // TIME STRING -> INTEGER CONVERSION
   int TimeHourInt = 0;
   // HOUR
   string TimeHourStr = StringSubstr(timeFormat,0,2);
   string TimeHourFirstStr = StringSubstr(timeFormat,0,1);
   string TimeHourSecondStr = StringSubstr(timeFormat,1,1);
  
   if ( TimeHourStr=="00" )
   {  // 0
      TimeHourInt = 0;  
   } else if ( TimeHourFirstStr=="0" )
   {
      // 1-9
      TimeHourInt = StrToInteger(TimeHourSecondStr);
   } else { // 10-24
      TimeHourInt = StrToInteger(TimeHourStr);
   }
   
   return (TimeHourInt);
 
}

int TimeToMinute(string timeFormat)
{
   int TimeMinuteInt = 0;

   // MINUTE
   string TimeMinuteStr = StringSubstr(timeFormat,3,2);
   string TimeMinuteFirstStr = StringSubstr(timeFormat,3,1);
   string TimeMinuteSecondStr = StringSubstr(timeFormat,4,1);   
    
   if ( TimeMinuteStr=="00" )
   {  // 0
      TimeMinuteInt = 0;  
   } else if ( TimeMinuteFirstStr=="0" )
   {
      // 1-9
      TimeMinuteInt = StrToInteger(TimeMinuteSecondStr);
   } else { // 10-60
      TimeMinuteInt = StrToInteger(TimeMinuteStr);
   }
   
   return (TimeMinuteInt);
}

bool NewBar()
{
   static datetime time_stamp = 0;
  
   if( time_stamp != iTime(Symbol(),0,0) ) 
   {
      time_stamp = iTime(Symbol(),0,0);
      return (true);
   }
      
   return (false);
}

bool GapExit(Cycle* &siklus) {
   if ( siklus.tp_price_buy>0 ) {
      if ( Bid>siklus.tp_price_buy ) {
         return true;
      }
   } 
   
   if ( siklus.tp_price_sell>0 ) {
      if ( Ask<siklus.tp_price_sell ) {
         return true;
      }
   }
   
   return false;
}

bool EarlyExit(Cycle* &siklus) {
   if ( siklus.early_exit ) {
      if ( siklus.early_exit_status==false ) {
         if ( siklus.loss>=Cycle::early_exit_lv ) {
            siklus.early_exit_status = true;
         }
      } else if ( siklus.early_exit_status==true ) {
         if ( GetProfitTotal(siklus.magic_1, siklus.magic_2, siklus.magic_3)>=siklus.early_exit_usd+siklus.loss )
            return true;
      }
   }
   
   return false;
}

double GetProfitTotal(int mn1, int mn2, int mn3) {
   double profit = 0;
   
   for ( int i=0; i<OrdersTotal(); i++ ) {
      if ( OrderSelect(i,SELECT_BY_POS) ) {
         if ( OrderSymbol()==Symbol() && ( OrderMagicNumber()==mn1 || OrderMagicNumber()==mn2 || OrderMagicNumber()==mn3 ) ) {
            profit += OrderProfit();
         }
      }
   }
   
   return profit;
}

bool CloseAll(int mn1, int mn2, int mn3, int slippage) {
   for ( int i=0; i<OrdersTotal(); i++ ) {
      if ( OrderSelect(i,SELECT_BY_POS) ) {
         if ( OrderSymbol()==Symbol() && ( OrderMagicNumber()==mn1 || OrderMagicNumber()==mn2 || OrderMagicNumber()==mn3 ) ) {
            if ( OrderType()==OP_BUY ) {
               if ( OrderClose(OrderTicket(),OrderLots(),Bid,slippage,clrAliceBlue)==false )
                  Alert( "#" + IntegerToString(OrderMagicNumber()) + " BUY CLOSE ERR: " + IntegerToString(GetLastError()) );
            } else if ( OrderType()==OP_SELL ) {
               if ( OrderClose(OrderTicket(),OrderLots(),Ask,slippage,clrAliceBlue)==false )
                  Alert( "#" + IntegerToString(OrderMagicNumber()) + " SELL CLOSE ERR: " + IntegerToString(GetLastError()) );
            }
         }
      }
   }
   
   if ( OrderCount(mn1, OP_BUY, Symbol())==0 && OrderCount(mn2, OP_BUY, Symbol())==0 && OrderCount(mn3, OP_BUY, Symbol())==0 && 
        OrderCount(mn1, OP_SELL, Symbol())==0 && OrderCount(mn2, OP_SELL, Symbol())==0 && OrderCount(mn3, OP_SELL, Symbol())==0 ) 
        return true;
   
   return false;
}

int OrderCount(int mn, int type, string symbol)
{
   int count = 0;
   for ( int i=0; i<OrdersTotal(); i++ )
   {
      if ( OrderSelect(i,SELECT_BY_POS) )
      {
         if ( OrderMagicNumber()==mn && OrderType()==type && OrderSymbol()==symbol )
         {
            count++;
         }
      }
   }
   
   return (count);
}

bool buy_tp_hit(Cycle* &siklus) {
   double profit = 0;
   // SB3
   if ( siklus.profit_3==0 && siklus.lot_sb3>0 ) {
      int tk3 = siklus.ticket_sb3;
      if ( IsTP(tk3)==true ) {
         siklus.profit_3 = GetProfit(tk3);
         profit += siklus.profit_3;
         //siklus.lot_sb3 = 0;
      }
   }
   // SB2
   if ( siklus.profit_2==0 && siklus.lot_sb2>0 ) {
      int tk2 = siklus.ticket_sb2;
      if ( IsTP(tk2)==true ) {
         siklus.profit_2 = GetProfit(tk2);
         profit += siklus.profit_2;
         //siklus.lot_sb2 = 0;
      }
   }
   // SB1
   if ( siklus.profit_1==0 && siklus.lot_sb1>0 ) {
      int tk1 = siklus.ticket_sb1;
      if ( IsTP(tk1)==true ) {
         siklus.profit_1 = GetProfit(tk1);
         profit += siklus.profit_1;
         //siklus.lot_sb1 = 0;
      }
   }
   
   if ( profit>0 ) 
      return true;
      
   return false;
}

bool sell_tp_hit(Cycle* &siklus) {
   double profit = 0;
   // SS3
   if ( siklus.profit_3==0 && siklus.lot_ss3>0 ) {
      int tk3 = siklus.ticket_ss3;
      if ( IsTP(tk3)==true ) {
         siklus.profit_3 = GetProfit(tk3);
         profit += siklus.profit_3;
         //siklus.lot_ss3 = 0;
      }
   }
   // SS2
   if ( siklus.profit_2==0 && siklus.lot_ss2>0 ) {
      int tk2 = siklus.ticket_ss2;
      if ( IsTP(tk2)==true ) {
         siklus.profit_2 = GetProfit(tk2);
         profit += siklus.profit_2;
         //siklus.lot_ss2 = 0;
      }
   }
   // SS1
   if ( siklus.profit_1==0 && siklus.lot_ss1>0 ) {
      int tk1 = siklus.ticket_ss1;
      if ( IsTP(tk1)==true ) {
         siklus.profit_1 = GetProfit(tk1);
         profit += siklus.profit_1;
         //siklus.lot_ss1 = 0;
      }
   }
   
   if ( profit>0 ) 
      return true;
      
   return false;
}

void Normalize_SS1_SS2(Cycle* &siklus) {
   // SS1 & SS3 DEMPET
   if ( siklus.lot_ss1==0 && delta_pt(siklus.op_price_ss1,siklus.op_price_ss3)<0.5*siklus.step_pt ) { 
      if ( Bid > siklus.op_price_ss3 + ( 3 * siklus.step_pt * Point() ) ) {
         Print("SS1 BACK TO LV1");
         siklus.op_price_ss1 = siklus.op_price_ss3 + ( 2 * siklus.step_pt * Point() ); // KE LEVEL 1
      } else if ( Bid > siklus.op_price_ss3 + ( 2 * siklus.step_pt * Point() ) ) {
         Print("SS1 NORMED TO LV2");
         siklus.op_price_ss1 = siklus.op_price_ss3 + ( 1 * siklus.step_pt * Point() ); // KE LEVEL 2
      }
      
   }
   
   // SS2 & SS3 DEMPET
   if ( siklus.lot_ss2==0 && delta_pt(siklus.op_price_ss3, siklus.op_price_ss2) < 0.5*siklus.step_pt ) {
      
      if ( Bid > siklus.op_price_ss3 + ( 2 * siklus.step_pt * Point() ) ) {
         Print("SS2 BACK TO LV2");
         siklus.op_price_ss2 = siklus.op_price_ss3 + ( 1 * siklus.step_pt * Point() ); // KE LEVEL 2
      }
   }
   
   // SS1 & SS2 DEMPET
   if ( siklus.lot_ss1==0 && delta_pt(siklus.op_price_ss1, siklus.op_price_ss2) < 0.5*siklus.step_pt ) {
      
      if ( Bid > siklus.op_price_ss2 + ( 2 * siklus.step_pt * Point() ) ) {
         Print("SS1 BACK TO LV1");
         siklus.op_price_ss1 = siklus.op_price_ss2 + ( 1 * siklus.step_pt * Point() ); // KE LEVEL 1
      }
   }
}

void Normalize_SB1_SB2(Cycle* &siklus) {
   // SB1 & SB3 DEMPET
   if ( siklus.lot_sb1==0 && delta_pt(siklus.op_price_sb1,siklus.op_price_sb3)<0.5*siklus.step_pt ) { 
      if ( Ask < siklus.op_price_sb3 - ( 3 * siklus.step_pt * Point() ) ) {
         Print("SB1 BACK TO LV1");
         siklus.op_price_sb1 = siklus.op_price_sb3 - ( 2 * siklus.step_pt * Point() ); // KE LEVEL 1
      } else if ( Ask < siklus.op_price_sb3 - ( 2 * siklus.step_pt * Point() ) ) {
         Print("SB1 NORMED TO LV2");
         siklus.op_price_sb1 = siklus.op_price_sb3 - ( 1 * siklus.step_pt * Point() ); // KE LEVEL 2
      }
      
   }
   
   // SB2 & SB3 DEMPET
   if ( siklus.lot_sb2==0 && delta_pt(siklus.op_price_sb3, siklus.op_price_sb2) < 0.5*siklus.step_pt ) {
      
      if ( Ask < siklus.op_price_sb3 - ( 2 * siklus.step_pt * Point() ) ) {
         Print("SB2 BACK TO LV2");
         siklus.op_price_sb2 = siklus.op_price_sb3 - ( 1 * siklus.step_pt * Point() ); // KE LEVEL 2
      }
   }
   
   // SB1 & SB2 DEMPET
   if ( siklus.lot_sb1==0 && delta_pt(siklus.op_price_sb2, siklus.op_price_sb1) < 0.5*siklus.step_pt ) {
      
      if ( Ask < siklus.op_price_sb2 - ( 2 * siklus.step_pt * Point() ) ) {
         Print("SB1 BACK TO LV1");
         siklus.op_price_sb1 = siklus.op_price_sb2 - ( 1 * siklus.step_pt * Point() ); // KE LEVEL 1
      }
   }
}

void ModifBEP(int type, int lv) {

   int mn = magic1;
   
   if ( lv==2 )
      mn = magic2;
   else if ( lv==3 ) 
      mn = magic3;

   for ( int i=0; i<OrdersTotal(); i++ ) {
      if ( OrderSelect(i, SELECT_BY_POS)==true ) {
         if ( OrderSymbol()==Symbol() && OrderType()==type && OrderMagicNumber()==mn ) {
            if ( OrderModify(OrderTicket(),OrderOpenPrice(),OrderOpenPrice(),OrderTakeProfit(),0,clrAliceBlue)==false )
               Alert("MODIF TK:" + IntegerToString(OrderTicket()) + " ERR:" + IntegerToString(GetLastError()) );
         }
      }
   }
}

void SB1_SB2_SB3_op_monitor(Cycle* &siklus) {
   // SB1
   if ( siklus.lot_sb1==0 && Ask>=siklus.op_price_sb1 ) {
      set_order_buy.execute = true;
      set_order_buy.lot = calculate_lot_buy(1,siklus);
      set_order_buy.tp_price = siklus.tp_price_buy;
      
      set_order_buy.sl_pt = siklus.step_pt;
      
      set_order_buy.magic = magic1;
      set_order_buy.comment = "SB1";
      set_order_buy.level = 1;
      return; 
   }
   // SB2 
   if ( siklus.lot_sb2==0 && Ask>=siklus.op_price_sb2 ) {
      set_order_buy.execute = true;
      set_order_buy.lot = calculate_lot_buy(2,siklus);
      set_order_buy.tp_price = siklus.tp_price_buy;
      
      set_order_buy.sl_pt = siklus.step_pt;
      
      set_order_buy.magic = magic2;
      set_order_buy.comment = "SB2";
      set_order_buy.level = 2;
      
      // TRAIL BEP
      if ( delta_pt(siklus.op_price_sb1, siklus.sl_price_sb1)>0.5*siklus.step_pt ) { // ONLY IF SL IS STILL 1 STEP BEHIND
         if ( delta_pt(siklus.op_price_sb1, siklus.op_price_sb2) > 0.5 * siklus.step_pt ) { // ONLY IF SB1 IS NOT BUMPED TO LV2
            siklus.sl_price_sb1 = siklus.op_price_sb1;
            set_bep_buy.execute = true;
            set_bep_buy.magic = magic1; // ONLY IF SB1 NOT BUMPED
         }
      }
            
      // TRAIL GRID SELL
      double mid = siklus.op_price_sb2-(siklus.step_pt * Point());
      grid_sell_up(siklus, mid);
      return; 
   }
   // SB3
   if ( siklus.lot_sb3==0 && Ask>=siklus.op_price_sb3 ) {
      set_order_buy.execute = true;
      set_order_buy.lot = calculate_lot_buy(3,siklus);
      set_order_buy.tp_price = siklus.tp_price_buy;
      
      set_order_buy.sl_pt = siklus.step_pt;
      
      set_order_buy.magic = magic3;
      set_order_buy.comment = "SB3";
      set_order_buy.level = 3;
      
      // TRAIL BEP
      if ( delta_pt(siklus.op_price_sb2, siklus.sl_price_sb2)>0.5*siklus.step_pt ) { // ONLY IF SB2 SL STILL 1 STEP BEHIND
         if ( delta_pt(siklus.op_price_sb2, siklus.op_price_sb3) > 0.5 * siklus.step_pt ) { // ONLY IF SB2 IS NOT BUMPED TO LV3
            siklus.sl_price_sb2 = siklus.op_price_sb2;
            set_bep_buy.execute = true;
            set_bep_buy.magic = magic2; // ONLY IF SB2 NOT BUMPED
         }
      }
            
      // TRAIL GRID SELL
      double mid = siklus.op_price_sb3-(siklus.step_pt * Point());
      grid_sell_up(siklus, mid);      
      return; 
   }
}

void SS1_SS2_SS3_op_monitor(Cycle* &siklus) {
   // SS1
   if ( siklus.lot_ss1==0 && Bid<=siklus.op_price_ss1 ) {
      set_order_sell.execute = true;
      set_order_sell.lot = calculate_lot_sell(1,siklus);
      set_order_sell.tp_price = siklus.tp_price_sell;
      
      set_order_sell.sl_pt = siklus.step_pt;
      
      set_order_sell.magic = magic1;
      set_order_sell.comment = "SS1";
      set_order_sell.level = 1;
      return; 
   }
   // SS2 
   if ( siklus.lot_ss2==0 && Bid<=siklus.op_price_ss2 ) {
      set_order_sell.execute = true;
      set_order_sell.lot = calculate_lot_sell(2,siklus);
      set_order_sell.tp_price = siklus.tp_price_sell;
      
      set_order_sell.sl_pt = siklus.step_pt;
      
      set_order_sell.magic = magic2;
      set_order_sell.comment = "SS2";
      set_order_sell.level = 2;
      
      // TRAIL BEP
      if ( delta_pt(siklus.op_price_ss1, siklus.sl_price_ss1)>0.5*siklus.step_pt ) { // IF SL STILL 1 STEP BEHIND
         // ONLY IF SS1 NOT BUMPED
         if ( delta_pt(siklus.op_price_ss1, siklus.op_price_ss2) > 0.5 * siklus.step_pt ) {
            siklus.sl_price_ss1 = siklus.op_price_ss1;
            set_bep_sell.execute = true;
            set_bep_sell.magic = magic1;
            Print("SS1 BEP'ED"); 
         }
      }
            
      // TRAIL GRID SELL
      double mid = siklus.op_price_ss2+(siklus.step_pt * Point());
      grid_buy_down(siklus, mid);
      
      //terminate = true;
      
      return; 
   }
   // SS3
   if ( siklus.lot_ss3==0 && Bid<=siklus.op_price_ss3 ) {
      set_order_sell.execute = true;
      set_order_sell.lot = calculate_lot_sell(3,siklus);
      set_order_sell.tp_price = siklus.tp_price_sell;
      
      set_order_sell.sl_pt = siklus.step_pt;
      
      set_order_sell.magic = magic3;
      set_order_sell.comment = "SS3";
      set_order_sell.level = 3;
      
      // TRAIL BEP
      if ( delta_pt(siklus.op_price_ss2, siklus.sl_price_ss2)>0.5*siklus.step_pt ) { // IF SL STILL 1 STEP BEHIND
         // ONLY IF SS2 NOT BUMPED
         if ( delta_pt(siklus.op_price_ss2, siklus.op_price_ss3) > 0.5 * siklus.step_pt ) {
            siklus.sl_price_ss2 = siklus.op_price_ss2;
            set_bep_sell.execute = true;
            set_bep_sell.magic = magic2;
            Print("SS2 BEP'ED"); // ONLY IF SS2 NOT BUMPED
         }
      }
            
      // TRAIL GRID SELL
      double mid = siklus.op_price_ss3+(siklus.step_pt * Point());
      grid_buy_down(siklus, mid);      
      return; 
   }
}

void grid_sell_up(Cycle* &siklus, double new_mid_price) {

   if ( siklus.mid_price<new_mid_price ) {
      siklus.mid_price = new_mid_price;   
      siklus.op_price_ss1 = new_mid_price - (siklus.step_pt * Point());
      siklus.op_price_ss2 = siklus.op_price_ss1 - (siklus.step_pt * Point());
      siklus.op_price_ss3 = siklus.op_price_ss2 - (siklus.step_pt * Point());
      
      siklus.sl_price_ss1 = siklus.op_price_ss1 + (siklus.step_pt * Point());
      siklus.sl_price_ss2 = siklus.op_price_ss2 + (siklus.step_pt * Point());
      siklus.sl_price_ss3 = siklus.op_price_ss3 + (siklus.step_pt * Point());
      
      siklus.tp_price_sell = siklus.mid_price - (4*siklus.step_pt * Point());
   }
}

void grid_buy_down(Cycle* &siklus, double new_mid_price) {

   if ( siklus.mid_price>new_mid_price ) {
      siklus.mid_price = new_mid_price;   
      siklus.op_price_sb1 = new_mid_price + (siklus.step_pt * Point());
      siklus.op_price_sb2 = siklus.op_price_sb1 + (siklus.step_pt * Point());
      siklus.op_price_sb3 = siklus.op_price_sb2 + (siklus.step_pt * Point());
      
      siklus.sl_price_sb1 = siklus.op_price_sb1 - (siklus.step_pt * Point());
      siklus.sl_price_sb2 = siklus.op_price_sb2 - (siklus.step_pt * Point());
      siklus.sl_price_sb3 = siklus.op_price_sb3 - (siklus.step_pt * Point());
      
      siklus.tp_price_buy = siklus.mid_price + (4*siklus.step_pt * Point());
   }
}

void SB1_SB2_SB3_sl_monitor(Cycle* &siklus) {
   // SB1 
   if ( siklus.lot_sb1>0 ) {
      if ( IsSL(siklus.ticket_sb1)==true ) {
         siklus.loss += MathAbs( GetProfit(siklus.ticket_sb1) );
         
         // BUMP ?
         if ( delta_pt(GetOpenPrice(siklus.ticket_sb1),GetClosePrice(siklus.ticket_sb1))< 0.5*siklus.step_pt ) {
            siklus.op_price_sb1 += siklus.step_pt * Point();
            siklus.sl_price_sb1 = siklus.op_price_sb1 - (siklus.step_pt*Point());
            Print("SB1 BUMPED");
         }
         
         siklus.ticket_sb1 = 0;
         siklus.lot_sb1 = 0;
         
         Print("SB1 SL HIT");
      } 
   } // END SB1
   
   // SB2 
   if ( siklus.lot_sb2>0 ) {
   
      if ( IsSL(siklus.ticket_sb2)==true ) {
      
         //Print("IsSL(siklus.ticket_sb2)==true");
      
         siklus.loss += MathAbs( GetProfit(siklus.ticket_sb2) );
         
         // BUMP ?
         if ( delta_pt(GetOpenPrice(siklus.ticket_sb2),GetClosePrice(siklus.ticket_sb2))< 0.5*siklus.step_pt ) {
            siklus.op_price_sb2 += siklus.step_pt * Point();
            siklus.sl_price_sb2 = siklus.op_price_sb2 - (siklus.step_pt*Point());
            Print("SB2 BUMPED");
         }
         
         siklus.ticket_sb2 = 0;
         siklus.lot_sb2 = 0;
         
         Print("SB2 SL HIT");
         //terminate = true;
      } 
   } // END SB2
   
   // SB3 
   if ( siklus.lot_sb3>0 ) {
      if ( IsSL(siklus.ticket_sb3)==true ) {
         siklus.loss += MathAbs( GetProfit(siklus.ticket_sb3) );
         
         siklus.ticket_sb3 = 0;
         siklus.lot_sb3 = 0;
         
         Print("SB3 SL HIT");
      } 
   } // END SB3        
}

void SS1_SS2_SS3_sl_monitor(Cycle* &siklus) {
   // SS1 
   if ( siklus.lot_ss1>0 ) {
      if ( IsSL(siklus.ticket_ss1)==true ) {
         siklus.loss += MathAbs( GetProfit(siklus.ticket_ss1) );
         
         // BUMP ?
         if ( delta_pt(GetOpenPrice(siklus.ticket_ss1),GetClosePrice(siklus.ticket_ss1))< 0.5*siklus.step_pt ) {
            siklus.op_price_ss1 -= siklus.step_pt * Point();
            siklus.sl_price_ss1 = siklus.op_price_ss1 + (siklus.step_pt*Point());
            Print("SS1 BUMPED");
         }
         
         siklus.ticket_ss1 = 0;
         siklus.lot_ss1 = 0;
         Print("SS1 HIT");
      } 
   } // END SS1
   
   // SS2 
   if ( siklus.lot_ss2>0 ) {
      if ( IsSL(siklus.ticket_ss2)==true ) {
         siklus.loss += MathAbs( GetProfit(siklus.ticket_ss2) );
         
         // BUMP ?
         if ( delta_pt(GetOpenPrice(siklus.ticket_ss2),GetClosePrice(siklus.ticket_ss2))< 0.5*siklus.step_pt ) {
            siklus.op_price_ss2 -= siklus.step_pt * Point();
            siklus.sl_price_ss2 = siklus.op_price_ss2 + (siklus.step_pt*Point());
            Print("SS2 BUMPED");
         }
         
         siklus.ticket_ss2 = 0;
         siklus.lot_ss2 = 0;
         Print("SS2 HIT");
      } 
   } // END SS2
   
   // SS3 
   if ( siklus.lot_ss3>0 ) {
      if ( IsSL(siklus.ticket_ss3)==true ) {
         siklus.loss += MathAbs( GetProfit(siklus.ticket_ss3) );
         
         siklus.ticket_ss3 = 0;
         siklus.lot_ss3 = 0;
      } 
   } // END SS3        
}

bool IsExit(int ticket) {
   if ( OrderSelect(ticket,SELECT_BY_TICKET) ) {  
      if ( OrderCloseTime()>0 )
         return true;
   }   
   return false;
}

void update_open_buy(Cycle* &siklus, int ticket, int level) {
   if ( level==1 ) {
      siklus.ticket_sb1 = ticket;
      siklus.real_price_sb1 = GetOpenPrice(ticket);
      siklus.lot_sb1 = GetLotsize(ticket);
      siklus.sl_price_sb1 = siklus.op_price_sb1 - (siklus.step_pt*Point());
   } else if ( level==2 ) {
      siklus.ticket_sb2 = ticket;
      siklus.real_price_sb2 = GetOpenPrice(ticket);
      siklus.lot_sb2 = GetLotsize(ticket);
      siklus.sl_price_sb2 = siklus.op_price_sb2 - (siklus.step_pt*Point());
   } else if ( level==3 ) {
      siklus.ticket_sb3 = ticket;
      siklus.real_price_sb3 = GetOpenPrice(ticket);
      siklus.lot_sb3 = GetLotsize(ticket);
      siklus.sl_price_sb3 = siklus.op_price_sb3 - (siklus.step_pt*Point());
   }
}

void update_open_sell(Cycle* &siklus, int ticket, int level) {
   if ( level==1 ) {
      siklus.ticket_ss1 = ticket;
      siklus.real_price_ss1 = GetOpenPrice(ticket);
      siklus.lot_ss1 = GetLotsize(ticket);
      siklus.sl_price_ss1 = siklus.op_price_ss1 + (siklus.step_pt*Point());
   } else if ( level==2 ) {
      siklus.ticket_ss2 = ticket;
      siklus.real_price_ss2 = GetOpenPrice(ticket);
      siklus.lot_ss2 = GetLotsize(ticket);
      siklus.sl_price_ss2 = siklus.op_price_ss2 + (siklus.step_pt*Point());
   } else if ( level==3 ) {
      siklus.ticket_ss3 = ticket;
      siklus.real_price_ss3 = GetOpenPrice(ticket);
      siklus.lot_ss3 = GetLotsize(ticket);
      siklus.sl_price_ss3 = siklus.op_price_ss3 + (siklus.step_pt*Point());
   }
}

bool IsTP(int ticket)
{
   if ( OrderSelect(ticket,SELECT_BY_TICKET) ) {  
      if ( OrderCloseTime()>0 && OrderProfit()>0 )
         return true;
   }   
   return false;
}


bool IsSL(int ticket)
{
   if ( OrderSelect(ticket,SELECT_BY_TICKET) ) {  
      if ( OrderCloseTime()>0 && delta_pt(OrderStopLoss(),OrderClosePrice())<50 ) {
         return true;
      }
   }   
   return false;
}

double GetOpenPrice(int tk) {
   if ( OrderSelect(tk,SELECT_BY_TICKET)==true ) {
      return OrderOpenPrice();
   }
   Print("GetOP ERR:" + IntegerToString(GetLastError()) );
   return Ask;
}

double GetClosePrice(int tk) {
   if ( OrderSelect(tk,SELECT_BY_TICKET)==true ) {
      return OrderClosePrice();
   }
   Print("GetClose ERR:" + IntegerToString(GetLastError()) );
   return Ask;
}

double GetProfit(int tk) {
   if ( OrderSelect(tk,SELECT_BY_TICKET)==true ) {
      return OrderProfit();
   }
   Print("GetProfit ERR:" + IntegerToString(GetLastError()) );
   return 0.0;
}

double GetLotsize(int tk) {
   if ( OrderSelect(tk,SELECT_BY_TICKET)==true ) {
      return OrderLots();
   }
   Print("GetLot ERR:" + IntegerToString(GetLastError()) );
   return 0.0;
}

void CreateCycle(Cycle* &array[], FIRST_POS first, double price, double trans_loss, double step) {
   // INSTANTIATE
   Cycle* new_cycle = new Cycle(first, price, trans_loss, step); 
    
   // INJECT TO ARRAY
   int size = ArraySize(array);
   int new_size = size + 1; 
   ArrayResize(array,new_size);
   array[size] = new_cycle; 
}

bool OpenBuy(double lot,int slippage,double sl_pt,double tp_price,string cmt,int mn,int& ticket) {
   int tk = OrderSend(Symbol(),OP_BUY,lot,Ask,slippage,0,0,cmt,mn,0,clrAliceBlue);
   
   if ( tk>0 ) {      
      ticket = tk;
      if ( OrderSelect(tk,SELECT_BY_TICKET) )
      {
         double op_price = OrderOpenPrice(); 
         double sl_price = 0;
         
         if ( sl_pt>0 )
            sl_price = NormalizeDouble(op_price-(sl_pt*Point()),Digits());

         if ( OrderModify(tk,op_price,sl_price,tp_price,0,clrAliceBlue)==false )
         {
            Alert("BUY MODIF ERR:" + IntegerToString(GetLastError()) + "sl_price:" + DoubleToStr(sl_price,Digits()) + " tp_price:" + DoubleToStr(tp_price,Digits()) );                  
         }         
      }    
      
      return true;  
   }
   
   return false;
} // END OP SELL

bool OpenSell(double lot,int slippage,double sl_pt,double tp_price,string cmt,int mn,int& ticket) {
   int tk = OrderSend(Symbol(),OP_SELL,lot,Bid,slippage,0,0,cmt,mn,0,clrAliceBlue);
   
   if ( tk>0 ) {  
      ticket = tk;    
      if ( OrderSelect(tk,SELECT_BY_TICKET) )
      {
         double op_price = OrderOpenPrice(); 
         double sl_price = 0;
         
         if ( sl_pt>0 )
            sl_price = NormalizeDouble(op_price+(sl_pt*Point()),Digits());

         if ( OrderModify(tk,op_price,sl_price,tp_price,0,clrAliceBlue)==false )
         {
            Alert("SELL MODIF ERR:" + IntegerToString(GetLastError()) + "sl_price:" + DoubleToStr(sl_price,Digits()) + " tp_price:" + DoubleToStr(tp_price,Digits()) );                  
         }         
      }    
      
      return true;  
   }
   
   return false;
} // END OP SELL

double calculate_lot_buy(int level, Cycle* &go) {
   double total_buy_slot_pt = 0;
   double total_buy_lot = 0;
   
   double total_running_buy = 0;
   
   // SB1 
   if ( go.lot_sb1==0 ) {
      total_buy_slot_pt += (go.step_pt/Cycle::kb_ratio) * 3 * Cycle::ratio_lv1;
      Print("total_buy_slot_pt #1:" + DoubleToStr(total_buy_slot_pt,2) );
   } else if ( go.lot_sb1>0 ) {
      double bm1_pt = ( go.tp_price_buy - go.real_price_sb1 )/Point();
      total_running_buy += (bm1_pt/Cycle::kb_ratio) * go.lot_sb1;
      Print("total_running_buy #1:" + DoubleToStr(total_running_buy,2) );
   }
   
   // SB2 
   if ( go.lot_sb2==0 ) {
      total_buy_slot_pt += (go.step_pt/Cycle::kb_ratio) * 2 * Cycle::ratio_lv2;
      Print("total_buy_slot_pt #2:" + DoubleToStr(total_buy_slot_pt,2) );
   } else if ( go.lot_sb2>0 ) {
      double bm2_pt = ( go.tp_price_buy - go.real_price_sb2 )/Point();
      total_running_buy += (bm2_pt/Cycle::kb_ratio) * go.lot_sb2;
      Print("total_running_buy #2:" + DoubleToStr(total_running_buy,2) );
   }
   
   // SB3 
   if ( go.lot_sb3==0 ) {
      total_buy_slot_pt += (go.step_pt/Cycle::kb_ratio) * Cycle::ratio_lv3;
      Print("total_buy_slot_pt #3:" + DoubleToStr(total_buy_slot_pt,2) );
   }
   
   Print("Cycle::target_usd:" + DoubleToStr(Cycle::target_usd,2) + " go.loss:" + DoubleToStr(go.loss,2) + " total_buy_slot_pt:" + DoubleToStr(total_buy_slot_pt,2) );
   
   if ( total_buy_slot_pt>0 ) {
      Print("total_buy_slot_pt>0");
      total_buy_lot = ( Cycle::target_usd + go.loss - total_running_buy ) / total_buy_slot_pt;
   }
   
   Print("total_buy_lot:" + DoubleToStr(total_buy_lot,2));
   
   double lot_normal_sb1 = total_buy_lot * Cycle::ratio_lv1;
   double lot_normal_sb2 = total_buy_lot * Cycle::ratio_lv2;
   double lot_normal_sb3 = NormalizeDouble( total_buy_lot * Cycle::ratio_lv3, 2);
   
   Print("lot_normal_sb1:" + DoubleToStr(lot_normal_sb1,2) );
   Print("lot_normal_sb2:" + DoubleToStr(lot_normal_sb1,2) );
   Print("lot_normal_sb3:" + DoubleToStr(lot_normal_sb1,2) );
   
   double step_normal_sb1 = 3 * go.step_pt;
   double step_normal_sb2 = 2 * go.step_pt;
   
   double step_actual_sb1 = (go.tp_price_buy - go.op_price_sb1) / Point();
   double step_actual_sb2 = (go.tp_price_buy - go.op_price_sb2) / Point();
   
   if ( level==1 ) {
      double lot_actual_sb1 = lot_normal_sb1 * ( step_normal_sb1 / step_actual_sb1 );
      if ( lot_actual_sb1<=0.01 )
         lot_actual_sb1 = 0.01;
      lot_actual_sb1 = NormalizeDouble(lot_actual_sb1,2);
      Print("lot_actual_sb1:" + DoubleToStr(lot_actual_sb1,2) );
      return lot_actual_sb1;
   } else if ( level==2 ) {
      double lot_actual_sb2 = lot_normal_sb2 * ( step_normal_sb2 / step_actual_sb2 );
      if ( lot_actual_sb2<=0.01 )
         lot_actual_sb2 = 0.01;
      lot_actual_sb2 = NormalizeDouble(lot_actual_sb2,2);
      Print("lot_actual_sb2:" + DoubleToStr(lot_actual_sb2,2) );
      return lot_actual_sb2;
   }
   //terminate = true;
   if ( lot_normal_sb3<=0.01 )
      lot_normal_sb3 = 0.01;
   Print("lot_normal_sb3: " + DoubleToStr(lot_normal_sb3,2) );
   return lot_normal_sb3;
}

double calculate_lot_sell(int level, Cycle* &go) {
   double total_sell_slot_pt = 0;
   double total_sell_lot = 0;
   
   double total_running_sell = 0;
   
   // SS1 
   if ( go.lot_ss1==0 ) {
      total_sell_slot_pt += go.step_pt * 3 * Cycle::ratio_lv1;
      Print("total_sell_slot_pt #1: " + DoubleToStr(total_sell_slot_pt,2) );
   } else if ( go.lot_ss1>0 ) {
      double sm1_pt = ( go.real_price_ss1 - go.tp_price_sell )/Point();
      total_running_sell += (sm1_pt/Cycle::kb_ratio) * go.lot_ss1;
   }
   
   // SS2 
   if ( go.lot_ss2==0 ) {
      total_sell_slot_pt += go.step_pt * 2 * Cycle::ratio_lv2;
      Print("total_sell_slot_pt #2: " + DoubleToStr(total_sell_slot_pt,2) );
   } else if ( go.lot_ss2>0 ) {
      double sm2_pt = ( go.real_price_ss2 - go.tp_price_sell )/Point();
      total_running_sell += (sm2_pt/Cycle::kb_ratio) * go.lot_ss2;
   }
   
   // SS3 
   if ( go.lot_ss3==0 ) {
      total_sell_slot_pt += go.step_pt * Cycle::ratio_lv3;
      Print("total_sell_slot_pt #3: " + DoubleToStr(total_sell_slot_pt,2) );
   }
   
   if ( total_sell_slot_pt>0 ) {
      total_sell_lot = ( Cycle::target_usd + go.loss - total_running_sell ) / total_sell_slot_pt;
      Print("total_sell_lot: " + DoubleToStr(total_sell_lot,2) );
   }
   
   double lot_normal_ss1 = total_sell_lot * Cycle::ratio_lv1;
   double lot_normal_ss2 = total_sell_lot * Cycle::ratio_lv2;
   double lot_normal_ss3 = NormalizeDouble( total_sell_lot * Cycle::ratio_lv3, 2);
   
   double step_normal_ss1 = 3 * go.step_pt;
   double step_normal_ss2 = 2 * go.step_pt;
   
   double step_actual_ss1 = (go.op_price_ss1-go.tp_price_sell) / Point();
   double step_actual_ss2 = (go.op_price_ss2-go.tp_price_sell) / Point();
   
   if ( level==1 ) {
      double lot_actual_ss1 = lot_normal_ss1 * ( step_normal_ss1 / step_actual_ss1 );
      if ( lot_actual_ss1<=0.01 )
         lot_actual_ss1 = 0.01;
      lot_actual_ss1 = NormalizeDouble(lot_actual_ss1,2);
      Print("lot_actual_ss1: " + DoubleToStr(lot_actual_ss1,2) );
      return lot_actual_ss1;
   } else if ( level==2 ) {
      double lot_actual_ss2 = lot_normal_ss2 * ( step_normal_ss2 / step_actual_ss2 );
      Print("PRE lot_actual_ss2: " + DoubleToStr(lot_actual_ss2,5) );
      if ( lot_actual_ss2<=0.01 )
         lot_actual_ss2 = 0.01;
      lot_actual_ss2 = NormalizeDouble(lot_actual_ss2,2);
      Print("POST lot_actual_ss2: " + DoubleToStr(lot_actual_ss2,2) );
      return lot_actual_ss2;
   }
   
   if ( lot_normal_ss3<=0.01 )
      lot_normal_ss3 = 0.01;
   
   Print("lot_normal_ss3: " + DoubleToStr(lot_normal_ss3,2) );
   return lot_normal_ss3;
}

double delta_pt(double price1, double price2) {
   double value = MathMax(price1,price2) - MathMin(price1,price2);
   return value/Point();
}

void DestroyCycle(Cycle* &array[]) {
   int size = ArraySize(array);
   
   for ( int i=0; i<size; i++ ) {
      delete(array[i]);
      array[i]=NULL;
   } // END FOR
}
