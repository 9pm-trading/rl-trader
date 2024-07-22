//+------------------------------------------------------------------+
//|                                                        H1Bar.mq4 |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

#include <MQLMySQL.mqh>

input string host = "localhost";
input string user = "root";
input string pass = "";
input string database = "mt4";
input int port = 3306;

input int slip = 50;
input int magic = 12345;

input string table = "eurusd_h1"; // TABLE NAME

int DB;
string socket = "0";
int flag = 0;

bool buy_open;
bool sell_open;
bool buy_close;
bool sell_close;

int buy_action;
int sell_action;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   //---
   DB = MySqlConnect(host, user, pass, database, port, socket, flag);
   
   if (DB == -1) 
      { Print ("Connection failed! Error: "+MySqlErrorDescription); } 
   else 
      { Print ("Connected! DBID#",DB);} 
   
   if ( IsTableExist(DB, table)==false ) {
      if ( CreateTable(table)==true ) {
      
         // INJECT FIRST
      
      }
   }   
   
   int row = CountRow(table);
   
   if ( row<=0 ) {
      //Alert("hi there i'm zero");
      InjectOnce(table);
   }
   
   buy_open = false;
   sell_open = false;
   
   buy_close = false;
   sell_close = false;
   
   buy_action = get_status(DB, "orders", "OP_BUY");
   sell_action = get_status(DB, "orders", "OP_SELL");
   //---
   return(INIT_SUCCEEDED);
}
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
//---
   MySqlDisconnect(DB);
}
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
//---
   if ( NewBar() ) {
      // POPULATE TABLE
      //Print( CountRow(table) );
      UpdateDB(DB,table);
   }

   if ( buy_action==1 ) {
      int tk = OrderSend(Symbol(),OP_BUY,1.0,Ask,slip,0,0,"RLv1",magic,0,clrAliceBlue);
      
      if ( tk>0 ) {
         // UPDATE TK TO DB
         buy_action = 2;
         UpdateStatus(DB,"orders",Ask,2,tk,"OP_BUY");
      }
   } else if ( buy_action==3 ) {
      if ( OrderCount(magic,OP_BUY,Symbol())>0 ) {
         CloseOrders(OP_BUY,magic,Symbol(),slip);
      }
      
      if ( OrderCount(magic,OP_BUY,Symbol())==0 ) {
         buy_action = 0;
         UpdateStatus(DB,"orders",0,0,0,"OP_BUY");
      }
   }  
   
   if ( sell_action==1 ) {
      int tk = OrderSend(Symbol(),OP_SELL,1.0,Bid,slip,0,0,"RLv1",magic,0,clrAliceBlue);
      
      if ( tk>0 ) {
         sell_action = 2;
         UpdateStatus(DB,"orders",Bid,2,tk,"OP_SELL");
      }
   } else if ( sell_action==3 ) {
      if ( OrderCount(magic,OP_SELL,Symbol())>0 ) {
         CloseOrders(OP_SELL,magic,Symbol(),slip);
      }
      
      if ( OrderCount(magic,OP_SELL,Symbol())==0 ) {
         sell_action = 0;
         UpdateStatus(DB,"orders",0,0,0,"OP_SELL");
      }
   }  
   
   // TIME FILTER - MINUTE 0-15
   int minute = TimeMinute(TimeCurrent());
   //if ( minute>=0 && minute<=5 ) {     
      // RETRIEVE ORDER STATUS 0 - NOT YET EXECUTED
      buy_action = get_status(DB, "orders", "OP_BUY");
      sell_action = get_status(DB, "orders", "OP_SELL");     
   //}
}
//+------------------------------------------------------------------+

void CloseOrders(int type, int mn, string symbol, int slippage) {
   for ( int i=0; i<OrdersTotal(); i++ ) {
      if ( OrderSelect(i,SELECT_BY_POS)==true ) {
         if ( OrderSymbol()==symbol && OrderMagicNumber()==mn ) {
            if ( OrderType()==type && type==OP_BUY ) {
               if ( OrderClose(OrderTicket(),OrderLots(),Bid,slippage,clrAliceBlue)==false ) {
                  Alert("BUY CLOSE ERR:" + IntegerToString(GetLastError()));
               }
            } else if ( OrderType()==type && type==OP_SELL ) {
               if ( OrderClose(OrderTicket(),OrderLots(),Ask,slippage,clrAliceBlue)==false ) {
                  Alert("SELL CLOSE ERR:" + IntegerToString(GetLastError()));
               }
            }
         }
      }  
   }
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

int get_status(int db, string tb, string type) {
   int status = -1;
 
   if (db == -1) { 
      Print ("Connection failed! Error: "+MySqlErrorDescription); 
   } else {
      string query_select;
      
      query_select = "SELECT * FROM `" + tb + "` WHERE type = \'" + type + "\'";
       
      int Cursor,Rows;
      Cursor = MySqlCursorOpen(db, query_select);
    
      if (Cursor >= 0) {
         Rows = MySqlCursorRows(Cursor);

         if (MySqlCursorFetchRow(Cursor)) {
            status = MySqlGetFieldAsInt(Cursor, 9); // index 9 - status
         }    
         MySqlCursorClose(Cursor); // NEVER FORGET TO CLOSE CURSOR !!!
  
      } else if ( Cursor==-1 ) {            
         Print ("Cursor opening failed. Error: ", MySqlErrorDescription);
      }    
      MySqlCursorClose(Cursor); 
   }
   
   return status;  
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

bool IsTableExist(int db,string tb) {
   //////////////////////////////////////////////////////////////////////////////////////////////
   ////////////////////////////////////////// CHECK TABLE EXISTENCE /////////////////////////////
   string query_select;
   query_select = "SELECT * FROM `" + tb + "` LIMIT 1";
   
   int Cursor,Rows;
   Cursor = MySqlCursorOpen(db, query_select);
   
   //Print("IsTableExist Cursor->" + IntegerToString(Cursor) );
   
   if (Cursor>=0 ) {
      Rows = MySqlCursorRows(Cursor);
      if ( Rows>=1 ) {
         //Print (Rows, " row(s) selected. Table Exists");
         
         MySqlCursorClose(Cursor); // NEVER FORGET TO CLOSE CURSOR !!!
         return true;
      }
   }
   else if ( Cursor<0 ) {   
      Print ("Cursor opening failed. Error: ", MySqlErrorDescription);
   }
   
   MySqlCursorClose(Cursor); // NEVER FORGET TO CLOSE CURSOR !!!
   //Print ("Cursor closing Error: ", MySqlErrorDescription);
   
   ////////////////////////////////////////// CHECK TABLE EXISTENCE /////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////

   return false;
}

bool CreateTable(string tbl) {  
   if (DB == -1) { 
      Print ("Connection failed! Error: "+MySqlErrorDescription);  
   } else {
      string query_create;
      
      query_create = StringConcatenate( "CREATE TABLE `" , tbl , "` (" , 
         "id int AUTO_INCREMENT PRIMARY KEY, " ,
         "symbol varchar(50), " ,
         "timestamp int, ",
         "open_1 varchar(50), ", 
         "high_1 varchar(50), ",
         "low_1 varchar(50), ",
         "close_1 varchar(50), ",
         "open_2 varchar(50), ", 
         "high_2 varchar(50), ",
         "low_2 varchar(50), ",
         "close_2 varchar(50), ",
         "open_3 varchar(50), ", 
         "high_3 varchar(50), ",
         "low_3 varchar(50), ",
         "close_3 varchar(50), ",
         "D50_1 varchar(50), ",
         "D50_2 varchar(50), ",
         "D50_3 varchar(50), ",
         "D21_1 varchar(50), ",
         "D21_2 varchar(50), ",
         "D21_3 varchar(50), ",
         
         "balance varchar(50)" ,
         ")"
      );
      
      if (MySqlExecute(DB, query_create)) {      
         Print ("Table `" + tbl + "` created.");
         
         return true;     
      } else {     
         Print ("Table `" + tbl +"` cannot be created. Error: ", MySqlErrorDescription);
      }
   }

   return false;
}

void InjectOnce(string tbl) {
   //////////////////////////////////////////////////////////////////////////////////////////////
   ////////////////////////////////////////// CHECK TABLE EXISTENCE /////////////////////////////    
   if (DB == -1) {
      Print ("Connection failed! Error: "+MySqlErrorDescription); 
   } else {
      
      // GET PRICE DATA HERE
      string symbol = Symbol();
      string timestamp = IntegerToString( iTime(Symbol(),PERIOD_H1,0) );
      string open_1 = DoubleToStr( Open[1] , Digits());
      string high_1 = DoubleToStr( High[1] , Digits());
      string low_1 = DoubleToStr( Low[1] , Digits());
      string close_1 = DoubleToStr( Close[1] , Digits());
      
      string open_2 = DoubleToStr( Open[2] , Digits());
      string high_2 = DoubleToStr( High[2] , Digits());
      string low_2 = DoubleToStr( Low[2] , Digits());
      string close_2 = DoubleToStr( Close[2] , Digits());
      
      string open_3 = DoubleToStr( Open[3] , Digits());
      string high_3 = DoubleToStr( High[3] , Digits());
      string low_3 = DoubleToStr( Low[3] , Digits());
      string close_3 = DoubleToStr( Close[3] , Digits());
      
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
      
      string D50_1 = DoubleToStr( delta_ma50_1, Digits() );
      string D50_2 = DoubleToStr( delta_ma50_2, Digits() );
      string D50_3 = DoubleToStr( delta_ma50_3, Digits() );
      
      string D21_1 = DoubleToStr( delta_ma21_1, Digits() );
      string D21_2 = DoubleToStr( delta_ma21_2, Digits() );
      string D21_3 = DoubleToStr( delta_ma21_3, Digits() );
      
      string bal = DoubleToStr(AccountBalance(), 2);
      
      string query_select;
      
      query_select = "INSERT INTO `" + tbl + "` (symbol,timestamp,open_1,high_1,low_1,close_1,open_2,high_2,low_2,close_2,open_3,high_3,low_3,close_3,D50_1,D50_2,D50_3,D21_1,D21_2,D21_3,balance) VALUES (" 
                     + "\'" + symbol + "\'"
                     + "," + timestamp
                     + "," + "\'" + open_1 + "\'"
                     + "," + "\'" + high_1 + "\'" 
                     + "," + "\'" + low_1 + "\'" 
                     + "," + "\'" + close_1 + "\'"
                     + "," + "\'" + open_2 + "\'"
                     + "," + "\'" + high_2 + "\'" 
                     + "," + "\'" + low_2 + "\'" 
                     + "," + "\'" + close_2 + "\'" 
                     + "," + "\'" + open_3 + "\'"
                     + "," + "\'" + high_3 + "\'" 
                     + "," + "\'" + low_3 + "\'" 
                     + "," + "\'" + close_3 + "\'"
                     + "," + "\'" + D50_1 + "\'"
                     + "," + "\'" + D50_2 + "\'"
                     + "," + "\'" + D50_3 + "\'"
                     + "," + "\'" + D21_1 + "\'"
                     + "," + "\'" + D21_2 + "\'"
                     + "," + "\'" + D21_3 + "\'"
                     + "," + "\'" + bal + "\'"
                     + ")";  
      
      if (MySqlExecute(DB, query_select)) {    
         Print ("Succeeded: ", query_select);    
      } else {    
         Print ("Error: ", MySqlErrorDescription);
         Print ("Query: ", query_select);
      }
      
      ////////////////////////////////////////// CHECK TABLE EXISTENCE /////////////////////////////
      //////////////////////////////////////////////////////////////////////////////////////////////
   }
}

void UpdateDB(int db,string tb) {
   // GET PRICE DATA HERE
   string timestamp = IntegerToString( iTime(Symbol(),PERIOD_H1,0) );
   string open_1 = "\'" + DoubleToStr( Open[1] , Digits()) + "\'";
   string high_1 = "\'" + DoubleToStr( High[1] , Digits()) + "\'";
   string low_1 = "\'" + DoubleToStr( Low[1] , Digits()) + "\'";
   string close_1 = "\'" + DoubleToStr( Close[1] , Digits()) + "\'";
   
   string open_2 = "\'" + DoubleToStr( Open[2] , Digits()) + "\'";
   string high_2 = "\'" + DoubleToStr( High[2] , Digits()) + "\'";
   string low_2 = "\'" + DoubleToStr( Low[2] , Digits()) + "\'";
   string close_2 = "\'" + DoubleToStr( Close[2] , Digits()) + "\'";
   
   string open_3 = "\'" + DoubleToStr( Open[3] , Digits()) + "\'";
   string high_3 = "\'" + DoubleToStr( High[3] , Digits()) + "\'";
   string low_3 = "\'" + DoubleToStr( Low[3] , Digits()) + "\'";
   string close_3 = "\'" + DoubleToStr( Close[3] , Digits()) + "\'";
   
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
   
   string D50_1 = "\'" + DoubleToStr( delta_ma50_1, Digits() ) + "\'";
   string D50_2 = "\'" + DoubleToStr( delta_ma50_2, Digits() ) + "\'";
   string D50_3 = "\'" + DoubleToStr( delta_ma50_3, Digits() ) + "\'";
   
   string D21_1 = "\'" + DoubleToStr( delta_ma21_1, Digits() ) + "\'";
   string D21_2 = "\'" + DoubleToStr( delta_ma21_2, Digits() ) + "\'";
   string D21_3 = "\'" + DoubleToStr( delta_ma21_3, Digits() ) + "\'";
   
   string bal = "\'" + DoubleToStr( AccountBalance(), 2 ) + "\'";

   //string cols = "timestamp,open_1,high_1,low_1,close_1,open_2,high_2,low_2,close_2,open_3,high_3,low_3,close_3,D50_1,D50_2,D50_3,D21_1,D21_2,D21_3";
   string query_select = "UPDATE `" + tb + "` SET timestamp=" + timestamp 
                                         + ",open_1=" + open_1 
                                         + ",high_1=" + high_1
                                         + ",low_1=" + low_1
                                         + ",close_1=" + close_1
                                         
                                         + ",open_2=" + open_2 
                                         + ",high_2=" + high_2
                                         + ",low_2=" + low_2
                                         + ",close_2=" + close_2
                                         
                                         + ",open_3=" + open_3 
                                         + ",high_3=" + high_3
                                         + ",low_3=" + low_3
                                         + ",close_3=" + close_3
                                         
                                         + ",D50_1=" + D50_1 
                                         + ",D50_2=" + D50_2
                                         + ",D50_3=" + D50_3
                                         
                                         + ",D21_1=" + D21_1 
                                         + ",D21_2=" + D21_2
                                         + ",D21_3=" + D21_3   
                                         + ",balance=" + bal
   + " WHERE id = 1";

   if (MySqlExecute(db, query_select)) {    
      Print ("Succeeded: ", query_select);    
   } else {    
      Print ("Error: ", MySqlErrorDescription);
      Print ("Query: ", query_select);
   }     
}

void UpdateStatus(int db,string tb,double price, int status,int ticket, string type) {
   
   string query_select = "UPDATE `" + tb + "` SET ticket=" + IntegerToString(ticket) 
                                         + ",status=" + IntegerToString(status) 
                                         + ",op=" + DoubleToStr(price,Digits())                                        
   + " WHERE type=\'" + type + "\'";

   if (MySqlExecute(db, query_select)) {    
      Print ("Succeeded: ", query_select);    
   } else {    
      Print ("Error: ", MySqlErrorDescription);
      Print ("Query: ", query_select);
   }     
}

int CountRow(string tbl) {
   //////////////////////////////////////////////////////////////////////////////////////////////
   ////////////////////////////////////////// CHECK TABLE EXISTENCE /////////////////////////////  
   if (DB == -1) {
      Print ("Connection failed! Error: "+MySqlErrorDescription); }
   else if ( DB>=0 ) {
   
      string query_select;
      
      // SELECT COUNT(*) FROM code_values;
      
      query_select = "SELECT COUNT(*) FROM `" + tbl + "`";
      
      int Cursor,Rows;
      Cursor = MySqlCursorOpen(DB, query_select);
    
      if (Cursor >= 0) {
         Rows = MySqlCursorRows(Cursor);
            
         //if ( Rows>=1 ) {
         
            int count = 0;
         
            if (MySqlCursorFetchRow(Cursor)) {
               count = MySqlGetFieldAsInt(Cursor, 0); // id
            }
         
            MySqlCursorClose(Cursor); // NEVER FORGET TO CLOSE CURSOR !!!
            return count;
         //}
      } else {   
         
         Print ("Cursor opening failed. Error: ", MySqlErrorDescription);
      }
      
      MySqlCursorClose(Cursor); // NEVER FORGET TO CLOSE CURSOR !!!*/
      //Print ("Cursor closing Error: ", MySqlErrorDescription);
      
      ////////////////////////////////////////// CHECK TABLE EXISTENCE /////////////////////////////
      //////////////////////////////////////////////////////////////////////////////////////////////
   }
   
   return -1;
}
