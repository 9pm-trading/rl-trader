import mysql.connector

def cnx():
  mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="mt4"
  )
  return mydb

def get_timestamp():
  mydb = cnx()
  mycursor = mydb.cursor()
  mycursor.execute("SELECT * FROM eurusd_h1")
  myresult = mycursor.fetchall() # myresult -> list  
  timestamp = myresult[0][2]
  mycursor.close()
  mydb.close()
  return timestamp

def get_data():
  mydb = cnx()
  mycursor = mydb.cursor()
  mycursor.execute("SELECT * FROM eurusd_h1")
  myresult = mycursor.fetchall() # myresult -> list 
  dic = {
    'timestamp' : myresult[0][2] ,
    'open_1' : myresult[0][3] ,
    'high_1' : myresult[0][4] ,
    'low_1' : myresult[0][5] ,
    'close_1' : myresult[0][6] ,

    'open_2' : myresult[0][7] ,
    'high_2' : myresult[0][8] ,
    'low_2' : myresult[0][9] ,
    'close_2' : myresult[0][10] ,

    'open_3' : myresult[0][11] ,
    'high_3' : myresult[0][12] ,
    'low_3' : myresult[0][13] ,
    'close_3' : myresult[0][14] ,

    'D50_1' : myresult[0][15] ,
    'D50_2' : myresult[0][16] ,
    'D50_3' : myresult[0][17] ,
    
    'D21_1' : myresult[0][18] ,
    'D21_2' : myresult[0][19] ,
    'D21_3' : myresult[0][20] ,
    'balance' : myresult[0][21]
  }
  mycursor.close()
  mydb.close()
  return dic

def create_table_orders():
  table_name = 'orders'
  mydb = cnx()
  mycursor = mydb.cursor()
  query = f'CREATE TABLE {table_name} (id INT AUTO_INCREMENT PRIMARY KEY, symbol VARCHAR(10), type VARCHAR(10), lot VARCHAR(10), op VARCHAR(10), tp VARCHAR(10), sl VARCHAR(10), ticket INT(11), comment VARCHAR(10), status INT(11))'
  print(f'table {table_name} created')
  mycursor.execute(query)
  mycursor.close()
  mydb.close()
  
def is_table_exist(table_name):
  mydb = cnx()
  mycursor = mydb.cursor()
  mycursor.execute("SHOW TABLES")

  for x in mycursor:
      if table_name in x[0]:
        print('table exists')
        return True  

  print('table not found')

  mycursor.close()
  mydb.close()
  return False 

def get_orders(table_name):
  mydb = cnx()
  mycursor = mydb.cursor()
  query = f"SELECT * FROM {table_name}"
  mycursor.execute(query)
  myresult = mycursor.fetchall() # myresult -> list 
  orders = [] # prep list
  for order in myresult:
    dic = {
      'id' : order[0] ,
      'symbol' : order[1] ,
      'type' : order[2] ,
      'lot' : order[3] ,
      'op' : order[4] ,
      'tp' : order[5] ,
      'sl' : order[6] ,
      'ticket' : order[7] , 
      'comment' : order[8] ,
      'status' : order[9]
    }
    orders.append(dic)
  # END FOR 
  mycursor.close()
  mydb.close()
  return orders # list of dic

def update_order(table_name, field, value, type):
  mydb = cnx()
  mycursor = mydb.cursor()
  query = f"UPDATE {table_name} SET {field} = {value} WHERE type = '{type}'"
  mycursor.execute(query)
  mydb.commit()
  print(mycursor.rowcount, "record(s) affected")
  mycursor.close()  
  mydb.close()

def bikin_table_bridge():
  table_name = 'bridge'
  mydb = cnx()
  mycursor = mydb.cursor()
  query = f'CREATE TABLE {table_name} (id INT PRIMARY KEY, action_vector INT(10), cycle_status int(10), target_usd FLOAT(10), max_dd_pct INT(10))'
  print(f'table {table_name} created')
  mycursor.execute(query)
  mycursor.close()
  mydb.close()  

def init_table_bridge(target, max_dd):
  table_name = 'bridge'
  mydb = cnx()
  mycursor = mydb.cursor()
  query = f'INSERT INTO {table_name} (id, action_vector, cycle_status, target_usd, max_dd_pct) VALUES (0, 0, 0, {target}, {max_dd})'
  mycursor.execute(query)
  mydb.commit()
  mycursor.close()
  mydb.close()  

def get_bridge_item(col_name):
  table_name = 'bridge'
  mydb = cnx()
  mycursor = mydb.cursor()
  mycursor.execute(f"SELECT {col_name} FROM {table_name} WHERE id=0")
  myresult = mycursor.fetchall() # myresult -> list 
  mycursor.close()
  mydb.close()
  
  return myresult[0][0]

def update_bridge_item(col_name, value):
  table_name = 'bridge'
  mydb = cnx()
  mycursor = mydb.cursor()
  query = f"UPDATE {table_name} SET {col_name} = {value} WHERE id = 0"
  mycursor.execute(query)
  mydb.commit()
  print(mycursor.rowcount, "record(s) affected")
  mycursor.close()  
  mydb.close()