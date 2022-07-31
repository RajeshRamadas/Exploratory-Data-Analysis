# -*- coding: utf-8 -*-
"""
@Author: Rajesh Kumar Ramadas 
@Date: 7/19/2022
@Email: Rajeshkumar1988r@gmail.com
@Github: https://github.com/rakskumar/develop

This module performs task related to database operation  
and creating table, updating table, connection and disconnection of database are few functions

"""
#external import
from sqlalchemy import create_engine,inspect,text
import pandas as pd

class DataAccess():
    """ Class DataAccess is used for accessing database  """
    def __init__(self):
        #dict to provide access to dataframe with resoective key value pair
        self.table_name_dict = {"_test_data_df":"test_data_tb","_train_data_df":"train_data_tb","_ideal_data_df":"ideal_data_tb"}
        self.df_dict = {"_test_data_df":'',"_train_data_df":'',"_ideal_data_df":''}
            
    def load_csv_to_db(self,train_data_csv, ideal_data_csv, test_data_csv,db_path):
        """
        load csv raw file data to database 
        input:
        return:
        """
        #analysis data file path 
        self.ideal_data_csv = ideal_data_csv
        self.train_data_csv = train_data_csv
        self.test_data_csv  = test_data_csv
        
        #relative path for database 
        self.db_path = db_path
        #create database 
        self.connect_db()
        #update database with all the data file
        self.update_rawdata_to_db(self.ideal_data_csv,self.train_data_csv,self.test_data_csv)
        # close db engine/connection
        self.disconnect_db()
    
   
    def update_rawdata_to_db(self,ideal_data_csv,train_data_csv,test_data_csv):
        """update Database from CSV file"""
        #convert csv to dataframe
        self.df_dict["_ideal_data_df"] = pd.read_csv(ideal_data_csv)
        self.df_dict["_train_data_df"] = pd.read_csv(train_data_csv)
        self.df_dict["_test_data_df"] = pd.read_csv(test_data_csv)
        
        
        def _create_table_from_df():
            """update database from dataframe"""
            
            #inspect used to get database related info
            inspect_db = inspect(self.engine)
            table_names_db = inspect_db.get_table_names()
            
            #Delete table if exist, create new table every execution
            for table_name in table_names_db:
                if table_name in list(self.table_name_dict.values()):
                    sql_cmd = text(f"DROP TABLE {table_name}")
                    self.engine.execute(sql_cmd)
                    
            #save data in database from dataframe 
            self.df_dict["_ideal_data_df"].to_sql(self.table_name_dict["_ideal_data_df"], self.engine)
            self.df_dict["_train_data_df"].to_sql(self.table_name_dict["_train_data_df"], self.engine)
            self.df_dict["_test_data_df"].to_sql(self.table_name_dict["_test_data_df"], self.engine) 
        
        _create_table_from_df()
    
    def connect_db(self, db_path = ".\\database\\data_analysis.db"):
        """
           create database and engine w.r.t path
           input relative path
           return engine object
        """
        try:
            self.engine = create_engine(f'sqlite:///{db_path}', echo=True)
            return self.engine
        except Exception as _error:
            print(f"Error: DATABASE ENGINE NOT CREATED: {_error}")
        
    
    def disconnect_db(self):
        """Disconnect database """
        try:  
            self.engine.dispose()
        except Exception as _error:
            print(f"Error: DISCONNECT UNSUCCESSFUL: {_error}")

    def get_test_data_df(self):
        """ get test dataset from database """
                                    
        _conn_db = self.connect_db()
        test_data_df = pd.read_sql_table(self.table_name_dict['_test_data_df'], _conn_db)
        self.disconnect_db()
        return test_data_df
        
    
    def get_ideal_data_df(self):
        """ get ideal dataset from database """
        _conn_db = self.connect_db()
        test_ideal_df = pd.read_sql_table(self.table_name_dict['_ideal_data_df'], _conn_db)
        self.disconnect_db()
        return test_ideal_df
    
    def get_train_data_df(self):
        """ get train dataset from database """
        _conn_db = self.connect_db()
        test_train_df = pd.read_sql_table(self.table_name_dict['_train_data_df'], _conn_db)
        
        self.disconnect_db()
        """disconnect database"""
        return test_train_df
    
    def create_tb(self,db_table_name,data_df):
        """ create table w.r.t engine"""
        try:
            data_df.to_sql(db_table_name, self.engine)
        except Exception as _error:
            print(f"Error: UNABLE TO CREATE TABLE : {_error}")
        