
# -*- coding: utf-8 -*-
"""
@Author: Rajesh Kumar Ramadas 
@Date: 7/19/2022
@Email: Rajeshkumar1988r@gmail.com
@Github: 

This module demonstrates accessing dataset from database(ideal,train and test dataset) 
and analysing dataset and chart generation.

"""
#external import
import yaml
import math 
from scipy import stats
from yaml.loader import SafeLoader 

#internal import
from access_db import DataAccess
from linear_regression_algo import Linear_regr
from chart_report import Chart

class Analysis():
    """
    Class is used for accessing datase and analysing data with chart
    """
    def __init__(self):
        #init all the class from different module
        #access database class
        self.db_access = DataAccess()
        #access linear function algo
        self.linear_regr = Linear_regr()
        #access chart class, for creating chart
        self.chart = Chart()
        self.config_file = {}
        #load all the input files(ideal.csv, test.csv, train.csv) by triggering below function 
        self.config_data = self.load_config_data()
        #copy csv file to database
        self.load_db_from_csv(self.config_data)
        
    #get project Config details from config.yaml
    def load_config_data(self):
        """
        Load Configration file, which consists of path for test analysis files(ideal.csv, test.csv, train.csv)
        """
        
        with open('config.yaml') as file_desc:
            self.config_data = list(yaml.load_all(file_desc, Loader=SafeLoader))
            self.config_data = self.config_data[0]
            #copy all the Config details to dict config_file, config file contains all file path
            for file_key, file_path in self.config_data.items():
                self.config_file[file_key] = self.config_data[file_key]
            
        return self.config_file
        
    def load_db_from_csv(self,config_data):
        """
        Load csv data to database
        """
        # copy train data from train.csv
        train_data_csv = config_data["train_data_csv"]
        # copy ideal data from ideal.csv
        ideal_data_csv = config_data["ideal_data_csv"]
        # copy test data from test.csv
        test_data_csv = config_data["test_data_csv"]
        #stores path of database to be created
        relative_path_to_db = config_data["relative_path_to_db"]
        #copy csv files to database
        self.db_access.load_csv_to_db(train_data_csv,ideal_data_csv,test_data_csv,relative_path_to_db)
        
    def least_square_best_fit_line(self) :
        """
        create best fit line using least square method
        """
        #set path for saving generated html charts
        ideal_chart_save_path = self.config_data["least_square_chart_path"]
        #generate line equation for train data
        self.line_equ_train_data_dict = self.linear_regr.line_equation_generator(self.linear_regr.train_data_df)
        #generate least square error and return root mean square error for easy representation 
        self.least_square_dict = self.linear_regr.least_squares_method(self.linear_regr.ideal_data_df,self.line_equ_train_data_dict)
        #generte barchart for ideal dataset, Y1 to Y50
        self.chart.generate_least_square_barchart(self.least_square_dict,ideal_chart_save_path)
        #generte best fit line for ideal dataset with respect ot train data
        best_fit_line_train_ideal_dict = self.linear_regr.best_fit_line_ideal_func(self.least_square_dict)
        #returns best fit line in ideal dataset with respect to train dataset
        #best_fit_line_train_ideal_dict :{'y1': {'y40': 119.1491}, 'y2': {'y44': 0.0249}, 'y3': {'y3': 0.7004}, 'y4': {'y44': 0.0221}}
        return best_fit_line_train_ideal_dict
    
   
    def generate_ideal_best_fit_chart(self,best_fit_line_dict ):
        """
        Generate best fit line, scatter plot for ideal and train dataset
        """
        # copy path to save train chart 
        train_chart_save_path = self.config_data["data_analysis_chart_path"]
        # copy path to save ideal chart 
        ideal_chart_save_path = self.config_data["least_square_chart_path"]
        # genertate chart for ideal and train dataset, showing similarity in data pattern
        self.chart.generate_bestfit_chart(best_fit_line_dict,ideal_chart_save_path,train_chart_save_path)
    
    def max_deviation_train_ideal_data(self,best_fit_line_dict,line_equ_train_data_dict):
        """
        calculate max deviation in ideal dataset 
        """
        # copy path to save mapped dataset chart 
        max_deviation_chart_save_path = self.config_data["mapping_chart_path"]
        #save dict of max deviation
        max_deviation_train_ideal_dict = {}
        #iterate for ideal test dataset which has a match in train dataset
        #best_fit_line_dict :{'y1': {'y40': 119.1491}, 'y2': {'y44': 0.0249}, 'y3': {'y3': 0.7004}, 'y4': {'y44': 0.0221}}
        for train_y_idx in best_fit_line_dict:
            #copy selected train colunm for each iteration
            train_column_y = train_y_idx
            #iterate for ideal test dataset 
            for ideal_y_idx in best_fit_line_dict[train_y_idx]:
                max_deviation_dict = {}
                #copy selected ideal colunm w.r.t given train column
                ideal_column_y = ideal_y_idx
                #copy respective train data line equation 
                #{'intercept': 233.28429, 'slope': -20.099, 'r_value': -0.8895, 'p_value': 1.598e-137, 'std_err': 0.517}
                train_line_eq = line_equ_train_data_dict[train_column_y]
                
                #calculate max deviation for the given train and ideal dataset
                max_deviation = self.linear_regr.max_deviation_calc(train_column_y,ideal_column_y,train_line_eq,max_deviation_chart_save_path)
                #calculate final max deviation by multiply square root with priviously calculate max deviation 
                max_deviation_train_ideal = max_deviation * math.sqrt(2)
                #rounding off to 4 decimal points
                max_deviation_dict[ideal_y_idx] = round(max_deviation_train_ideal,4)
                #create a dict of max deviation for respective selected ideal dataset
                max_deviation_train_ideal_dict[train_column_y] = max_deviation_dict
        
        return max_deviation_train_ideal_dict
    
    def map_test_data(self,best_fit_line_dict,line_equ_train_data_dict):
        """
        map the test data i=on scatter plot
        """
        # copy path to save test mapped chart 
        mapping_chart_save_path = self.config_data["mapping_chart_path"]
        # calculate the max deviation for ideal dataset with respect to trai dataset
        max_deviation_train_ideal_dict = self.max_deviation_train_ideal_data(best_fit_line_dict,line_equ_train_data_dict)
        #validate the test data with in range of max deviation
        map_test_dataset_dict = self.linear_regr.validate_max_deviation_test_data(max_deviation_train_ideal_dict)
        #Map the test data in chart
        self.chart.generate_map_test_data_chart(map_test_dataset_dict,max_deviation_train_ideal_dict,mapping_chart_save_path)
        


#start of main code
if __name__ == "__main__":
    
   analysis = Analysis()
   #analysis of train dataset and generate chart
   best_fit_line_dict = analysis.least_square_best_fit_line()
   #analysis of ideal dataset and generate chart
   analysis.generate_ideal_best_fit_chart(best_fit_line_dict)
   #analysis of mapper dataset and generate chart
   analysis.map_test_data(best_fit_line_dict,analysis.line_equ_train_data_dict)
   