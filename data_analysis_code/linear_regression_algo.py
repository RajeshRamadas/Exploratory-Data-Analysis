
# -*- coding: utf-8 -*-
"""
@Author: Rajesh Kumar Ramadas 
@Date: 7/19/2022
@Email: Rajeshkumar1988r@gmail.com
@Github: 

This module performs task related to analysis and consist of Algorith form least square method,
max deviation etc
 
"""
#external import
import pandas as pd
import numpy as np
from scipy import stats
import math

#internal import 
from access_db import DataAccess
from chart_report import Chart

class Linear_regr():
    """
    class Linear_regr is used perform calculation related to linear regression like least square method etc
    """
    def __init__(self):
        self.db_access = DataAccess()
        self.ideal_data_df = self.db_access.get_ideal_data_df()    
        self.train_data_df = self.db_access.get_train_data_df()
        self.test_data_df = self.db_access.get_test_data_df()
    
    def max_deviation_calc(self,train_column_y,ideal_column_y,train_line_eq,max_dev_save_chart):
        """
        calculate max deviation between ideal and train dataset
        """
        #create init max deviation dataframe
        self.max_deviation_df = pd.DataFrame()
        #copy independent variable data from 'X' column from train dataset
        self.max_deviation_df['x'] = self.train_data_df['x']
        #copy dependent variable data from 'Y' train dataset
        self.max_deviation_df[train_column_y] = self.train_data_df[train_column_y]
        #copy dependent variable data from 'Y' ideal dataset
        self.max_deviation_df[ideal_column_y] = self.ideal_data_df[ideal_column_y]
        # calculate 'Y' best fit line using slope and intercept from train dataset
        self.max_deviation_df['y(bestfit)']   = (train_line_eq['slope'] * self.train_data_df['x']) + train_line_eq['intercept']
        # calculate deviation/ difference of best fit line(train data) and ideal dataset pair
        self.max_deviation_df['deviation']    =  round(abs( self.ideal_data_df[ideal_column_y] - self.max_deviation_df['y(bestfit)']),4)
        #calculate max deviation from list of deviation
        max_deviation = self.max_deviation_df['deviation'].max()
        #rounding off to 4 decimal point
        max_deviation = round(max_deviation,4)
        #generate graph for max deviation and mapping test data
        Chart.generate_max_deviation_graph(self,self.max_deviation_df ,max_deviation,train_column_y,ideal_column_y,max_dev_save_chart)
           
        return max_deviation
               
    def line_equation_generator(self, train_data_df ):
        """
        Line equation for train data is calcuated 
        """
        # dict for line equation of train dataset
        train_data_line_equation_dict = {}
        #Iterate all the column from X, Y1,y2,y3,y4       
        for col_name in train_data_df.columns[:]:
            #Check for dependent variable 
            if col_name.find('y') != -1:
                
                slope_intercept_dict = {}

                dependant_dataset = train_data_df[col_name]
                independent_dataset = train_data_df['x']
                #copy slope, intercept, r_value, p_value, std_err
                slope, intercept, r_value, p_value, std_err = stats.linregress(independent_dataset, dependant_dataset)
                
                #copy all regression parameter to dict
                slope_intercept_dict["intercept"] = intercept
                slope_intercept_dict["slope"] = slope
                slope_intercept_dict["r_value"] = r_value
                slope_intercept_dict["p_value"] = p_value
                slope_intercept_dict["std_err"] = std_err
                #make dict of line equation for all the train dataset
                train_data_line_equation_dict[col_name] = slope_intercept_dict
        
        return train_data_line_equation_dict
        
    def least_squares_method(self,ideal_data_df,line_equ_train_data_dict):
        """
        function to calcuate the least square method
        """
        least_square_dict = {}
        #iterate all the line equation of the train dataset
        for line_eq_train_data_idx in line_equ_train_data_dict:
            #copy slope and intercept for the train dataset
            intercept = line_equ_train_data_dict[line_eq_train_data_idx]['intercept']
            slope = line_equ_train_data_dict[line_eq_train_data_idx]['slope']
            
            ideal_least_square_dict = {}
            #iterate for all the ideal dataset(y1 to y50)
            for col_name in ideal_data_df.columns[:]:
                
                least_square_df = pd.DataFrame() 
                #copy 'x' column from ideal dataset
                least_square_df['x'] = ideal_data_df['x']
                #check for 'y' column
                if col_name.find('y') != -1:
                    least_square = 0
                    #ideal dataset for finding Residual error 
                    least_square_df['y_actual'] = ideal_data_df[col_name]
                    #best fit line for train data set
                    least_square_df['y_predicted'] = (least_square_df['x'] * slope) + intercept
                    #calculate Residual error w.r.t ideal data point
                    least_square_df['Residual_err'] = least_square_df['y_actual'] - least_square_df['y_predicted']
                    #calculate square error 
                    least_square_df['Residual_err_square'] = least_square_df['Residual_err'] * least_square_df['Residual_err']
                    #calculate sum of all the square error 
                    least_square = least_square_df['Residual_err_square'].sum()
                    #calculate mean of summation of square error
                    Mean_least_square = least_square/len(least_square_df['y_actual'])
                    #calculate root of average square error 
                    Root_mean_square_error = math.sqrt(Mean_least_square)
                    #ideal_least_square_dict[col_name] = least_square
                    ideal_least_square_dict[col_name] = Root_mean_square_error
            #make dict pair of matching train dataset and ideal dataset
            least_square_dict[line_eq_train_data_idx] = ideal_least_square_dict
            
        return least_square_dict

    def best_fit_line_ideal_func(self,least_square_dict):
        """
        Function to identifiy best fit ideal function w.r.t train function
        """
        best_fit_line = {}
        #iterate through the list of square error 
        for train_line in least_square_dict:
            # find the minimum from the list of square error  
            least_square = min(least_square_dict[train_line].values())
            # loop through ideal function to find match for train function
            least_square_key_value = {key:value for key, value in least_square_dict[train_line].items() if value == least_square}
            #create dict of all the min square error for graph creation
            best_fit_line[train_line] = least_square_key_value

        return best_fit_line
    
    def validate_max_deviation_test_data(self,max_deviation_train_ideal_dict):
        """
        function calclates the test data within the range of max deviation 
        """
        mapping_test_data_dict = {}
        #iterate for dict of max deviation
        #max_deviation_train_ideal_dict : {'y1': {'y40': 373.8001}, 'y2': {'y44': 0.0687}, 'y3': {'y3': 1.5206}, 'y4': {'y44': 0.0597}}
        for max_deviation_idx in max_deviation_train_ideal_dict:
            #iterate for all the selcted ideal dataset 
            for ideal_col_y_idx in (max_deviation_train_ideal_dict[max_deviation_idx]):
                mapping_data_set_dict = {}
                ideal_col_y = ideal_col_y_idx
                # copy max deviation
                max_deviation = max_deviation_train_ideal_dict[max_deviation_idx][ideal_col_y]
                
                max_deviation_mapper_df = pd.DataFrame() 
            
                ideal_x = self.ideal_data_df['x'].to_numpy()
                ideal_y = self.ideal_data_df[ideal_col_y].to_numpy()
                #generate all linear regression parmater
                ideal_slope, ideal_intercept, r_value, p_value, std_err = stats.linregress(ideal_x, ideal_y)

                ideal_intercept = round(ideal_intercept,4)
                ideal_slope = round(ideal_slope,4)
                # copy independent and dependent variables
                max_deviation_mapper_df['x'] = self.ideal_data_df['x']
                max_deviation_mapper_df[ideal_col_y] = self.ideal_data_df[ideal_col_y]
                # generate best fit line with ideal function
                max_deviation_mapper_df['y_bestfit'] = (ideal_slope*self.ideal_data_df['x']) + ideal_intercept
                #create upper and lower band with support of max deviation
                max_deviation_mapper_df['y_upperband'] = max_deviation_mapper_df['y_bestfit'] + max_deviation
                max_deviation_mapper_df['y_lowerband'] = max_deviation_mapper_df['y_bestfit'] - max_deviation
                
                max_deviation_mapper_df.set_index('x', inplace = True)
                
                for index, test_data_row in self.test_data_df.iterrows():
                    # test data points
                    x_index =test_data_row['x']
                    y_value =test_data_row['y']
                    ideal_data = (max_deviation_mapper_df.loc[x_index])
                    #mapping test data with in range of max deviation
                    if (y_value >= ideal_data['y_lowerband'] and y_value <= ideal_data['y_upperband']):
                        mapping_data_point_dict = {}
                        mapping_data_point_dict["x"] = x_index
                        mapping_data_point_dict["y"] = y_value
                        mapping_data_point_dict["ideal_column"] = ideal_col_y_idx
                        mapping_data_point_dict["y_upperband"] = ideal_data['y_upperband']
                        mapping_data_point_dict["y_lowerband"] = ideal_data['y_lowerband']
                        #create a dict of test data with in range of max deviation
                        mapping_data_set_dict[x_index] = mapping_data_point_dict
            #dict of all the test data point for all ideal functions
            mapping_test_data_dict[max_deviation_idx+"_"+ideal_col_y_idx] = mapping_data_set_dict
                
        return mapping_test_data_dict

 