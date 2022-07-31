
# -*- coding: utf-8 -*-
"""
@Author: Rajesh Kumar Ramadas
@Date: 7/19/2022
@Email: Rajeshkumar1988r@gmail.com
@Github: 

This module performs task related to chart generation 

"""
#external import
import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt

from bokeh.io import show
from bokeh.layouts import row
from bokeh.plotting import figure, output_file, save,show
from bokeh.models import Band, ColumnDataSource,FactorRange
from bokeh.palettes import magma
from scipy import stats

#internal import
from access_db import DataAccess

#create inhetance by adding new class for report 
class Chart:
    """
    class used for chart generation 
    """
    def __init__(self):
        #access data from database, with below class
        self.db_access = DataAccess()
        self.ideal_data_df = self.db_access.get_ideal_data_df()    
        self.train_data_df = self.db_access.get_train_data_df()
        self.test_data_df = self.db_access.get_test_data_df()
        
    
    def generate_bestfit_chart(self,best_fit_line_dict,ideal_chart_save_path,train_chart_save_path):
        """
        analysis best fit line for train and ideal dataset
        """
        #create a dataframe for regression 
        self.create_regression_dataframe(best_fit_line_dict,ideal_chart_save_path,train_chart_save_path)
    
    def create_regression_dataframe(self,best_fit_line_dict,ideal_chart_save_path,train_chart_save_path):
        """
        Create regression dataset for ideal and train dataset
        """
        #iterate through pair of train and ideal function
        for train_data_idx in best_fit_line_dict:
            chart_info = {}
            #dict contains ideal function with residual error or square mean
            chart_info[train_data_idx] = best_fit_line_dict[train_data_idx]
            #create init a dataframe 
            regression_data_df = pd.DataFrame()
            #train function from train dataset(Y1..Y4), 
            train_data_y_data = train_data_idx
            #selected ideal function from ideal dataset pair of train dataset
            ideal_data_y_data = (list(best_fit_line_dict[train_data_idx].items())[0][0])
            #copy train and ideal function pair, for generating regression graph
            regression_data_df['x_train'] = self.train_data_df['x']
            regression_data_df['y_train'] = self.train_data_df[train_data_y_data]
            regression_data_df['x'] = self.ideal_data_df['x']
            regression_data_df['y'] = self.ideal_data_df[ideal_data_y_data]
            #generate regresssion line for train dataset
            self.generate_train_line_chart(regression_data_df,chart_info,train_chart_save_path)
            #generate regresssion line for ideal dataset
            self.generate_ideal_line_chart(regression_data_df,chart_info,ideal_chart_save_path)
            
    
    def generate_train_line_chart(self,regression_data_df,chart_info,train_chart_save_path):
        """
        generate line chart for train dataset
        """
        #copy dependent and independent dataset for generating chart
        x_train = regression_data_df['x_train'].to_numpy()
        y_train = regression_data_df['y_train'].to_numpy()
        
        #regression parameter of given dataset
        slope_train, intercept_train, r_value, p_value, std_err = stats.linregress(x_train, y_train)
        
        _r_value =  (r_value)
        _p_value =  (p_value)
        _r_square = (r_value) * (r_value)
        #calculate 'Y' value for respective "x" value for given slope and intercept
        y_predicted_train = [slope_train * idx + intercept_train  for idx in x_train]
       
        _train_slope =  round(slope_train,4)
        _train_intercept =  round(intercept_train,4)
        
        #create line equation to be displayed in graph
        train_line_eq_str = f" y = {_train_slope}x + {_train_intercept}"
        
        
        for train_data_column_idx_key, ideal_least_square_info_value in chart_info.items():
            #copy 'Y' column from train datatset
            train_data_column_idx = train_data_column_idx_key
            
            
        #tool features to be enabled by Bokeh graph
        TOOLS = "pan,box_zoom,reset,save,zoom_out,zoom_in"
        #init the graph
        train_p1 = figure(tools=TOOLS,title=f"Train data function ({train_data_column_idx})",sizing_mode="stretch_width", max_width=500, height=500, x_axis_label='x ', y_axis_label=f'y = {train_data_column_idx}')
        #display Graph legend for line equation
        train_p1.line(x_train, y_predicted_train, legend_label=f"line eq : {train_line_eq_str}",line_color="red", line_width=2)
        train_p1.circle(x_train, y_train, fill_color="red", size=2)
        #display Graph legend for r value
        train_p1.line(x_train, y_predicted_train, legend_label=f"r_value : {round(_r_value,5)}",line_color="red", line_width=2)
        train_p1.circle(x_train, y_train, fill_color="red", size=2)
        #display Graph legend for r-square value
        train_p1.line(x_train, y_predicted_train, legend_label=f"r_square : {round(_r_square,5)}",line_color="red", line_width=2)
        train_p1.circle(x_train, y_train, fill_color="red", size=2)
        #display Graph legend for p value
        train_p1.line(x_train, y_predicted_train, legend_label=f"p_value : {round(_p_value,5)}",line_color="red", line_width=2)
        train_p1.circle(x_train, y_train, fill_color="red", size=2)
        #display Graph legend for std err
        train_p1.line(x_train, y_predicted_train, legend_label=f"std_err : {round(std_err,5)}",line_color="red", line_width=2)
        train_p1.circle(x_train, y_train, fill_color="red", size=2)
        # create name for save html file
        train_chart_filename = train_chart_save_path + f"train_{train_data_column_idx}.html"
        #save html file
        output_file(filename = train_chart_filename, title=f"Train data({train_data_column_idx}) ")
        #show graph in brower
        try:  
            show(row(train_p1)) 
        except Exception as _error:
            print(f"Error: GRAPH GENERATED UNSUCCESSFUL: {_error}")  
            
    def generate_least_square_barchart(self,least_square_dict,ideal_chart_save_path):
       
        """
        generate bar chart for identiifng least square or residual error
        """
        #iterate dict contain train and ideal pair, 
        for train_idx, ideal_least_square in least_square_dict.items():
            ideal_idx_lst = []
            least_square_value_lst = []
            color_lst = []
            #iterate dict contain ideal with least square,
            for ideal_idx, least_square_value in ideal_least_square.items(): 
                #list of ideal 'y' with its least square
                #ideal_idx_lst : 0.6283:y50
                ideal_idx_lst.append(f"{round((least_square_value),4)}:{ideal_idx}") 
                #list of least square
                least_square_value_lst.append(round((least_square_value),4))
                #color blue for all the bar
                color_lst.append("blue")
                
            #find the minimum least square value
            index = least_square_value_lst.index(min(least_square_value_lst))
            #color the bar with red in case of minimum least square value
            color_lst[index] = "red"
            #tool feature for graph
            TOOLS = "pan,box_zoom,reset,save,zoom_out,zoom_in"
            #init the  graph
            p = figure(y_range=FactorRange(factors=ideal_idx_lst), max_width=500, height=500, title=f"least square for ideal function w.r.t  train dataset : {train_idx}",
            toolbar_location=None, tools=TOOLS)
            #init horizontal bar chart
            p.hbar(y=ideal_idx_lst, right=least_square_value_lst,height = 0.5,color = color_lst)
            #create file name
            file_header = f"least_square_ideal_dataset_vs_train_dataset_{train_idx}"
            chart_filename = ideal_chart_save_path + file_header+".html"
            #save file 
            output_file(filename=chart_filename, title=file_header )
            #show graph in brower 
            try:  
                show(row(p)) 
                pass
            except Exception as _error:
                print(f"Error: GRAPH GENERATED UNSUCCESSFUL: {_error}")
                
            
    def generate_ideal_line_chart(self,regression_data_df,chart_info,ideal_chart_save_path):
        """
        plot chart from ideal and train dataset using least square method
        """
        #copy train and ideal pair dataset 
        x_train = regression_data_df['x_train'].to_numpy()
        y_train = regression_data_df['y_train'].to_numpy()
        x_ideal = regression_data_df['x'].to_numpy()
        y_ideal = regression_data_df['y'].to_numpy()
        
        #regression parameter for ideal and train pair dataset
        slope_train, intercept_train, r_value, p_value, std_err = stats.linregress(x_train, y_train)
        slope_ideal, intercept_ideal, r_value, p_value, std_err = stats.linregress(x_ideal, y_ideal)
        
        #create an predicted or best fit line for train and ideal pair dataset
        y_predicted_train = [slope_train * idx + intercept_train  for idx in x_train]
        y_predicted_ideal = [slope_ideal * idx + intercept_ideal  for idx in x_ideal]
        
        line_eq_train_ideal_dict = self.generate_line_equation(chart_info)
        #copy slope and intercept for ideal function
        _line_eq_train_dict = line_eq_train_ideal_dict['ideal_function']
         #copy slope and intercept for train function
        _line_eq_ideal_dict = line_eq_train_ideal_dict['train_function']  
        
        #copy slope and intercept for train and ideal data pair
        _train_slope = _line_eq_train_dict['slope'] 
        _train_intercept = _line_eq_train_dict['intercept']
        _ideal_slope = _line_eq_ideal_dict['slope'] 
        _ideal_intercept = _line_eq_ideal_dict['intercept'] 
        
        #create line function for ideal and train pair
        ideal_line_eq_str = f" y = {_ideal_slope}x + {_ideal_intercept}"
        train_line_eq_str = f" y = {_train_slope}x + {_train_intercept}"
        
        #iterate dict containing ideal and train pair datset with least square 
        for train_data_column_idx_key, ideal_least_square_info_value in chart_info.items():
            #copy train column
            train_data_column_idx = train_data_column_idx_key
            for ideal_data_column_idx_key, ideal_least_square_value in ideal_least_square_info_value.items():
                #copy ideal column
                ideal_data_column_idx = ideal_data_column_idx_key
                #copy least square for ideal function
                ideal_least_square = round(ideal_least_square_value,4)
        
        #tool features for Bokeh
        TOOLS = "pan,box_zoom,reset,save,zoom_out,zoom_in"
        #init graph for train dataset
        ideal_p1 = figure(tools=TOOLS,title=f"Train data function ({train_data_column_idx})",sizing_mode="stretch_width", max_width=500, height=500, x_axis_label='x ', y_axis_label=f'y = {train_data_column_idx}')
        #init graph for ideal dataset
        ideal_p2 = figure(tools=TOOLS,title=f"Ideal data function ({ideal_data_column_idx})",sizing_mode="stretch_width", max_width=500, height=500, x_axis_label='x', y_axis_label=f'y = {ideal_data_column_idx}')
        #init graph for Train Vs Ideal data function
        ideal_p3 = figure(tools=TOOLS,title=f"Train Vs Ideal data function ({train_data_column_idx} & {ideal_data_column_idx})",sizing_mode="stretch_width", max_width=500, height=500, x_axis_label='x', y_axis_label=f'y = {train_data_column_idx} & {ideal_data_column_idx} ')
        #display Graph legend for line equation train dataset
        ideal_p1.line(x_train, y_predicted_train, legend_label=f"line eq : {train_line_eq_str}",line_color="red", line_width=2)
        ideal_p1.circle(x_train, y_train, fill_color="red", size=2)
        #display Graph legend for line equation ideal dataset
        ideal_p2.line(x_ideal, y_predicted_ideal, legend_label=f"line eq : {ideal_line_eq_str}",line_color="green", line_width=2)
        ideal_p2.circle(x_ideal, y_ideal, fill_color="green", size=2)
        #display Graph legend for line equation ideal vs train dataset
        ideal_p3.line(x_train, y_predicted_train, legend_label=f"line eq : {train_line_eq_str}",line_color="red", line_width=2)
        ideal_p3.circle(x_train, y_train, fill_color="red", size=2)
        ideal_p3.line(x_ideal, y_predicted_ideal, legend_label=f"line eq : {ideal_line_eq_str},least square :{ideal_least_square}",line_color="green", line_width=2)
        ideal_p3.circle(x_ideal, y_ideal, fill_color="green", size=2)
        #name the save file path      
        ideal_chart_filename = ideal_chart_save_path + f"train_{train_data_column_idx}_ideal_{ideal_data_column_idx}.html"
        #save the generated graph 
        output_file(filename=ideal_chart_filename, title=f"Train data({train_data_column_idx}) vs Ideal data function({ideal_data_column_idx}) ")
        #display the graph in browser
        try:  
            show(row(ideal_p1, ideal_p2, ideal_p3)) 
        except Exception as _error:
            print(f"Error: GRAPH GENERATED UNSUCCESSFUL: {_error}")

    def generate_line_equation(self,chart_info):
        """
        generate line equation for given dataset
        """
        #dict for line equation(slope and intercept)
        line_eq_train_ideal_dict = {}
        line_eq_ideal_dict = {}
        line_eq_train_dict = {}
        #iterate through the ideal and train data function with least square
        for train_data_column_idx_key, ideal_least_square_info_value in chart_info.items():
            #copy train 'y' column
            _train_data_column_y = train_data_column_idx_key
            for ideal_data_column_idx_key, ideal_least_square_value in ideal_least_square_info_value.items():
                #copy ideal 'y' column
                _ideal_data_column_y = ideal_data_column_idx_key
        #copy dataset for ideal and train pair
        ideal_x = self.ideal_data_df['x'].to_numpy()
        ideal_y = self.ideal_data_df[_ideal_data_column_y].to_numpy()
        train_x = self.train_data_df['x'].to_numpy()
        train_y = self.train_data_df[_train_data_column_y].to_numpy()
        #generate regression pair for ideal and train
        ideal_slope, ideal_intercept, r_value, p_value, std_err = stats.linregress(ideal_x, ideal_y)
        train_slope, train_intercept, r_value, p_value, std_err = stats.linregress(train_x, train_y)
        
        #copy slope and intercept for ideal and train pair
        line_eq_train_dict['slope'] =  round(train_slope,4)
        line_eq_train_dict['intercept'] = round(train_intercept,4)
        line_eq_ideal_dict['slope'] = round(ideal_slope,4)
        line_eq_ideal_dict['intercept'] = round(ideal_intercept,4)
        
        #line equation(slope and intercept) for ideal and train function
        line_eq_train_ideal_dict['ideal_function'] = line_eq_train_dict
        line_eq_train_ideal_dict['train_function'] = line_eq_ideal_dict
        
        return line_eq_train_ideal_dict
         
    def generate_mapper_graph(self,mapper_point_data,max_deviation_train_ideal_dict,mapping_chart_save_path):
        """
        Generate mapped chart for test dataset, within range of max deviation 
        """
        df_in_range_point = pd.DataFrame()
        x = []
        y = []
        mapper_point_data_tag  = ""
        # mapper_point_data contains information related to mapping points on chart
        # {'y2_y44': [{'x': -9.8, 'y': 0.04520088, 'ideal_column': 'y44', 'y_upperband': 0.059579999999999994, 'y_lowerband': -0.07782},
        # {'x': -0.3, 'y': -0.026510444, 'ideal_column': 'y44', 'y_upperband': 0.05863, 'y_lowerband': -0.07876999999999999},
        # {'x': 3.5, 'y': -0.012937355, 'ideal_column': 'y44', 'y_upperband': 0.058249999999999996, 'y_lowerband': -0.07915}]}
        for mapper_point_data_idx,mapper_point_data_value in mapper_point_data.items():
            mapper_point_data_tag = mapper_point_data_idx
            # mapper_point_data[mapper_point_data_idx] contains dict of tester points and repective info about band and ideal function
            #[{'x': -9.8, 'y': 0.04520088, 'ideal_column': 'y44', 'y_upperband': 0.059579999999999994, 'y_lowerband': -0.07782},
            # {'x': -0.3, 'y': -0.026510444, 'ideal_column': 'y44', 'y_upperband': 0.05863, 'y_lowerband': -0.07876999999999999},
            # {'x': 3.5, 'y': -0.012937355, 'ideal_column': 'y44', 'y_upperband': 0.058249999999999996, 'y_lowerband': -0.07915}]
            for in_range_idx in mapper_point_data[mapper_point_data_idx]:
                # test data points
                x.append(in_range_idx['x'])
                y.append(in_range_idx['y'])
                #ideal column
                ideal_column_y = in_range_idx['ideal_column']
        #create a dict of tester data points
        dict = {'x':x,'y':y}
        
        df_in_range_point = pd.DataFrame(dict)
        
        #all the test data points
        """
             x         y
        0 -9.8  0.045201
        1 -0.3 -0.026510
        2  3.5 -0.012937
        """
        x =  df_in_range_point['x'].to_numpy()
        y =  df_in_range_point['y'].to_numpy()
        
        ideal_col_y = 0
        max_deviation = 0
        max_deviation_tag = ''
        
        #iterate through dict of max deviation calcuated
        for map_idx in max_deviation_train_ideal_dict:
            #iterate through dict of ideal function and its max dec=viation
            for ideal_idx in max_deviation_train_ideal_dict[map_idx]:
                #ideal function for max deviation
                ideal_col_y = ideal_idx 
                #max deviation
                max_deviation = max_deviation_train_ideal_dict[map_idx][ideal_idx]
                #generate train and ideal function pair(y1_y40)
                max_deviation_tag = f'{map_idx}_{ideal_idx}'
                
                
            #copy independent and dependent variables from ideal dataset
            x_data_pt = self.ideal_data_df['x'].to_numpy()
            y_data_pt = self.ideal_data_df[ideal_col_y].to_numpy()
              
            if (mapper_point_data_tag == max_deviation_tag):
        
                df = pd.DataFrame()
                #copy independent and dependent variables
                df['x'] = x_data_pt
                df['y'] = y_data_pt
                #regression parameter
                ideal_slope, ideal_intercept, r_value, p_value, std_err = stats.linregress( df['x'], df['y'])
     
                #generate best fit line for ideal function
                df['y_bestfit'] = (ideal_slope * df['x']) + ideal_intercept
                #dataframe for upper and lower band
                df['y_upperband'] = df['y_bestfit'] + max_deviation
                df['y_lowerband'] = df['y_bestfit'] - max_deviation
                
                source = ColumnDataSource(df.reset_index())
                #create tool feature
                TOOLS = "pan,box_zoom,reset,save,zoom_out,zoom_in"
                
                #init graph with dimension
                p = figure(sizing_mode="stretch_width", max_width=500, height=500,tools=TOOLS)
                #create scatter plot
                p.scatter(x='x', y='y_bestfit', line_color=None, fill_alpha=0.5, size=5, source=source)
                #create upper and lowe band
                band = Band(base='x', lower='y_lowerband', upper='y_upperband', source=source, level='underlay',
                    fill_alpha=1.0, line_width=1, line_color='black')
                
                #graph features 
                p.add_layout(band)
                p.title.text = f"x vs {ideal_col_y}, max deviation :{max_deviation}"
                p.xgrid[0].grid_line_color=None
                p.ygrid[0].grid_line_alpha=0.5
                p.xaxis.axis_label = 'X'
                p.yaxis.axis_label = 'Y'
                size = 10
                color = magma(256)
                file_header = f"mapped_testdata_train_function_{map_idx}_ideal_function_{ideal_col_y}"
                #generate legend for 
                p.scatter(x, y,size = size, color = "red",legend_label="Mapped testdata within max deviation") 
                #file path to save graph 
                chart_filename = mapping_chart_save_path + file_header+".html"
                #save graph in save path
                output_file(filename=chart_filename, title=file_header )
                #show the graph in browser.    
                try:  
                    show(row(p)) 
                    pass
                except Exception as _error:
                    print(f"Error: GRAPH GENERATED UNSUCCESSFUL: {_error}")
    
    def generate_map_test_data_chart(self,map_data_set_dict,max_deviation_train_ideal_dict,mapping_chart_save_path):
        """
        calculate max deviation for ideal and train data
        """ 
        max_deviation_tag_list = []
        #iterate dict of deviation of ideal and train 
        for train_col_y in max_deviation_train_ideal_dict:
            for ideal_col_y in max_deviation_train_ideal_dict[train_col_y]:
                #create a tag list for train and respective pair of ideal function
                max_deviation_tag_list.append(f"{train_col_y}_{ideal_col_y}")
        
        #iterate through the deviation tag example '0.0085(deviation) : 19.9(test point)'
        for max_deviation_idx in max_deviation_tag_list:
        
            in_range_data_points_list = []
            in_range_data_points_dict = {}
            #iterate through the dict containing test dataset, alonf side ideal function and band dataset
            #3.5: {'x': 3.5, 'y': -0.012937355, 'ideal_column': 'y44', 'y_upperband': 0.04925, 'y_lowerband': -0.07015}
            for in_range_max_max_deviation in map_data_set_dict[max_deviation_idx]:
                in_range_data_points = (map_data_set_dict[max_deviation_idx][in_range_max_max_deviation])
                in_range_data_points_list.append(in_range_data_points)
            #list consist of points with in range of upper and lower limit
            in_range_data_points_dict[max_deviation_idx] = in_range_data_points_list
            #generate the test datapoint graph
            self.generate_mapper_graph(in_range_data_points_dict,max_deviation_train_ideal_dict,mapping_chart_save_path)
            
    def generate_max_deviation_graph(self,max_deviation_df,max_deviation,train_column_y,ideal_column_y, save_chart):
        """
        Generate chart for max deviation (horizontal bar chart)
        """ 
        x_axis_list = []
        #create string list of 'X' axis points
        x_axis = max_deviation_df['x'].tolist()
        x_axis = list(map(str,x_axis))
        
        #list of deviation for ideal functiom at different points[0-400]
        deviation = max_deviation_df['deviation'].tolist()
        #iterate list of deviation and create a deviation and test point pair
        for idx in range(0,len(max_deviation_df['deviation'])):
            # '0.0085(deviation) : 19.9(test point)' 
            x_axis_list.append(f"{deviation[idx]} : {x_axis[idx]}")     
            
        #set default color blue
        color = ['blue'] * 400
        #find the max deviation from the list
        index_max_deviation = deviation.index(max(deviation))
        #change to clor to red in case of max deviation
        color[index_max_deviation] = 'red'
        #tool feature for graph
        TOOLS = "pan,box_zoom,reset,save,zoom_out,zoom_in"
        #init the graph feature
        p = figure(y_range=FactorRange(factors=x_axis_list), max_width=5000, height=5000, title=f"Max Deviation : train dataset: {train_column_y} vs Ideal dataset{ideal_column_y}",
        toolbar_location=None, tools=TOOLS)
        #generate horizontal bar
        p.hbar(y=x_axis_list, right=deviation,height = 0.2,color = color)
        #generate file name
        file_header = f"max_deviation_train_data_{train_column_y}_vs_ideal function_{ideal_column_y}"
        chart_filename = save_chart + file_header+".html"
        output_file(filename=chart_filename, title=file_header )
        #show graph in browers
        try:  
            show(row(p)) 
            pass
        except Exception as _error:
            print(f"Error: GRAPH GENERATED UNSUCCESSFUL: {_error}")
        
    