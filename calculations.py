
import pandas as pd
import os
import re

from scipy import stats


import data_functions


def calculate_MACD(data):

  num_rows = data.shape[0]

  EMA_values = {} # {row_num: value }
  MACD_values = {}

  def calculate_MACD_for_row(row):

    return row




  if num_rows > 27:

    # we need a loop through every row
    for row_num, row in data.iterrows():

      #if row_num == 20:
      pass




  return data


#__________________________________


def calculate_MA50_for_data(data):
  num_rows = data.shape[0]

  all_values = []


  def calculate_MA50(row):

    row_num = row.name
    closing_value = row['Close']

    print(row_num)

    if row_num >= 49:

      # get the previous 49 points
      previous_49_indexes_start_point = row_num - 49

      # calculate Moving Avg using a subset of values starting at value that is minus 49 points or positions prior
      Moving_Avg = sum( all_values[previous_49_indexes_start_point: row_num+1])  / 50
      row['MA_50'] = Moving_Avg


    # add each closing stock value so we access it later
    all_values.append(closing_value)

    return row




  if num_rows > 50:
    data = data.apply(calculate_MA50, axis=1, result_type='expand')



  return data

#_____________________________________



def calculate_MA100_for_data(data):
  num_rows = data.shape[0]

  all_values = []


  def calculate_MA100(row):

    row_num = row.name
    closing_value = row['Close']

    print(row_num)

    if row_num >= 99:

      # get the previous 99 points
      previous_99_indexes_start_point = row_num - 99

      # calculate Moving Avg using a subset of values starting at value that is minus 99 points or positions prior
      Moving_Avg = sum( all_values[previous_99_indexes_start_point: row_num+1])  / 100
      row['MA_50'] = Moving_Avg


    # add each closing stock value so we access it later
    all_values.append(closing_value)

    return row




  if num_rows > 100:
    data = data.apply(calculate_MA100, axis=1, result_type='expand')



  return data


#___________________________________






def calculate_slope(filtered_data, col_x, col_y, x_is_date=False):

  # Filter DataFrame
  # filtered_data = data[(data[col_x] >= start_date) & (data[col_x] <= end_date)]

  if x_is_date:
    # Convert 'date' to ordinal (numerical representation)
    filtered_data[f'{col_x}_ordinal'] = filtered_data[col_x].apply(lambda x: x.toordinal())
    slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_data[f'{col_x}_ordinal'], filtered_data[col_y])
  else:
    slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_data[col_x], filtered_data[col_y])


  return slope, intercept, r_value, p_value, std_err
#_________________________________


def calculate_slope_date_filter(data, col_x, col_y, start_date_str, end_date_str):


  # Convert strings to datetime objects
  start_date = pd.to_datetime(start_date_str)
  end_date = pd.to_datetime(end_date_str)

  # Convert 'date' to ordinal (numerical representation)
  data[f'{col_x}_ordinal'] = data[col_x].apply(lambda x: x.toordinal())

  # Filter DataFrame
  filtered_data = data[(data[col_x] >= start_date) & (data[col_x] <= end_date)]

  slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_data[f'{col_x}_ordinal'], filtered_data[col_y])


  return slope, intercept, r_value, p_value, std_err


#__________________________________


def calculate_slope_MA_50_for_previous_N_days(data, N_days_prior=30):

  #slope, intercept, r_value, p_value, std_err = calculate_slope(data, col_x='Date', col_y='Close')

  num_rows = data.shape[0]

  all_values = []

  def apply_function(row):

    row_num = row.name
    closing_value = row['Close']

    print(row_num)

    # we need minimum number of days to allow enough data points for N_days_prior. We must start at 49th index, because indexes before that don't ahve MA50 values
    if row_num >= (49 + N_days_prior):
      # get the previous 49 points
      number_previous_days_start_point = row_num - N_days_prior

      filtered_data = data_functions.filter_data_last_n_points(data, target_position=row_num,
                                                               number_of_positions_prior=number_previous_days_start_point)

      # calculate slope for values of MA50 from all the previous points (starting at 50th point) up to this current one
      result = calculate_slope(filtered_data=filtered_data, col_x='Date', col_y='MA_50', x_is_date=True)

      slope, intercept, r_value, p_value, std_err = result
      row[f'slope_MA_50_last_{N_days_prior}_days'] = slope



    # add each closing stock value so we access it later
    all_values.append(closing_value)

    return row


  if num_rows > (50 + N_days_prior):
    data = data.apply(apply_function, axis=1, result_type='expand')

  
  return data


#_____________________________________


#def get_slope_