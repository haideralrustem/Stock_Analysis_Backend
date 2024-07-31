
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



def calculate_MA200_for_data(data):
  num_rows = data.shape[0]

  all_values = []


  def calculate_MA200(row):

    row_num = row.name
    closing_value = row['Close']

    print(row_num)

    if row_num >= 199:

      # get the previous 99 points
      previous_99_indexes_start_point = row_num - 199

      # calculate Moving Avg using a subset of values starting at value that is minus 99 points or positions prior
      Moving_Avg = sum( all_values[previous_99_indexes_start_point: row_num+1])  / 200
      row['MA_200'] = Moving_Avg


    # add each closing stock value so we access it later
    all_values.append(closing_value)

    return row




  if num_rows > 100:
    data = data.apply(calculate_MA200, axis=1, result_type='expand')



  return data


#___________________________________


def calculate_MA50_MA200_diff_for_data(data):
  num_rows = data.shape[0]

  all_values = []


  def calculate_diff(row):

    row_num = row.name
    closing_value = row['Close']

    print(row_num)

    if row_num >= 199:
      row['MA_50_MA_200_diff'] = row['MA_50'] - row['MA_200']

    else:
      row['MA_50_MA_200_diff'] = None



    return row




  if num_rows >= 200:
    data = data.apply(calculate_diff, axis=1, result_type='expand')



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
    if row_num >= (49 + N_days_prior-1):
      # get the previous 49 points
      number_previous_days_start_point = row_num - N_days_prior

      filtered_data = data_functions.filter_data_last_n_points(data, target_position=row_num,
                                                               number_of_positions_prior=N_days_prior-1)

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

def calculate_slope_MA_200_for_previous_N_days(data, N_days_prior=30):
  # slope, intercept, r_value, p_value, std_err = calculate_slope(data, col_x='Date', col_y='Close')

  num_rows = data.shape[0]

  all_values = []

  def apply_function(row):

    row_num = row.name
    closing_value = row['Close']

    print(row_num)

    # we need minimum number of days to allow enough data points for N_days_prior. We must start at 49th index, because indexes before that don't ahve MA50 values
    if row_num >= (199 + N_days_prior - 1):
      # get the previous 49 points
      number_previous_days_start_point = row_num - N_days_prior

      filtered_data = data_functions.filter_data_last_n_points(data, target_position=row_num,
                                                               number_of_positions_prior=N_days_prior - 1)

      # calculate slope for values of MA50 from all the previous points (starting at 50th point) up to this current one
      result = calculate_slope(filtered_data=filtered_data, col_x='Date', col_y='MA_200', x_is_date=True)

      slope, intercept, r_value, p_value, std_err = result
      row[f'slope_MA_200_last_{N_days_prior}_days'] = slope

    # add each closing stock value so we access it later
    all_values.append(closing_value)

    return row

  if num_rows > (200 + N_days_prior):
    data = data.apply(apply_function, axis=1, result_type='expand')

  return data


#____________________________________________


def detect_MA_crossover(data, days_prior_for_detection_window=30):

  num_rows = data.shape[0]

  all_values = []

  def apply_detect_MA_crossover_function(row):
    row_num = row.name
    closing_value = row['Close']



    # we need minimum number of days to allow enough data points for days_prior_for_detection_window.
    # We must start at 199th index, because indexes before that don't have MA200 values
    if row_num >= (199 + days_prior_for_detection_window - 1):
      # get the previous 49 points
      number_previous_days_start_point = row_num - days_prior_for_detection_window

      filtered_data = data_functions.filter_data_last_n_points(data, target_position=row_num,
                                                               number_of_positions_prior=days_prior_for_detection_window - 1)

      # detect if MA50 is higher than MA200 at any point during the days_prior_for_detection_window period

      row['MA_Crossover'] = 0

      # detect points where MA50 changes from

      occurrences_where_MA50_higher_MA200 = []
      occurrences_where_MA50_lower_MA200 = []


      def count_MA50_MA200_occurrences(subset_row):

        position = subset_row.name
        # get occurrences_where_MA50_higher_MA200
        if subset_row['MA_50'] > subset_row['MA_200']:
          # record
          occurrences_where_MA50_higher_MA200.append(position)


        if subset_row['MA_50'] < subset_row['MA_200']:
          occurrences_where_MA50_lower_MA200.append(position)


        return


      filtered_data.apply(count_MA50_MA200_occurrences, axis=1, result_type='expand')

      # make decision about how many instances we detect of MA50 > MA200 or MA50 < MA200
      MA_Crossover = 0
      tanked_stock = 0



      last_position_of_MA50_higher_MA200 = -1
      last_position_of_MA50_lower_MA200 = -1


      if len(occurrences_where_MA50_higher_MA200) > 0:
        # get the highest postion (last position)
        last_position_of_MA50_higher_MA200 = max(occurrences_where_MA50_higher_MA200)

      else:
        # stock is tanked
        MA_Crossover = 0
        tanked_stock = 1



      if len(occurrences_where_MA50_lower_MA200) > 0:
        # get the highest postion (last position)
        last_position_of_MA50_lower_MA200 = max(occurrences_where_MA50_lower_MA200)
      else:
        # there is no crossover happening in that time period, but stock is healthy, since there are no points MA50 lower than MA200
        MA_Crossover = 0
        tanked_stock = 0
        pass



      if last_position_of_MA50_higher_MA200 != -1 and last_position_of_MA50_lower_MA200 != -1:

        if last_position_of_MA50_higher_MA200 > last_position_of_MA50_lower_MA200:
          # good case. MA50 was recorded to be higher at a point later than when it was lower
          MA_Crossover = 1

        if last_position_of_MA50_higher_MA200 < last_position_of_MA50_lower_MA200:
          MA_Crossover = 0
          tanked_stock = 1


      row['MA_Crossover'] = MA_Crossover
      row['tanked_stock'] = tanked_stock


    return row


  if num_rows > (200 + days_prior_for_detection_window):
    data = data.apply(apply_detect_MA_crossover_function, axis=1, result_type='expand')



  return data


#__________________________________________





def calculate_range_diff_for_previous_N_days(data, N_days_prior=7, custom_column_name=""):
  # slope, intercept, r_value, p_value, std_err = calculate_slope(data, col_x='Date', col_y='Close')

  num_rows = data.shape[0]

  all_values = []

  def apply_function(row):

    row_num = row.name
    closing_value = row['Close']



    # we need minimum number of days to allow enough data points for N_days_prior.
    if row_num >= ( N_days_prior ):

      number_previous_days_start_point = row_num - N_days_prior

      filtered_data = data_functions.filter_data_last_n_points(data, target_position=row_num,
                                                               number_of_positions_prior=N_days_prior )

      average = filtered_data['Close'].mean()

      max_value = filtered_data['Close'].max()
      min_value = filtered_data['Close'].min()


      # Get the indices of the max and min values
      max_index = filtered_data['Close'].idxmax()
      min_index = filtered_data['Close'].idxmin()


      range_diff_percentage = 0

      if max_index > min_index:  # huge increase in the past N days, so positive diff
        range_diff_percentage = (max_value - min_value) / min_value


      if max_index < min_index:  # huge decrease in the past N days, so negative diff
        range_diff_percentage = (min_value - max_value) / max_value



      row[f"range_diff_percentage_past_{N_days_prior}_days"] = range_diff_percentage



    return row

  if num_rows > (N_days_prior):
    data = data.apply(apply_function, axis=1, result_type='expand')

  return data

#________________________________________________



def calculate_MA50_MA200_gap_in_percent(data):
  # slope, intercept, r_value, p_value, std_err = calculate_slope(data, col_x='Date', col_y='Close')

  num_rows = data.shape[0]

  all_values = []

  def apply_function(row):

    row_num = row.name
    closing_value = row['Close']

    row['MA50_MA200_gap_in_percent'] = None

    # we need minimum number of days to allow enough data points for N_days_prior.
    if row_num >= ( 199  ):
      row['MA50_MA200_gap_in_percent'] = row['MA_50_MA_200_diff'] / row['MA_200']


    return row



  if num_rows > (200):
    data = data.apply(apply_function, axis=1, result_type='expand')

  return data

#________________

