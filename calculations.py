
import pandas as pd
import os
import re






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


def calculate_MA50(data):
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