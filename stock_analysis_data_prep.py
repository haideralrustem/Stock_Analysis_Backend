
import os
import pandas as pd
import datetime


import data_functions
import calculations




data_folder = os.path.join(os.getcwd(), "data")



def read_csv_all_encodings(file_path, dtype=str, keep_default_na=False):
  dataframe = pd.DataFrame()
  try:
    dataframe = pd.read_csv(file_path, dtype=dtype, keep_default_na=keep_default_na, index_col=False)
  except Exception as e:
    print(e)
    try:
      dataframe = pd.read_csv(file_path, dtype=dtype, keep_default_na=keep_default_na, encoding="cp1252", index_col=False)
    except Exception as e:
      print(e)
      try:
        dataframe = pd.read_csv(file_path, dtype=str, keep_default_na=keep_default_na, encoding="ISO-8859-1", index_col=False)
      except Exception as e:
        print(e)

  if not dataframe.empty:
    dataframe = data_functions.remove_unknown_characters(dataframe)

  return dataframe


#_____________________________________________





def prepare_data():

  # for file in os.listdir(data_folder):
  #   if "copy" not in file.lower() or "lock" not in file.lower():
  #     print(file)

  data = read_csv_all_encodings(file_path=os.path.join(data_folder, "PSA.csv"))

  data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

  data['Date'] = pd.to_datetime(data['Date'])

  data = calculations.calculate_MA50_for_data(data)
  data = calculations.calculate_MA200_for_data(data)

  data = calculations.calculate_MA50_MA200_diff_for_data(data)

  # calculate MA_50 trending slope in prior month for each point
  data = calculations.calculate_slope_MA_50_for_previous_N_days(data, N_days_prior=30)
  # calculate MA_50 trending slope in prior 3 months for each point
  data = calculations.calculate_slope_MA_50_for_previous_N_days(data, N_days_prior=90)


  # calculate MA_200 trending slope in prior 1 month for each point
  data = calculations.calculate_slope_MA_200_for_previous_N_days(data, N_days_prior=30)
  # calculate MA_200 trending slope in prior 3 months for each point
  data = calculations.calculate_slope_MA_200_for_previous_N_days(data, N_days_prior=90)


  data = calculations.detect_MA_crossover(data, days_prior_for_detection_window=30)

  data = calculations.calculate_range_diff_for_previous_N_days(data, N_days_prior=7)
  data = calculations.calculate_range_diff_for_previous_N_days(data, N_days_prior=30)

  data = calculations.calculate_MA50_MA200_gap_in_percent(data)


  data = calculations.calculate_percentile_and_standard_dev_for_previous_N_days(data, N_days_prior=30)
  data = calculations.calculate_percentile_and_standard_dev_for_previous_N_days(data, N_days_prior=60)
  data = calculations.calculate_percentile_and_standard_dev_for_previous_N_days(data, N_days_prior=90)

  data = calculations.calculate_abs_percent_change_for_previous_N_days(data, N_days_prior=30)
  data = calculations.calculate_abs_percent_change_for_previous_N_days(data, N_days_prior=60)
  data = calculations.calculate_abs_percent_change_for_previous_N_days(data, N_days_prior=90)
  data = calculations.calculate_abs_percent_change_for_previous_N_days(data, N_days_prior=200)
  # data = calculations.calculate_abs_percent_change_for_previous_N_days(data, N_days_prior=5*360, custom_column_name=)

  data = calculations.calculate_status_next_day_outcome(data)

  return data





if __name__ == "__main__":
  prepare_data()
  pass