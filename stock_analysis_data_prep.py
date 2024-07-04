
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

  data = calculations.calculate_MA50(data)

  return data





if __name__ == "__main__":
  prepare_data()
  pass