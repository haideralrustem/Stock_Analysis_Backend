
import os
import pandas as pd

import stock_analysis_data_prep as sadp

import database_specific_functions




# download csv file of 5yr data from yahoo

# pull data from DB (if exists, if not -> proceed to full data processing)

# get most recent data point

# subtract the difference in rows between new csv file and most recent data point

# get the subset new data only

# add that to old data

# restrict time window to 5yr for everything

# calculate metrics (feature) for only the new data rows



def perform_calcuations_for_select_data_rows(data, row_indexes_to_calculate):
  # data should contain new and old data rows.
  # row_indexes_to_calculate should indicate the new rows

  #
  return





#__________________________________

# find diff
def find_unprocessed_data_points(table_from_database, table_from_download):

  unprocessed_data_points_indexes = []

  #database_specific_functions

  return unprocessed_data_points_indexes

#__________________________________

def partial_data_processing(full_data_table):

  # data_source_file_path = os.path.join(sadp.data_folder, "testing_data_sets", "samples", "WELL_2024_08_12.csv")
  data_source_file_path = os.path.join(sadp.data_folder, "testing_data_sets", "samples", "WELL_2024_08_12_prepared.csv")

  full_data_table = sadp.read_csv_all_encodings(file_path=data_source_file_path)
  prepared_data  = sadp.prepare_data_features(full_data_table, row_indexes_to_calculate=[i for i in range(1250, 1255)])

  # prepared_data.to_csv(os.path.join(sadp.data_folder, "testing_data_sets", "samples", "WELL_2024_08_12_prepared.csv") , index=False)
  return

#__________________________________


def full_data_processing():
  return

#___________________________________





if __name__ == "__main__":
  partial_data_processing()

