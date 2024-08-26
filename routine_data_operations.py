
import os
import pandas as pd

import stock_analysis_data_prep as sadp




# download csv file of 5yr data from yahoo

# pull data from DB (if exists, if not -> proceed to full data processing)

# get most recent data point

# subtract the difference in rows between new csv file and most recent data point

# get the subset new data only

# add that to old data

# restrict time window to 5yr for everything

# calculate metrics (feature) for only the new data rows




#__________________________________

def partial_data_processing():

  # data_source_file_path = os.path.join(sadp.data_folder, "testing_data_sets", "samples", "WELL_2024_08_12.csv")
  data_source_file_path = os.path.join(sadp.data_folder, "testing_data_sets", "samples", "WELL_2024_08_12_prepared.csv")

  data = sadp.read_csv_all_encodings(file_path=data_source_file_path)
  prepared_data  = sadp.prepare_data_features(data, row_indexes_to_calculate=[i for i in range(1250, 1255)])

  # prepared_data.to_csv(os.path.join(sadp.data_folder, "testing_data_sets", "samples", "WELL_2024_08_12_prepared.csv") , index=False)
  return


def full_data_processing():
  return

#___________________________________


if __name__ == "__main__":
  partial_data_processing()

