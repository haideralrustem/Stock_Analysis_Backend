
import os


import database_basic_methods as dbm
import stock_analysis_data_prep as sadp

import data_functions






def check_ticker_exists(ticker):

  # ticker
  tablename = 'stocks'
  where_statement_conditions = f"ticker = '{ticker}'"
  rows_exist, records = dbm.check_if_rows_exist_matching_criteria(tablename, where_statement_conditions=where_statement_conditions)

  return rows_exist, records



#_______________________________


def retirve_data_for_ticker(ticker):

  return


#________________________________

#update
#delete


if __name__ == "__main__":
  check_ticker_exists(ticker='WEL')
  exit(1)
  tablename = 'stocks'
  # data_source_file_path = os.path.join(sadp.data_folder, "testing_data_sets", "samples", "WELL_2024_08_12.csv")
  data_source_file_path = os.path.join(sadp.data_folder, "testing_data_sets", "samples", "WELL_2024_08_12_prepared.csv")

  full_data_table = sadp.read_csv_all_encodings(file_path=data_source_file_path)
  prepared_data = sadp.prepare_data_features(full_data_table, row_indexes_to_calculate=[i for i in range(1250, 1255)])

  prepared_data['ticker'] = 'WELL'
  modified_data = data_functions.create_data_point_id(prepared_data)
  modified_data = data_functions.add_backticks_to_column_names_with_spaces(modified_data)

  print('')

  dbm.update_Table(tablename, dataframeForUpdate=modified_data, ID_column_for_update='data_point_id', ignoreNULL=True)