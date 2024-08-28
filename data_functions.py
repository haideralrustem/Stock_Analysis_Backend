
import pandas as pd




def parse_credentials():
  credentials = {}

  with open(f'./credentials.txt', 'r') as file:
    lines = file.readlines()

    for line in lines:
      # Split the string on '='
      key, value = line.split('=')
      key = key.strip()
      value = value.strip()

      credentials[key] = value

  return credentials



def remove_unknown_characters(dataframe):
    characters_to_remove = [u'\uFFFD']

    def function(value):
        if isinstance(value, str):
            new_cleaned_value = value
            for character in characters_to_remove:
                if character in str(value):
                    new_cleaned_value = str(new_cleaned_value).replace(character, '')

            return new_cleaned_value

        else:
            return value

    dataframe = dataframe.applymap(lambda value: function(value))


    return dataframe


#_________________________________________

def isEmpty(value, additonal_null_values=None):
  default_null_values = ['', 'none', 'nan', 'nat', 'null', 'n/a']

  if additonal_null_values:
    if isinstance(additonal_null_values, list):
      default_null_values = default_null_values + additonal_null_values

  if str(value).strip().lower() in default_null_values:
    return True

  else:
    return False

#____________________________

def force_convert_to_datetime(dataframe, column):
  # Function to convert non-datetime values to 0
  def convert_to_datetime_or_zero(value):
    try:
      return pd.to_datetime(value)
    except (ValueError, TypeError):
      return 0

  # Apply the function to each value in the DataFrame
  dataframe[column] = dataframe[column].apply(convert_to_datetime_or_zero)

  return dataframe
#_____________________________________



def force_convert_to_numeric(dataframe, column):
  # Function to convert non-numeric values to 0
  def convert_to_zero(value):
    if not pd.to_numeric(value, errors='coerce'):
      return 0
    else:
      return pd.to_numeric(value, errors='coerce')

  # Apply the function to each value in the DataFrame
  dataframe[column] = dataframe[column].apply(convert_to_zero)

  return dataframe


#__________________________________

def filter_data_last_n_points(data, target_position, number_of_positions_prior):
  start_position = target_position - number_of_positions_prior
  subset_data = data.iloc[start_position: target_position + 1]

  return subset_data


#___________________________________


def filter_data_future_n_points(data, current_position, number_of_positions_ahead):
  start_position = current_position
  end_position = current_position + number_of_positions_ahead
  subset_data = data.iloc[start_position: end_position ]

  return subset_data
#______________________________________

def filter_data_last_50_points(data, target_position):

  start_position = target_position - 49
  subset_data = data.iloc[start_position: target_position + 1]


  return subset_data

#_________________________________

def filter_data_last_100_points(data, target_position):
  start_position = target_position - 100
  subset_data = data.iloc[start_position: target_position + 1]

  return subset_data