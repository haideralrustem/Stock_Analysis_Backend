
import os
import pandas as pd
import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

from sklearn.model_selection import GridSearchCV
from sklearn import svm


from sklearn.utils.validation import column_or_1d

import statsmodels.api as sm

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





def prepare_data(data):

  # for file in os.listdir(data_folder):
  #   if "copy" not in file.lower() or "lock" not in file.lower():
  #     print(file)

  #data = read_csv_all_encodings(file_path=os.path.join(data_folder, "PSA.csv"))

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


#__________________________________________


def select_rows(data):



  # keep rows from 288 and row before last (which has no outcome variable available). This is done to ensure we have values available for each variable in the analysis

  selected_data = data.iloc[288: -2].reset_index(drop=True)

  #x = data.drop(columns=['status_next_day'])


  return selected_data


#___________________________________

def combine_data_files():

  list_of_dataframes = []
  combined_data = pd.DataFrame()

  for file in os.listdir(f"./data/"):

    if file.endswith("csv") and "copy" not in file.lower():

      data =  read_csv_all_encodings(file_path=os.path.join(data_folder, file))
      data = prepare_data(data)
      data = select_rows(data)
      list_of_dataframes.append(data)


  result = pd.concat(list_of_dataframes, ignore_index=True)
  combined_data = result

  combined_data.to_csv(os.path.join(data_folder, 'combined_data', 'all_data.csv'), index=False)

  return combined_data


#_____________________________________

def prepare_splits():
  all_data_df = pd.read_csv(os.path.join(data_folder, 'combined_data', 'all_data.csv'), index_col=False)

  all_data_df = all_data_df.drop(columns=['Close', 'Date', 'Adj Close', 'High', 'Low'])
  X = all_data_df.drop(columns=['status_next_day'])
  y = all_data_df[['status_next_day']]


  return all_data_df, X, y

#________________________________
def analyze_all_data(model):



  all_data_df, X, y = prepare_splits()

  y = y.values.ravel()

  # standared_scaler = StandardScaler()
  # X = standared_scaler.fit_transform(X)

  power_transformer = PowerTransformer()
  X = power_transformer.fit_transform(X)


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  # instantiate the model (using the default parameters)
  #logreg_model = LogisticRegression(random_state=1234)


  if model == 'LogisticRegression':
    logreg_model = LogisticRegression(class_weight='balanced', max_iter=50, random_state=1234,
                   solver='saga')

    # fit the model with data
    result = logreg_model.fit(X_train, y_train)

    y_pred = logreg_model.predict(X_test)

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    score = logreg_model.score(X_test, y_test)


    print(score)

    print(cnf_matrix)

    print(logreg_model.coef_, logreg_model.intercept_)

    target_names = ['no_increase', 'increased']
    print(classification_report(y_test, y_pred, target_names=target_names))

    coef_dict = {}
    feature_names = []
    for name in logreg_model.feature_names_in_:
      feature_names.append(name)

    c = 0
    for value in logreg_model.coef_[0]:
      coef_dict[feature_names[c]] = value

      c += 1


    # logreg_model.feature_names_in_

    for k,v in coef_dict.items():
      print(f"{k}  =>  {v}")
    # print(tabulate(coef_df, headers='keys', tablefmt='psql'))


  #_______________________________


  if model == 'SVC':
    clf = svm.SVC(kernel='linear', C=10)  # Linear Kernel
    # Train the model using the training sets
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))

  return

#_____________________________________________


def grid_search(model):
  all_data_df, X, y = prepare_splits()

  y = y.values.ravel()

  standared_scaler = StandardScaler()
  # X = standared_scaler.fit_transform(X)





  if model == 'LogisticRegression':

    power_transformer = PowerTransformer()
    X = power_transformer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    param_grid_lr = {
      'max_iter': [20, 50, 100, 200, 500, 1000],
      'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
      'class_weight': ['balanced']
    }

    logModel_grid = GridSearchCV(estimator=LogisticRegression(random_state=1234), param_grid=param_grid_lr, verbose=1,
                                 cv=10, n_jobs=-1)

    logModel_grid.fit(X_train, y_train)
    print(logModel_grid.best_estimator_)



  if model == 'SVC':
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid,  verbose=100)
    grid_search.fit(X, y)
    result = grid_search.best_params_

    print(result)


  pass
#___________________________________________

if __name__ == "__main__":
  #stage_data_for_ananlysis()

  #combine_data_files()
  #grid_search()
  model = 'LogisticRegression'
  model= 'SVC'

  grid_search(model)
  # analyze_all_data(model)



