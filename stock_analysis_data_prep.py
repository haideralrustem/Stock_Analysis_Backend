
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

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.impute import SimpleImputer
import pickle

from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.utils.validation import column_or_1d

import statsmodels.api as sm

import data_functions
import calculations




data_folder = os.path.join(os.getcwd(), "data")

outcome_columns = ['status_next_day', 'status_increase_in_next_7_days', 'status_increase_in_next_14_days']



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

def prepare_data_features(data, row_indexes_to_calculate=None):


  data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

  data['Date'] = pd.to_datetime(data['Date'])

  data = calculations.calculate_MA50_for_data(data, row_indexes_to_calculate=row_indexes_to_calculate)
  data = calculations.calculate_MA200_for_data(data, row_indexes_to_calculate=row_indexes_to_calculate)

  data = calculations.calculate_MA50_MA200_diff_for_data(data, row_indexes_to_calculate=row_indexes_to_calculate)

  # calculate MA_50 trending slope in prior month for each point
  data = calculations.calculate_slope_MA_50_for_previous_N_days(data, N_days_prior=30, row_indexes_to_calculate=row_indexes_to_calculate)
  # calculate MA_50 trending slope in prior 3 months for each point
  data = calculations.calculate_slope_MA_50_for_previous_N_days(data, N_days_prior=90, row_indexes_to_calculate=row_indexes_to_calculate)

  # calculate MA_200 trending slope in prior 1 month for each point
  data = calculations.calculate_slope_MA_200_for_previous_N_days(data, N_days_prior=30, row_indexes_to_calculate=row_indexes_to_calculate)
  # calculate MA_200 trending slope in prior 3 months for each point
  data = calculations.calculate_slope_MA_200_for_previous_N_days(data, N_days_prior=90, row_indexes_to_calculate=row_indexes_to_calculate)

  data = calculations.detect_MA_crossover(data, days_prior_for_detection_window=30, row_indexes_to_calculate=row_indexes_to_calculate)

  data = calculations.calculate_range_diff_for_previous_N_days(data, N_days_prior=7, row_indexes_to_calculate=row_indexes_to_calculate)
  data = calculations.calculate_range_diff_for_previous_N_days(data, N_days_prior=30, row_indexes_to_calculate=row_indexes_to_calculate)

  data = calculations.calculate_MA50_MA200_gap_in_percent(data, row_indexes_to_calculate=row_indexes_to_calculate)

  data = calculations.calculate_percentile_and_standard_dev_for_previous_N_days(data, N_days_prior=30, row_indexes_to_calculate=row_indexes_to_calculate)
  data = calculations.calculate_percentile_and_standard_dev_for_previous_N_days(data, N_days_prior=60, row_indexes_to_calculate=row_indexes_to_calculate)
  data = calculations.calculate_percentile_and_standard_dev_for_previous_N_days(data, N_days_prior=90, row_indexes_to_calculate=row_indexes_to_calculate)

  data = calculations.calculate_abs_percent_change_for_previous_N_days(data, N_days_prior=30, row_indexes_to_calculate=row_indexes_to_calculate)
  data = calculations.calculate_abs_percent_change_for_previous_N_days(data, N_days_prior=60, row_indexes_to_calculate=row_indexes_to_calculate)
  data = calculations.calculate_abs_percent_change_for_previous_N_days(data, N_days_prior=90, row_indexes_to_calculate=row_indexes_to_calculate)
  data = calculations.calculate_abs_percent_change_for_previous_N_days(data, N_days_prior=200, row_indexes_to_calculate=row_indexes_to_calculate)

  data['status_next_day'] = np.nan

  data['status_increase_in_next_7_days'] = np.nan

  data['status_increase_in_next_14_days'] = np.nan


  return data

#___________________________________________
def prepare_data_outcomes(data):

  data = calculations.calculate_status_next_day_outcome(data)

  data = calculations.calculate_status_next_N_days_outcome(data, N_days=7)
  data = calculations.calculate_status_next_N_days_outcome(data, N_days=14)
  return data

#____________________________________________

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

  data =  calculations.calculate_status_next_N_days_outcome(data, N_days=7)
  data = calculations.calculate_status_next_N_days_outcome(data, N_days=14)


  return data




#__________________________________________


def prepare_single_data_point(data):

  # for file in os.listdir(data_folder):
  #   if "copy" not in file.lower() or "lock" not in file.lower():
  #     print(file)

  #data = read_csv_all_encodings(file_path=os.path.join(data_folder, "PSA.csv"))

  data = prepare_data_features(data)


  return data

#__________________________________________


def select_rows(data, outcome='', outcome_unavailable=False):
  # keep rows from 288 and row before last (which has no outcome variable available). This is done to ensure we have values available for each variable in the analysis

  selected_data = pd.DataFrame()

  if outcome_unavailable:
    selected_data = data.iloc[288: ].reset_index(drop=True)


  else:

    if outcome == 'status_next_day':
      selected_data = data.iloc[288: -2].reset_index(drop=True)

    if outcome == 'status_increase_in_next_7_days':
      selected_data = data.iloc[288: -8].reset_index(drop=True)


    if outcome == 'status_increase_in_next_14_days':
      selected_data = data.iloc[288: -15].reset_index(drop=True)

  #x = data.drop(columns=['status_next_day'])



  return selected_data


#___________________________________

def combine_data_files(data_files_path, output_path, output_file_name):

  list_of_dataframes = []
  combined_data = pd.DataFrame()

  for file in os.listdir(  data_files_path ):

    if file.endswith("csv") and "copy" not in file.lower():

      data =  read_csv_all_encodings(file_path=os.path.join(data_files_path, file))
      data = prepare_data(data)

      list_of_dataframes.append(data)


  result = pd.concat(list_of_dataframes, ignore_index=True)
  combined_data = result

  combined_data.to_csv(os.path.join(output_path, output_file_name), index=False)

  return combined_data


#_____________________________________

def data_cleanup(data):
  data = data.drop(columns=['Close', 'Date', 'Adj Close', 'High', 'Low', 'MA_50', 'MA_200'])
  data = data.dropna()

  return data
#___________________________________


def split_data_test_train(data, outcome='status_next_day'):


  X = data.drop(columns=['status_next_day'])
  y = data[['status_next_day']]

  if outcome == 'status_increase_in_next_7_days':
    X = data.drop(columns=['status_next_day', 'status_increase_in_next_14_days'])
    y = data[['status_increase_in_next_7_days']]

  if outcome == 'status_increase_in_next_14_days':
    X = data.drop(columns=['status_next_day', 'status_increase_in_next_7_days'])
    y = data[['status_increase_in_next_14_days']]

  return data, X, y

#____________________________________


def drop_outcome_columns(data):
  cols_to_drop = []
  for column in outcome_columns:
    if column in data.columns.to_list():
      cols_to_drop.append(column)

  if cols_to_drop:
    data = data.drop(columns=cols_to_drop)
  return data
#__________________________

def prepare_splits(full_file_path, outcome='status_next_day'):

  all_data_df = pd.read_csv(full_file_path, index_col=False)

  all_data_df = select_rows(all_data_df, outcome=outcome)

  all_data_df = data_cleanup(all_data_df)

  all_data_df = all_data_df.dropna(subset=[outcome])

  # all_data_df = all_data_df.dropna()

  all_data_df, X, y = split_data_test_train(data=all_data_df, outcome=outcome)

  X = drop_outcome_columns(X)

  return all_data_df, X, y




#_____________________________________

def preprocess_data_for_single_prediction(data):
  processed_data = prepare_data_features(data)
  processed_data = select_rows(processed_data, outcome, outcome_unavailable=True)

  processed_data_X = processed_data.drop(columns=['status_next_day', 'status_increase_in_next_7_days', 'status_increase_in_next_14_days'])

  processed_data_X = data_cleanup(processed_data_X)
  power_transformer = PowerTransformer()
  processed_data_X = power_transformer.fit_transform(processed_data_X)
  return processed_data_X

#__________________________





#________________________________
def analyze_all_data(model, outcome=''):



  all_data_df, X, y = prepare_splits(full_file_path=os.path.join(data_folder, 'combined_data', 'all_data.csv'), outcome=outcome)



  y = y.values.ravel()

  # standared_scaler = StandardScaler()
  # X = standared_scaler.fit_transform(X)

  power_transformer = PowerTransformer()
  X = power_transformer.fit_transform(X)

  # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
  # if model == 'LogisticRegression':
  #
  #   X = imp.fit_transform(X)
  #
  # if model == 'SVC':
  #
  #   X = imp.fit_transform(X)


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  # instantiate the model (using the default parameters)
  #logreg_model = LogisticRegression(random_state=1234)

  classifier =None
  if model == 'LogisticRegression':



    logreg_model = LogisticRegression(class_weight='balanced', max_iter=20, random_state=1234,
                   solver='sag')

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


    classifier = logreg_model
  ###

  if model == 'SVC':
    clf = svm.SVC(kernel='rbf', C=10, gamma=0.01, probability=True, class_weight='balanced')  # Linear Kernel
    # Train the model using the training sets
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))

    y_prob_train = clf.predict_proba(X_train)[:, 1]
    y_prob_test = clf.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_train, y_prob_train)
    plt.fill_between(recall, precision)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title("Train Precision-Recall curve")
    # plt.show()

    classifier = clf



  if model == 'DecisionTreeClassifier':
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    target_names = ['no_increase', 'increased']
    print(classification_report(y_test, y_pred, target_names=target_names))

    classifier = clf




  return classifier

#_____________________________________________


def grid_search(model, outcome=''):
  all_data_df, X, y = prepare_splits(full_file_path=os.path.join(data_folder, 'combined_data', 'all_data.csv'), outcome=outcome)

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

    logModel_grid = GridSearchCV(estimator=LogisticRegression(random_state=1234), param_grid=param_grid_lr, refit=True, scoring="precision", verbose=55,
                                 cv=10, n_jobs=-1)

    logModel_grid.fit(X_train, y_train)
    print(logModel_grid.best_estimator_)



  if model == 'SVC':
    power_transformer = PowerTransformer()
    X = power_transformer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Cs = [0.001, 0.01, 0.1, 1, 10]
    # gammas = [0.001, 0.01, 0.1, 1]
    Cs = [0.1, 1, 10, 100, 1000]
    gammas = [1, 0.1, 0.01, 0.001, 0.0001]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, refit = True, scoring="precision", verbose=100)
    grid_search.fit(X, y)
    result = grid_search.best_params_

    print(result)

    print(grid_search.best_estimator_)


  exit(1)
  pass

#___________________________________


def save_classifier(classfier, name):

  with open(f"{name}.pkl", "wb") as f:
    pickle.dump(classfier, f, protocol=5)

  # joblib.dump(classfier, 'my_model.pkl', compress=9)

  return
#________________________________
def load_classifier(classifier_filename):
  filename = f'{classifier_filename}'
  # load the model from disk

  with open(filename, "rb") as f:
    loaded_model = pickle.load(f)


  return loaded_model

#_______________________

def preditct_testing_data(classifier, outcome=''):
  
  all_data_df, X, y = prepare_splits(full_file_path=os.path.join(data_folder, 'testing_data_sets', 'combined_data', 'all_data.csv'), outcome=outcome)

  y = y.values.ravel()

  power_transformer = PowerTransformer()
  X = power_transformer.fit_transform(X)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  y_pred = classifier.predict(X_test)
  print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
  print("Precision:", metrics.precision_score(y_test, y_pred))
  print("Recall:", metrics.recall_score(y_test, y_pred))
  target_names = ['no_increase', 'increased']
  print(classification_report(y_test, y_pred, target_names=target_names))
  
  return


#__________________________________




def preditct_single_data_point(data, classifier, outcome=''):

  processed_data_X = preprocess_data_for_single_prediction(data)

  observation_to_predict = processed_data_X[-1].reshape(1, -1)

  result = classifier.predict(observation_to_predict)

  print(result)

  probability_pred = classifier.predict_proba(observation_to_predict)
  print(probability_pred)


  print(f"Likely outcome for {data.iloc[-1]} \n"
        f"Outcome = {result[0]}\n"
        f"Chance of INcrease = {probability_pred[0, 1]}")
  # y_pred = classifier.predict(X_test)
  # print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
  # print("Precision:", metrics.precision_score(y_test, y_pred))
  # print("Recall:", metrics.recall_score(y_test, y_pred))
  # target_names = ['no_increase', 'increased']
  # print(classification_report(y_test, y_pred, target_names=target_names))

  return


#__________________________________________



def main_runner():
  ml_model = analyze_all_data(model, outcome=outcome)
  # save_classifier(classfier=ml_model, name=f"{model}_{outcome}")

  classifier = load_classifier(f'{model}_{outcome}.pkl')
  # preditct_testing_data(classifier, outcome=outcome)

  data_source_file_path = os.path.join(data_folder, "testing_data_sets", "samples", "WELL_2024_08_12.csv")

  all_data = read_csv_all_encodings(file_path=data_source_file_path)

  for i in range(-7, 0):
    data = all_data.iloc[:i]
    print()
    preditct_single_data_point(data, classifier, outcome=outcome)

#___________________________________________

if __name__ == "__main__":
  #"WELL_2024_08_01.csv"
  # data = read_csv_all_encodings(file_path=os.path.join(data_folder, "PSA.csv"))
  # prepare_data(data)


  # combine_data_files(data_files_path=data_folder, output_path=os.path.join(data_folder, 'combined_data'), output_file_name='all_data.csv')
  #
  # combine_data_files(data_files_path=os.path.join(data_folder, 'testing_data_sets') , output_path=os.path.join(data_folder, 'testing_data_sets', 'combined_data'),
  #                    output_file_name='all_data.csv')



  model = 'LogisticRegression'
  model= 'SVC'
  # model = 'DecisionTreeClassifier'
  # grid_search(model, outcome='status_increase_in_next_7_days')


  # outcome = 'status_next_day'
  outcome = 'status_increase_in_next_7_days'

  # grid_search(model, outcome=outcome)








#{'C': 10, 'gamma': 0.01}


# {'C': 100, 'gamma': 0.1}
# SVC(C=100, gamma=0.1)
