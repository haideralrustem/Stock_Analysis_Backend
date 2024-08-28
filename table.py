import pandas as pd
from tabulate import tabulate




def format(object):
    if isinstance(object, pd.Series):
        dataframe = pd.DataFrame(object).transpose()
    else:
        dataframe = object

    headers = dataframe.columns.to_list()
    table = tabulate(dataframe, headers=headers, tablefmt="presto", showindex=False)

    return table