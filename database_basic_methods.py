import pyodbc
import mysql.connector

import numpy
import pandas as pd
import re
import dateutil.parser
import datetime
import traceback


import data_functions
import table





creds = data_functions.parse_credentials()


def connect_to_MariaDB(database_query_to_execute):
    # Establish a connection
    conn = mysql.connector.connect(
        host=creds["db_hostname"],
        port=creds["db_port"],
        user=creds["db_user"],
        password=creds["db_password"],
        database=creds["database_name"]
    )

    # Create a cursor
    cursor = conn.cursor()

    # Execute a query
    # cursor.execute('SELECT * FROM samples')

    # execute any logic here through the passed parameter : database_query_to_execute which should be a function
    database_query_to_execute(cursor)
    # for row in cursor:
    #     print(" => ", row)

    cursor.close()
    conn.close()
    return
#_________________________________

def connect_to_DB():
    # Establish a connection
    conn = mysql.connector.connect(
        host=creds["db_hostname"],
        port=creds["db_port"],
        user=creds["db_user"],
        password=creds["db_password"],
        database=creds["database_name"]
    )

    # Create a cursor
    cursor = conn.cursor()

    print(conn)

    return conn, cursor


#--------------------------------------

def execute_DB():
    conn, cursor = connect_to_DB()
    table_name = "stocks"
    describe_query = f"DESCRIBE {table_name}"
    cursor.execute(describe_query)

    # Fetch and print the results
    for column in cursor.fetchall():
        print(column)

    cursor.close()
    conn.close()

#______________________________________


def update_DB():
    conn, cursor = connect_to_DB()
    table_name = "qc_table"
    describe_query = f"DESCRIBE {table_name}"
    cursor.execute(describe_query)

    # Fetch and print the results
    for column in cursor.fetchall():
        print(column)

    cursor.close()
    conn.close()


#-------------------------------

def update_Table(tablename, dataframeForUpdate, ID_column_for_update, ignoreNULL=True):

    conn, cursor = connect_to_DB()

    columns = dataframeForUpdate.columns.tolist()

    columns_when_updating = [col for col in columns if col != ID_column_for_update]

    if ignoreNULL:
        dataframeForUpdate = dataframeForUpdate.fillna('')

    dataframeForUpdate = format_dates_for_database(dataframeForUpdate, ignoreNULL=ignoreNULL)

    successful_queries = 0
    unsuccessful_queries = 0
    update_queries= 0
    insert_queries = 0

    print(f"\nExecuting update_Table:\n{len(dataframeForUpdate)} rows  \n\n")
    # Iterating through each row
    for index, row in dataframeForUpdate.iterrows():
        values = []

        check_exists_query = cursor.execute(f"SELECT * FROM {tablename} WHERE {ID_column_for_update} = %s", (row[ID_column_for_update],))
        # cursor.execute(check_exists_query, (value_to_check,))
        result = cursor.fetchall()

        # Check if the row exists
        if   len(result) > 0:
            # print(f"Row with value {row[ID_column_for_update]} exists in the table... so update it....")

            # update
            updateSQLQuery = []
            # update_query = f"UPDATE your_table SET column1 = %s WHERE some_condition_column = %s"

            updateSuccess = False
            updateSQLQuery.append(f"UPDATE {tablename} SET ")

            for column in columns_when_updating:
                if str(row[column]).lower().strip() not in ['', 'none', 'nat', 'nan', 'null']:
                    updateSQLQuery.append(f"{column} = %s , ")
                else:
                    if ignoreNULL==False:
                        updateSQLQuery.append(f"{column} = NULL , ")



            updateSQLQuery[len(updateSQLQuery) - 1] = re.sub(r",", "", updateSQLQuery[len(updateSQLQuery) - 1])

            updateSQLQuery.append(f"WHERE {ID_column_for_update} = %s")

            for column in columns_when_updating:
                if str(row[column]).lower().strip() not in ['', 'none', 'nat', 'nan', 'null']:
                    values.append(row[column])

            sql_update_statement = ''.join(updateSQLQuery)

            # print(sql_update_statement)
            # print(values)

            # this is for the where clause : WHERE ID_column_for_update = %s
            values.append(row[ID_column_for_update])

            try:

                print(f"\rProgress: row {index}", end='', flush=True)
                # Execute the query with the specified values
                cursor.execute(sql_update_statement, values)
                # Commit the changes to the database
                conn.commit()
                # print(f"{cursor.rowcount} record(s) affected")
                successful_queries += 1
                update_queries += 1

            except mysql.connector.Error as error:

                print(f"\n\n [DB_ERROR] >>> Failed to update record in table: {error}\nQuery is: \n{sql_update_statement} \n\t\t {values} \n")

                print(f"{table.format(row)}")
                unsuccessful_queries += 1
                # Print the stack trace
                traceback.print_exc()


        else:
            # print(f"Row with value {row[ID_column_for_update]} does not exist in the table... insert it...")

            # insert
            insertSQLQuery = []
            insertSuccess = False
            insertSQLQuery.append(f"INSERT INTO {tablename} ( ")

            # insertSQLQuery = f"INSERT INTO {tablename} (column1, column2) VALUES (%s, %s)"

            for column in columns:

                if str(row[column]).lower().strip() not in ['', 'none', 'nat', 'nan']:
                    insertSQLQuery.append(f"{column}, ")

            insertSQLQuery[len(insertSQLQuery) - 1] = re.sub(r",", "", insertSQLQuery[len(insertSQLQuery) - 1])


            # CLOSE paranthesis for columns
            insertSQLQuery.append(f") ")

            insertSQLQuery.append(f" VALUES (")

            for column in columns:
                if str(row[column]).lower().strip() not in ['', 'none', 'nat', 'nan']:
                    insertSQLQuery.append(f" %s, ")

            insertSQLQuery[len(insertSQLQuery)-1] =  re.sub(r",", "", insertSQLQuery[len(insertSQLQuery)-1])
            # CLOSE paranthesis for VALUES()
            insertSQLQuery.append(f") ")

            for column in columns:
                if str(row[column]).lower().strip() not in ['', 'none', 'nat', 'nan']:
                    values.append(row[column])



            sql_insert_statement =''.join(insertSQLQuery)

            try:
                print(f"\rProgress: row {index}", end='', flush=True)
                # Execute the query with the specified values
                result = cursor.execute(sql_insert_statement, values)
                # Commit the changes to the database
                conn.commit()

                # print("Insert was successful.")

                insertSuccess = True
                successful_queries += 1
                insert_queries += 1

            except mysql.connector.Error as error:
                print(''.join(insertSQLQuery))
                print(values)
                print()
                print(f"\n \n [DB_ERROR] >>> Failed to insert record into table: {error}\n")
                print(f"{table.format(row)}")
                insertSuccess = False
                unsuccessful_queries += 1
                # Print the stack trace
                traceback.print_exc()


    cursor.close()
    conn.close()

    msg = (f"\n\n--------------------------\n"
           f"update_Table  with dataframe ({len(dataframeForUpdate)}) : \n\n"
           f"successful_queries = {successful_queries}, unsuccessful_queries={unsuccessful_queries}, "
           f"update_queries={update_queries}, "
           f"insert_queries={insert_queries}"
           f"\n-------------------------\n\n")

    # print(f"{msg}")

    return successful_queries, unsuccessful_queries, update_queries, insert_queries, msg
#------------------------------


# this function does not attempt to insert records if not found
def update_Table_no_insert(tablename, dataframeForUpdate, ID_column_for_update):

    conn, cursor = connect_to_DB()

    columns = dataframeForUpdate.columns.tolist()
    columns = [col for col in columns if col != ID_column_for_update]

    dataframeForUpdate = dataframeForUpdate.fillna('')
    dataframeForUpdate = format_dates_for_database(dataframeForUpdate)

    successful_queries = 0
    unsuccessful_queries = 0
    update_queries= 0
    insert_queries = 0

    print(f"\nExecuting update_Table_no_insert:\n{len(dataframeForUpdate)} rows \n  \n\n")

    # Iterating through each row
    for index, row in dataframeForUpdate.iterrows():
        values = []

        check_exists_query = cursor.execute(f"SELECT * FROM {tablename} WHERE {ID_column_for_update} = %s",
                                            (row[ID_column_for_update],))
        # cursor.execute(check_exists_query, (value_to_check,))
        result = cursor.fetchall()

        # Check if the row exists
        if len(result) > 0:
            # print(f"Row with value {row[ID_column_for_update]} exists in the table... so update it....")

            # update
            updateSQLQuery = []
            # update_query = f"UPDATE your_table SET column1 = %s WHERE some_condition_column = %s"

            updateSuccess = False
            updateSQLQuery.append(f"UPDATE {tablename} SET ")

            for column in columns:
                if str(row[column]).lower().strip() not in ['', 'none', 'nat']:
                    updateSQLQuery.append(f"{column} = %s , ")

            updateSQLQuery[len(updateSQLQuery) - 1] = re.sub(r",", "", updateSQLQuery[len(updateSQLQuery) - 1])

            updateSQLQuery.append(f"WHERE {ID_column_for_update} = %s")

            for column in columns:
                if str(row[column]).lower().strip() not in ['', 'none', 'nat']:
                    values.append(row[column])

            sql_update_statement = ''.join(updateSQLQuery)



            # this is for the where clause : WHERE ID_column_for_update = %s
            values.append(row[ID_column_for_update])

            try:
                print(f"\rProgress: row {index}", end='', flush=True)
                # Execute the query with the specified values
                cursor.execute(sql_update_statement, values)
                # Commit the changes to the database
                conn.commit()
                # print(f"{cursor.rowcount} record(s) affected")
                successful_queries += 1
                update_queries += 1

            except mysql.connector.Error as error:
                print(sql_update_statement)
                print(values)
                print(
                    f"\n\n [DB_ERROR] >>> Failed to update record in table: {error}\nQuery is: {sql_update_statement} \t\t {values} \n")

                print(f"{table.format(row)}")
                unsuccessful_queries += 1
                # Print the stack trace
                traceback.print_exc()


    cursor.close()
    conn.close()

    feedback_msg = (f"\n---------------------\n"
                    f"UPDATE DB SUMMARY  with dataframe ({len(dataframeForUpdate)}): \n{dataframeForUpdate.head(1)} \n\n"
                    f"successful_queries: {successful_queries}, unsuccessful_queries: {unsuccessful_queries}, update_queries: {update_queries}"
                    f"\n--------------------- ")
    return successful_queries, unsuccessful_queries, update_queries, feedback_msg

#________________________________________

# This function will use NULL values this function does not attempt to insert records if not found
def update_Table_force_null_no_insert(tablename, dataframeForUpdate, ID_column_for_update):

    conn, cursor = connect_to_DB()

    columns = dataframeForUpdate.columns.tolist()
    columns = [col for col in columns if col != ID_column_for_update]

    # dataframeForUpdate = dataframeForUpdate.fillna('')
    dataframeForUpdate = format_dates_for_database(dataframeForUpdate, ignoreNULL=False)

    successful_queries = 0
    unsuccessful_queries = 0
    update_queries= 0
    insert_queries = 0

    print(f"\nExecuting update_Table_no_insert:\n{len(dataframeForUpdate)} rows \n  \n\n")

    # Iterating through each row
    for index, row in dataframeForUpdate.iterrows():
        values = []

        check_exists_query = cursor.execute(f"SELECT * FROM {tablename} WHERE {ID_column_for_update} = %s",
                                            (row[ID_column_for_update],))
        # cursor.execute(check_exists_query, (value_to_check,))
        result = cursor.fetchall()

        # Check if the row exists
        if len(result) > 0:
            # print(f"Row with value {row[ID_column_for_update]} exists in the table... so update it....")

            # update
            updateSQLQuery = []
            # update_query = f"UPDATE your_table SET column1 = %s WHERE some_condition_column = %s"

            updateSuccess = False
            updateSQLQuery.append(f"UPDATE {tablename} SET ")

            for column in columns:
                if str(row[column]).lower().strip() in ['', 'none', 'nat', 'null']:
                    updateSQLQuery.append(f"{column} = NULL , ")
                else:
                    updateSQLQuery.append(f"{column} = %s , ")

            updateSQLQuery[len(updateSQLQuery) - 1] = re.sub(r",", "", updateSQLQuery[len(updateSQLQuery) - 1])

            updateSQLQuery.append(f"WHERE {ID_column_for_update} = %s")

            for column in columns:
                if str(row[column]).lower().strip() not in ['', 'none', 'nat', 'null']:
                    values.append(row[column])

            sql_update_statement = ''.join(updateSQLQuery)



            # this is for the where clause : WHERE ID_column_for_update = %s
            values.append(row[ID_column_for_update])

            try:
                print(sql_update_statement)
                print(f"\rProgress: row {index}", end='', flush=True)
                # Execute the query with the specified values
                cursor.execute(sql_update_statement, values)
                # Commit the changes to the database
                conn.commit()
                # print(f"{cursor.rowcount} record(s) affected")
                successful_queries += 1
                update_queries += 1

            except mysql.connector.Error as error:
                print(sql_update_statement)
                print(values)
                print(
                    f"\n\n [DB_ERROR] >>> Failed to update record in table: {error}\nQuery is: {sql_update_statement} \t\t {values} \n")

                print(f"{table.format(row)}")
                unsuccessful_queries += 1
                # Print the stack trace
                traceback.print_exc()


    cursor.close()
    conn.close()

    feedback_msg = (f"\n---------------------\n"
                    f"UPDATE DB SUMMARY  with dataframe ({len(dataframeForUpdate)}): \n{dataframeForUpdate.head(1)} \n\n"
                    f"successful_queries: {successful_queries}, unsuccessful_queries: {unsuccessful_queries}, update_queries: {update_queries}"
                    f"\n--------------------- ")
    return successful_queries, unsuccessful_queries, update_queries, feedback_msg

#____________________________________________


# this function will only insert records that do not exist in the database. This function does not attempt to update records if already existing.
def insert_into_Table_no_update(tablename, dataframeForUpdate, ID_column):

    conn, cursor = connect_to_DB()
    columns = dataframeForUpdate.columns.tolist()


    dataframeForUpdate = dataframeForUpdate.fillna('')
    dataframeForUpdate = format_dates_for_database(dataframeForUpdate)

    successful_queries = 0
    unsuccessful_queries = 0
    insert_queries = 0

    existing_records = 0

    print(f"\nExecuting insert_into_Table_no_update:\n{len(dataframeForUpdate)} rows. \n \n\n")

    # Iterating through each row
    for index, row in dataframeForUpdate.iterrows():
        values = []

        cursor.execute(f"SELECT * FROM {tablename} WHERE {ID_column} = %s", (row[ID_column],))
        # cursor.execute(check_exists_query, (value_to_check,))
        result = cursor.fetchall()

        # Check if the row exists, we skip, because we are not interested in any existing records
        if len(result) > 0:
            print(f"{index} - Row with value {row[ID_column]} already exists in the table... so skip it....")
            existing_records += 1


        # Insert if record is not already in the database
        else:
            # print(f"Row with value {row[ID_column_for_update]} does not exist in the table... insert it...")
            # insert
            insertSQLQuery = []
            insertSuccess = False
            insertSQLQuery.append(f"INSERT INTO {tablename} ( ")

            # insertSQLQuery = f"INSERT INTO {tablename} (column1, column2) VALUES (%s, %s)"

            for column in columns:

                if str(row[column]).lower().strip() not in ['', 'none', 'nat']:
                    insertSQLQuery.append(f"{column}, ")

            insertSQLQuery[len(insertSQLQuery) - 1] = re.sub(r",", "", insertSQLQuery[len(insertSQLQuery) - 1])

            # CLOSE paranthesis for columns
            insertSQLQuery.append(f") ")

            insertSQLQuery.append(f" VALUES (")

            for column in columns:
                if str(row[column]).lower().strip() not in ['', 'none', 'nat']:
                    insertSQLQuery.append(f" %s, ")

            insertSQLQuery[len(insertSQLQuery) - 1] = re.sub(r",", "", insertSQLQuery[len(insertSQLQuery) - 1])
            # CLOSE paranthesis for VALUES()
            insertSQLQuery.append(f") ")

            for column in columns:
                if str(row[column]).lower().strip() not in ['', 'none', 'nat']:
                    values.append(row[column])


            sql_insert_statement = ''.join(insertSQLQuery)


            try:
                # print(f"{index} -  {sql_insert_statement}")
                # print(f"{index} -  {values}")
                # print("\n")

                print(f"\rProgress: row {index}", end='', flush=True)

                # Execute the query with the specified values
                result = cursor.execute(sql_insert_statement, values)
                # Commit the changes to the database
                conn.commit()

                # print("Insert was successful.")

                insertSuccess = True
                successful_queries += 1
                insert_queries += 1


            except mysql.connector.Error as error:
                print(f"\n\n [DB_ERROR] >>> Failed to insert record at index {index} into table: {error}\n\n")
                print(f"{table.format(row)}")
                insertSuccess = False
                unsuccessful_queries += 1
                # Print the stack trace
                traceback.print_exc()

    cursor.close()
    conn.close()
    if successful_queries == None:
        print()

    feedback_msg = (f"\n--------------------------\n"
                    f"INSERT_ONLY DB SUMMARY with dataframe ({len(dataframeForUpdate)}): \n \n\n"
                    f"successful_queries: {successful_queries}, unsuccessful_queries: {unsuccessful_queries}, "
                    f"existing_records (un-touched): {existing_records} "
                    f"\n------------------------- ")

    return successful_queries, unsuccessful_queries, insert_queries, existing_records, feedback_msg


#________________________________________



# this function pdates records in the database using two ID columns. It also does not attempt to insert records if not found
def update_Table_two_ids(tablename, dataframeForUpdate, ID_column_for_update1, ID_column_for_update2):

    conn, cursor = connect_to_DB()

    columns = dataframeForUpdate.columns.tolist()
    columns = [col for col in columns if col != ID_column_for_update1 or col != ID_column_for_update2]

    dataframeForUpdate = dataframeForUpdate.fillna('')
    dataframeForUpdate = format_dates_for_database(dataframeForUpdate)

    successful_queries = 0
    unsuccessful_queries = 0
    update_queries= 0
    insert_queries = 0

    print(f"\nExecuting update_Table_two_ids :\n{len(dataframeForUpdate)} rows \n  \n\n")


    # Iterating through each row
    for index, row in dataframeForUpdate.iterrows():
        values = []

        check_exists_query = cursor.execute(f"SELECT * FROM {tablename} WHERE {ID_column_for_update1} = %s AND {ID_column_for_update2} = %s" ,
                                            (row[ID_column_for_update1], row[ID_column_for_update2]))
        # cursor.execute(check_exists_query, (value_to_check,))
        result = cursor.fetchall()

        # Check if the row exists
        if len(result) > 0:
            # print(f"Row with value {row[ID_column_for_update]} exists in the table... so update it....")

            # update
            updateSQLQuery = []
            # update_query = f"UPDATE your_table SET column1 = %s WHERE some_condition_column = %s"

            updateSuccess = False
            updateSQLQuery.append(f"UPDATE {tablename} SET ")

            for column in columns:
                if str(row[column]).lower().strip() not in ['', 'none', 'nat']:
                    updateSQLQuery.append(f"{column} = %s , ")

            updateSQLQuery[len(updateSQLQuery) - 1] = re.sub(r",", "", updateSQLQuery[len(updateSQLQuery) - 1])

            updateSQLQuery.append(f"WHERE {ID_column_for_update1} = %s AND {ID_column_for_update2} = %s ")

            for column in columns:
                if str(row[column]).lower().strip() not in ['', 'none', 'nat']:
                    values.append(row[column])

            sql_update_statement = ''.join(updateSQLQuery)



            # this is for the where clause : WHERE ID_column_for_update = %s AND {ID_column_for_update2} = %s
            values.append(row[ID_column_for_update1])
            values.append(row[ID_column_for_update2])

            try:
                print(f"\rProgress: row {index}", end='', flush=True)
                # Execute the query with the specified values
                cursor.execute(sql_update_statement, values)
                # Commit the changes to the database
                conn.commit()
                # print(f"{cursor.rowcount} record(s) affected")
                successful_queries += 1
                update_queries += 1

            except mysql.connector.Error as error:
                print(sql_update_statement)
                print(values)
                print(
                    f"\n\n [DB_ERROR] >>> Failed to update record in table: {error}\nQuery is: {sql_update_statement} \t\t {values} \n")

                print(f"{table.format(row)}")
                unsuccessful_queries += 1
                # Print the stack trace
                traceback.print_exc()


    cursor.close()
    conn.close()

    feedback_msg = (f"\n------------\n"
                    f"UPDATE DB SUMMARY with dataframe ({len(dataframeForUpdate)}):  \n \n\n"
                    f"successful_queries: {successful_queries}, unsuccessful_queries: {unsuccessful_queries}, update_queries: {update_queries} \n"
                    f"----------------\n")
    return successful_queries, unsuccessful_queries, update_queries, feedback_msg


#_________________________________________

# uses a WHERE statement conditions to update specific rows that meet these conditions
def update_Table_with_where_statement(tablename, dataframeForUpdate, where_statement):

    conn, cursor = connect_to_DB()

    columns = dataframeForUpdate.columns.tolist()
    # columns = [col for col in columns if col != ID_column_for_update]

    dataframeForUpdate = dataframeForUpdate.fillna('')
    dataframeForUpdate = format_dates_for_database(dataframeForUpdate)

    successful_queries = 0
    unsuccessful_queries = 0
    update_queries= 0
    insert_queries = 0

    print(f"\nExecuting update_Table_with_where_statement:\n{len(dataframeForUpdate)} rows \n \n\n")


    # Iterating through each row
    for index, row in dataframeForUpdate.iterrows():
        values = []

        check_exists_query = cursor.execute(f"SELECT * FROM {tablename} WHERE {where_statement}")
        # cursor.execute(check_exists_query, (value_to_check,))
        result = cursor.fetchall()

        # Check if the row exists
        if len(result) > 0:
            # print(f"Row with value {row[ID_column_for_update]} exists in the table... so update it....")

            # update
            updateSQLQuery = []
            # update_query = f"UPDATE your_table SET column1 = %s WHERE some_condition_column = %s"

            updateSuccess = False
            updateSQLQuery.append(f"UPDATE {tablename} SET ")

            for column in columns:
                if str(row[column]).lower().strip() not in ['', 'none', 'nat']:
                    updateSQLQuery.append(f"{column} = %s , ")

            updateSQLQuery[len(updateSQLQuery) - 1] = re.sub(r",", "", updateSQLQuery[len(updateSQLQuery) - 1])

            updateSQLQuery.append(f"WHERE {where_statement}")

            for column in columns:
                if str(row[column]).lower().strip() not in ['', 'none', 'nat']:
                    values.append(row[column])

            sql_update_statement = ''.join(updateSQLQuery)


            try:
                print(f"\rProgress: row {index}", end='', flush=True)
                # Execute the query with the specified values
                cursor.execute(sql_update_statement, values)
                # Commit the changes to the database
                conn.commit()
                # print(f"{cursor.rowcount} record(s) affected")
                successful_queries += 1
                update_queries += 1

            except mysql.connector.Error as error:
                print(sql_update_statement)
                print(values)
                print(f"\n\n [DB_ERROR] >>> Failed to update record in table: {error}\nQuery is: {sql_update_statement} \t\t {values} \n")
                unsuccessful_queries += 1
                # Print the stack trace
                traceback.print_exc()

    cursor.close()
    conn.close()

    feedback_msg = (f"\n---------------------\n"
                    f"UPDATE DB [WITH WHERE] SUMMARY (rows: {len(dataframeForUpdate)}): \n  \n\n"
                    f"successful_queries: {successful_queries}, unsuccessful_queries: {unsuccessful_queries}, update_queries: {update_queries} \n"
                    f"---------------------\n ")
    return successful_queries, unsuccessful_queries, update_queries, feedback_msg


#_________________________________________

def retrieve_record_from_database(tablename, search_by_column_name, search_value):

    records = []
    column_names = []
    conn, cursor = connect_to_DB()

    select_query = f"SELECT * FROM {tablename} WHERE {search_by_column_name} = %s"
    print(select_query)
    cursor.execute(select_query, [search_value])

    column_names = [i[0] for i in cursor.description]

    # Fetch all the rows that satisfy the conditions
    result = cursor.fetchall()

    # Print the result
    for row in result:
        print(row)
        records.append(row)


    cursor.close()
    conn.close()

    # records is a list of tuples:  [ (x,x,x,x) , (x,x,x,x) ]
    # convert list of tuples to dataframe for better readability and access

    results_dataframe = pd.DataFrame(records, columns=column_names)

    return results_dataframe, column_names



#________________________________


def dev_empty_table(tablename):
    conn = mysql.connector.connect(
        host=creds["db_hostname"],
        port=creds["db_port"],
        user=creds["db_user"],
        password=creds["db_password"],
        database=creds["database_name"]
    )

    # Create a cursor
    cursor = conn.cursor()

    sql = f"DELETE FROM {tablename}"
    cursor.execute(sql)

    conn.commit()

    cursor.close()
    conn.close()


#-------------------------

def format_dates_for_database(dataframe, ignoreNULL=True):

    date_columns = [column for column in dataframe.columns if "date_" in column.lower() or "_date" in column.lower()]

    def function(value):
        format = '%m/%d/%Y'
        if isinstance(value, str):
            try:
                date_obj = dateutil.parser.parse(str(value))
                return date_obj
            except Exception as e:
                print(f" dateutil.parser.parse ERROR for value: [ {value} ] : \n", e)
                if ignoreNULL:
                    return ''
                else:
                    return 'NULL'
        else:
            return value



    for date_col in date_columns:
        dataframe[date_col] = dataframe[date_col].apply(lambda value: function(value))
        # dataframe[date_col] = pd.to_datetime(dataframe[date_col], format='%m/%d/%Y %H:%M:%S')

    return dataframe




#____________________________________

def retrieve_records_from_db_where(tablename, where_statement_conditions='', table_columns_to_show=None ):
    # where_statement_conditions example  ==> WHERE column1 = 'value1'  AND  column2 > 100
    records_dataframe = None

    conn, cursor = connect_to_DB()

    # check if where_statement_conditions is empty (no-specified conditions). In such case, just get all records
    if str(where_statement_conditions).lower() in ['none', '']:
        # this syntax in SQL: "WHERE 1" means get all records with no conditions specified
        where_statement_conditions = "1"


    columns_to_include = ''
    if table_columns_to_show:
        columns_to_include = ', '.join(table_columns_to_show)
    else:
        columns_to_include = "*"  # include all columns of table in resulting query

    # Define the SQL query
    query = f"SELECT {columns_to_include} FROM {tablename} WHERE {where_statement_conditions}"
    # print(query)
    # Execute the SQL query
    cursor.execute(query)
    column_names = [i[0] for i in cursor.description]

    # Fetch all the records
    results = cursor.fetchall()
    records = []
    # Display the retrieved records
    for row in results:
        records.append(row)

    records_dataframe = pd.DataFrame(records, columns=column_names)

    # Close the cursor and connection
    cursor.close()
    conn.close()

    return records_dataframe



#_________________________________


def retrieve_records_from_db_complex_query(tablename, where_statement_conditions='', table_columns_to_show=None, additional_sql_logic=''):
    # where_statement_conditions example  ==> WHERE column1 = 'value1'  AND  column2 > 100
    records_dataframe = None

    conn, cursor = connect_to_DB()

    # check if where_statement_conditions is empty (no-specified conditions). In such case, just get all records
    if str(where_statement_conditions).lower() in ['none', '']:
        # this syntax in SQL: "WHERE 1" means get all records with no conditions specified
        where_statement_conditions = "1"


    columns_to_include = ''
    if table_columns_to_show:
        columns_to_include = ', '.join(table_columns_to_show)
    else:
        columns_to_include = "*"  # include all columns of table in resulting query

    # Define the SQL query
    query = f"SELECT {columns_to_include} FROM {tablename} WHERE {where_statement_conditions}"

    if additional_sql_logic:
        query = query + f" {additional_sql_logic} "

    # print(query)


    # Execute the SQL query
    cursor.execute(query)
    column_names = [i[0] for i in cursor.description]

    # Fetch all the records
    results = cursor.fetchall()
    records = []
    # Display the retrieved records
    for row in results:
        records.append(row)

    records_dataframe = pd.DataFrame(records, columns=column_names)

    # Close the cursor and connection
    cursor.close()
    conn.close()

    return records_dataframe

#_________________________

### This method retrieves records where a column you specify has any of the possible values that the column can have
def retrieve_records_matching_list_of_values_in_column(tablename, dataframe, column_name_for_matching):

    values_list = dataframe[column_name_for_matching].tolist()

    qutoed_values_list = [f'"{num}"' for num in values_list]
    where_conditions = f" {column_name_for_matching} IN ( {', '.join(f'{num}' for num in qutoed_values_list)} ) "
    matched_records = retrieve_records_from_db_where(tablename=tablename, where_statement_conditions=where_conditions)

    return matched_records



#_____________________________________________



def delete_rows_using_where_statement_conditions(tablename, where_statement_conditions):
    conn, cursor = connect_to_DB()

    delete_query = f"DELETE FROM  {tablename}  WHERE {where_statement_conditions} "


    print(delete_query)

    success = 0
    failed = 0
    num_rows_affected= -1
    try:
        cursor.execute(delete_query)
        num_rows_affected = cursor.rowcount
        # Commit the changes to the database
        conn.commit()
        success +=1
    except Exception as e:
        failed += 1
        print(f"[DELETE FROM DB]-[ERROR] : {e} ")

    feedback = f"[delete_rows_using_where_statement_conditions] findished with rows affected = {num_rows_affected} success = {success}, failed = {failed}"

    # Close the cursor and connection
    cursor.close()
    conn.close()

    return feedback


#________________________________________


def delete_duplicate_rows_using_column(tablename, target_duplicate_column, keep_policy):
    conn, cursor = connect_to_DB()

    # keep_policy  should loo like : "s1.id > s2.id"
    if keep_policy in [None, '']:
        # this will keep the higher Internal_Sample_Number, and delete the other duplicates
        keep_policy = "s1.Internal_Sample_Number < s2.Internal_Sample_Number"


    delete_query = f"DELETE s1 FROM  {tablename} s1, {tablename} s2 WHERE {keep_policy} AND s1.{target_duplicate_column} = s2.{target_duplicate_column} "


    # print(delete_query)

    success_outcome = 0
    fail_outcome = 0
    num_rows_affected= -1
    try:
        cursor.execute(delete_query)
        num_rows_affected = cursor.rowcount
        # Commit the changes to the database
        conn.commit()
        success_outcome +=1
    except Exception as e:
        fail_outcome += 1
        print(f"[DELETE FROM DB]-[ERROR] : {e} ")

    feedback = f"[delete_duplicate_rows_using_column] finished with rows affected = {num_rows_affected} success = {success_outcome}, fail = {fail_outcome}"

    # Close the cursor and connection
    cursor.close()
    conn.close()

    return feedback

#--------------------------------------

if __name__ == "__main__":
    # dataframeForUpdate = pd.DataFrame([{"Accession_Number": "12346", "Lab_Assigned_Identifier": "CS-12345", "fastq_file_code": "SHJ-SHS"}])
    tablename = "qc_table"
    ID_column_for_update = "Accession_Number"
    # update_Table(tablename, dataframeForUpdate, ID_column_for_update)

    execute_DB()