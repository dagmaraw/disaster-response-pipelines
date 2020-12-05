import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath - (string) the filepath to the messages .csv
    categories_filepath - (string) the filepath to the categories .csv

    OUTPUT:
    df - (pandas dataframe) a dataframe with messages and categories combined

    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id')
    return df

def clean_data(df):
    '''
    INPUT:
    df - (pandas dataframe) the dataframe made in load_data

    OUTPUT:
    df - (pandas dataframe) the cleaned df

    '''
    # split the values in the categories column on the ; character so that each value becomes a separate column
    categories = df.categories.str.split(pat=';',expand=True)
    row = categories.iloc[0]
    slice_names = lambda row: row[:-2] # strip off - character and 1/0 for each category name
    category_colnames = row.apply(slice_names)
    categories.columns = category_colnames # rename the columns of `categories`
    # iterate through the category columns in df to keep only the last character of each string (1 or 0)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)

    # the 'related' column has values of 2
    # since we don't know what this is supposed to mean, drop these rows
    indices_2 = df[df.related==2].index
    df.drop(indices_2,inplace=True)
    
    return df

def save_data(df, database_filename):
    '''
    INPUT:
    df - (pandas dataframe) the cleaned data
    database_filename - (string) the desired file name of the database

    Description:
    Saves df in a sqlite database

    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(database_filename[5:-3], engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()