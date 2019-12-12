import pandas as pd
def clean(df):
    "calculate the number of all teh null values"
    
    df.isna().sum()
    df.dropna(inplace = True)
    
    "convert time columns to datetime and subtract to get the number of days"
    
    df['launched'] = pd.to_datetime(df['launched'])
    df['deadline'] = pd.to_datetime(df['deadline'])
    df['number_of_days'] = df['deadline'] - df['launched']
    
    "extracting the number of days
    
    df['number_of_days'] = df['number_of_days'].dt.days
    
    "return columns we are not interested in predicting"
    
    df = df[df['state'] != 'live']
    df = df[df['state'] != 'suspended']
    df = df[df['state'] != 'canceled']
    
    "drop the columns we do not need"
    
    del df['goal']
    del df['pledged']
    del df['usd pledged']
    
    "calculate the amount between the pledged and goal amount"
    
    df['pledged_goal_diff'] = df['usd_pledged_real'] - df['usd_goal_real']
    
    "The next step is to create a separate column for the binary classification which is the state of the project, failed and   successful"
    
    df['classes_state'] = df.state.astype('category').cat.codes
    
    return df