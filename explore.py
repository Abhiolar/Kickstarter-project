import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
    
def question_viz(kick):
    print( """Q1. Which categories have the most successful number of projects and 
    which categories have the most failed number of projects based on counts and percentage?
    The client is not interested in categories of projects which has less than 2500
    counts as they want a wide range of data to base their decisions on.""")
    
    print('\n')
    
    shape = (30, 30)
    fig, ax = plt.subplots(figsize = shape)
    sns.countplot(x='category', palette="Set2",data= kick[kick['state'] == 'successful'].groupby("category").filter(lambda x: len(x) > 2500),
                   ax=ax)
    plt.title('Q1.Frequency counts of successful categories') 
    
    print("""According to the plot below, product deign has the 
    highest number of successful projects, followed closely by table
    top games and then shorts in third.""")
    
    print('\n')

    shape = (30, 30)
    fig, ax = plt.subplots(figsize = shape)
    sns.countplot(x='category', palette="Set2",data= kick[kick['state'] == 'failed'].groupby("category").filter(lambda x: len(x) > 2500),
                   ax=ax)
    plt.title('Q1.Frequency counts of failed categories')
    
   
    
    print("""The plot below shows that product design is 
    also the category that has failed the most followed by documentary
    and food in second and third place.Now that we know the counts of 
    each category with regards to failure and success, 
    what to calculate next is the percentage of success and failure divided by the outcomes of the whole projects""")
    print('\n')
    
    counts = kick.category.value_counts()
    succ = kick[kick.state == 'successful'].category.value_counts()


    rows_list = []
    for i in counts.keys():
        dict1 = {}
        # get input row in dictionary format
        # key = col_name
        dict1['category'] = i
        dict1['Percent'] = succ[i] / counts[i] * 100
        rows_list.append(dict1)

    frame = pd.DataFrame(rows_list)   
    most_success = frame.sort_values('Percent').tail(10)
    least_success = frame.sort_values('Percent').head(10)
    print("For the top 10 project most success rates and least success rates:")
    
    print('\n')
    
    shape = (16, 10)
    fig, ax = plt.subplots(figsize = shape)
    sns.barplot(x='category',y = 'Percent', palette="pastel",data = most_success, ax=ax)
    plt.title('Q1.Success rates of the top 10 categories')
    
    print("""Clearly from the plot below the most success 
    rates for the category of a project is chiptunes followed by residencies""")

    print('\n')

    shape = (16, 10)
    fig, ax = plt.subplots(figsize = shape)
    sns.barplot(x='category',y = 'Percent', palette="pastel",data = least_success, ax=ax)
    plt.title('Q1.success rates of least 10 categories');
    
    
    print("""As evident by the plot, apps, web and Mobile games are the least successful categories.""")
    print('\n')
    
    
    
    counts = kick.category.value_counts()
    succ = kick[kick.state == 'failed'].category.value_counts()


    rows_list = []
    for i in counts.keys():
        dict1 = {}
        # get input row in dictionary format
        # key = col_name
        dict1['category'] = i
        dict1['Percent'] = succ[i] / counts[i] * 100
        rows_list.append(dict1)

    frame = pd.DataFrame(rows_list)   
    most_fail = frame.sort_values('Percent').tail(10)
    least_fail = frame.sort_values('Percent').head(10)
    
    
    shape = (16, 10)
    fig, ax = plt.subplots(figsize = shape)
    sns.barplot(x='category',y = 'Percent', palette="Set2",data = most_fail, ax=ax)
    plt.title('Q1.failure rates of top 10 failed catergories');
    
    
    print("""Just as we guessed in the previous plots, Apps, web and Mobile games are the categories that failed the most""")
    
    print('\n')
    
    
    
    
    print(""" Q2 . Which main categories have the most successful 
    and most failed number of projects based on the counts and success rates?""")
    
    print('\n')
    
    
    shape = (30, 30)
    fig, ax = plt.subplots(figsize = shape)
    sns.countplot(x='main_category', palette="Set2",data= kick[kick['state'] == 'successful'].groupby("main_category").filter(lambda x: len(x) > 2500),
                   ax=ax)
    plt.title('Q2.Highest performing main_categories in terms of frequency counts ');
    
    print("""As evident Music has the highest number of successful main_categories on Kickstarter
          followed by film&video, games and art.""")
    print('\n')
    
    
    
    
    shape = (30, 30)
    fig, ax = plt.subplots(figsize = shape)
    sns.countplot(x='main_category', palette="Set2",data= kick[kick['state'] == 'failed'].groupby("main_category").filter(lambda x: len(x) > 2500),
                   ax=ax)
    plt.title('Q2.lowest performing main_categories in terms of frequency counts ');
    
    print("""Moving on to the highest number of failed projects in a main_category turns 
    out to be Film&video followed by publishing and Music
    The next step is to calculate the success rates of these main categories.""")
    print('\n')
    
    
    
    counts = kick.main_category.value_counts()
    succ = kick[kick.state == 'successful'].main_category.value_counts()


    rows_list = []
    for i in counts.keys():
        dict1 = {}
        # get input row in dictionary format
        # key = col_name
        dict1['main_category'] = i
        dict1['Percent'] = succ[i] / counts[i] * 100
        rows_list.append(dict1)

    frame = pd.DataFrame(rows_list)   
    most_success = frame.sort_values('Percent').tail(10)
    least_success = frame.sort_values('Percent').head(10)
    
    
    shape = (16, 10)
    fig, ax = plt.subplots(figsize = shape)
    sns.barplot(x='main_category',y = 'Percent', palette="BrBG",data = most_success, ax=ax)
    plt.title('Q2.High success rates for main category ');
    
    print("""As evident by the plot above, the most success rates for a main category is Dance followed
    by theatre and comics.""")
    
    print('\n')
    
    
    shape = (16, 10)
    fig, ax = plt.subplots(figsize = shape)
    sns.barplot(x='main_category',y = 'Percent', palette="BrBG",data = least_success, ax=ax)
    plt.title('Q2.Low success rates for main categories');
    
    
    
    print("""And lastly, as evidenced by the plot above, the least successful 
main category is Technology followed by journalism and crafts.""")
    
    print('\n')
    
    
    
    print(""" Q3.What main_categories have the most amount of money pledged amongst successful and 
    failed projects and the least amount of money pledged? 
    This gives us an insight into what each different main category goal amount is needed to realise its goals and kickstart the project""")
    
    plt.figure(figsize=(20,10))
    sns.set(style="ticks", palette="pastel")

    sns.boxplot(x="main_category", y="usd_goal_real",
                hue="state", palette=["m", "g"],
                data=kick, showfliers = False)
    sns.despine(offset=10, trim=True)
    plt.title('Q3.Main categories with the most goal amount target including the success and failure');
    
    print('\n')
    
    
    print("""Judging by the measures of centrality in the box plot below, 
    technology requires the most amount in pledged and the least amount needed for a main category is crafts. Seeing as the last 
    question has indicated that technology main category
    has the least success rates and in terms of the money
    required to kickstart the project, it migh not be the most viable domian for our angel investors.""")
    
    print('\n')
   
    
    
    "Wordcloud function to get the buzz words for successful projects"
    
    
def show_wordcloud(data, title = None):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
    
        
   
        
    
    
    
    

    
    
    


    


    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





    
    
    
    
    
    