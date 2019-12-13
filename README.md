# Mod5_project
Kickstarter success

## BUSINESS PROBLEM 
An angel investor company has approached our data science company to help them find out what it takes to make a kickstarter project a success or a failure on the kickstarter platform which is a global crowdfunding platform that gives you access to backers from different parts of the world and they pledge a certain amount of money to help your project realise its goals in terms of amount needed to fulfill the project.
This is a all or nothing approach which means if you do not realise your goal amount within the certain deadline, it is a failure and the money is returned back to the backers. Our customer which is looking to back some projects on the platform but they need our help to answer four business questions that will be give them insights into what kind of projects to back and most importantly develop a classifier model that can predict the success or failure of a project on the kickstarter platform.


## DELIVERABLES

The main aim of this project is to build a model that predicts whether the kickstarter will be successful or not.

- Documents to review for this repository are- Kickstarter project.ipynb
-cleaning.py (contains the functions for cleaning)
-model.py(contains the functions for modelling)


## 6 BUSINESS QUESTIONS WE DELIEVERED ANSWERS TO 
1.Which categories have the most successful number of projects and which categories have the most failed number of projects based on counts and percentage? The client is not interested in categories of projects which has less than 2500 counts as they want a wide range of data to base their decisions on.

Answers - "Product design" has the highest number of successful projects, followed closely by "table top games" and "shorts" in third.
Product design is also the category that has failed the most followed by documentary and food in second and third place.

the most success rates for the category of a project is chiptunes followed by residencies and moblie apps and games are the least successful categories.

2.Which main categories have the most successful and most failed number of projects based on the counts and success rates?
Answers- Music has the highest number of successful main_categories on Kickstarter followed by film&video, games and art.Highest number of failed projects in a main_category turns out to be Film&video followed by publishing and Music.

The most success rates for a main category is Dance followed by theatre and comics and the least successful main category is Technology followed by journalism and crafts.

3.What main_categories have the most amount of money pledged amongst successful and failed projects and the least amount of money pledged? This gives us an insight into what each different main category goal amount is needed to realise its goals and kickstart the project?
Answers - Technology requires the most amount in pledged money and the least amount needed for a main category is crafts. Seeing as the last question has indicated that technology main category has the least success rates and in terms of the money required to kickstart the project, it might not be the most viable domian for our angel investors.

4.For the successful and failed projects in our respective main_categories, which categories see more money than was originally pledged?
Answers- even though technology main_category has least success rates in a project kickstarting, some of the successful projects end up superseding target goal than any other category followed by design and games main category

5.The next question our client asked- what is the deadline bench mark for most successful projects on kickstarter? and what duration of days does have a lower cahnce of a project being successful?

Answers- The failed and successful main_categories is around 30 days and between 40 and 60 days there is a higher chance of your kickstarter project ending up unsuccessful rather than successful.

6.Our client would like to know the most popular words in the failed and successful projects

Answers - for successful projects, the buzz words are ZEN, CD, NEW, RECORD, ALBUM and for unsuccessful projects the buzz words are 
BREW, COMEDY, NOVEL, NEW , MOBILE

## MODELLING
Building a Baseline Logistic Regression model, the train score is 0.708 and test score is 0.694 which is not bad for the first model. Because we have an imbalanced dataset(60%- unsuccessful and 40% successful) we pass in several weight class arguments to see how it improves the model and at a 2:1 ratio of the successful projects with respect to the unsuceesful we get the best evaluation metric at train score is 0.7202180186572597 test score is  0.7049521355739365.

Using more advanced classifier models such as Decision trees and Random Forest, we used the GridSerachCV to best parameters that gives the best results. None of the model overfits and the best model with Area under the curve evaluation metric, the bigger the value the more the model predicts True positive cases. The value was 0.721.

The optimal threshold calculated to decrease the fpr which is alpha as we want to decrease the number of false positives rate  in our model prediction is 0.64.

## CONCLUSION -

It is relevant to our clients that the model tends towards increasing alpha to penalise the false positive predictions.

















