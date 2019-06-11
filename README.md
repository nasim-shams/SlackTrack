# SlackTrack
Predicting user activity on social platforms 


=== background ===
This project was done as a consulting project for Insight Data Science. Insight uses a social platform (formerly slack) as the main communication platform among fellows, staff and alumni. This platform serves not only as a communcation tool, but also as a community building tool. This is where fellows from the previous sessions can stay in touch, connect with the new fellows or exchange knowloedge. So, it is very important to Insight that the platform remains active and fellows keep using it even after the official length of the prgram (which is 8 weeks). However, this is not always the case. The aim of this project was to use records of useres activity (i.e., time stamps of the messages, and the channels these messages were posted to) , and build predictive models for future user activity.

== method ==
Two approaches was applied in this project. The first one consisted of applying random forest/gradient boosting on a set of features extracted from the dataset. The goal of the analysis was to make a binary prediction on weather the user is still active six weeks after the end of the program. 
The second approach consisted of a deep learning model, run on set of features based on graph analysis and time series analysis techniques. The model was trained as a regressor to make forcast about user's daily activity. i.e., if a user is going to make any post at given date in the future. 

== code description ==
Includes scripts for 
- data preprocessing 
- modeling (including random forest/gradient boosting and deep learning models)
- training
- evaluation