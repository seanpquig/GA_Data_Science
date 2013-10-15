##Final Project Proposals

####*Proposal 1: Twitter and Instagram collage*

* This project would attempt to create a method to group a series of Tweets and Instagram photos together by topic.  This could create some kind of collage that melds text from Twitter with pictures from Instagram by similar topic, trend, or event.
* Data could be accessed through the Twitter and Instagram API's.  I do need to spend a little more time acquainting myself with each to determine that I can get all the data I would need.
* I could use clustering to create groups of similar tweets and photos based on the text in their corresponding messages and hashtags.  I could also predefine some kind of groups and manually classify some data to provide a training set on which I could use some supervised classification algorithims.
* I would hypothesize that this could be somewhat effective, but it might depend on limiting analysis to tweets and photos of a particular source or type.

####*Proposal 2: Artist/Brand pairing alogrithm*
* This would try to devise a method that can pair artists with a brand message based on text data surrounding the particular artists and brands.
* Artist twitter handles are available through the echo nest API.  This could provide a source of text data from tweeets that could be matched against similar brand sources of text data (twitter, wikipedia, amazon?).
* I could use clustering on a set of artist and brand text features to extract groupings of similar messages.
* I would hypothesize that this could work quite well, and have some potential use as a tool in advertising decision making.  I think data preparation could be the hardest part of this problem.



####*Proposal 3: Long/Short stock portfolio constructor*

* This project would attempt to use a machine learning alorithms to develop a method for classifying a group of stocks as "buys" or "sells" based on various fundamental or technical measures.  The goal being to have the buys outperform and the sell underperform some benchmark over some forward-looking time period.
* Data can be gathered through Yahoo Finance's API.  I have used this previously.  I also have some personal datasets from prior work and projects that I could use.
* I could use SVM to create a plane that seperates buy's and sells.  Maybe a decision tree, or simply try a regression model that just tries to predict performance rather than classifying.  Supervised methods will make sense as I can use historical data where the foward looking performance is know.
* I hypothesize that certain proven measures such as price momentum or Joel Greenblatt's "Magic Formula"" metrics should show some predictive capacity on historical data.