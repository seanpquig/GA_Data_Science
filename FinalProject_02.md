##Final Project Outline

###*Twitter text-based, Music Reccomendation Engine*
***

####Overview
* The goal of this project is to build a feature set of text data relating to artists from their Twitter feeds.  This data will be the basis for a recommendation engine that will be able to take any twitter handle, retrieve recent tweets for it, and suggest artists that display the highest level of feature similarity.  
* This could make individual artist recommendations for personal twitter handles, or it could pair artists and brands by taking in tweets from brand twitter pages.

####Plan of attack
1. __Get artist list__ by tapping into echonest API.  Retrieve twitter handles for most relevent artists as defined by their "hotttnesss" metric.  How many artists?

2. Using Twitter's API, __rertieve and save a collection of recent tweets from each artist's feed__.  How should I decide between using tweets, retweets, and mentions?  How many should I get/ how far back should I look?  How should I store all this text data for artists?  JSON?  CSV?

3. __Vectorize this text data__, in order to extract features that can be used by a model.  A simple token count matrix could be used, but a tf-idf matrix may be more appropriate, as it will give more weight to unique words that do not appear frequently in documents.

4. __Build recommendation engine__.  First gather tweet data for a given brand or individual twitter handle.  Vectorize and map these features into the same space as the artist text data.  Recommend artists whose features are most similar.  Similarity could be measured by cosine similarity or minkowski distance (probably Euclidian generalization).

5. __Interpret results__.  It is not a supervised learning problem, so it is hard to quantitatively measure my success.  Maybe do some kind of qualitative audit to see if my brand/artist pairings make some kind of sense?  It looks like Amazon, and other business recommendation implementations measure success by improvement in user and business metrics.  Is there some clever way to test this?  Maybe do a survey that allows people to get recommendations for their twitter handle and rate them?

6. __Variations/concerns__.  Maybe try some clustering to create artist like categories?  Would it be easier to find a way to use the same data for a supervised learning problem?  I'm concerned that I could set all this up and then end up with a final result that makes crap/useless recommendations.  Is there a way to prevent/anticipate this?  Might these pairings mean something else besides matching preferences?  Maybe writing/speaking style?  Could it be some kind of personality matcher?




