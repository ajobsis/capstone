# Capstone: Predicting Useful Yelp Reviews

## Contents

- [Problem Statement](#Problem-Statement)
- [Data Dictionary](#Data-Dictionary)
- [Executive Summary](#Executive-Summary)
- [Conclusion](#Conclusion)


## Problem Statement

Yelp is a platform that allows users to make reservations, leave reviews, and find businesses. In addition, users can mark which reviews they find useful.  Reviews can be both very helpful and very harmful for businesses. Being able to determine what makes a useful review can help businesses create better, more  targeted listings. 


## Data Dictionary

|Feature|Type|Dataset|Description|
|---|---|---|---|
|review_id|object|Yelp|Unique review id key|
|user_id|object|Yelp|User id key of the reviewer who left the review|
|business_id|object|Yelp|Business id key of the business being reviewed|
|stars|int64|Yelp|Yelp review star rating (between 1-5 stars|
|useful|int64|Yelp|Number of "useful" votes received|
|funny|int64|Yelp|Number of "funny" votes received|
|cool|int64|Yelp|Number of "cool" votes on the review|
|text|object|Yelp|The review text|
|date|datetime64|Yelp|Date review was posted|
|target|int64|Calculated|Binary 1 = useful, 0 = not useful|
|num_words|int64|Calculated|Number of words in the review text|
|num_chars|int64|Calculated|Number of characters in the review text|


## Executive Summary

The data was sourced from Yelp’s [Open Dataset](https://www.yelp.com/dataset), an all-purpose dataset for learning.  The original dataset had 6,990,280 reviews, so Perl was used to dump out 21,032 random samples for analysis.  The data was originally in JSON file format, and was separated into a reviews dataset, a business dataset, and a user dataset.  There were keys in the reviews dataset that could be used to join the business, and user data, but I deemed that unnecessary to the scope of this study.  

Once the data had been sampled, it was checked for whether null values needed to be imputed.  As there were no null values in the dataset, imputation was not required.  To prepare the data for EDA and modeling, the review text was cleaned up to remove any new lines, and punctuation.  A target column was calculated from the useful column in preparation for binary classification (0 = no useful votes, 1 = 1+ useful votes).

Following EDA, it was determined that the “useful” attribute had a large number of outliers that needed to be cleaned up.  Additionally, both “funny” and “cool” columns were dropped altogether, as the data was too sparsely populated. 

(note a random seed was set at 73)

The baseline of the target was calculated at 0.56 (the majority class which is 0, the reviews that didn’t receive a useful vote).  Once the baseline was calculated, the reviews corpus was then passed through a stemming process, which is where words were “reduced” to their word stems. The stemmed data was then passed into a vectorizer, which converts documents into word vectors.  Once the data was vectorized, it was then passed into a LassoCV model to determine which columns can be safely dropped, thus ruling out multicollinearity.  Once the columns have been reduced, the remainder were concatenated with the original dataframe to arrive at the final columns for modeling.  

Four models were chosen to model the data.  The first one selected was the logistic regression model.  The following parameters were set for the model: the solver was set to liblinear, and the penalty chosen was l2.  Once the model was run, the top 10 coefficients were analyzed.  The highest five coefficients were num_chars, need, new, store, and shop, while the bottom five were table, server, food, atmosphere, and num_words. Next, metrics were generated.  A confusion matrix detailed that a high number of true negatives and false negatives were generated with this model (1798 true negatives, and 1002 false negatives).  This is compared to 529 false positives, and 829 true positives.  Overall the model appears to be better at detecting true negatives than positives.  When graphing the true positive rate versus the false positive rate, the resultant ROC curve was 0.67, which is below the acceptable level of 0.7.  When compared to baseline, the model outperforms with a score of 0.62 (as compared to 0.56 baseline).  

The next model run was the random forest model.  The top five words for random forest were num_words, num_chars, great, food, and like. The bottom five words were inside, sushi, soup, house and rice.  Looking at the confusion matrix for this run, by far, the clearest success for predicting, was the true negatives, with 1659 correctly guessed non-useful reviews.  True positives (the reviews most useful) were next in ranking at 984 useful reviews.  There  were a comparatively small number of predicted false positives at 668.  When it comes to graphing true positive rates against true negative rates, the ROC curve is 0.68, which is slightly below the acceptable rate of 0.70.  The model performed a score of 0.64, which is above the model baseline of 0.56.  

The model run after the random forest was an SVM.  The top five feature importance words for the SVM were num_chars, num_words, need, store, and new.  The bottom five words for SVM were atmosphere, food, wonderful, server, and table.  Generating metrics, the confusion matrix again shows that a large portion of the correct guesses were the true negative classifications, with 1561 true negative guesses.  The true positives, or the reviews that are deemed useful, came in at second place with 1055 classifications.  The false negative and false positives were pretty close together in count.  The true positive rate versus true negative rate graphs about the same as the previous two models, with an ROC curve of 0.67, which falls below acceptable parameters.  Finally, for this model, we had a score of 0.62, which exceeds the baseline score of 0.56.  

Our final model was the neutral network model.  This model required a fair amount of tweaking to get it to this point.  It was set up with one input layer, three hidden layers, and one output layer.  The input and hidden layers used ‘ReLU’, while the output layer used ‘sigmoid’ as the activation layer.   The model was trained for ten epochs.  The score varies from run to run, but seems to land on around 0.63.  

## Conclusion

In determining whether a yelp review is useful or not, over 20,000 features were analyzed.  These features were passed through TF-IDF to vectorize, which generated 200 vectors for analysis.  These vectors were subset with lasso, then fed into four models: logistic regression, random forest, SVM, and a neural network.  

The models all seemed to perform comparatively when it came to producing scores.  The neural network and random forest models both performed slightly better with scores of and 0.63 and 0.64.  All models beat the baseline of 0.56.  

When predicting the true negatives (successfully predicting reviews that had no useful votes), the logistic regression model performed better than SVM and random forest. However, when it came to predicting true positives (successfully predicting reviews that had useful votes), SVM had the best results.  Random forest had the lowest score for predicting true positives.  

The logistic regression model had the best precision for predicting useful reviews.  In other words,  this model had the best ability to determine what measure of “useful” identifications were actually correctly identified as useful.  While SVM, on the other hand, had the worst precision. SVM did, however, have the best recall/sensitivity rate, which is the models ability to find all relevant cases of usefulness.  Finally, SVM also had the best f1-score.  

When it comes to analyzing the word vectors themselves, there is extensive overlap between the highest logistic regression coefficients, and the SVM feature importances.  The following word tokens are shared between the two models: ‘shop', 'new', 'need', 'way', ‘num_chars’, 'decided', 'store', 'review', ‘place’.  While ‘place’ and ‘num_chars’ is shared among logistic regression, random forest, and SVM.  

As far as themes, there seem to be at least a couple of themes in the word tokens with the highest values (either in coefficients, or feature importances).  The first theme is location.  Words such as ‘place’, ‘store’, and ‘shop’, for example, are all examples of location-based word tokens.  The second theme that stands out is compliments.  Words such as ‘great’, ‘good’, and (possibly, depending on context) ‘like’.  One interesting note, is that the compliments theme comes exclusively from the random forest model.  There doesn’t seem to be features with high values that are compliments in either the logistic regression model or the SVM.  There is only one location-based word token for the random forest model.  There **are** a couple of temporal word tokens (‘new’, and ‘time’) that appear in the logistic regression and random forest model respectively. Finally, tokens such as ‘service’ and ‘food’ are interesting too, because they’re representative of transaction.  

Finally, I would say that the best performing model was the SVM model.  It had the most consistent results with precision/recall/f1-score.  It also had a high number of true positive **and** true negative predictions, and only had a 1-2% drop in score compared to the neural network and the random forest model.
