# Yahoo Troll Detection Challenge

## East India Company

## December 2022

## 1 Authors

1. Kaushik Mishra (IMT2020137)
2. Monjoy Narayan Choudhury (IMT2020502)

## 2 Preliminary Steps

There were no null attirbutes and duplicate rows. We followed our analysis with
generaton of visualisations like

1. Word Clouds
2. Word Count plot
3. Word Count plot for bigrams

## 3 Preprocessing

We did the following steps in the first run:

1. Checked for NULL values: None found
2. Checked for duplicate rows: None found
3. Performed Feature Engineering to enhance our decisions
4. Data Cleaning
5. Foreign Language Detection

The feature engineering consisted of:

1. Word Count
2. Capital letters count
3. Unique Words
4. Sentence Count
5. Number of special characters
6. Number of numerical values
7. Number of stopwords

We used these parameters once to train but had no success in it. Data cleaning
methods that we implemented are:

1. Lower Case Conversion
2. Contraction removal
3. Removing Stopwords
4. Removing Whitespaces
5. Removing Numerical values
6. Removing URL
7. Stemming using PorterStemmer
8. Lemmatizing using WordNetLemmatizer

Due to the feature engineering, we did in the earlier half we noticed that punc-
tuations matter, as well as symbols due to the detection of foreign languages in
the dataset. So we were bound to remove all pre-processing steps which he had
done in the initial phase and our final submissions have no preprocessing done.

## 4 Feature Generation

We tried 3 models to convert words into a feature vector which can be used to
train the ML models:

1. CountVectorizer(): We stuck to this for most of our analysis and took
    ngram_range(1,4). We tried fitting separately on only train data and trans-
    form validation and test data as well as fitting on a combined data frame
    of train and test data.
2. Word2Vec: We built a vocabulary from the given tokens of our dataset
    with a window size of 10 (which limited our accuracy). Even if Word2Vec
    captured semantic similarity pretty well, it didn’t contribute much to im-
    proving our model
3. TFIDF: This provided us with the best results. However, we had ap-
    proached using the tool pretty differently. We took the analyzer of TFIDF
    with respect to words as well as characters and combine both of them to
    create a custom TFIDF transformer. The parameters for the rest of the
    attributes involve ngramrange=(1,3), lowercase=False, strip_accents =
    ’unicode’, maxdf = 0.8, maxfeatures = 150000.


4. Naive Bayes Transformer∗: This is an interesting approach that we found
    on Kaggle for a similar contest and also a paper from MIT where it is
    discussed in detail. In this, we try to estimate the relative importance
    of various created features by TF-IDF or CountVectorizer by modeling
    other features as an appropriate distribution. Here they use Naive Bayes
    to model that distribution. This gave a similar accuracy to what we were
    getting with the above approaches but we believe that this could have
    been pursued more to get better results. However, we felt that due to
    our lack of understanding of the approach in its entirety it would be an
    injustice to use an elegant solution by blatant copy-pasting.

## 5 Resampling

We noticed that the dataset given is highly skewed, i.e the data is biased towards
the majority class. We tried the following standard techniques we found:

1. Random Undersampling
2. Random Oversampling
3. SMOTE

We believed this would improve the outcomes, however, the results deteriorated
and we had to drop this approach too.

## 6 Models Tried

1. Naive Bayes - MultinomialNB
2. Logistic Regression with GridSearchCV and RandomisedSearchCV ap-
    plied on
a) C
(b) penalty
(c) solver
(d) Class Weights and StratifiedKFold to find the best split. Also, we used predict_proba for
this to control the threshold to accommodate class imbalance
3. Perceptron with classweight = balanced
4. Gradient Boosting Classifier
5. XGBoost
6. Random Forest Classifier with entropy


7. Support Vector Classifier with Linear kernel and class weight balanced.
8. Stochastic Gradient Descent Classifier with hinge loss, penalty = l

In the case of Random Forest, the model timed out on Kaggle as well as colab.
Support Vector Classifier could give results (unsatisfactory) on small data only,
so we had to use our undersampled data. We tried to employ grid search on
perceptron but didn’t notice much progress. Our best performance is covered
by Logistic Regression with custom parameters for threshold and class weights.

## 7 Best submission

We came 4th on the private leaderboard with score: 0.64208, 0.003 less than
the first. Our submission consists of:

1. No Preprocessing
2. Feature Extraction using stack 2 TFIDF vectorizer one with analyzer
    ’word’ and other with analyzer ’char’ with parameters ngramrange=(1,3),
    lowercase=False, analyzer=”char”,stripaccents = ’unicode’, maxdf =
    0.8, maxfeatures = 150000
3. No Resampling
4. Validation size = 0.
5. Logistic Regression with maxiter = 300,classweight={0:1,1:2}
6. Prediction probability threshold: 0.

## 8 Analysing our final model

We tried to use LIME (Local Interpretable Model-agnostic Explanations) to
provide explanations of predictions made by models, especially misclassifica-
tions. We did the analysis for False Positives and False Negatives and noted the
following observation:

1. Proper nouns throw the model off balance. Eg: Donald Trump makes
    an appearance in both troll and not troll and contributes heavily to the
    decision.
2. The model fails to take long statements where it is made of 2 sentences
    which makes no sense.
3. Need to include domain knowledge to decide a good threshold. This was
    evident in the case of false positives where all misclassification had a prob-
    ability above the threshold of 0.4 even if they were less than the probability
    of being negative


4. The model sometimes fails to take in the factor that there were more words
    citing it to be ”troll” and considering an extreme word like ”rape” to be
    involved with something ”not-troll” to be the decider




