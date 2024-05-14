# Importing necessary libraries
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
import string

# Defining needed functions
def preprocess_emails(df):
    # Shuffles the dataset
    df = df.sample(frac = 1, ignore_index = True, random_state = 42)
    # Removes the "Subject:" string, which comprises the first 9 characters of each email. Also, convert it to a numpy array.
    X = df.text.apply(lambda x: x[9:]).to_numpy()
    # Convert the labels to numpy array
    Y = df.spam.to_numpy()
    return X, Y

def preprocess_text(X):
    """
    Preprocesses a collection of text data by removing stopwords and punctuation.
    """
    # Make a set with the stopwords and punctuation
    stop = set(stopwords.words('english') + list(string.punctuation))

    # The next lines will handle the case where a single email is passed instead of an array of emails.
    if isinstance(X, str):
        X = np.array([X])

    # The result will be stored in a list
    X_preprocessed = []

    for i, email in enumerate(X):
        email = np.array([i.lower() for i in word_tokenize(email) if i.lower() not in stop]).astype(X.dtype)
        X_preprocessed.append(email)

    if len(X) == 1:
        return X_preprocessed[0]
    return X_preprocessed

def get_word_frequency(X,Y):
    """
    Calculate the frequency of each word in a set of emails categorized as spam (1) or not spam (0).

    """
    word_dict = {}

    num_emails = len(X)

    for i in range(num_emails):
        email = X[i]
        cls = Y[i]
        email = set(email)
        for word in email:
            if word not in word_dict.keys():
                word_dict[word] = {'spam': 1, 'ham': 1}
            if cls == 0:
                word_dict[word]['ham'] += 1
            if cls == 1:
                word_dict[word]['spam'] += 1

    return word_dict

def prob_word_given_class(word, cls, word_frequency, class_frequency):
    """
    Calculate the conditional probability of a given word occurring in a specific class.

    """
    # Get the amount of times the word appears with the given class (class is stores in spam variable)
    amount_word_and_class = word_frequency[word][cls]
    p_word_given_class = amount_word_and_class/class_frequency[cls]

    return p_word_given_class


def prob_email_given_class(treated_email, cls, word_frequency, class_frequency):
    """
    Calculate the probability of an email being of a certain class (e.g., spam or ham) based on treated email content.

    """

    prob = 1

    for word in treated_email:

        if word in word_frequency.keys():

            prob *= prob_word_given_class(word, cls, word_frequency = word_frequency, class_frequency = class_frequency)

    return prob

def naive_bayes(treated_email, word_frequency, class_frequency, return_likelihood = False):
    """
    Naive Bayes classifier for spam detection.

    This function determines whether an email is likely to be spam (1) or not spam (0) using the Naive Bayes algorithm.
    It relies on the conditional probabilities associated with the treated email being classified as spam or not spam,
    along with the prior probabilities of spam and not spam classes. The ultimate classification is determined by comparing these calculated probabilities.

    """

    prob_email_given_spam = prob_email_given_class(treated_email, 'spam', word_frequency, class_frequency)

    prob_email_given_ham = prob_email_given_class(treated_email, 'ham', word_frequency, class_frequency)

    p_spam = class_frequency['spam']/(class_frequency['spam']+class_frequency['ham'])

    p_ham = class_frequency['ham']/(class_frequency['spam']+class_frequency['ham'])

    spam_likelihood = p_spam * prob_email_given_spam

    ham_likelihood = p_ham * prob_email_given_ham

    if return_likelihood == True:
        return (spam_likelihood, ham_likelihood)

    elif spam_likelihood >= ham_likelihood:
        return 1
    else:
        return 0


def get_true_positives(Y_true, Y_pred):
    """
    Calculate the number of true positive instances in binary classification.

    """
    if len(Y_true) != len(Y_pred):
        return "Number of true labels and predict labels must match!"
    n = len(Y_true)
    true_positives = 0
    for i in range(n):
        true_label_i = Y_true[i]
        predicted_label_i = Y_pred[i]
        if true_label_i == 1 and predicted_label_i == 1:
            true_positives += 1
    return true_positives

def get_true_negatives(Y_true, Y_pred):
    """
    Calculate the number of true negative instances in binary classification.

    """
    if len(Y_true) != len(Y_pred):
        return "Number of true labels and predict labels must match!"
    n = len(Y_true)
    true_negatives = 0
    for i in range(n):
        true_label_i = Y_true[i]
        predicted_label_i = Y_pred[i]
        if true_label_i == 0 and predicted_label_i == 0:
            true_negatives += 1
    return true_negatives

# Main Body

dataframe_emails = pd.read_csv('emails.csv')
dataframe_emails.head()

#exploring the dataset
print(f"Number of emails: {len(dataframe_emails)}")
print(f"Proportion of spam emails: {dataframe_emails.spam.sum()/len(dataframe_emails):.4f}")
print(f"Proportion of ham emails: {1-dataframe_emails.spam.sum()/len(dataframe_emails):.4f}")

X, Y = preprocess_emails(dataframe_emails)

X_treated = preprocess_text(X)

#Splitting data into train/test
TRAIN_SIZE = int(0.80*len(X_treated)) # 80% of the samples will be used to train.
X_train = X_treated[:TRAIN_SIZE]
Y_train = Y[:TRAIN_SIZE]
X_test = X_treated[TRAIN_SIZE:]
Y_test = Y[TRAIN_SIZE:]

# Building the word_frequency dictionary using the training set.
word_frequency = get_word_frequency(X_train,Y_train)

# Counting the spam and ham emails
class_frequency = {'ham': sum(Y_train == 0), 'spam': sum(Y_train == 1)}

Y_pred = []
for email in X_test:
    prediction = naive_bayes(email, word_frequency, class_frequency)
    Y_pred.append(prediction)
print(f"Y_test and Y_pred matches in length? Answer: {len(Y_pred) == len(Y_test)}")


true_positives = get_true_positives(Y_test, Y_pred)
true_negatives = get_true_negatives(Y_test, Y_pred)
print(f"The number of true positives is: {true_positives}\nThe number of true negatives is: {true_negatives}")
accuracy = (true_positives + true_negatives)/len(Y_test)
print(f"Accuracy is: {accuracy:.4f}")
