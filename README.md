# Naive-Bayes
The Naive Bayes algorithm holds a pivotal role in Machine Learning and Data Science, employing Bayes' Theorem to classify data points into specific classes. This algorithm operates under the "naive assumption" that each feature is independent of the others, a simplification that may not hold true for all datasets. However, despite this simplification, Naive Bayes offers a straightforward implementation and often yields remarkably useful outcomes. It's crucial to understand that Naive Bayes is a supervised algorithm, necessitating labeled data for effective operation. For instance, in the context of email classification, it would require a dataset comprising emails already categorized as "spam" or "ham" to train the model.

### Naive Bayes for Spam Detection

The focus is on a binary classification problem: distinguishing between spam and non-spam emails, colloquially referred to as "ham." The spam emails will be labeled as $1$, and non-spam (ham) emails as $0$.

The probability of interest for a given email is denoted as:

$$ P(\text{spam} \mid \text{email}) $$

The higher this probability, the more likely the email is to be classified as spam. Bayes' Theorem, which you saw in the lectures, is used in the calculation in the following way:

$$ P(\text{spam} \mid \text{email}) = \frac{P(\text{email} \mid \text{spam}) \cdot P(\text{spam})}{P(\text{email})} $$

Here's a breakdown of the terms:

- $ P(\text{spam}) $: Probability of a randomly selected email being spam, equivalent to the proportion of spam emails in the dataset.
- $ P(\text{email} \mid \text{spam}) $: Probability of a specific email occurring given that it is known to be spam.
- $ P(\text{email}) $: Overall probability of the email occurring.

An interesting early "shortcut" you can take in this approach is just ignore the $ P(\text{email}) $ term. The goal of this calculation will be to compare the probability an email is spam to the probability is ham. Here's the expression for both $ P(\text{spam} \mid \text{email}) $ and $ P(\text{ham} \mid \text{email}) $:

$$ P(\text{spam} \mid \text{email}) = \frac{P(\text{email} \mid \text{spam}) \cdot P(\text{spam})}{P(\text{email})} $$

$$ P(\text{ham} \mid \text{email}) = \frac{P(\text{email} \mid \text{ham}) \cdot P(\text{ham})}{P(\text{email})} $$

Since $ P(\text{email}) > 0 $ and it appears in both expressions, comparing the two probabilities only requires evaluating the numerators and you can ignore this denominator.
