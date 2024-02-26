# Chess Player Strength Identification from PGN Files

## Introduction
This project aims to develop a model capable of identifying the strength of chess players based solely on the moves recorded in PGN (Portable Game Notation) files. By analyzing games from players of varying skill levels, our model seeks to infer the ratings of participants without prior knowledge of their actual Elo scores. This approach could revolutionize chess coaching strategies and refine the human-centric aspects of engine evaluations by revealing systematic differences in gameplay across different rating bands.

## Data Source
We utilize the Lichess database, specifically a 33.5 GB dataset from January 2023 containing over 100 million games, 7% of which include engine analysis. Our preprocessing efforts focused on extracting games with engine annotations and balancing the dataset across various rating bands and time controls to minimize biases.

## Feature Engineering
Each game is broken down into moves, with each move undergoing a detailed analysis to extract a comprehensive set of features. These features include both intrinsic aspects of the moves themselves and external factors like time pressure and computer assessments of the position. The goal is to capture the essence of player decisions in a structured format suitable for machine learning.

## Supervised Learning
Our machine learning approach involves comparing different models, including Naïve Bayes, Random Forests, and LSTM networks, to understand how chess players of different levels differ in their gameplay. The LSTM model, chosen for its ability to capture temporal dependencies, shows promising results in predicting player strength accurately.

## Supervised Evaluation
We measure our models' performance using raw accuracy and mean cumulative accuracy metrics. The LSTM model outperforms other techniques, demonstrating the importance of considering the sequential nature of chess moves for player strength prediction.

## Feature Analysis
Using the IntegratedGradients method, we analyze the contribution of each feature to the model's predictions. This analysis helps identify patterns and traits associated with different levels of chess play, offering insights into strategic elements that correlate with player strength.

## Ablation and Learning Curve Analysis
We conduct ablation studies to understand the impact of different feature sets on the model's predictive performance. Additionally, learning curve analysis reveals how the model's accuracy improves with increasing dataset sizes, highlighting the importance of a large and diverse training set for reliable predictions.

## Sensitivity Analysis
A comprehensive hyperparameter tuning process identifies key variables affecting model performance. This analysis helps optimize the model by balancing the trade-offs between overfitting and underfitting.

## Failure Analysis
We examine instances of misclassification to understand the limitations of our model. These analyses provide valuable insights into areas for improvement, particularly in handling the complexities of chess strategy and the nuances of game phases.

## Conclusion
This project represents a significant step towards understanding chess gameplay through machine learning. By accurately predicting player strength from PGN files, we unlock new possibilities for coaching, game analysis, and the study of chess as a cognitive activity.

For detailed instructions and code, feel free to contact me or refer to the accompanying Jupyter notebooks.
