# E1I4_MachinLearning

Our team performed wine quality classification using randomforest regressor.


First, We got the data of redwine and whitewine.

![1.png](ReadMe%20md%2059ae4d4c8f24482bb9982fdfc4732ba5/1.png)

Next, we used matplot library to check the distribution of quality and the correlation between quality and other variables in an graph.

![2.png](ReadMe%20md%2059ae4d4c8f24482bb9982fdfc4732ba5/2.png)

![3.png](ReadMe%20md%2059ae4d4c8f24482bb9982fdfc4732ba5/3.png)

Thereafter, the data was separated into training set and testing set, and learning was conducted using the RandomForestRegressor model.

![4.png](ReadMe%20md%2059ae4d4c8f24482bb9982fdfc4732ba5/4.png)

![5.png](ReadMe%20md%2059ae4d4c8f24482bb9982fdfc4732ba5/5.png)

The relationship between features is as follows.

![6.png](ReadMe%20md%2059ae4d4c8f24482bb9982fdfc4732ba5/6.png)

To improve model performance, compute permutation importance for feature selection.

![Untitled](ReadMe%20md%2059ae4d4c8f24482bb9982fdfc4732ba5/Untitled.png)

Result of leaving only 6 top features

![Untitled](ReadMe%20md%2059ae4d4c8f24482bb9982fdfc4732ba5/Untitled%201.png)

However, there was no significant difference in the model learning results.

![Untitled](ReadMe%20md%2059ae4d4c8f24482bb9982fdfc4732ba5/Untitled%202.png)

![Untitled](ReadMe%20md%2059ae4d4c8f24482bb9982fdfc4732ba5/Untitled%203.png)

Finally, we tested the applicability of the model to real-world data. We used wine information data and consumer ratings provided by [the sites](https://www.wine21.com/main.html). The real world data used for testing is as follows.

![Untitled](ReadMe%20md%2059ae4d4c8f24482bb9982fdfc4732ba5/Untitled%204.png)

You can see the distribution of consumer ratings.

![Untitled](ReadMe%20md%2059ae4d4c8f24482bb9982fdfc4732ba5/Untitled%205.png)

Put the real world data above into a model to predict scores. score_df is a dataFrame where predicted consumer scores of 10 real-world data are stored.

```python
score_df = pd.DataFrame(columns=range(10))

for i in range (0, 10) :
  data = real_test.iloc[i]
  data.columns = ['volatile acidity', 'residual sugar', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates', 'alcohol']
  score = forest.predict([data])
  score_df[i] = score
```

You can compare the actual consumer ratings with the distribution of ratings predicted by the model.

![Untitled](ReadMe%20md%2059ae4d4c8f24482bb9982fdfc4732ba5/Untitled%206.png)
