# E1I4_MachineLearning

Our team performed wine quality classification using randomforest regressor.  

First, We got the data of redwine and whitewine.

![dataoverview](https://user-images.githubusercontent.com/115680658/208426990-2b10c28d-f399-4f1e-bf5e-faec467823da.png)

Next, we used matplot library to check the distribution of quality and the correlation between quality and other variables in an graph.

![qualitydistribution](https://user-images.githubusercontent.com/115680658/208427046-c28e9d90-34a8-485c-b930-5629387cd808.png)

![heatmap](https://user-images.githubusercontent.com/115680658/208427086-1e20cbb6-ed75-4d5e-ba11-e66caf7852df.png)

Thereafter, the data was separated into training set and testing set, and learning was conducted using the RandomForestRegressor model.

<img width="372" alt="tracc" src="https://user-images.githubusercontent.com/115680658/208422702-8e4c46e0-3c95-438d-9b3f-4fc2e9fe7b31.png">
<img width="359" alt="teacc" src="https://user-images.githubusercontent.com/115680658/208422750-9b08fd61-bfc5-43ff-ac0a-3f9ce486ece6.png">

The relationship between features is as follows.

![6.png](ReadMe%20md%2059ae4d4c8f24482bb9982fdfc4732ba5/6.png)

To improve model performance, compute permutation importance for feature selection.

![Untitled](https://user-images.githubusercontent.com/115680658/208420916-3b064d43-5f08-4351-a70e-778043f16cd6.png)

Result of leaving only 6 top features

[<img width="929" alt="top6features" src="https://user-images.githubusercontent.com/115680658/208422501-6d3c9704-c230-436e-b1f5-472b004bd585.png">](https://github.com/clairew99/E1I4_MachinLearning/issues/2#issue-1502834703)

However, there was no significant difference in the model learning results.

[<img width="372" alt="tracc" src="https://user-images.githubusercontent.com/115680658/208422702-8e4c46e0-3c95-438d-9b3f-4fc2e9fe7b31.png">](https://github.com/clairew99/E1I4_MachinLearning/issues/3#issue-1502836586)

[![Untitled](ReadMe%20md%2059ae4d4c8f24482bb9982fdfc4732ba5/Untitled%203.png)](https://github.com/clairew99/E1I4_MachinLearning/issues/4#issue-1502837142)

Finally, we tested the applicability of the model to real-world data. We used wine information data and consumer ratings provided by [the sites](https://www.wine21.com/main.html). The real world data used for testing is as follows.

![real-world data](https://user-images.githubusercontent.com/29935149/208426604-a2db50da-9271-4cd4-bcc5-5ef8f4dddfd5.PNG)

You can see the distribution of consumer ratings.

![consumer_score](https://user-images.githubusercontent.com/29935149/208426912-0d9a1153-4de4-44e1-bbc5-0fdbae6c22fb.png)

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

![model_predict score](https://user-images.githubusercontent.com/29935149/208426930-d4226b2f-172c-4f0e-ab17-9605a00580db.png)
