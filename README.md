# Tomoro AI Take-home Assignment (ConvFinQA)

## Data Exploration

In my exploration of the data, I noticed that the data itself has some imperfections. Firstly, there are rows where the "table" doesn't make sense. For example, the first datapoint in the JSON has it's columns given as: 
["2008",
"year ended june 30 2009 2008",
"year ended june 30 2009 2008",
"year ended june 30 2009"],
a clear mistake. Given more time, I would do a deeper dive into how we might go about cleaning up these imperfections, or at the very least flagging bad datapoints.

## Accuracy Metrics

The answers in the dataset are all numeric. I chose to use the absolute percentage error as my accuracy metric, calculated as:
\[ APE = \left|\frac{\text{actual} - \text{expected}}{\text{expected}}\right| \times 100\% \]

I chose this for a few reasons. Firstly, the dataset's "answer" values represent financial metrics like revenue, profit margins, and growth rates. Financial data can vary greatly in magnitude (from small percentages to billions in revenue). Percentage error normalizes these differences, making errors comparable across different scales.

Secondly, I thought it important to consider the use-case of such a model. The nature of the dataset means this model will most likely be used in business cases, where I believe the absolute percentage error's interpretability would be better than other metrics for numerical errors (RMSE etc.). We can easily make a statement like "the prediction was off by X%" and can reasonably assume this would be understood by non-technical stakeholders.

Finally, in financial calculations we can assume that both over and under-predictions are equally as problematic for model performance, so we need some form of symmetry. The absolute value ensures this is the case.

## Further Development Options

In my output I highlight the datapoints used. Given more time, I would try to do some validation of these against the question, and potentially also against the table to make sure they are being extracted properly.