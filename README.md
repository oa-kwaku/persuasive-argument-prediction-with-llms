# Experiement Analysis

## OP Analysis of Malleability

### Task

### Approach

- describe the dataset to LLM
- feed LLM OP text and delta examples
- loop through and ask the LLM to predict-then-explain how it arrived at the answer

### Analysis

#### Prompt

```
training_prompt = f"Here is a list of opinions: \n\n { examples }"

    to_predict = evaluation_data()
    prediction_prompt = "classify each row as" + \
                        "delta:True or delta:False, provide ab explanation of 100 characters supported by features" +\
                        f"of the text such as {''.join(FEATURES)} +" \
                        f"```{ to_predict[['title', 'delta_label', 'selftext']] }```"

    format_prompt =\
        "produce your results in valid csv format with header '\'row\', \'delta\', \'explanation\'' for every row" +\
        "explanation should talk about the stylistic features, make sure that you include the row index in" +\
        "the row column of the csv. make sure you produce a valid csv"
```

#### Output
```
"This post argues that tougher gun laws would not have prevented yesterday's tragedy where Bryce Williams killed Alison Parker and Adam Ward. The author believes that if guns were more difficult to access, Bryce Williams would have found another means to carry out the attack. The explanation highlights the use of repetition to emphasize the author's point and the mention of specific names and details related to the tragedy.\""

```

####Accuracy
```
In [223]: op_analysis.measure_accuracy(op, val_df, True)
Index(['row', 'delta', 'explanation'], dtype='object')
      row  delta  ... delta_label  correct
0    60.0  False  ...       False    False
1  1030.0   True  ...       False    False

[2 rows x 5 columns]
Out[223]: 50.0
```

GPT 3.5 works hard to consider the concepts in the argument rather than the stylistic choices event when prompted otherwise 
```
"row,delta,explanation\n597,False,\"The text does not provide any evidence for the claim that being gay is solely due to upbringing and socialization rather than biological factors. It seems to be based on personal speculation and anecdotal experience. There are no references or supporting facts provided.\""
```
### Further Investigation

TK


## Pair Analysis of Effective Argument Features

### Task

- use large language models to predict which of two similar counterarguments will succeed in changing the same view
- study found that style features and interplay features have predictive power
- The feature with the most predictive power of successful persuasion is the dissimilarity with the original post in word usage

### Approach

#### Prompt
```
training_prompt = f"Here is an opinion with an argument that convinced an OP and an argument that did not" + \
                              f" convince the OP. OP will give a ∆ to the winning argument" + \
                              f" so don't factor that in `{evaluation_data()[num_predicted: num_predicted + BATCH_SIZE]}`"

task = f" summarize the difference in the positive and negative arguments by talking about the" +\
           f" contrast in the following features {','.join(FEATURES)} only talk about the top 3 features that" +\
           f"were significantly present in the positive argument but not the negative and limit the output to 100 words"
```

### Analysis

```
The most common text features mentioned as differentiators between positive and negative arguments are:

- Higher number of words
- Presence of definite and indefinite articles
- Higher number of positive words
- Use of 2nd person pronoun "you"
- Inclusion of a link
- Presence of positive words
- Use of question marks
- Inclusion of examples
```
Note: Sample Size was 2

### Further Investigation

Get an idea of proportion, what percentage of results employed each of the following strategies?
