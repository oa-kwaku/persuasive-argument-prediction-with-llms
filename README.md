# Persuasive argument prediction

After Reading Tan et al. I set out to replicate their study on persuasion observed on r/changemyview. 

There were two objectives for this exercise:

1. Assess the Malleability of the Original Poster (OP) via GPT 3.5 API's
   
2. Analyze pairs of arguments attempting to change an OP's view and then assess the difference between the argument that convinced the OP versus the argument that did not also via GPT 3.5 API's.

For more context, please refer to [https://chenhaot.com/papers/changemyview.html](https://chenhaot.com/papers/changemyview.html).

## 1. OP Analysis of Malleability

### Task

For this objective, I had a JSON dataset of OP opinions. I wanted to see if I could predict which OP's had "malleable" minds with the help of GPT 3.5 and prompting

### Approach

I tried two different strategies to try to predict the malleability of OP's:

1. **Unprimed Approach** Rely on the encoded knowledge GPT 3.5 might have about open-mindedness and produce an unrimed prompt

2. **Feature Primed Approach** Tan et al. found a battery of statistically significant predictive text features that indicated an OP's malleability. I set out to prime GPT 3.5 with those features to see if I could beat the unprimed approach. Features are found below

```
FEATURES = ['#words', '#definite articles', '#indefinite articles', '#positive words', '#2nd person pronoun', '#links',
            '#negative words', '#hedges', '#1st person pronouns', '#1st person plural pronoun', '#.com links',
            'frac. links',
            'frac. .com links', '#examples', 'frac. definite articles', '#question marks', '#PDF links', '#.edu links',
            'frac. positive words', 'frac. question marks', '#quotations', 'arousal', 'valence', 'word entropy',
            '#sentences',
            'type-token ratio', '#paragraphs', 'Flesch-Kincaid grade levels', '#italics', 'bullet list', '#bolds',
            'numbered words',
            'frac. italics']
```

#### Unprimed Approach

I prompted GPT 3.5 with the following prompt

>I have a set of predictions here \n\n{examples}\n\n I want you to explain what characteristics make an OP malleable, now tell me, out of the following opinions that I have given you, how many of these op's are malleable based on your predictions. append to your explanation the number correct then list the row numbers with the variable { examples } I passed in `heldout_op_data` from `'cmv/op_task/heldout_op_data.jsonlist.bz2'` for GPT 3.5 to assess

**Example explanation output from GPT 3.5**

>"it means that the original poster (OP) had their view changed by the responses and arguments presented by others. In this context, a malleable OP refers to an OP who is open-minded and willing to consider different perspectives and potentially change their initial view based on the arguments and evidence presented. Characteristics that make an OP malleable include:\n\n1. Openness to different viewpoints: The OP is receptive to hearing different perspectives and is willing to engage in a discussion about their beliefs.\n\n2. Willingness to evaluate evidence: The OP is open to examining evidence and considering logical arguments that challenge their initial view. They are not dismissive of opposing viewpoints but rather seek to understand and evaluate the validity of alternative arguments.\n\n3. Adaptability to new information: The OP is willing to incorporate new information and insights into their thinking process. They may realize that their initial view was based on incomplete or flawed information and are open to changing their perspective based on more accurate or comprehensive information.\n\n4. Reflective and analytical thinking: The OP engages in critical thinking and reflects on their own beliefs and assumptions. They are able to consider different angles and perspectives, weighing the strengths and weaknesses of their own arguments and the arguments presented by others.\n\n5. Non-defensive attitude: The OP does not become defensive or dismissive when their beliefs are challenged. They engage in respectful and constructive discussions, valuing the exchange of ideas and insights rather than seeking to defend their position at all costs.\n\nIt is important to note that having a malleable OP does not guarantee a change in view, as the OP may still maintain their original perspective after considering different arguments. However, a malleable OP is more open to the possibility of changing their view based on the convincing arguments and evidence presented by others."

#### Feature Primed Approach

I prompted GPT 3.5 with the following prompt

>Here is a list of opinions: \n\n {examples} classify each row as delta:True or delta:False, provide an explanation of 100 characters supported by features of the text such as 
>{''.join(FEATURES)} ```{to_predict[['title', 'delta_label', 'selftext']]}``` produce your
>results in valid csv format with header '\'row\', \'delta\', \'explanation\'' for every row >explanation should talk about the stylistic features, make sure that you include the row index in >the row column of the csv. make sure you produce a valid csv"

the variable `{ examples }` in this case are 20 rows from the training set fed into the prompt in text format.
(sadly GPT 3.5 did not produce a consistent csv, although it was simple enough to engineer it to one in PyCharm)

**Sample Explanation**
>"This post argues that tougher gun laws would not have prevented yesterday's tragedy where Bryce Williams killed Alison Parker and Adam Ward. The author believes that if guns were more difficult to access, Bryce Williams would have found another means to carry out the attack. The explanation highlights the use of repetition to emphasize the author's point and the mention of specific names and details related to the tragedy."

### Analysis

**Unprimed Model Fit**

I performed an Area Under the Receiver Operating Characteristic (AUROC) analysis and got a result under 0.5. This essentially means the fit of the model was worse than what one expects from a human naively predicting.
```
In [389]: auc_score_unprimed = roc_auc_score(validation_numeric["delta_label_numeric"], validation_numeric[
     ...: "unprimed_numeric"])

In [390]: auc_score_unprimed
Out[390]: 0.48747769667477703
```

I then asked the LLM to take the explanations of each row and summarize what characteristics make the OP malleable

>The most common factors that made this group of OPs less malleable are:\n\n1. Strong opinions without mentioning openness to alternative viewpoints.\n2. Firm beliefs without expressing a desire to hear counter-arguments or be persuaded otherwise.\n3. Lack of mention of being open to changing their views.\n4. Clear stance on an issue, leaving little room for malleability.\n5. Stating a belief without indicating a willingness to question or self-reflect.\n\nThese factors suggest that a lack of openness to considering new information, perspectives, or evidence, as well as a strong emotional attachment to their opinions, contribute to the OPs being less malleable


What is interesting here is that GPT 3.5 mainly described the disposition of the OP rather than textual or argumentative features. See the example of a single opinion below.
>* Malleable: Yes. This opinion has the potential to be changed if the OP is open to considering different perspectives on Snowden's support from the left.

I suspect this might be because of the model's fine-tuning to be conversational.

**Feature Primed Model Fit**

In the feature-primed model, we saw modestly better performance. per the AUROC analysis. I would like to take further samples of a similar size and see if we are statistically significant in our finding that the feature-primed approach performed better than the unprimed approach. I did find a seemingly high false positive rate of `0.9610` for the unprimed approach versus `0.5428` for the feature-primed approach.
```
In [391]: auc_score_feature_primed = roc_auc_score(validation_numeric["delta_labe
     ...: l_numeric"], validation_numeric["feature_primed_numeric"])

In [392]: auc_score_feature_primed
Out[392]: 0.5280129764801298
```

I then asked the LLM to take the explanations of each row and summarize what characteristics make the OP malleable

> This collection of posts all demonstrates different viewpoints and perspectives on various topics. However, they all have commonalities in how they express 
> personal opinions and beliefs without providing evidence or counterarguments. They also tend to use emotional language and may have dismissive tones towards 
> opposing arguments. Some posts mention specific events or examples to support their viewpoint, while others include personal experiences or anecdotes. 
> Overall, these posts reflect the subjective nature of personal opinions and the role that emotions can play in shaping them.

> Commonalities among the explanations of why a person's post indicates they are malleable include the use of personal beliefs or opinions, the mention of 
> supporting evidence or sources, and the inclusion of specific words or phrases that indicate a positive or negative sentiment. Additionally, some explanations
>  mention the presence of logical arguments or reasoning to support the views expressed in the posts.

When primed with features the LLM considers argumentative structures and rhetorical devices used in the text. Language like "They also tend to use emotional language and may have dismissive tones towards opposing arguments" is indicative of this bias

>"row,delta,explanation\n597,False,\"The text does not provide any evidence for the claim that being gay is solely due to upbringing and socialization rather than biological factors. It seems to be based on personal speculation and anecdotal experience. There are no references or supporting facts provided.\""

In the above analysis, we see that the LLM was able to the assess that the OP was speculating and relying on anecdotes rather than facts. This is quite interesting considering that `anecdotes` or `# of facts` were not in the original list of features that I provided. By priming the LLM it seems as though I forced the LLM to consider the rhetorical structure of the text rather than the OP's disposition.

### Further Investigation

We received better model performance and a more cogent explanation when priming the LLM with features. I would like to test several prompting strategies in which I prime the LLM with different frameworks to consider when performing its analysis of the text. LLMs have been found to be good at analogical reasoning. I would try to see if incorporating a strategy akin to [ideal point estimation]([url](https://blogs.iq.harvard.edu/ideal_points_1#:~:text=Ideal%20point%20models%20attempt%20to,that%20they%20cast%20on%20legislation.)) might yield better results. Generally the thought would be to prime the large language model with characteristics of what a malleable OP's text would look like and what an unmalleable OP's text would look like. and see if that analogy could help the LLM better predict.

## Pair Analysis of Effective Argument Features

### Task

I set out to use LLMs to predict which of two similar counterarguments will succeed in changing the same view.
Tan et al. found that style features and interplay features have predictive power. 

- The feature with the most predictive power of successful persuasion is the dissimilarity with the original post in word usage


### Unprimed Approach
I prompted GPT 3.5 with the following:

 ```f"Please summarize and list the commonalities in this list of explanations {pair}"```

 the variable  `{ pair }` in this case in 500 rows, batched in a series of 5 rows from the held-out set, fed into the prompt in text format.

**Unprimed Prompt output from GPT 3.5**
>"To summarize the common ways to successfully change someone's mind:\n\n1. Present logical and rational arguments.\n2. Appeal to emotions.\n3. Provide alternative perspectives.\n4. Engage in respectful dialogue.\n5. Use storytelling and personal anecdotes.\n6. Focus on common ground.\n7. Provide evidence and examples.\n8. Be patient and persistent."

### Feature Primed Approach

I prompted GPT 3.5 with the following in order to perform the task

```
summarize the difference in the positive and negative arguments by talking about the contrast in the following features {','.join(FEATURES)} only talk about the top 3 features that were significantly present in the positive argument but not the negative and limit the output to 100 words
```

**Sample Output**
>"In summary, the positive argument was more convincing and persuasive compared to the negative argument due to several key features:\n\n1. Use of positive words: The positive argument used language that conveyed an optimistic and supportive tone, which likely influenced the reader's perception of the argument.\n\n2. Inclusion of links: The positive argument provided additional evidence and supporting information through the use of links, which enhanced the credibility and persuasiveness of the argument.\n\n3. Use of examples: The positive argument included specific examples to illustrate and strengthen its points, which made the argument more relatable and persuasive.\n\nOverall, the positive argument utilized language, evidence, and examples to create a compelling and persuasive case. In contrast, the negative argument lacked these persuasive elements, making"

In the sample output of one of the batches, one can get a sense of how the LLM is thinking of arguments when primed by features.

I concatenated the outputs and asked GPT 3.5 to give me a list of the top features

>The most common text features mentioned as differentiators between positive and negative arguments are:
   - Higher number of words
   - Presence of definite and indefinite articles
   - Higher number of positive words
   - Use of 2nd person pronoun "you"
   - Inclusion of a link
   - Presence of positive words
   - Use of question marks
   - Inclusion of examples


### Analysis

Contrasting the pairwise analysis from the unprimed and primed approach we get results that are quite differential. The only features conceptually similar are evidence and examples in the unprimed output and evidence of links in the feature-primed output.  In the unprimed output, GPT 3.5 focuses on how to appeal emotionally to an OP, "appeal to emotions", "respectful dialogue", and "common ground" are all strategies to maximize the agreeableness of OP and positive poster. In the feature-primed output, the LLM appears to suggest that appearing authoritative is the key to changing views. "Higher number of words, use of question marks (sign of rhetorical questions), use of 2nd person pronoun "you" illustrate this.

### Further Investigation

In further investigration I would have liked to get a distributional sense of how often each top strategy was used and also a temporal sense of when strategies were deployed. Tan et al.'s cutting up responses into quantiles was effective in this. perhaps taking a handful of positive-negative pairs and feeding them into the LLM quadrant by quadrant could give us a temporal sense of when the strategies were deployed.
