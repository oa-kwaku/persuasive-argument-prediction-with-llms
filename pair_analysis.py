import pandas as pd
from openai_models import gpt3_5 as gpt

pd.set_option('display.max_colwidth', 10000)

EVAL_SAMPLE = 500
BATCH_SIZE = 5

FEATURES = ['#words', '#definite articles', '#indefinite articles', '#positive words', '#2nd person pronoun', '#links',
 '#negative words', '#hedges', '#1st person pronouns','#1st person plural pronoun','#.com links', 'frac. links',
 'frac. .com links', '#examples', 'frac. definite articles', '#question marks', '#PDF links', '#.edu links',
 'frac. positive words', 'frac. question marks', '#quotations', 'arousal', 'valence', 'word entropy', '#sentences',
 'type-token ratio', '#paragraphs', 'Flesch-Kincaid grade levels', '#italics', 'bullet list', '#bolds','numbered words',
 'frac. italics']


def evaluation_data():
    """
    :return: str, the prompt containing example data for the model
    """
    heldout_pair_data = pd.read_json('cmv/pair_task/heldout_pair_data.jsonlist.bz2', compression='bz2', lines=True)

    heldout_pair_data = heldout_pair_data[["positive", "negative", "op_text"]] \
        .apply(lambda x: f" {x.op_text} was convinced by following interaction:" +
                         f" {x.positive} but not by this interaction: {x.negative}", axis=1)
    return heldout_pair_data.sample(n=EVAL_SAMPLE, random_state=1)


def unprimed_prediction(batch=False, verbose=False):
    to_predict = evaluation_data()
    task = f" summarize the difference in the positive and negative arguments by talking about the contrast in argument," +\
    "limit the output to 100 words \n\n"

    if batch:
        num_predicted = 0
        batched_responses = ""

        while num_predicted < len(to_predict):
            training_prompt = str(evaluation_data()[num_predicted: num_predicted + BATCH_SIZE])

            gpt_response = gpt.response(content="".join([task, training_prompt]))
            if gpt_response:
                if verbose:
                    print(gpt_response)

                batched_responses += (gpt_response.choices[0].message.content + ",")
                num_predicted += BATCH_SIZE

            else:
                continue

        return batched_responses
    else:
        response = gpt.response(content="".join([str(evaluation_data()), task]))
        return response.choices[0].message.content


def feature_primed_predictions(batch=False, verbose=False):
    '''GPT analysis'''
    to_predict = evaluation_data()

    task = f" summarize the difference in the positive and negative arguments by talking about the" +\
           f" contrast in the following features {','.join(FEATURES)} only talk about the top 3 features that" +\
           f"were significantly present in the positive argument but not the negative and limit the output to 100 words"

    if batch:
        num_predicted = 0
        batched_responses = ""

        while num_predicted < len(to_predict):
            training_prompt = evaluation_data()[num_predicted: num_predicted + BATCH_SIZE]

            gpt_response = gpt.response(content="".join([training_prompt, task]))
            if gpt_response:
                if verbose:
                    print(gpt_response)

                batched_responses += (gpt_response.choices[0].message.content + ",")
                num_predicted += BATCH_SIZE

            else:
                continue

        return batched_responses
    else:
        training_prompt = f"Here is an opinion with an argument that convinced an OP and an argument that did not" + \
                          f" convince the OP. OP will give a ∆ to the winning argument" + \
                          f" so don't factor that in `{evaluation_data()}`"
        response = gpt.response(content="".join([training_prompt, task]))
        return response.choices[0].message.content
