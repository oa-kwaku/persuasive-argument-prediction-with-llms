import pandas as pd
from openai_models import gpt3_5 as gpt
from io import StringIO

pd.set_option('display.max_colwidth', 10000)

TRAIN_SAMPLE = 25
EVAL_SAMPLE = 2
BATCH_SIZE = 1

FEATURES = ['#words', '#definite articles', '#indefinite articles', '#positive words', '#2nd person pronoun', '#links',
 '#negative words', '#hedges', '#1st person pronouns','#1st person plural pronoun','#.com links', 'frac. links',
 'frac. .com links', '#examples', 'frac. definite articles', '#question marks', '#PDF links', '#.edu links',
 'frac. positive words', 'frac. question marks', '#quotations', 'arousal', 'valence', 'word entropy', '#sentences',
 'type-token ratio', '#paragraphs', 'Flesch-Kincaid grade levels', '#italics', 'bullet list', '#bolds','numbered words',
 'frac. italics']


def prompt_example_data():

    """
    :return: str, the prompt containing example data for the model
    """
    train_op_data = pd.read_json('cmv/op_task/train_op_data.jsonlist.bz2', compression='bz2', lines=True)

    sample_op_data = train_op_data.sample(n=TRAIN_SAMPLE, random_state=1)

    return sample_op_data[["delta_label", "title", "selftext"]]\
        .apply(lambda x: f"'{x.title}: {x.selftext}' is defined as delta={x.delta_label}", axis=1)\
        .to_string(index=False, header=False)


def evaluation_data():
    """
    :return: str, the prompt containt the example data for the model to predict
    """
    heldout_op_data = pd.read_json('cmv/op_task/heldout_op_data.jsonlist.bz2', compression='bz2', lines=True)
    return heldout_op_data.sample(n=EVAL_SAMPLE, random_state=1)


def predictions(batch=False, verbose=False):
    examples = prompt_example_data()
    training_prompt = f"Here is a list of opinions: \n\n { examples }"

    to_predict = evaluation_data()
    prediction_prompt = "classify each row as" + \
                        "delta:True or delta:False, provide a explanation of 100 characters supported by features" +\
                        f"of the text such as {''.join(FEATURES)} " +\
                        f"```{ to_predict[['title', 'delta_label', 'selftext']] }```"

    format_prompt =\
        "produce your results in valid csv format with header '\'row\', \'delta\', \'explanation\'' for every row" +\
        "explanation should talk about the stylistic features, make sure that you include the row index in" +\
        "the row column of the csv. make sure you produce a valid csv"

    if batch:
        num_predicted = 0
        batched_responses = ""
        while num_predicted < len(to_predict):
            to_predict = evaluation_data()
            to_process = to_predict[num_predicted: num_predicted + BATCH_SIZE]
            prediction_prompt = "classify each row as" + \
                                "delta:True or delta:False, provide a explanation of 100 characters supported by features" +\
                                f" of the text such as {''.join(FEATURES)} " +\
                                f"{''.join(FEATURES)} ``` {to_process[['title', 'delta_label', 'selftext']]} ```"

            gpt_response = gpt.response(content="".join([training_prompt, prediction_prompt, format_prompt]))
            if gpt_response:
                if verbose:
                    print(gpt_response)
                batched_responses += (gpt_response.choices[0].message.content + "\n")
                num_predicted += BATCH_SIZE
            else:
                continue

        return batched_responses
    else:
        response = gpt.response(content="".join([training_prompt, prediction_prompt, format_prompt]))
        return response.choices[0].message.content


def fix_csv(csv):
    return gpt.response(content=f"here is the following csv, please reformat so that it is in valid format {csv}")\
        .choices[0].message.content


def measure_accuracy(gpt_response, validation_df, verbose=False):
    prediction_result = pd.read_csv(StringIO(gpt_response))

    prediction_result["delta"] = prediction_result["delta"].astype(bool)

    prediction_result["row"] = pd.to_numeric(prediction_result.row, errors='coerce')
    prediction_result = prediction_result.dropna(subset=["row"])

    prediction_result = pd.merge(prediction_result, validation_df[["delta_label", "row"]], on='row')
    prediction_result["correct"] = prediction_result["delta"] == prediction_result["delta_label"]

    if verbose:
        print(prediction_result)

    return len(prediction_result[prediction_result["correct"] == True]) / len(prediction_result)
