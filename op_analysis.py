import pandas as pd
from openai_models import gpt3_5 as gpt


TRAIN_SAMPLE = 30
EVAL_SAMPLE = 10
BATCH_SIZE = 1

pd.set_option('display.max_colwidth', 10000)


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


def predictions(batch=False):
    examples = prompt_example_data()
    training_prompt = f"Here is a list of opinions: \n\n { examples }"

    to_predict = evaluation_data()
    prediction_prompt = f"predict if each each row of the csv is delta:True or delta:False" +\
        f"then explain why you came at the predictions\n\n { to_predict[['title', 'delta_label', 'selftext']] } \n\n"

    format_prompt=\
        "please produce your results in this valid JSON format ```{ row: '(the dataframe index)' delta: '' explanation: '' }``` for every row \n " \
        "please ensure that what you print is valid json and you remove newlines" +\
        "make sure that explanation is 100 characters and has no escape literals"

    if batch:
        num_predicted = 0
        batched_responses = ""
        while num_predicted < len(to_predict):
            to_predict = evaluation_data()
            to_process = to_predict[num_predicted: num_predicted + BATCH_SIZE]
            prediction_prompt = f"predict if each each row of the csv is delta:True or delta:False" + \
                                f"then explain why you came at the predictions\n\n { to_process[['title', 'delta_label', 'selftext']]} \n\n"

            gpt_response = gpt.response(content="".join([training_prompt, prediction_prompt, format_prompt]))
            if gpt_response:
                print(gpt_response)
                batched_responses += (gpt_response.choices[0].message.content + "\n")
                num_predicted += BATCH_SIZE
            else:
                continue

        return batched_responses
    else:
        response = gpt.response(content="".join([training_prompt, prediction_prompt, format_prompt]))
        return response.choices[0].message.content


def measure_accuracy(predictions, validation_df, verbose=False):
    prediction_result = pd.json_normalize(predictions)
    prediction_result["delta"] = prediction_result.delta == 'True'  # hack to force boolean from json

    validation_df["row"] = validation_df.index

    prediction_result = pd.merge(prediction_result, validation_df[["delta_label", "row"]], on='row')
    prediction_result["correct"] = prediction_result["delta"] == prediction_result["delta_label"]

    if verbose:
        print(prediction_result)

    return len(prediction_result[prediction_result["correct"] == True]) / len(prediction_result)
