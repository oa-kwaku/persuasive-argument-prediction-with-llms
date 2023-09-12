import pandas as pd
from openai_models import gpt3_5 as gpt
from io import StringIO

pd.set_option('display.max_colwidth', 10000)

EVAL_SAMPLE = 2
BATCH_SIZE = 1


def evaluation_data():
    """
    :return:
    """
    heldout_pair_data = pd.read_json('cmv/pair_task/heldout_pair_data.jsonlist.bz2', compression='bz2', lines=True)

    heldout_pair_data = heldout_pair_data[["positive", "negative", "op_text"]] \
        .apply(lambda x: f" {x.op_text} was convinced by following interaction:" +
                         f" {x.positive} but not by this interaction: {x.negative}", axis=1)
    return heldout_pair_data.sample(n=EVAL_SAMPLE, random_state=1)


def predictions(batch=False):
    to_predict = evaluation_data()
    prediction_prompt = f"explain for each row why the positive performed better than the negative"

    format_prompt = f"out put the response in csv format the headings should be " + \
                    f" 'op_text, positive_strategies, negative_strategies' then for each row" + \
                    f" `op_text: 'op_text', positive_strategies: 1. strategy1 2. strategy2 3. strategy3," + \
                    f" negative_strategies: 1. strategy1 2. strategy2 3. strategy3`"

    if batch:
        num_predicted = 0
        batched_responses = ""

        while num_predicted < len(to_predict):
            training_prompt = f"Here is an opinion with an argument that convinced an OP and an argument that did not" + \
                              f" convince the OP. OP will give a ∆ to the winning argument" + \
                              f" so don't factor that in `{evaluation_data()[num_predicted: num_predicted + BATCH_SIZE]}`"

            gpt_response = gpt.response(content="".join([training_prompt, prediction_prompt, format_prompt]))
            if gpt_response:
                print(gpt_response)
                batched_responses += (gpt_response.choices[0].message.content + ",")
                num_predicted += BATCH_SIZE

            else:
                continue

        return pd.concat([pd.read_csv(StringIO(x)) for x in batched_responses])
    else:
        training_prompt = f"Here is an opinion with an argument that convinced an OP and an argument that did not" + \
                          f" convince the OP. OP will give a ∆ to the winning argument" + \
                          f" so don't factor that in `{evaluation_data()}`"
        response = gpt.response(content="".join([training_prompt, prediction_prompt, format_prompt]))
        return response.choices[0].message.content
