import os
import os.path as osp
import json

REASONING_RESP_INST = {
    1: """{}
    * Your final answer must be grounded to some text that is explicitly written and relevant to the question in the chart.
    * If you need to answer multiple terms, separate them with commas.
    * Unless specified in the question (such as answering with a letter), you are required to answer the full names of subplots and/or labels by default.
    """,
    2: """{}
    * If there are options in the question, your final answer must conform to one of the options.
    * If there are additional instructions in the question, follow them accordingly.
    * If there are neither options nor additional instructions, you are allowed to respond with a short phrase only.
    """,
    3: """{}
    * Your final answer must be grounded to a number that is exlicitly written and relevant to the question in the chart, even if it's an approximate value.
    * You are allowed to extract numbers within some text when needed.
    """,
    4: """{}
    {}
    """,
}

DESCRIPTIVE_RESP_INST = {
    1: """{}what is its title?
    * Your final answer should be the most relevant title of the plot that is explicitly written.
    * If the plot does not have an explicit title or contains only a letter, answer 'Not Applicable'.
    """,
    2: """{}what is the label of the x-axis?
    * Your final answer should be the label of the x-axis that is explicitly written, including the case when x-axis is shared across multiple subplots. When the x-axis is present on both the top and bottom of the plot, answer the label of the x-axis at the bottom.
    * If the plot does not have an explicit x-axis label, answer 'Not Applicable'.
    """,
    3: """{}what is the label of the y-axis?
    * Your final answer should be the label of the y-axis that is explicitly written, including the case when y-axis is shared across multiple subplots. When the y-axis is present on both the left and right of the plot, answer the label of the y-axis at the left.
    * If the plot does not have an explicit y-axis label, answer 'Not Applicable'.""",
    4: """{}what is the leftmost labeled tick on the x-axis?
    * Your final answer should be the tick value on the x-axis that is explicitly written, including the case when x-axis is shared across multiple subplots. When the x-axis is present on both the top and bottom of the plot, answer based on the axis at the bottom. Ignore units or scales that are written separately from the tick, such as units and scales from the axis label or the corner of the plot.""",
    5: """{}what is the rightmost labeled tick on the x-axis?
    * Your final answer should be the tick value on the x-axis that is explicitly written, including the case when x-axis is shared across multiple subplots. When the x-axis is present on both the top and bottom of the plot, answer based on the axis at the bottom. Ignore units or scales that are written separately from the tick, such as units and scales from the axis label or the corner of the plot.""",
    6: """{}what is the spatially lowest labeled tick on the y-axis?
    * Your final answer should be the tick value on the y-axis that is explicitly written, including the case when y-axis is shared across multiple subplots. When the y-axis is present on both the left and right of the plot, based on the axis at the left. Ignore units or scales that are written separately from the tick, such as units and scales from the axis label or the corner of the plot.""",
    7: """{}what is the spatially highest labeled tick on the y-axis?
    * Your final answer should be the tick value on the y-axis that is explicitly written, including the case when y-axis is shared across multiple subplots. When the y-axis is present on both the left and right of the plot, based on the axis at the left. Ignore units or scales that are written separately from the tick, such as units and scales from the axis label or the corner of the plot.""",
    8: """{}what is difference between consecutive numerical tick values on the x-axis?
    * Your final answer should be the difference between consecutive numerical tick values of the x-axis, including the case when x-axis is shared across multiple subplots. When the x-axis is present on both the top and bottom of the plot, answer based on the axis at the bottom. Ignore units or scales that are written separately from the tick, such as units and scales from the axis label or the corner of the plot.
    * If the plot does not have an explicit x-axis tick value, or if the tick values are not numerical, or if the difference is not constant between all consecutive tick values, answer "Not Applicable".""",
    9: """{}what is difference between consecutive numerical tick values on the y-axis?
    * Your final answer should be the difference between consecutive numerical tick values of the y-axis, including the case when y-axis is shared across multiple subplots. When the y-axis is present on both the left and right of the plot, answer based on the axis at the left. Ignore units or scales that are written separately from the tick, such as units and scales from the axis label or the corner of the plot.
    * If the plot does not have an explicit y-axis tick value, or if the tick values are not numerical, or if the difference is not constant between all consecutive tick values, answer "Not Applicable".""",
    10: """{}how many lines are there?
    * Your final answer should be the number of lines in the plot. Ignore grid lines, tick marks, and any vertical or horizontal auxiliary lines.
    * If the plot does not contain any lines or is not considered a line plot, answer "Not Applicable".""",
    11: """{}do any lines intersect?
    * Your final answer should be "Yes" if any lines intersect, and "No" otherwise. Ignore grid lines, tick marks, and any vertical or horizontal auxiliary lines.
    * If the plot does not contain any lines or is not considered a line plot, answer "Not Applicable".""",
    12: """{}how many discrete labels are there in the legend?
    * Your final answer should account for only labels relevant to the plot in the legend, even if the legend is located outside the plot.
    * If the plot does not have a legend or no legend is not considered relevant to this plot, answer "Not Applicable".""",
    13: """{}what are the names of the labels in the legend?
    * You should write down the labels from top to bottom, then from left to right and separate the labels with commas. Your final answer should account for only labels relevant to the plot in the legend, even if the legend is located outside the plot.
    * If the plot does not have a legend or no legend is not considered relevant to this plot, answer "Not Applicable".""",
    14: """{}what is the difference between the maximum and minimum values of the tick labels on the continuous legend (i.e., colorbar)?
    * You should remove the percentage sign (if any) in your answer.
    * If the plot does not have an explicit colorbar-based continuous legend or the legend is not considered relevant to this subplot, answer "Not Applicable".""",
    15: """{}what is the maximum value of the tick labels on the continuous legend (i.e., colorbar)?
    * You should remove the percentage sign (if any) in your answer.
    * If the plot does not have an explicit colorbar-based continuous legend or the legend is not considered relevant to this subplot, answer "Not Applicable".""",
    16: """{}what is the general trend of data from left to right?
    * Your final answer should be within a few words, such as "increases", "increases then stabilizes".""",
    17: """{}What is the total number of explicitly labeled ticks across all axes?
    * Your final answer should be the total number of explicitly labeled ticks across all axes, including the case when any axis is shared across multiple subplots.""",
    18: """What is the layout of the subplots?
    * Your final answer should follow "n by m" format, where n is the number of rows and m is the number of columns.
    * If the plot does not contain subplots, answer "1 by 1".""",
    19: """What is the number of subplots?
    * Your final answer should be the total number of subplots in the plot.
    * If the plot does not contain subplots, answer "1".""",
}


def get_number_instruction(answer):
    base = answer.split(".")
    whole, decimal = base[0], None if len(base) == 1 else base[1]
    # check if it contains decimal places
    if whole is not None and decimal is None:
        inst = "* Your final answer must be an exact integer."
    elif whole is not None and decimal is not None:
        num_decimal = len(decimal)
        inst = (
            f"* Your final answer must be a number with {num_decimal} decimal places."
        )
    else:
        raise ValueError(f"Invalid answer: {answer}")
    return inst


def build_reasoning_queries(reasoning_data, descriptive_data, image_meta):
    queries = []
    cnt = 0
    for k, d in reasoning_data.items():
        figure_path = os.path.join("images", f"{d['figure_id']}.jpg")
        inst_category = d["inst_category"]
        # 1: text-in-chart, 2: text-in-general, 3: number-in-chart
        if inst_category in [1, 2, 3]:
            question = REASONING_RESP_INST[inst_category].format(d["query"])
        # 4: number-in-general -> need to specify the number of decimal places
        elif inst_category == 4:
            question = REASONING_RESP_INST[inst_category].format(
                d["query"], get_number_instruction(d["answer"])
            )
        else:
            raise ValueError(f"Invalid instruction category: {inst_category}")
        query = {
            "figure_id": d["figure_id"],  # figure_id
            "img_path": figure_path,  # figure_path
            "inst_category": inst_category,  # instruction category
            "raw_question": d["query"],  # question @@@ without @@@ instruction
            "question": question,  # question with instruction
            "question_id": f"reasoning_{cnt}",
            "answer": d["answer"],  # answer
            "question_type": "reasoning",
            "image_meta": image_meta[str(d["figure_id"])],
            "num_subplots": descriptive_data[k]["num_subplots"],
            "qa_source": d["qa_source"],
        }
        cnt += 1
        queries.append(query)
    print("quries number: ", len(queries))
    return queries


def descriptive_query_helper(qid, subplot_loc):
    if qid in [18, 19]:
        # skip subplot location when asking about the layout of the subplots
        return DESCRIPTIVE_RESP_INST[qid]
    if isinstance(subplot_loc, list):
        if subplot_loc[0] == 0:
            # when there is only one subplot
            prefix = "For the current plot, "
        else:
            # when there are multiple subplots
            prefix = (
                f"For the subplot at row {subplot_loc[0]} and column {subplot_loc[1]}, "
            )
    # when subplots do not form a grid
    elif isinstance(subplot_loc, str):
        prefix = f"For {subplot_loc}, "
    else:
        raise ValueError(f"Invalid subplot_loc: {subplot_loc}")
    # return the question with the subplot location
    return DESCRIPTIVE_RESP_INST[qid].format(prefix)


def build_descriptive_quries(data, image_meta):
    queries = []
    cnt = 0
    for _, d in data.items():
        figure_path = os.path.join("images", f"{d['figure_id']}.jpg")
        for i in range(len(d["qids"])):
            # mapping from template id and subplot location to the question
            question = descriptive_query_helper(d["qids"][i], d["subplot_loc"])
            curr_query = {
                "figure_id": d["figure_id"],  # figure_id
                "img_path": figure_path,  # figure_path (dropped later)
                "subq_idx": i,  # index of the (4) questions for the given figure
                "qid": d["qids"][i],  # template id
                "question": question,  # question content
                "question_id": f"descriptive_{cnt}",
                "answer": d["answers"][i],  # answer
                "question_type": "descriptive",
                "resp_key": f'{d["figure_id"]}_{i}',
                "image_meta": image_meta[str(d["figure_id"])],
                "num_subplots": d["num_subplots"],
            }
            cnt += 1
            queries.append(curr_query)
    print("quries number: ", len(queries))
    return queries


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    data_dir = osp.expanduser(data_dir)
    name = cfg.get("dataset_name", "")
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    os.makedirs(osp.join(output_dir, "images"), exist_ok=True)
    # run git clone https://github.com/princeton-nlp/CharXiv to data_dir
    os.system(f"git clone https://github.com/princeton-nlp/CharXiv {data_dir}")
    os.system(
        f"cd {output_dir}/images && wget https://huggingface.co/datasets/princeton-nlp/CharXiv/resolve/main/images.zip && unzip images.zip && rm images.zip"
    )
    contents = []
    image_meta = json.load(
        open(osp.join(data_dir, f"data/image_metadata_{split}.json"))
    )
    descriptive_data = json.load(
        open(osp.join(data_dir, f"data/descriptive_{split}.json"))
    )
    queries = build_descriptive_quries(descriptive_data, image_meta)

    reasoning_data = json.load(open(osp.join(data_dir, f"data/reasoning_{split}.json")))
    queries.extend(
        build_reasoning_queries(reasoning_data, descriptive_data, image_meta)
    )

    contents.extend(queries)
    with open(osp.join(output_dir, "data.json"), "w") as f:
        json.dump(contents, f, indent=2)
