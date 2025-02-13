
import ray
from datasets import load_dataset
from typing import Type, Any, Dict
from ray.data.llm import build_llm_processor, HttpRequestProcessorConfig, Processor, ProcessorConfig
import os
import pprint
import copy

from skythought_evals.util.math_parsing_util import (
    extract_answer,
    math_equal,
    strip_answer_string,
)

##### READ DATASET #####
def read_dataset():
    hf_ds = load_dataset("qq8933/MATH500", split="test")
    ray_ds = ray.data.from_huggingface(hf_ds)
    return ray_ds


##### LLM PROCESSOR #####
template = [
    {
        "role": "user",
        "content": "Return your final response within \\boxed{{}}. {problem}"
    }
]

def format_prompt(template, row):
    formatted_prompt = []
    for conv in template:
        conv = copy.deepcopy(conv)
        conv["content"] = conv["content"].format(**row)
        formatted_prompt.append(conv)
    return formatted_prompt


llm_processor = build_llm_processor(
    HttpRequestProcessorConfig(
        url="https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        qps=100,
    ),
    preprocess=lambda row: {
        "payload": {
            "model": "gpt-4o-mini",
            "messages": format_prompt(template, row)
        }
    },
    postprocess=lambda row: {**row, "generated_text": row["http_response"]["choices"][0]["message"]["content"]},
)

##### Testing READ DATASET and LLM PROCESSOR #####
ds = read_dataset()
ds = ds.limit(1)

ds = llm_processor(ds)
# row = out_ds.take(1)[0]


##### Example of checking correctness #####
# This can be used as simple map udf
# But what is the problem with this? 
# 1. It's not reusable in contexts where `answer` and `generated_text` are not the desired columns in the dataset
# 2. It does not carry over the input. So after this stage is applied you lose input <> output relationship

# def check_correctness(row):
#     answer = strip_answer_string(row["answer"])
#     pred = extract_answer(row["generated_text"])
#     pred = strip_answer_string(pred)
#     return {"correctness": math_equal(pred, answer)}
# ds = ds.map(check_correctness)

##### Making the reusable stages #####
############## Strategy: 
############## 1. One stage does extraction. configurable on which column to extract and strip or not for reusablity
############## 2. One stage does equality check. configurable on which columns to check
############## 3. For MATH500, we need to extract final boxed answer from both `generated_text` and `solution` and then do a math_equal check
############## 4. This will be a processor with 3 stages. To instances of the first stage with different config and one instance of the third stage
############## 5. To make a reusable processor, we can make more configurations on whether to extract answer from solution or not, etc.

# Let's make the stateful stage udfs and in the end processors
from ray.llm._internal.batch.stages.base import StatefulStageUDF, StatefulStage

# Extractor stage
class MathAnswerExtractorUDF(StatefulStageUDF):
    
    def __init__(
        self, 
        data_column: str,
        col_to_extract: str,
        strip: bool = True,
    ):
        super().__init__(data_column)
        self.strip = strip
        self.col_to_extract = col_to_extract
        
    async def udf(self, rows):
        for row in rows:
            out = extract_answer(row[self.col_to_extract])
            if self.strip:
                out = strip_answer_string(out)
            yield {f"extraced_{self.col_to_extract}": out, self.IDX_IN_BATCH_COLUMN: row[self.IDX_IN_BATCH_COLUMN]}
    
    @property
    def expected_input_keys(self):
        return [self.col_to_extract]
        
class MathAnswerExtractorStage(StatefulStage):
    fn: Type[MathAnswerExtractorUDF] = MathAnswerExtractorUDF
    fn_constructor_kwargs: Dict[str, Any] = {}
    map_batches_kwargs: Dict[str, Any] = {"concurrency": 1}

##### Testing the extractor stage #####

stage1 = MathAnswerExtractorStage(
    fn_constructor_kwargs=dict(
        col_to_extract="generated_text",
        strip=True,
    ),
)

###### Side note: Testing an individual stage #####
# NOTE: Standalone test of stage is not possible. So what is the quickest way to test an individual stage? 
# Did not work!!
# ds = ds.map_batches(
#     MathAnswerExtractor,
#     **config.get_dataset_map_batches_kwargs(
#         batch_size=1,
#         data_column="__data",
#     )
# )

# This seems to be the way to test individual stage
# processor = Processor(
#     # batch size is required. A bit weird.
#     config=ProcessorConfig(batch_size=64), 
#     stages=[stage1],
#     # ray needs to be fixed to allow none values for pre/postprocess
#     # preprocess=lambda row: row,
#     # postprocess=lambda row: row,
# )
# ds = processor(ds)

##### Creating the second stage #####
stage2 = MathAnswerExtractorStage(
    fn_constructor_kwargs=dict(
        col_to_extract="solution",
        strip=True,
    ),
)

# processor = Processor(
#     config=ProcessorConfig(batch_size=64),
#     stages=[stage1, stage2],
# )

# ds = processor(ds)


##### Creating the third stage definition #####

class MathEqualUDF(StatefulStageUDF):
    def __init__(self, data_column: str, ground_truth_col: str, pred_col: str):
        super().__init__(data_column)
        self.ground_truth_col = ground_truth_col
        self.pred_col = pred_col

    async def udf(self, rows):
        for row in rows:
            gt = row[self.ground_truth_col]
            pred = row[self.pred_col]
            math_equal_res = math_equal(gt, pred)
            yield {f"math_equal_{self.ground_truth_col}_vs_{self.pred_col}": math_equal_res, self.IDX_IN_BATCH_COLUMN: row[self.IDX_IN_BATCH_COLUMN]}


class MathEqualStage(StatefulStage):
    fn: Type[MathEqualUDF] = MathEqualUDF
    fn_constructor_kwargs: Dict[str, Any] = {}
    map_batches_kwargs: Dict[str, Any] = {"concurrency": 1}



stage3 = MathEqualStage(
    fn_constructor_kwargs=dict(
        ground_truth_col="extraced_solution",
        pred_col="extraced_generated_text",
    ),
)


##### Testing the processor #####
score_processor = Processor(
    config=ProcessorConfig(batch_size=64),
    stages=[stage1, stage2, stage3],
)

# ds = score_processor(ds)


##### Creating the processor config #####
# This is the reusable processor config
class MathVerifyProcessorConfig(ProcessorConfig):
    
    generated_column: str
    answer_column: str
    should_extract_answer: bool = False
    
    
def build_math_verify_processor(config: MathVerifyProcessorConfig, **kwargs):
    
    stages = [
        MathAnswerExtractorStage(
            fn_constructor_kwargs=dict(
                col_to_extract=config.generated_column,
                strip=True,
            ),
        ),
    ]
    pred_col = f"extraced_{config.generated_column}"
    
    gt_col = f"{config.answer_column}"
    if config.should_extract_answer:
        stages.append(
            MathAnswerExtractorStage(
                fn_constructor_kwargs=dict(
                    col_to_extract=config.answer_column,
                    strip=True,
                ),
            ),
        )
        gt_col = f"extraced_{config.answer_column}"
    
    stages.append(
        MathEqualStage(
            fn_constructor_kwargs=dict(
                ground_truth_col=gt_col,
                pred_col=pred_col,
            ),
        )
    )
    
    return Processor(
        config=config,
        stages=stages,
        **kwargs
    )
    
##### Testing the reusable processor #####
# Assuming we have bunch of these processors we can just import them and chain them together 
processor = build_math_verify_processor(
    MathVerifyProcessorConfig(
        batch_size=1,
        generated_column="generated_text",
        answer_column="answer",
        should_extract_answer=False,
    )
)
ds = processor(ds)

row = ds.take(1)[0]
pprint.pprint(row)



##### Pretending the library is implemnted ##### 
from .scores.math import MathVerifyProcessorConfig, build_math_verify_processor

llm = EndpointLLM(model="gpt-4o-mini", template=template, http_config={"url": "https://api.openai.com/v1/chat/completions", "headers": {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}})
# or llm = VLLM(model=..., template=..., kwargs=...)

def read_dataset():
    ...
    
    
ds = read_dataset()
ds = llm(ds)


processor = build_math_verify_processor(
    MathVerifyProcessorConfig(
        batch_size=1,
        generated_column="generated_text",
        answer_column="answer",
        should_extract_answer=False,
    )
)

ds = processor(ds)

row = ds.take(1)[0]
pprint.pprint(row)