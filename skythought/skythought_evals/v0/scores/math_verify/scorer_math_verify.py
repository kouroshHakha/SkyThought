
from ray.llm._internal.batch.processor import Processor, ProcessorConfig, ProcessorBuilder
from pydantic import Field

from .stages_math_verify import MathVerifyParserStage, MathVerifyScoreStage

class MathVerifyScoreProcessorConfig(ProcessorConfig):
    """The configuration for the MathVerifyScoreProcessor."""
    
    answer_column: str = Field(
        description="The column name of the answer."
    )
    prediction_column: str = Field(
        description="The column name of the prediction."
    )
    score_concurrency: int = Field(
        default=1,
        description="The concurrency of the score stage."
    )
        


def build_math_verify_score_processor(
    config: MathVerifyScoreProcessorConfig,
    **kwargs,
) -> Processor:
    
    stages = [
        MathVerifyParserStage(
            fn_constructor_kwargs=dict(
                data_column=config.answer_column,
            ),
            map_batches_kwargs=dict(
                concurrency=config.score_concurrency,
            ),
        ),
        MathVerifyScoreStage(
            fn_constructor_kwargs=dict(
                data_column=config.prediction_column,
            ),
            map_batches_kwargs=dict(
                concurrency=config.score_concurrency,
            ),
        ),
    ]
    return Processor(config, stages, **kwargs)



ProcessorBuilder.register(MathVerifyScoreProcessorConfig, build_math_verify_score_processor)