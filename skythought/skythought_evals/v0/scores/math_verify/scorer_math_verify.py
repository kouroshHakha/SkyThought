
from ray.llm._internal.batch.processor import Processor, ProcessorConfig, ProcessorBuilder


class MathVerifyScoreProcessorConfig(ProcessorConfig):
    pass
        


def build_math_verify_score_processor(
    config: MathVerifyScoreProcessorConfig,
    **kwargs,
) -> Processor:
    
    stages = [
        MathVerifyParserStage(...),
        MathVerifyScoreStage(...)
    ]
    return Processor(config, stages, **kwargs)



ProcessorBuilder.register(MathVerifyScoreProcessorConfig, build_math_verify_score_processor)