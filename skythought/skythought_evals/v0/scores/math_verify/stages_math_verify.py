
from typing import Any, Dict, AsyncIterator, List
from ray.llm._internal.batch.stages.stateful_stage import StatefulStage, StatefulStageUDF


class MathVerifyParserUDF(StatefulStageUDF):
    def __init__(self, data_column: str):
        super().__init__(data_column)
        
        
    def parse_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ground_truth": item["answer"],
            "prediction": item["student_answer"],
        }
    
    async def udf(self, batch: List[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        for input_ in batch:
            yield {
                self.IDX_IN_BATCH_COLUMN: input_[self.IDX_IN_BATCH_COLUMN],
                **self.parse_item(input_)
            }
            
    def expected_input_keys(self) -> List[str]:
        return ["generated_text", "answer"]


class MathVerifyScoreUDF(StatefulStageUDF):

    def __init__(
        self,
        data_column: str,
        **kwargs
    ):
        super().__init__(data_column)
        
        
    def score_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {"score": 1.0}
    
    async def udf(self, batch: List[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:        
        
        for input_ in batch:
            yield {
                self.IDX_IN_BATCH_COLUMN: input_[self.IDX_IN_BATCH_COLUMN],
                **self.score_item(input_)
            }
            

class MathVerifyParserStage(StatefulStage):
    fn: StatefulStageUDF = MathVerifyParserUDF


class MathVerifyScoreStage(StatefulStage):
    fn: StatefulStageUDF = MathVerifyScoreUDF

