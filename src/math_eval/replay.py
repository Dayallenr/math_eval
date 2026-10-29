"""Harbor agent that transports saved model predictions, with no model access."""

import json
from pathlib import Path
import tempfile

from harbor.agents.base import BaseAgent
from harbor.agents.options import AgentOptions
from harbor.models.agent.context import AgentContext
from pydantic import field_validator

from .suite import digest, read_jsonl


class ReplayOptions(AgentOptions):
    predictions_file: str

    @field_validator('predictions_file')
    @classmethod
    def file_exists(cls, value: str) -> str:
        if not Path(value).is_file():
            raise ValueError('Predictions file does not exist on the Harbor host')
        return value


class SavedPredictionAgent(BaseAgent):
    options_model = ReplayOptions

    @staticmethod
    def name() -> str:
        return 'math-saved-prediction'

    def version(self) -> str:
        return '1.0.0'

    async def setup(self, environment) -> None:
        pass

    async def run(self, instruction: str, environment, context: AgentContext) -> None:
        fingerprint = digest(instruction)
        records = [row for row in read_jsonl(Path(self.options.predictions_file))
                   if row['instruction_sha256'] == fingerprint]
        if len(records) != 1:
            raise ValueError('Expected exactly one prediction matching this task instruction')
        record = records[0]
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / 'prediction.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        # Upload data rather than interpolating an answer into an executable command.
        with tempfile.TemporaryDirectory(prefix='math-answer-') as tmp:
            answer = Path(tmp) / 'answer.txt'
            answer.write_text(record['answer'] or '', encoding='utf-8')
            await environment.upload_file(answer, '/app/answer.txt')
        context.n_input_tokens = record.get('input_tokens')
        context.n_output_tokens = record.get('output_tokens')
        context.metadata = {'task_id': record['task_id'], 'finish_reason': record['finish_reason'],
                            'mode': 'saved prediction; no LLM calls during Harbor grading'}
