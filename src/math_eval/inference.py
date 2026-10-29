"""One model load per run; batches of unaided, text-only model responses."""

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time
from uuid import uuid4

from .grading import extract_final_answer
from .suite import digest, load_suite, write_json


def identify_model(model: str, revision: str | None = None) -> dict:
    path = Path(model).expanduser()
    if path.is_dir():
        path = path.resolve()
        if revision:
            raise ValueError('--revision applies to Hub models, not local directories')
        if not (path / 'config.json').is_file():
            raise ValueError('Model needs config.json. Export/merge raw training checkpoints or adapters first.')
        weights = sorted(path.glob('*.safetensors')) + sorted(path.glob('pytorch_model*.bin'))
        weights = [p for p in weights if not p.name.startswith('adapter_')]
        if not weights:
            raise ValueError('No full model weights found; export/merge the training checkpoint first')
        manifest_path = path / 'export_manifest.json'
        export = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        return {'kind': 'local', 'model': str(path), 'revision': None,
                'label': f"{export.get('training_run', path.name)} / step {export.get('step', '?')}" if export else path.name,
                'export_manifest': export,
                'config_sha256': digest((path / 'config.json').read_bytes()),
                # Cheap file fingerprint, NOT a cryptographic hash of every weight.
                'weight_files': [{'name': p.name, 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in weights]}
    if path.is_absolute() or model.startswith(('./', '../', '~')):
        raise FileNotFoundError(f'Model directory not found: {model}')
    from huggingface_hub import HfApi
    resolved = HfApi().model_info(model, revision=revision).sha
    return {'kind': 'huggingface', 'model': model, 'revision': resolved,
            'label': f'{model}@{resolved[:12]}'}


class TransformersGenerator:
    """The only component that imports Torch or loads weights onto a GPU."""

    def __init__(self, identity: dict, settings: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        self.torch = torch
        self.settings = settings
        device = settings['device']
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU unavailable. Run generation in an NRP GPU job, or use --device cpu for a tiny smoke test.')
        set_seed(settings['seed'])
        dtype = settings['dtype']
        if dtype == 'auto':
            dtype = 'bfloat16' if device == 'cuda' and torch.cuda.is_bf16_supported() else ('float16' if device == 'cuda' else 'float32')
        kwargs = {'revision': identity['revision']} if identity['revision'] else {}
        self.tokenizer = AutoTokenizer.from_pretrained(identity['model'], trust_remote_code=False, **kwargs)
        if not self.tokenizer.chat_template:
            raise ValueError('Model tokenizer has no chat template; provide the exported SFT tokenizer/template')
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError('Tokenizer has neither padding nor EOS token')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            identity['model'], dtype=getattr(torch, dtype), low_cpu_mem_usage=True,
            trust_remote_code=False, **kwargs,
        ).to(device).eval()
        self.model.config.use_cache = True
        self.metadata = {
            'resolved_dtype': dtype, 'device': device,
            'chat_template_sha256': digest(json.dumps(self.tokenizer.chat_template, sort_keys=True)),
            'tokenizer_class': type(self.tokenizer).__name__,
            'model_class': type(self.model).__name__,
            'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers')},
            'generation_config': self.model.generation_config.to_dict(),
        }

    def generate(self, prompts: list[str]) -> list[dict]:
        torch, settings = self.torch, self.settings
        rendered = [self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True,
            enable_thinking=settings['enable_thinking'],
        ) for prompt in prompts]
        batch = self.tokenizer(rendered, return_tensors='pt', padding=True,
                               truncation=False, add_special_tokens=False)
        width = batch['input_ids'].shape[1]
        if width > settings['max_input_tokens']:
            raise ValueError(f'Prompt has {width} tokens, exceeding max_input_tokens; no silent truncation')
        context = getattr(self.model.config, 'max_position_embeddings', None)
        if context and width + settings['max_new_tokens'] > context:
            raise ValueError('Prompt plus generation budget exceeds model context; adjust token limits')
        batch = batch.to(self.model.device)
        # Explicit greedy/sampling controls override any inherited training config.
        generation = dict(max_new_tokens=settings['max_new_tokens'], do_sample=settings['temperature'] > 0,
                          num_beams=1, num_return_sequences=1, repetition_penalty=1.0,
                          pad_token_id=self.tokenizer.pad_token_id)
        if settings['temperature'] > 0:
            generation.update(temperature=settings['temperature'], top_p=settings['top_p'], top_k=0)
        with torch.inference_mode():
            sequences = self.model.generate(**batch, **generation)
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        results = []
        for index, sequence in enumerate(sequences[:, width:]):
            ids = sequence.tolist()
            stop = next((j for j, token in enumerate(ids) if token in eos_ids), None)
            used = ids if stop is None else ids[:stop + 1]
            results.append({
                'response': self.tokenizer.decode(used, skip_special_tokens=True),
                'rendered_prompt': rendered[index],
                'input_tokens': int(batch['attention_mask'][index].sum().item()),
                'output_tokens': len(used),
                'finish_reason': 'eos' if stop is not None else 'length',
            })
        return results


def generate_run(suite: Path, model: str, output_root: Path, settings: dict,
                 revision: str | None = None, generator_factory=TransformersGenerator) -> Path:
    manifest, questions, suite_hash = load_suite(suite)
    identity = identify_model(model, revision)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    run = {'schema_version': 1, 'run_id': run_id, 'status': 'generating',
           'created_at': datetime.now(timezone.utc).isoformat(), 'model': identity,
           'suite_sha256': suite_hash, 'suite_counts': manifest['counts'],
           'subset': manifest['subset'], 'settings': settings, 'attempts_per_question': 1,
           'generation_kind': 'model', 'suite_path_at_generation': str(suite.resolve()),
           'runner_sha256': {p.name: digest(p.read_bytes()) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    write_json(output / 'run.json', run)
    started = time.monotonic()
    try:
        engine = generator_factory(identity, settings)
        run['inference'] = engine.metadata
        write_json(output / 'run.json', run)
        with (output / 'predictions.jsonl').open('x', encoding='utf-8') as handle:
            for start in range(0, len(questions), settings['batch_size']):
                batch = questions[start:start + settings['batch_size']]
                responses = engine.generate([row['prompt'] for row in batch])
                if len(responses) != len(batch):
                    raise RuntimeError('Generator returned an incorrect response count')
                for question, response in zip(batch, responses, strict=True):
                    record = {key: question[key] for key in ('task_id', 'benchmark', 'instruction_sha256')}
                    record.update(response)
                    record.update({'prompt': question['prompt'], 'suite_sha256': suite_hash,
                                   'answer': extract_final_answer(response['response'])})
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                handle.flush()
                print(f"Generated {min(start + len(batch), len(questions))}/{len(questions)}", flush=True)
        run.update(status='generated', generated_count=len(questions),
                   generation_seconds=time.monotonic() - started,
                   predictions_sha256=digest((output / 'predictions.jsonl').read_bytes()))
    except BaseException as error:
        run.update(status='generation_failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'run.json', run)
        print(f'Run directory: {output}', flush=True)
    return output
