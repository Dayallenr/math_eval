"""Exercise real save/load/generate with tiny random Qwen3 weights on CPU.

This checks inference plumbing only; random weights have no benchmark meaning.
"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from math_eval.inference import TransformersGenerator, identify_model


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'Install the inference extra for the real model smoke test')
class TransformersSmoke(unittest.TestCase):
    def test_load_local_qwen_and_generate_a_padded_batch(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vocabulary = {'<unk>': 0, '<pad>': 1, '<eos>': 2, 'user': 3, 'assistant': 4,
                          'Compute': 5, 'two': 6, 'plus': 7, 'three': 8, '.': 9, '5': 10}
            raw = Tokenizer(WordLevel(vocabulary, unk_token='<unk>'))
            raw.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='<unk>',
                                                pad_token='<pad>', eos_token='<eos>')
            tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
            tokenizer.save_pretrained(root)
            config = Qwen3Config(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                 head_dim=8, max_position_embeddings=512, eos_token_id=2, pad_token_id=1)
            Qwen3ForCausalLM(config).save_pretrained(root)
            settings = dict(device='cpu', dtype='float32', seed=42, enable_thinking=False,
                            max_input_tokens=64, max_new_tokens=4, temperature=0.0, top_p=1.0)
            generator = TransformersGenerator(identify_model(str(root)), settings)
            outputs = generator.generate(['Compute two plus three.', 'Compute two.'])
            self.assertEqual(len(outputs), 2)
            self.assertGreater(outputs[0]['input_tokens'], outputs[1]['input_tokens'])
            for output in outputs:
                self.assertIn(output['finish_reason'], ('eos', 'length'))
                self.assertLessEqual(output['output_tokens'], 4)
                self.assertIsInstance(output['response'], str)
            settings['max_input_tokens'] = 1
            with self.assertRaisesRegex(ValueError, 'no silent truncation'):
                generator.generate(['Compute two plus three.'])
