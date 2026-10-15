#!/usr/bin/env python3
"""CPU-only audit of saved immutable checkpoint config and model source."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
evidence = root / 'results/long-draft'
config_receipt = json.loads((evidence / 'checkpoint-config.browser.json').read_text())
source_receipt = json.loads((evidence / 'upstream-model.browser.json').read_text())
config = json.loads(config_receipt['content'])
source = source_receipt['content']
tree = ast.parse(source)
classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
assert config['dflash_config'] == {
    'block_size': 8, 'conv_group_size': 16, 'conv_kernel_size': 2,
    'mask_token_id': 248070, 'selector_rank': 256, 'selector_top_k': 16,
    'target_layer_ids': [5, 19, 33, 47, 61],
}
assert config['hidden_size'] == 5120 and config['num_hidden_layers'] == 5
assert config['sliding_window'] == 2048 and config['is_causal'] is False
selector = ast.get_source_segment(source, classes['CandidateSelector'])
conv = ast.get_source_segment(source, classes['GroupedDynamicCausalConv'])
convolve = ast.get_source_segment(source, functions['_grouped_dynamic_convolve'])
generator = ast.get_source_segment(source, functions['dflash_generate'])
assert 'range(hidden.shape[1])' in selector
assert 'nn.Embedding(config.vocab_size, rank)' in selector
assert 'nn.Linear(config.hidden_size, rank, bias=False)' in selector
assert 'torch.empty(2, kernel_size, hidden_size)' in conv
assert 'batch, length, hidden_size = hidden.shape' in convolve
assert 'block_size = model.block_size if block_size is None else block_size' in generator
assert '[:, 1 - verify_size :, :]' in generator
result = {
    'checkpoint_config_url': config_receipt['finalUrl'],
    'model_source_url': source_receipt['finalUrl'],
    'checkpoint_configured_query_rows': 8,
    'checkpoint_configured_proposals': 7,
    'native_experimental_proposals': [7, 16, 32, 64],
    'native_query_and_verifier_rows': [8, 17, 33, 65],
    'native_window': 1024,
    'checkpoint_window': 2048,
    'checkpoint_max_position_embeddings': config['max_position_embeddings'],
    'native_draft_ring_rows': 1120,
    'source_allows_runtime_block_size_override': True,
    'selector_walk_uses_runtime_hidden_length': True,
    'conv_has_no_length_sized_parameters': True,
    'observed_length_sized_architecture_limit': None,
    'long_length_training_support': 'not established: immutable config declares 8 query rows; no training-distribution evidence for 17/33/65',
    'long_length_quality': 'unmeasured extrapolation, not equivalent-quality assertion',
    'source_browser_content_sha256': hashlib.sha256(source.encode()).hexdigest(),
    'note': 'Source browser transport redacts timing-variable names; architecture sections audited here are unaffected. No model weights were downloaded or expanded.',
}
print(json.dumps(result, indent=2))
