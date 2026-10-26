"""Load precomputed contexts and their split metadata."""
from pathlib import Path
import torch

_chunk_cache = {}


def load_prebuilt_subgraphs(subgraph_dir, split_name='train', load_all=False):
    path = Path(subgraph_dir) / f'subgraphs_{split_name}.pt'
    data = torch.load(path, map_location='cpu', weights_only=False)
    metadata = dict(data['metadata'])
    metadata['_chunk_files'] = list(data['chunk_files'])
    metadata['_chunk_dir'] = data['chunk_dir']
    if not load_all:
        return {}, metadata
    graphs = {}
    for filename in metadata['_chunk_files']:
        chunk_path = Path(filename)
        if not chunk_path.is_absolute():
            chunk_path = Path(metadata['_chunk_dir']) / chunk_path
        graphs.update(torch.load(chunk_path, map_location='cpu', weights_only=False))
    return graphs, metadata


def load_subgraph_from_chunks(window_id, metadata, chunk_cache=None):
    cache = _chunk_cache if chunk_cache is None else chunk_cache
    for filename in metadata.get('_chunk_files', []):
        path = Path(filename)
        if not path.is_absolute():
            path = Path(metadata['_chunk_dir']) / path
        key = str(path)
        if key not in cache:
            cache[key] = torch.load(path, map_location='cpu', weights_only=False)
        if window_id in cache[key]:
            return cache[key][window_id]
    return None


def check_subgraph_compatibility(metadata, num_hops, max_neighbors):
    return (metadata.get('protocol') == 'logbridge-context'
            and metadata.get('num_hops') == num_hops
            and metadata.get('max_neighbors') == max_neighbors)
