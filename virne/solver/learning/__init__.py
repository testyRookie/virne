try:
    import torch_sparse, torch_scatter, torch_cluster
except ImportError:
    print('PyTorch Geometric is not installed completely. Installing now...')
    import os
    import torch
    cuda_version = torch.version.cuda
    if cuda_version is None:
        cuda_suffix = 'cpu'
    else:
        if cuda_version in ['11.8', '12.1', '12.4']:
            cuda_suffix = 'cu' + cuda_version.replace('.', '')
        else:
            cuda_suffix = 'cu118'
    print(f'cuda version: {cuda_version} (suffix: {cuda_suffix})')
    torch_version = torch.__version__.split('+')[0]
    torch_version_parts = torch_version.split(".")
    torch_version_parts[-1] = "0"
    torch_version = ".".join(torch_version_parts)
    print(f'torch version: {torch_version}')
    print(f'Install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv based on torch-{torch_version}+{cuda_suffix}')
    os.system(f'pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_suffix}.html')
    # os.system('cls' if os.name == 'nt' else 'clear')
    import torch_sparse, torch_scatter, torch_cluster
    
from .hopfield_network import HopfieldNetworkSolver
from .gae_clustering import GaeClusteringSolver
from .mcts import MctsSolver
from .pg_mlp import PgMlpSolver
from .pg_cnn import PgCnnSolver
from .pg_cnn2 import PgCnn2Solver
from .ddpg_attention import DdpgAttentionSolver
from .pg_seq2seq import PgSeq2SeqSolver
from .a3c_gcn_seq2seq import A3CGcnSeq2SeqSolver
from .multiagent import MultiAgentAuctionSolver


__all__ = [
    # Unsupervised learning solvers
    'HopfieldNetworkSolver',
    'GaeClusteringSolver',
    # Reinforcement learning solvers
    'MctsSolver',
    'PgMlpSolver',
    'PgCnnSolver',
    'PgCnn2Solver',
    'PgSeq2SeqSolver',
    'A3CGcnSeq2SeqSolver',
    'DdpgAttentionSolver',
    'MultiAgentAuctionSolver',
]