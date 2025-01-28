import torch
import torch.nn as nn
from torch.autograd import Function

from lvq.grassmann import init_randn, compute_distances_on_grassmann_mdf


class PrototypeLayer(nn.Module):
    def __init__(self,
                 num_prototypes,
                 num_classes,
                 feature_dim,
                 subspace_dim,
                 metric_type='chordal',
                 dtype=torch.float32,
                 device='cpu'
                ):
        """
        Initialize the PrototypeLayer.

        Args:
            num_prototypes (int): Number of prototypes.
            num_classes (int): Number of classes.
            feature_dim (int): Dimension of data features.
            subspace_dim (int): Dimension of subspaces.
            dtype (torch.dtype, optional): Data type of the tensors. Defaults to torch.float32.
            device (str, optional): Device to use ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        super().__init__()
        self._feature_dim = feature_dim
        self._subspace_dim = subspace_dim
        self._metric_type = 'chordal'

        # Initialize prototypes
        self.xprotos, self.yprotos, self.yprotos_mat, self.yprotos_comp_mat = init_randn(
            self._feature_dim,
            self._subspace_dim,
            num_of_protos=num_prototypes,
            num_of_classes=num_classes,
            device=device,
        )

        self._number_of_prototypes = self.yprotos.shape[0]

        # Initialize relevance parameters
        self.relevances = nn.Parameter(
            torch.ones((1, self.xprotos.shape[-1]), dtype=dtype, device=device) / self.xprotos.shape[-1]
        )

        self.distance_layer = ChordalPrototypeLayer()

    def forward(self, xs_subspace: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PrototypeLayer.

        Args:
            xs_subspace (torch.Tensor): Input subspaces.

        Returns:
            torch.Tensor: Output from the ChordalPrototypeLayer.
        """
        # Compute distances between data and prototypes
        output = compute_distances_on_grassmann_mdf(
            xs_subspace,
            self.xprotos,
            self.relevances,
        )
        
        return output['distance'], output['Qw']
        
        
# class ChordalPrototypeLayer(Function):
#     # @staticmethod
#     def forward(self, xs_subspace, xprotos, relevances):
#         """
#         Forward pass of the ChordalPrototypeLayer.
#
#         Args:
#             ctx: Context object to save tensors for backward computation.
#             xs_subspace (torch.Tensor): Input subspaces.
#             xprotos (torch.Tensor): Prototypes.
#             relevances (torch.Tensor): Relevance parameters.
#
#         Returns:
#             tuple: Output distance and Qw.
#         """
#
#         # Compute distances between data and prototypes
#         output = compute_distances_on_grassmann_mdf(
#             xs_subspace,
#             xprotos,
#             relevances,
#         )
#
#         # ctx.save_for_backward(
#         #     xs_subspace, xprotos, relevances,
#         #     output['distance'], output['Q'], output['Qw'], output['canonicalcorrelation'])
#         return output['distance'], output['Qw']


