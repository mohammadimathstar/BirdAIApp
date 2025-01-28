
import torch
import torch.nn as nn
from torch import Tensor



def orthogonalize_batch(x_batch: Tensor) -> Tensor:
    """
    Orthogonalize each matrix in a batch of matrices using QR decomposition.

    Parameters:
    -----------
    x_batch : Tensor
        Input tensor of shape (batch_size, m, n), where each (m, n) matrix will be orthogonalized.

    Returns:
    --------
    Tensor
        A tensor of orthogonal matrices of shape (batch_size, m, min(m, n)), where each matrix in the batch is orthogonalized.

    Notes:
    ------
    This function uses the QR decomposition to orthogonalize each (m, n) matrix in the batch.
    The `mode='reduced'` option ensures that the resulting Q matrices are of shape (m, min(m, n)).
    """
    Q, _ = torch.linalg.qr(x_batch, mode='reduced')
    return Q


def grassmann_repr(batch_imgs: Tensor, dim_of_subspace: int) -> Tensor:
    """
    Generate Grassmann representations from a batch of images.

    Parameters:
    -----------
    batch_imgs : Tensor
        A batch of features of size (batch_size, num_of_channels, W, H).
    dim_of_subspace : int
        The dimensionality of the extracted subspace.

    Returns:
    --------
    Tensor
        An orthonormal matrix of size (batch_size, W*H, dim_of_subspace) representing the Grassmann subspaces.

    Raises:
    -------
    AssertionError
        If the input tensor `batch_imgs` does not have 4 dimensions.
    """
    assert batch_imgs.ndim == 4, f"batch_imgs should be of shape (batch_size, num_of_channels, W, H), but it is {batch_imgs.shape}"

    bsize, nchannel, w, h = batch_imgs.shape
    # Reshape the batch images to shape (batch_size, num_of_channels, W*H)
    xs = batch_imgs.view(bsize, nchannel, w * h)

    try:
        U, S, Vh = torch.linalg.svd(xs, full_matrices=False)
    except:
        eps = 1e-2
        batch_eye = torch.eye(nchannel, w * h).unsqueeze(0).repeat(bsize, 1, 1).to(xs.device)
        U, S, Vh = torch.linalg.svd(xs + batch_eye * eps, full_matrices=False)

    # Select the appropriate orthonormal matrix based on dimensions
    if U.shape[-2] > U.shape[-1]:
        return U[:, :, :dim_of_subspace]  # Shape: (batch_size, num_of_channels, dim_of_subspace)
    else:
        return Vh.transpose(-1, -2)[:, :, :dim_of_subspace]  # Shape: (batch_size, W*H, dim_of_subspace)


def grassmann_repr_full(batch_imgs: Tensor, dim_of_subspace: int) -> Tensor:
    """
    Generate Grassmann representations from a batch of images.
    It returns both singular values and left/right principal directions.
    """
    assert batch_imgs.ndim == 4, f"batch_imgs should be of shape (batch_size, num_of_channels, W, H), but it is {batch_imgs.shape}"

    bsize, nchannel, w, h = batch_imgs.shape
    # Reshape the batch images to shape (batch_size, num_of_channels, W*H)
    xs = batch_imgs.view(bsize, nchannel, w * h)


    # SVD: generate principal directions
    U, S, Vh = torch.linalg.svd(xs, full_matrices=False)

    assert U.shape[-2] > U.shape[-1], f"The matrix size is {U.shape[1:]}."

    # Shape: (batch_size, num_of_channels, dim_of_subspace)
    return U[:, :, :dim_of_subspace], Vh[:, :dim_of_subspace,:], S[:, :dim_of_subspace]


def init_randn(
        dim_of_data: int,
        dim_of_subspace: int,
        labels: Tensor = None,
        num_of_protos: [int, Tensor] = 1,
        num_of_classes: int = None,
        device='cpu'
) -> tuple:
    """
    Initialize prototypes randomly using a Gaussian distribution.

    Parameters:
    -----------
    dim_of_data : int
        Dimensionality of the data space.
    dim_of_subspace : int
        Dimensionality of the subspace.
    labels : Tensor, optional
        Tensor containing class labels. If None, `num_of_classes` must be provided.
    num_of_protos : int or Tensor, optional
        Number of prototypes per class if an integer, or a tensor specifying the number of prototypes for each class.
        Default is 1.
    num_of_classes : int, optional
        Number of classes. Required if `labels` is None.
    device : str, optional
        Device on which to place the tensors. Default is 'cpu'.

    Returns:
    --------
    tuple
        A tuple containing:
        - xprotos (Tensor): Initialized prototypes of shape (total_num_of_protos, dim_of_data, dim_of_subspace).
        - yprotos (Tensor): Labels of the prototypes of shape (total_num_of_protos,).
        - yprotos_mat (Tensor): One-hot encoded labels of the prototypes of shape (nclass, total_num_of_protos).
        - yprotos_mat_comp (Tensor): Complementary one-hot encoded labels of the prototypes of shape (nclass, total_num_of_protos).
    """
    if labels is None:
        assert num_of_classes is not None, "num_of_classes must be provided if labels are not given."
        classes = torch.arange(num_of_classes)
    else:
        classes = torch.unique(labels)

    if isinstance(num_of_protos, int):
        total_num_of_protos = len(classes) * num_of_protos
    else:
        total_num_of_protos = torch.sum(num_of_protos).item()

    nclass = len(classes)
    prototype_shape = (total_num_of_protos, dim_of_data, dim_of_subspace)

    # Initialize prototypes using QR decomposition for orthogonalization
    Q, _ = torch.linalg.qr(
        0.5 + 0.1 * torch.randn(prototype_shape, device=device),
        mode='reduced')
    xprotos = nn.Parameter(Q)

    # Set prototypes' labels
    yprotos = torch.repeat_interleave(classes, num_of_protos).to(torch.int32)

    yprotos_mat = torch.zeros((nclass, total_num_of_protos), dtype=torch.int32)
    yprotos_mat_comp = torch.ones((nclass, total_num_of_protos), dtype=torch.int32, device=device)

    # Setting prototypes' labels
    for i, class_label in enumerate(yprotos):
        yprotos_mat[class_label, i] = 1
        yprotos_mat_comp[class_label, i] = 0

    return xprotos, yprotos.to(device), yprotos_mat.to(device), yprotos_mat_comp.to(device)


def compute_distances_on_grassmann_mdf(
        xdata: Tensor,
        xprotos: Tensor,
        relevance: Tensor = None
) -> dict:
    """
    Compute the (geodesic or chordal) distances between an input subspace and all prototypes.

    Parameters:
    -----------
    xdata : Tensor
        Input tensor representing the subspaces, expected shape (batch_size, W*H, dim_of_subspace).
    xprotos : Tensor
        Prototype tensor representing the prototype subspaces, expected shape (num_of_prototypes, W*H, dim_of_subspace).
    relevance : Tensor, optional
        Tensor representing the relevance of each dimension in the subspace, expected shape (1, dim_of_subspace).
        If None, a uniform relevance is assumed. Default is None.
    """
    assert xdata.ndim == 3, f"xdata should be of shape (batch_size, W*H, dim_of_subspace), but it is {xdata.shape}"

    # If relevance is not provided, assume uniform relevance
    if relevance is None:
        relevance = torch.ones((1, xprotos.shape[-1])) / xprotos.shape[-1]

    xdata = xdata.unsqueeze(dim=1)  # Shape: (batch_size, 1, W*H, dim_of_subspace)

    # Compute the singular value decomposition of the product of the transposed xdata and xprotos
    U, S, Vh = torch.linalg.svd(
        torch.transpose(xdata, 2, 3) @ xprotos.to(xdata.dtype),
        full_matrices=False,
    )

    distance = 1 - torch.transpose(
        relevance @ torch.transpose(S, 1, 2).to(relevance.dtype),
        1, 2
    )

    if torch.isnan(distance).any():
        raise Exception('Error: NaN values! Using the --log_probabilities flag might fix this issue')

    output = {
        'Q': U,  # Shape: (batch_size, num_of_prototypes, WxH, dim_of_subspaces)
        'Qw': torch.transpose(Vh, 2, 3),  # Shape: (batch_size, num_of_prototypes, WxH, dim_of_subspaces)
        'canonicalcorrelation': S,  # Shape: (batch_size, num_of_prototypes, dim_of_subspaces)
        'distance': torch.squeeze(distance, -1),  # Shape: (batch_size, num_of_prototypes)
    }

    return output


# def winner_prototype_indices(
#         distances: Tensor
# ):
#     """
#     Find the first two closest prototypes (with different labels) to a batch of data.
#     """
#     assert distances.ndim == 2, (f"Distances should be a matrix of shape (batch_size, number_of_prototypes), "
#                                  f"but got {distances.shape}")
#     copied_distances = distances.clone().detach()
#
#     i_1st_closest = distances.argmin(axis=1)
#     y_1st_closest = model.prototype_layer.yprotos(i_1st_closest)
#
#     # Generate a mask for the prototypes corresponding to each image's label
#     mask = model.prototype_layer.yprotos_mat[y_1st_closest]
#
#     # Apply the mask to distances
#     distances_sparse = distances * mask
#
#     # Find the index of the closest prototype for each image
#     i_2nd_closest = torch.stack(
#         [
#             torch.argwhere(w).T[0,
#             torch.argmin(
#                 w[torch.argwhere(w).T],
#             )
#             ] for w in torch.unbind(distances_sparse)
#         ], dim=0
#     ).T
#
#     return i_1st_closest, i_2nd_closest
#
#
# def winner_prototype_distances(
#         ydata: Tensor,
#         yprotos_matrix: Tensor,
#         yprotos_comp_matrix: Tensor,
#         distances: Tensor
# ) -> tuple:
#     """
#     Find the distances between first two closest prototype (with different labels) to a data.
#     """
#     nbatch, nprotos = distances.shape
#
#     # Find the indices of winner and non-winner prototypes
#     iplus = winner_prototype_indices(ydata, yprotos_matrix, distances)
#     iminus = winner_prototype_indices(ydata, yprotos_comp_matrix, distances)
#
#     # Extract distances for winner and non-winner prototypes
#     Dplus = torch.zeros_like(distances)
#     Dminus = torch.zeros_like(distances)
#     Dplus[torch.arange(nbatch), iplus] = distances[torch.arange(nbatch), iplus]
#     Dminus[torch.arange(nbatch), iminus] = distances[torch.arange(nbatch), iminus]
#
#     return Dplus, Dminus, iplus, iminus