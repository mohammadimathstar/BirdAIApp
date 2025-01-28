import torch
# from utils import winner_prototypes_indices

def winner_prototypes_indices(
        model,
        distances
):
    """
    Find the closest prototypes to a batch of data.
    """
    assert distances.ndim == 2, (f"Distances should be a matrix of shape (batch_size, number_of_prototypes), "
                                 f"but got {distances.shape}")

    i_1st_closest = distances.argmin(axis=1)
    y_1st_closest = model.prototype_layer.yprotos[i_1st_closest]

    # TODO: Check : Generate a mask for the prototypes corresponding to each image's label
    mask = model.prototype_layer.yprotos_comp_mat[y_1st_closest]

    # Apply the mask to distances
    distances_sparse = distances * mask

    # Find the index of the closest prototype for each image
    i_2nd_closest = torch.stack(
        [
            torch.argwhere(w).T[0,
            torch.argmin(
                w[torch.argwhere(w).T],
            )
            ] for w in torch.unbind(distances_sparse)
        ], dim=0
    ).T

    return i_1st_closest, i_2nd_closest



def contribution_of_img_on_constructed_subspace(vh, s):
    """
    returns a (nbatch x n x d) matrix: each row captures the importance of an image
    It also shows the coordinate of each image on the reconstructed subspace
    """
    RS_1 = torch.bmm(vh.transpose(-1, -2), torch.diag_embed(1 / s))
    return RS_1


def contribution_of_img_on_princiapl_dir(RS_1, Q):
    """
    returns a (nbatch x n x d) matrix: each row captures the importance of an image
    It also shows the coordinate of each image on the reconstructed subspace
    """
    return torch.bmm(RS_1, Q)


def projection_of_img_on_prototype_subspace(X, rotated_proto):
    """it return W : (nbatch x n x d)"""
    W = torch.bmm(X.transpose(-1, -2), rotated_proto)
    return W


def effect_of_hidden_pixel_on_prototype(M, W, rel: torch.Tensor):
    """
    X (Dxn) = U(Dxd) S(dxd) Vh(d x n)
    M (nxd) = R(nxd) S^-1(dxd) Q(dxd)
    W (nxd) = X.T(nxD) V (Dxd)
    Result = (nxd) matrix capturing
    """
    nxd_mat = (M * W) @ torch.tile(torch.diag(rel[0]).unsqueeze(0), dims=(M.shape[0], 1, 1))

    # return 1 - nxd_mat.sum(axis=-1) #???????????????????????
    return nxd_mat.sum(axis=-1), nxd_mat
    # eps = 0.8
    # log_nxd_mat = torch.log10((-nxd_mat + 2)/(1-nxd_mat+eps))
    # return log_nxd_mat.sum(axis=-1)

def get_importance(model, feature, Vh, S, output_dic, xprotos, rel):

    nbatch, nchannel, nW, nH = feature.shape

    distance = output_dic['distance']
    iplus, iminus = winner_prototypes_indices(model, distance)

    plus = {'index': iplus, "Q": output_dic["Q"][0, iplus], "Qw": output_dic["Qw"][0, iplus]}
    minus = {'index': iminus, "Q": output_dic["Q"][0, iminus], "Qw": output_dic["Qw"][0, iminus]}



    print(f"iplus: {iplus}, iminus{iminus}")
    print(iplus, iminus)

    winners_type = ['plus', 'minus']
    winners = [plus, minus]

    print("before for loop")
    protos_rot = dict()
    dic = dict()
    for wtype, w in zip(winners_type, winners):

        RS_1 = contribution_of_img_on_constructed_subspace(Vh, S)
        M = contribution_of_img_on_princiapl_dir(RS_1, w['Q'])

        rotated_proto = xprotos[w['index']] @ w['Qw']

        #TODO: you should flatten sample
        x = feature.view(nbatch, nchannel, nW * nH)
        W = projection_of_img_on_prototype_subspace(x, rotated_proto)

        image_region_effect_on_proto, image_region_effect_on_proto_per_direction = effect_of_hidden_pixel_on_prototype(M, W, rel)

        image_region_effect_on_proto_per_direction = torch.reshape(
            torch.permute(
                image_region_effect_on_proto_per_direction, dims=(0, 2, 1)
            ), (nbatch, -1, nW, nH)
        ) # (nbatch, num_of_principal_dir, W, H)

        dic[wtype] = dict()
        dic[wtype]['image_region_effect_on_proto'] = image_region_effect_on_proto
        dic[wtype]['image_region_effect_on_proto_per_direction'] = image_region_effect_on_proto_per_direction

        protos_rot[wtype] = rotated_proto
    image_region_effect_on_distance = dic['plus']['image_region_effect_on_proto'] - dic['minus'][
        'image_region_effect_on_proto']

    return (
        image_region_effect_on_distance.view(nW, nH),
        dic['plus']['image_region_effect_on_proto_per_direction'],
        dic['minus']['image_region_effect_on_proto_per_direction']
    )


