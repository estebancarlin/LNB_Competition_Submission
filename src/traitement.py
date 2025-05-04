from skimage import io
import numpy as np
import torch


def load_image(path):
    image = io.imread(path)
    if len(image.shape) == 3:
        image = np.moveaxis(image, -1, 0)
    return torch.from_numpy(image)


def find_previous(Nom_id_2, Id):
    name_2 = Nom_id_2[Id]
    name_0 = ''
    name_1 = ''
    for i, x in enumerate(name_2.split('-')):
        if i != 5:
            name_0 += x + '-'
            name_1 += x + '-'
        else:
            name_0 += '0-'
            name_1 += '1-'

    return name_0[:-1], name_1[:-1]


def moyenne_s2(s2_0, s2_1, mask_0, mask_1):

    indice = [1, 3, 4, 8, 9, 10]

    output = np.zeros_like(s2_0)

    for i in range(s2_0.shape[0]):
        for j in range(s2_0.shape[1]):
            # Check if any pixel in both masks is a cloud
            if np.isin(mask_0[i, j], indice) and np.isin(mask_1[i, j], indice):
                output[i, j] = (s2_0[i, j] + s2_1[i, j]) / 2
            # Check if any pixel in one mask is a cloud
            elif np.isin(mask_0[i, j], indice) != np.isin(mask_1[i, j], indice):
                if np.isin(mask_1[i, j], indice):
                    output[i, j] = s2_1[i, j]
                else:
                    output[i, j] = s2_0[i, j]
            # Compute the average if neither or both masks contain clouds
            else:
                output[i, j] = (s2_0[i, j] + s2_1[i, j]) / 2

    return output


def normalisation_s1(tensor):
    return torch.clamp(tensor, -30, 0) / (-30)


def normalisation_s2(tensor):
    return np.log(torch.clamp(tensor, 0, 10) + 1) / np.log(11)


def modif_path(path):
    if path.endswith(" (1).tiff"):
        new_path = path[:-9] + ".tiff"
        print(f"New name : {new_path}")
        return '/content/' + new_path
    elif path.endswith(" .tiff"):
        return path[:-6] + ".tiff"

    else:
        return path
