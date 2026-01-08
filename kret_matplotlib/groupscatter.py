from kret_rosetta.to_pd_np import TO_NP_TYPE


class GroupScatter:
    """
    Take in y and y_hat, optional categorical column, # centroids, downsample, and regression funcion (OLS, Huber, etc)
    """

    def __init__(self, y: TO_NP_TYPE, y_hat: TO_NP_TYPE): ...
