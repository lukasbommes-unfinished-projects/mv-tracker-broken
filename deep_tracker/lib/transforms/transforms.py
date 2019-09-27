def standardize_motion_vectors(motion_vectors, mean, std):
    motion_vectors[..., 2] = (motion_vectors[..., 2] - mean[0]) / std[0]  # x channel
    motion_vectors[..., 1] = (motion_vectors[..., 1] - mean[1]) / std[1]  # y channel
    return motion_vectors
