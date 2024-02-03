from tools import feature_distance

def all_feature_distance(all_feature_distance_type,
                         anchor_embedding,
                         positive_embedding,
                         negative_embedding,
                         channel,
                         distance_net = None):
    if all_feature_distance_type == "euclidean":
        # euclidean_distance
        positive_learning_distance = feature_distance.euclidean_torch(anchor_embedding, positive_embedding)
        negative_learning_distance = feature_distance.euclidean_torch(anchor_embedding, negative_embedding)
        cross_learning_distance    = feature_distance.euclidean_torch(positive_embedding, negative_embedding)
    else:
        raise ValueError('Unsupported All Feature Distance Type: {}'.format(all_feature_distance_type))
    return positive_learning_distance, negative_learning_distance, cross_learning_distance
