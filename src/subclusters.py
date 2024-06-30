import numpy as np
from sklearn.cluster import KMeans


def sub_clusters(features):
    kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(features)
    output= kmeans.labels_
    clusters = [np.squeeze(np.array(features)[[np.where(output==i)[0]]],axis=0) for i in range(len(np.unique(output)))]
    return kmeans.cluster_centers_, clusters


def update_subclusters(all_dist, query, fea_label, id_pred, label_pred, features, decision, n_neighbours, cluster):
    max_ind=np.argmax(all_dist)
    features[max_ind]=np.concatenate((features[max_ind],np.expand_dims(query["image_features"], axis=0)),axis=0)
    # fea_label[cluster].append(np.expand_dims(query["image_features"], axis=0))  # Have doubt here
    id_pred[cluster].append(query["id"])
    label_pred[cluster].append((query['id'],decision.count(1)/n_neighbours))
    return features, fea_label, id_pred, label_pred