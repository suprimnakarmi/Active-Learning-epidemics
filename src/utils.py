def data_separation(dataset,label):
    add_data= []
    i=0
    while len(add_data)!=20:
        if dataset[i]["label"]==label:
            add_data.append(dataset[i]['image_features'])
            del dataset[i]
        i+=1
    return add_data

def mean_features(positive, negative):
    # print(f"pure_pf: {positive}")
    # print(f"p_type: {type(positive)}")
    # print(f"len_p: {len(positive)}")
    mpos_features=np.array([np.mean(i,axis=0) for i in positive])  # Mean of all positive sub clusters 
    mneg_features=np.array([np.mean(i,axis=0) for i in negative])  # Mean of all negative sub clusters
    # print(mpos_features)
    return mpos_features, mneg_features

def flatten_features(features):
    all_features = []
    for i in features:
        for j in i:
            all_features.append(j)
    return all_features