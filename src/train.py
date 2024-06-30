import argparse
import numpy as np
import os
from tqdm import tqdm
from statistics import mode
from utils import flatten_features
from subclusters import update_subclusters
from dataloader import dat

all_fea = []
model,feature_size= all_models(img_size, select_model)
for data in tqdm(range(len(batch_img))):
  try:
    features = model.predict(batch_img[data]).flatten().reshape(batch_size,feature_size)
  except:
    img_len=len(batch_img[data])
    features = model.predict(batch_img[data]).flatten().reshape(img_len,feature_size)
  all_fea.extend(features)


  def correct_mispredictions(query, fea_label,train_label, train_id, ind_data, decision,data_frame_1, count, pos_dist, neg_dist, pos_features, neg_features):
    if mode(decision) != query["label"]:
        # print("here")
        count +=1 
        data_frame_1["Image name"].append(query["filepath"].split("/")[-1])
        data_frame_1["Mistake ID"].append(query['id'])
        data_frame_1["Original label"].append(query['label'])
        data_frame_1["Predicted label"].append(mode(decision))
        data_frame_1["Mistake index"].append(ind_data)
        if query["label"]==1:
            pos_features_list= list(pos_features)
            pos_features_list.append(np.expand_dims(query["image_features"], axis=0))
            pos_features = np.array(pos_features_list)
            # pos_features= np.concatenate((pos_features,np.expand_dims(query["image_features"], axis=0)),axis=0)
        else:
            neg_features_list= list(neg_features)
            neg_features_list.append(np.expand_dims(query["image_features"], axis=0))
            neg_features = np.array(neg_features_list)
            # neg_features = np.concatenate((neg_features,np.expand_dims(query["image_features"], axis=0)),axis=0)
        train_label[query['label']].append(query["label"])
        train_id[query['label']].append(query['id'])

    else:
        if query['label'] == 0:
            max_ind = np.argmax(neg_dist)
            neg_features[max_ind] = np.concatenate((neg_features[max_ind],np.expand_dims(query["image_features"], axis=0)),axis=0)
            # fea_label[query['label']].append(np.concatenate((fea_label[query['label']],np.expand_dims(query["image_features"], axis=0)),axis=0))
        else:
            max_ind = np.argmax(pos_dist)

            pos_features[max_ind] = np.concatenate((pos_features[max_ind],np.expand_dims(query["image_features"], axis=0)),axis=0)
            # fea_label[query['label']].append(np.concatenate((mpos_features,np.expand_dims(query["image_features"], axis=0)),axis=0))
        train_label[query['label']].append(query["label"])
        train_id[query['label']].append(query['id'])
    return count,data_frame_1,fea_label,train_label,train_id,pos_features,neg_features
  

def distance2(query, fea_label, select_distance, id_pred, label_pred, n_neighbours, count, train_label, train_id, ind_data, data_frame_1, pos_features, neg_features, supervised_data): # Query is the raw dictionary (from pickle file) // fea_label is dictionary of {0: [], 1:[]} (distance) // select distance is int
  exp_query = np.expand_dims(query['image_features'], axis=0)
  pos_tup, neg_tup = [], []

  if select_distance==1: # Euclidean distance
    # print(f"Type: {type(fea_label[0])}")
    # print(f"Shape: {fea_label[0].shape}")
    neg_dist = np.linalg.norm(query['image_features']- fea_label[0], axis=1)  # Calculating the Euclidean distance using numpy (axis=1) to calculate all at ones   
    pos_dist = np.linalg.norm(query['image_features']- fea_label[1], axis=1)

  # elif select_distance==2: # Manhattan distance
  #   neg_dist = np.squeeze(manhattan_distances(fea_label[0],exp_query))  # convert (1,n) to (,n)
  #   pos_dist=np.squeeze(manhattan_distances(fea_label[1],exp_query))

  # elif select_distance==3: # Cosine distance
  #   neg_dist = np.squeeze(cosine_distances(exp_query,fea_label[0]))  # convert (1,n) to (,n)
  #   pos_dist=np.squeeze(cosine_distances(exp_query,fea_label[1]))
  
  for dist_single in pos_dist:
    # print(dist_single)
    pos_tup.append((dist_single,1))

  for dist_single in neg_dist:
    neg_tup.append((dist_single,0))

  pos_tup.extend(neg_tup)
  tup_dist = sorted(pos_tup)[:n_neighbours]
  
  decision = [y for (x,y) in tup_dist]

  if supervised_data:
    count,data_frame_1,fea_label,train_label,train_id, pos_features,neg_features=correct_mispredictions(query, fea_label,train_label,train_id, ind_data, decision,data_frame_1, count, pos_dist, neg_dist, pos_features, neg_features)
    
  else:
    if decision.count(0) > decision.count(1):
      neg_features, fea_label, id_pred, label_pred = update_subclusters(neg_dist,query,fea_label,id_pred,label_pred,neg_features, decision, n_neighbours, cluster=0)
      
    else:
      pos_features, fea_label, id_pred, label_pred = update_subclusters(pos_dist,query,fea_label,id_pred,label_pred,pos_features, decision,n_neighbours, cluster=1)
  
  return id_pred, label_pred, data_frame_1, count, train_label, train_id, pos_features, neg_features


n_neighbours=15

data_frame = {"Labeled data": [],
              "Dataset": [],
              "Accuracy": [],
              "Specificity": [],
              "Sensitivity": [],
              "AUC":[],
              "Dunn index": [],
              "Davies Bouldin": [],
              "Silhouette index":[],
              "TP":[],
              "TN":[],
              "FP":[],
              "FN":[],
              "pos_labeled_img":[],
              "neg_labeled_img":[],
              "corrected_count":[]
    
}
# fea_label1={0: [],
#             1:[]}


for size in labeled_size:
  labeled_data, unlabeled_data = data_loader(t_dataset, size)
#   print(f"labeled data length {len(labeled_data)}")
#   print(f"Unlabeled data length {len(unlabeled_data)}")
  select=0         # To select the dataset out of three sets ==> three sets: [d11, d12, d13] ==> eg: [200,200,200]




  while(select < 3):
    data_frame_1 = {  "Image name": [],
                  "Mistake index": [],
                  "Mistake ID": [],
                  "Original label": [],
                  "Predicted label": []
                  
    }
    pos_img, neg_img=0, 0

    fpos, fneg= [], []

    label_gt = {0: [],    
        1 :[]}    
                            # Collect the ground truth (label) of all the predicting images
    train_label = {0: [],    
        1 :[]}    

    label_pred = {0: [],
        1 :[]}               # Collect the predicted label for all the images

    id_gt = {0: [], 
            1: [] }         # Collect the ground truth (id) of all the predicting images

    id_pred = {0: [],
            1: []}        # Collect the predicted id for all the images 

    fea_label = {0: [],
            1: []}

    train_id ={0: [],
            1:[]}
        
    # print(type(labeled_data[0][0]))
    # for data in labeled_data[select]:
    #     if data["label"] == 1:
    #         fpos.append(data['image_features'])
    #         train_id[1].append(data['id'])
    #         train_label[1].append((data['id'],data['label']))
    #         pos_img +=1

    #     else:
    #         fneg.append(data['image_features'])
    #         train_id[0].append(data['id'])
    #         train_label[0].append((data['id'],data['label']))
    #         neg_img +=1

    # print(f"Blen: {len(labeled_data[select])}")
    fpositive = data_separation(labeled_data[select],1)    # Get 20 features of each class

    
    fnegative = data_separation(labeled_data[select],0)


    mneg_features,neg_features= sub_clusters(fnegative)  # Get the subclusters (Using K-means algorithm)
    mpos_features,pos_features= sub_clusters(fpositive)    

        

    count, ind_data=0, 40
    for data in labeled_data[select]:
        fea_label={0: mneg_features,
            1: mpos_features}
        id_pred, label_pred, data_frame_1, count, train_label, train_id, pos_features, neg_features= distance2(data,fea_label,1,id_pred,label_pred,n_neighbours, count, train_label, train_id, ind_data, data_frame_1, pos_features, neg_features, supervised_data=True)
        mpos_features, mneg_features = mean_features(pos_features, neg_features)    # Get the mean of the features
        ind_data +=1

    data_f_1 = pd.DataFrame.from_dict(data_frame_1)
    data_f_1.to_csv(f"./csv_results_x-ray_counts/new/resnet101_euclidean_mistake_{size}_{select}.csv",index=False)

    for data in tqdm(unlabeled_data[select]):
      if data["label"]==1:
        id_gt[1].append(data['id'])
        label_gt[1].append((data['id'],data['label']))
      
      else:
        id_gt[0].append(data['id'])
        label_gt[0].append((data['id'],data['label']))
      
      fea_label={0: mneg_features,
            1: mpos_features}

      id_pred, label_pred, _, _, _, _, pos_features, neg_features = distance2(data,fea_label,1,id_pred,label_pred,n_neighbours, count, train_label, train_id, ind_data, data_frame_1, pos_features, neg_features,supervised_data=False) # ind_data is the index of misclassification
      mpos_features, mneg_features = mean_features(pos_features, neg_features)    # Get the mean of the features

    accuracy, specificity, sensitivity,TP,TN,FP,FN= classification_metrics(id_gt,id_pred)
    flattened_pos_features = flatten_features(pos_features) 
    flattened_neg_features = flatten_features(neg_features)
    dunn_index, davies_bouldin_index, silhouette_index = cluster_metrics(flattened_pos_features, flattened_neg_features, train_label,id_pred)
    cl_auc = roc_auc_curve(label_gt,label_pred)
    data_frame["Labeled data"].append(size)
    data_frame["Dataset"].append(f"d_{select}")
    data_frame["Accuracy"].append(accuracy)
    data_frame["Specificity"].append(specificity)
    data_frame["Sensitivity"].append(sensitivity)
    data_frame["AUC"].append(cl_auc)
    data_frame["Dunn index"].append(dunn_index)
    data_frame["Davies Bouldin"].append(davies_bouldin_index)
    data_frame["Silhouette index"].append(silhouette_index)
    data_frame["TP"].append(TP)
    data_frame["TN"].append(TN)
    data_frame["FP"].append(FP)
    data_frame["FN"].append(FN)
    data_frame["pos_labeled_img"].append(pos_img)
    data_frame["neg_labeled_img"].append(neg_img)
    data_frame["corrected_count"].append(count)

    print(f"Labeled image: {size} \t Dataset: d_{select} \t Accuracy: {accuracy} \t Specificity: {specificity} \t Sensitivity: {sensitivity} \t Dunn index: {dunn_index}  \t Davies Bouldin: {davies_bouldin_index} \t Silhouette index: {silhouette_index} \t AUC: {cl_auc} \t Corrected count: {count}")
    select +=1 
  