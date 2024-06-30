from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
import numpy as np

def classification_metrics(label_gt,id_pred):
  TP,FP,FN,TN = 0,0,0,0

  for tp in id_pred[1]:   # TP --> When correctly classified covid
    if tp in label_gt[1]:
      TP +=1

  for tn in id_pred[0]:  # TN --> When correctly classified healthy (non-covid)
    if tn in label_gt[0]:
      TN +=1

  for fp in id_pred[1]: # FP --> When incorrectly classified healthy (Classified healthy as covid)
    if fp in label_gt[0]:
      FP +=1

  for fn in id_pred[0]: # FN --> When missed covid classification (Covid cases missed)
    if fn in label_gt[1]:
      FN +=1

  accuracy= (TP+TN)/(TP+TN+FP+FN)
  specificity = TN/(TN+FP)
  sensitivity = (TP)/(TP+FN)
  # f1_score = (2*precision*recall)/(precision + recall)
  
  print("TP: ", TP)
  print("FP: ", FP)
  print("FN: ", FN)
  print("TN: ", TN)

  return accuracy, specificity, sensitivity,TP,TN,FP,FN

def roc_auc_curve(label_gt,label_pred):
  gt_labels= sorted(label_gt[0]+ label_gt[1])  # Contains (id,labels) tuple of binary class 
  pred_labels = sorted(label_pred[0]+label_pred[1]) # Contains (id,labels) tuple of binary class --> sorted to match each element in gt_labels and pred_labels
  y_test = [y for (x,y) in gt_labels]   # Get only the labels
  y_scores = [y for (x,y) in pred_labels]
  fpr, tpr, threshold = roc_curve(y_test, y_scores)
  roc_auc = auc(fpr, tpr)
  return roc_auc

def cluster_metrics(pos_features, neg_features, train_label,id_pred):
  print("Calculating Dunn's index...")
  intra_dist1 = euclidean_distances(neg_features).max()
  intra_dist2 = euclidean_distances(pos_features).max()
  inter_dist = euclidean_distances(neg_features,pos_features).min()

  if intra_dist1>intra_dist2:
    max_intra_dist= intra_dist1  
  else:
    max_intra_dist = intra_dist2 

  Dunn_index = inter_dist/max_intra_dist

  print("Calculating Davies Bouldin index...")

  # Davies Bouldin and Silhouette score from sklearn library.
  class_0 =np.concatenate((np.zeros(shape=(len(train_label[0])),dtype=int),np.zeros(shape=(len(id_pred[0])),dtype=int),np.zeros(shape=(20),dtype=int)))
  class_1 = np.concatenate((np.ones(shape=(len(train_label[1])),dtype=int),np.ones(shape=(len(id_pred[1])),dtype=int),np.zeros(shape=(20),dtype=int)))
  class_all = np.concatenate((class_0,class_1))
  feature_all = np.concatenate((neg_features,pos_features))

  davies_bouldin_index = davies_bouldin_score(feature_all,class_all)
  silhouette_index = silhouette_score(feature_all,class_all)

  print("davies: ", davies_bouldin_index)
  print("silhouette_sklearn: ", silhouette_index)
  
  return Dunn_index,davies_bouldin_index, silhouette_index