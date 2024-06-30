import os
import numpy as np
import glob
import cv2

dataset_path = "./x-ray_dataset"

labels = [0,1] # 1 = Covid // 0 = Noncovid 

all_files=[]

for i in os.listdir(dataset_path):  # Get all the files from the directory in a two element list. First element is list of file location to covid images and second element is list of file location to non-covid images.
  file1 = glob.glob(os.path.join(dataset_path,i, "*.png"))
  file2 = glob.glob(os.path.join(dataset_path,i, "*.jpg")) # .jpg files are also present.
  file1.extend(file2)  # Only extends when there is .jpg file present
  all_files.append(file1)


  count=0     # Count to record the ids of files. Each file has a unique ID.
img_size = 224
def get_dataset(files, label,count):        
  dataset=[]  # List to hold all the dataset. Each element is a dictionary
  
  for j in tqdm(files):  # Loop over each file location
    data_dict = {}  
    data_dict["id"] = count
    data_dict["filepath"] = j
    img=cv2.imread(j)
    img = cv2.resize(img,(img_size,img_size))
    data_dict["image"]= img
    data_dict["label"]= label
    count +=1
    dataset.append(data_dict)
  return dataset, count


c_dataset, nc_dataset, t_dataset = [], [], []  

for i,data in enumerate(all_files[1:]):
  dataset,count=get_dataset(data,labels[i],count)
  if labels[i]==1:
    c_dataset = dataset
  else:
    nc_dataset = dataset
t_dataset = c_dataset + nc_dataset

image_only, label_only, id_only, img_name = [], [], [], []
for data in t_dataset:
  image_only.append(data["image"])
  label_only.append(data["label"]) 
  id_only.append(data['id'])
  img_name.append(data["filepath"].split("/")[-1])
image_only=np.array(image_only)


# labeled_size = [200,400,800,1100,1300,1550]
labeled_size = [1550]
def data_loader(dataset,n): # Method to return three sets of labeled dataset for experiment
  labeled_data, unlabeled_data = [], [] 

  l_data = dataset[:n]    # First dataset // labeled
  ul_data = dataset[n:]   # First dataset // unlabeled
  labeled_data.append(l_data)
  unlabeled_data.append(ul_data)

  l_data = dataset[1500:1500+n]    # second dataset // labeled
  ul_data = dataset[:1500]+dataset[1500+n:]
  labeled_data.append(l_data)
  unlabeled_data.append(ul_data)

  l_data = dataset[3000:3000+n]     # Third dataset // labeled
  ul_data = dataset[:3000]+dataset[3000+n:]
  labeled_data.append(l_data)
  unlabeled_data.append(ul_data)
  return labeled_data, unlabeled_data