# The code defines a custom dataset class that reads a label file and corresponding data files, processes the medical data into a structured format, 
# and provides it in a format suitable for use with PyTorch. The design allows for easy access to individual data entries, making it ideal for training machine learning models. 

from __future__ import print_function                                                                    # Ensures compatibility for the print version with future implementations of Python.
import torch.utils.data as data                                                                          # Provides classes and functions for handling datasets and data loading.
import torch                                                                                             # Used for building neural networks among other things.
import os                                                                                                # Used to assist with working with files and directory paths.

# Initialize the dataset.
class MyDataset(data.Dataset):
    def __init__(self, dataPath, labelPath):
        labelFile = open(labelPath)                                                                      # Open the label file containing data file names and labels.
        dataset=[]                                                                                       # Initialize an empty list for storing dataset items.
        #dataset: filenames,labels            
        line_num = 0                                                                                     # Initialize a counter for line number.
        for line in  labelFile.readlines():                                                              # Loop through each line in the label file.
        # rstrip() remove spaces in right end
            if line_num!=0:                                                                              # Skip the header (if any).
                words = line.strip().split(',')                                                          # Remove leading and trailing whitespace and create a list of strings split on commas.
                if os.path.isfile(os.path.join(dataPath, words[0]+".txt")):                              # Check if a corresponding data file exists for the entry.
                    dataset.append((words[0]+".txt", words[len(words)-1]))                               # Add the filename and its associated label to the dataset list.
            line_num=line_num+1 # advance line_num for loop                                              # Increment the line number counter.
        self.dataPath = dataPath                                                                         # Save the path to the data files. Holds the directory of data files.
        self.dataset = dataset                                                                           # Store the list of datasets. Holds teh dataset tuples of (file, label).
        dic={'time':0,'Age':1,'Gender':2,'Height':3,'ICUType':4,'Weight':5,'Albumin':6,\                 # Defines a dictionary (dic) that maps medical feature names (e.g., Age, Weight) to specific indices (0–41).
             'ALP':7,'ALT':8,'AST':9,'Bilirubin':10,'BUN':11,'Cholesterol':12,'Creatinine':13,\
             'DiasABP':14,'FiO2':15,'GCS':16,'Glucose':17,'HCO3':18,'HCT':19,'HR':20,\
             'K':21,'Lactate':22,'Mg':23,'MAP':24,'MechVent':25,'Na':26,'NIDiasABP':27,\
             'NIMAP':28,'NISysABP':29,'PaCO2':30,'PaO2':31,'pH':32,'Platelets':33,'RespRate':34,\
             'SaO2':35,'SysABP':36,'Temp':37,'TropI':38,'TropT':39,'Urine':40,'WBC':41}
    
        self.dic=dic
        #self.classes = class_names
        #self.transform = transform
        #self.target_transform = target_transform
        #self.loader = loader

    #  Used to retrieve a single data sample (and its label) from the dataset.
    def __getitem__(self, index):
        #print(self.dataset[index])
        fileName, label = self.dataset[index]                                                            # Get the filename and label at the given index.
        f=open(os.path.join(self.dataPath, fileName))                                                    # Open the corresponding data file.
        #read_csv is DataFrame, need to be transformed into ndarray
        count=0                                                                                          # Initialize counter.
        age=gender=height=icutype=weight=-1                                                              # Initialize key patient information to -1 (missing).
        lastTime=0                                                                                       # Track the last timestamp processed.
        totalData=[]                                                                                     # Initialize list to store time-series data.

        # This section processes each line of the file, extracting timestamps, features, and values.
        # For each new timestamp, a new entry is created in totalData. For repeated timestamps, the existing entry is updated with new feature values.
        for line in f.readlines():                                                                        
            if count > 1:
                words=line.split(",")                                                                    # Split the line into words (timestamp, feature, value).
                timestamp=words[0]
                feature=words[1]
                value=words[2]
                
                # -1 is missing value
                if timestamp == "00:00":                                                                 # If the timestamp is "00:00", the patient's baseline info is stored.
                    if feature=='Age':
                        age=value
                    if feature=='Gender':
                        gender=value
                    if feature=='Height':
                        height=value
                    if feature == 'ICUType':
                        icutype=value
                    if feature=='Weight':
                        weight=value
                else:
                    if timestamp!=lastTime:                                                               # If this is a new timestamp, start a new data entry.
                        data=[-1.0]*42                                                                    # Initialize the data entry with -1 (missing) values.
                        hourandminute=timestamp.split(":")
                        data[0]=float(hourandminute[0])*60+float(hourandminute[1])                        # Convert timestamp to minutes.
                        data[1]=float(age)                                                                # Add patient baseline info.
                        data[2]=float(gender)                                                             # Add patient baseline info.
                        data[3]=float(height)                                                             # Add patient baseline info.
                        data[4]=float(icutype)                                                            # Add patient baseline info.
                        data[5]=float(weight)                                                             # Add patient baseline info.
                        data[self.dic[feature]]=float(value)                                              # Update the corresponding feature using the feature dictionary.
                        totalData.append(data)                                                            # Add the data entry to the list.
                    else:
                        totalData[len(totalData)-1][self.dic[feature]]=float(value)                       # If the same timestamp, update the existing entry with new feature data.
                lastTime=timestamp      
            count+=1
            #if len(totalData)==24:
            #    break;
        #print(totalData)
        #print(label)
        return torch.FloatTensor(totalData), label, fileName                                              # Return the processed time-series data, label, and filename.

    # Returns the total number of entries in the dataset.
    def __len__(self):
        return len(self.dataset)

dataset = MyDataset("/home/lyh/Desktop/set-a/train/", "/home/lyh/Desktop/set-a/train/list.txt")           # Instantiate the dataset object. Creates an instance of the MyDataset class, loading data from the specified path.
train_loader = torch.utils.data.DataLoader(\                                                              # Initializes a DataLoader to handle batch loading with a batch size of 2. 
    dataset, batch_size=2, shuffle=False, num_workers=0)
print(len(train_loader))                                                                                  # Iterates through the batches, printing filenames and labels.
for dataset,label,filename in train_loader:
    print(filename)
    #print(dataset)
    print(label)

#for dataset in train_loader:
#    print(dataset)
    


