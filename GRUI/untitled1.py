# It iterates over all folders in the current directory. For each folder, it looks into subdirectories and checks for the presence of a file named result.
# It reads the result file, processes the third value in each line, and keeps track of the highest value found.
# It creates empty files named after the highest values found (nowMin for each folder, globalMin for all folders).
# It prints the highest value found across all the directories.

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 09:23:18 2018

@author: yonghong
"""
import os                                                                                 # Assists with file and directory handling.

def f():                                                                                  # Defines the function that performs the main task of the script.

    folders =  os.listdir("./")                                                           # Lists all files and folders in the current directory and store in 'folders'.
    globalMin=0.0                                                                         # This variable will hold the maximum value found across all folders.
    for folder in folders:                                                                # Loop over each folder in the folders list.
        if os.path.isdir(folder):                                                         # Checks if the current folder is a directory (i.e., not a file). If true, the script continues processing; otherwise, it skips the item.
            secondPaths=os.listdir("./"+folder)                                           # Lists the contents (files and subdirectories) of the current folder, storing the result in secondPaths.
            nowMin=0.0                                                                    # This variable tracks the highest value found in the current folder.
            for s in secondPaths:                                                         # Loop over each item s in secondPaths.
                #print(s)
                if os.path.isfile(os.path.join("./",folder,s,"result")):                  #  Checks if a file named result exists inside the subdirectory s (under the folder). If true, the script proceeds.
                    with open(os.path.join("./",folder,s,"result"),"r") as f:             # Opens the result file for reading.
                        temp=0.0                                                          # Initializes temp=0.0 to track the highest value in the file.
                        for line in f:                                                    # Loops through each line in the result file.
                            #print(line)
                            a=float(line.split(",")[2])                                   # Splits the line using commas, then convert the third value (line.split(",")[2]) into a float.
                            #print(a)
                            if a>nowMin:                                                  # Compares a with nowMin, globalMin, and temp. If a is greater, the corresponding variable is updated with a.
                                nowMin=a
                            if a>globalMin:
                                globalMin=a
                            if a>temp:
                                temp=a 
                        ggg=open(os.path.join(folder,s,str(temp)),"w")                    # After processing the file, the script creates a file named after the highest value found (temp) in the subdirectory. The file is opened and immediately closed, creating an empty file.
                        ggg.close()
            r=open(os.path.join(folder,str(nowMin)),"w")                                  # After processing all subdirectories, a file is created in the main folder with the name nowMin (the highest value found in the current folder). This file is opened and then closed.
            r.close()
            if nowMin>globalMin:                                                          # Updates globalMin to nowMin if the latter is greater.
                globalMin=nowMin
    d=open(os.path.join(str(globalMin)),"w")                                              # Creates a file named after globalMin (the highest value found across all folders).
    d.close()
    print(globalMin)                                                                      # Prints the globalMin.

if __name__=="__main__":
    f()
            
        
