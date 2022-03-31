#   < IMAGE SEARCH ENGINE >
import numpy as np
import matplotlib
import cv2
import matplotlib.pyplot as plt
import glob

#ENTER THE image DATA SET PATH HERE as "path/*.jpg"
#for exampple "/content/drive/MyDrive/Task 2 Image Dataset/BBT/*.jpg"
#' /*.jpg ' is to read images from folder

path = glob.glob("/content/drive/MyDrive/Task 2 Image Dataset/BBT/*.jpg")
#input image from user
query_image = input("please input image or path to image")
img = cv2.imread(query_image)
cv2.imshow(img)
print("\nsimilar images:\n")
# Calculate histogram
hist1_red = cv2.calcHist([img],[0],None,[256],[0,256])
hist2_green = cv2.calcHist([img],[1],None,[256],[0,256])
hist3_blue = cv2.calcHist([img],[2],None,[256],[0,256])
hist1 = hist1_red.reshape([1,-1])
hist2 = hist2_green.reshape([1,-1])
hist3 = hist3_blue.reshape([1,-1])
combine = np.concatenate((hist1,hist2,hist3),axis = 1)
#extract image set

for file in path:
  #print(file)
  img1 = cv2.imread(file)
  # Calculate histogram for data set individually and comparing
  hist4_red = cv2.calcHist([img1],[0],None,[256],[0,256])
  hist5_green = cv2.calcHist([img1],[1],None,[256],[0,256])
  hist6_blue = cv2.calcHist([img1],[2],None,[256],[0,256])
  hist4 = hist4_red.reshape([1,-1])
  hist5 = hist5_green.reshape([1,-1])
  hist6 = hist6_blue.reshape([1,-1])
  combine1 = np.concatenate((hist4,hist5,hist6),axis = 1)
  #cosine similarity(cosine_distance)
  #cose_similarity = a.b/(|a|*|b|)
  a = np.squeeze(combine)
  b= np.squeeze(combine1)
  dot_product=np.dot(a,b) #dot product of arrays
  norm_a = np.linalg.norm(a) #np.linalg.norm= |a|=square root of sum of squares of all elements
  norm_b = np.linalg.norm(b)
  cos_similarity = dot_product/(norm_a*norm_b) 
  #print(cos_similarity)
  threshold = 0.85
  if cos_similarity > threshold:
    print("similarity: ",cos_similarity)
    cv2.imshow(img1)
    print("\n")