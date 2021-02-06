from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.linalg as la
import math
import random
import os

size_x=50
size_y=50
n_train=90
n_test=10

accuracy_average_case=[]
accuracy_closest_matching_case=[]
accuracy_knn_case=[]
accuracy_hybrid_case=[]
KNN_parameter=3
total_iteration=10

#Function that splits the dataset into training images and test images
def train_test_listmaker():
        image_list = [image_name for image_name in os.listdir("dataset")]
        train_image_list = list(random.sample(image_list,n_train))
        test_image_list = [image for image in image_list if image not in train_image_list]
        return train_image_list,test_image_list

#Function that forms the data matrix
def data_matrix_maker ( image_list  ) :
        data_matrix = []
        for pic_name in image_list:
            image = io.imread("dataset/" + pic_name)
            data = [int(image[y][x]) for y in range(size_y) for x in range(size_x)]
            data_matrix.append(data)
        return data_matrix

# Function to calculate the average face  and output also
def average_image_maker ( data_matrix ) :
        average_image = []
        for i in range(0,size_x*size_y):
                    sum_feature_i = 0
                    for j in range(0,n_train):
                            sum_feature_i += data_matrix[j][i]
                    average=sum_feature_i/n_train
                    average_image.append(average)
        face=[]
        count=0
        for y in range(size_y):
            row=[]
            for x in range(size_x):
                row.append(average_image[count])
                count+=1
            face.append(row)
        fig=plt.figure(frameon=False,figsize=(3,3))
        plt.axis('off')
        plt.title("Average face of all the training faces is as :")
        plt.imshow(face,cmap=cm.gray)
        plt.show()
        return average_image

# Function that does centering of the images
def centred_matrix_maker(data_matrix,average_image):
        centred_data_matrix=[]
        for j in range(0,len(data_matrix)):
                    centred_data=[data_matrix[j][i] - average_image[i] for i in range(size_x*size_y)]
                    centred_data_matrix.append(centred_data)
        return centred_data_matrix

# Function for forming covariance matrix
def covariance_matrix_maker(centred_data_matrix):
        covariance_matrix=[]
        for i in range(0,size_x*size_y):
                    row=[]
                    for k in range(0,size_x*size_y):
                            su=0
                            for j in range(0,n_train):
                                        su+=centred_data_matrix[j][i]*centred_data_matrix[j][k]
                            row.append(su/(n_train))
                    covariance_matrix.append(row)
        return covariance_matrix

# Function that calculates eigenvector and eigenvalues
def eigen_value_and_vector_maker(covariance_matrix):
        eigen=la.eigh(covariance_matrix)
        eigen_vec=[]
        for i in range(len(eigen[1][0])):
                row=[eigen[1][j][i] for j in range(len(eigen[1]))]
                eigen_vec.append(row)
        eigen_values=eigen[0][::-1]
        eigen_vector=eigen_vec[::-1] 
        return eigen_values,eigen_vector


# Function that calculates variances explained by eigenfaces
def top_k_eigenfaces_calculator(eigen_values,eigen_vector):
        variance_explained  = 0
        top_k_eigenfaces    = 0
        x=[]
        y=[]
        while(variance_explained < 0.9999):
                    variance_explained += eigen_values[top_k_eigenfaces]/sum(eigen_values)
                    top_k_eigenfaces   += 1
                    x.append(top_k_eigenfaces)
                    y.append(variance_explained)
        plt.title("% of variance explained by the principal components")
        plt.xlabel("Number of principal eigenfaces ---->")
        plt.ylabel("% of variance explained")
        plt.plot(x , y, color="r")
        plt.show()
        return top_k_eigenfaces

# Function that outputs the principal eigenfaces
def show_principal_faces(eigenfaces,top_k_eigenfaces):
        fig=plt.figure(frameon=False,figsize=(15,15))
        faces_per_Row=10
        total_rows=math.ceil(top_k_eigenfaces/faces_per_Row)
        plt.axis('off')
        plt.title("Plotting of top " +str(top_k_eigenfaces) +" principal eigenfaces\n")
        subplot_num=1
        for i in range(0,top_k_eigenfaces):
            current_image=eigenfaces[i]
            image=[]
            index=0
            for j in range(0,size_y):
                row=[]
                for k in range(0,size_x):
                    row.append(current_image[index])
                    index+=1
                image.append(row)
            test=fig.add_subplot(total_rows,faces_per_Row,subplot_num)
            test.imshow(image,cmap=cm.gray)
            subplot_num += 1
            test.set_xticks([])
            test.set_yticks([])
        plt.show()

#Function that forms the projection matrix 
def projection_matrix_maker(data_matrix,eigen_vector,top_k_eigenfaces):
        projection_matrix=[]
        for j in range(0,len(data_matrix)):
                    row=[]
                    for v in range(0,top_k_eigenfaces):
                            su=0
                            for i in range(0,size_x*size_y):
                                        su += data_matrix[j][i]*eigen_vector[v][i]
                            row.append(su)
                    projection_matrix.append(row)
        return projection_matrix

# Accuracy Method 1 : average closet distance
def accuracy_average_closest_distance(test_projection_matrix,train_projection_matrix,test_image_list, train_image_list,top_k_eigenfaces):
        correct_predicted=0
        prediction_array=[]
        for  i in range(0,n_test):
                    test_person = test_image_list[i][:(len(test_image_list[i])-7)]
                    
                    distance_dict={ "John_Bolton":0,   "Kofi_Annan":0,
                                    "Gerhard_Schroeder":0,  "Atal_Bihari_Vajpayee":0,
                                    "John_Ashcroft":0,  "Donald_Rumsfeld":0,
                                    "Jean_Chretien":0,  "Serena_Williams":0,
                                    "David_Beckham":0,  "Winona_Ryder":0 }
                    count_dict={ "John_Bolton":0, "Kofi_Annan":0, "Gerhard_Schroeder":0,
                                "Atal_Bihari_Vajpayee":0, "John_Ashcroft":0, "Jean_Chretien":0,
                                "Donald_Rumsfeld":0, "Serena_Williams":0, "David_Beckham":0,
                                "Winona_Ryder":0 }

                    for j in range(0,n_train):
                            train_person=train_image_list[j][:(len(train_image_list[j])-7)]
                            dist=0
                            for k in range(0,top_k_eigenfaces):
                                        dist=dist+pow(train_projection_matrix[j][k]-test_projection_matrix[i][k],2)
                            distance_dict[train_person] += dist
                            count_dict[train_person] += 1
                    
                    for p in distance_dict : distance_dict[p] /= count_dict[p]

                    min_distance = min(distance_dict.values())
                    for person in distance_dict:
                            if distance_dict[person] == min_distance : closest_person = person
                    if closest_person == test_person : correct_predicted += 1
                    prediction_array.append(closest_person)

        return  correct_predicted,prediction_array

# Accuracy Method 2 :  closet matching person (simply KNN with parameter=1)
def accuracy_closest_matching_person(test_projection_matrix,train_projection_matrix,test_image_list, train_image_list,top_k_eigenfaces):
        correct_predicted=0
        prediction_array=[]
        for  i in range(0,n_test):
                    large = 1000000000
                    test_person = test_image_list[i][:(len(test_image_list[i])-7)]
                    min_distance = large

                    for j in range(0,n_train):
                            train_person=train_image_list[j][:(len(train_image_list[j])-7)]
                            dist=0
                            for k in range(0,top_k_eigenfaces):
                                        dist=dist+pow(train_projection_matrix[j][k]-test_projection_matrix[i][k],2)
                            if (dist < min_distance):
                                        min_distance   =  dist
                                        closest_person =  train_person
                        
                    if closest_person == test_person : correct_predicted += 1
                    prediction_array.append(closest_person)

        return  correct_predicted,prediction_array

# Accuracy Method 3 : KNN (k-nearest neighbor with parameter k=3) 
def accuracy_knn (test_projection_matrix,train_projection_matrix,test_image_list, train_image_list,top_k_eigenfaces,n_nearest_neighbor) :
        correct_predicted=0
        prediction_array=[]
        for  i in range(0,n_test):
                    large = 1000000000
                    test_person = test_image_list[i][:(len(test_image_list[i])-7)]
                    distance_list=[]
                    for j in range(0,n_train):
                            train_person=train_image_list[j][:(len(train_image_list[j])-7)]
                            dist=0
                            for k in range(0,top_k_eigenfaces):
                                        dist=dist+pow(train_projection_matrix[j][k]-test_projection_matrix[i][k],2)
                            distance_list.append((dist,train_person))
                    
                    distance_list.sort()
                    frequency_dict={}
                    for r in range(0,n_nearest_neighbor):
                        if distance_list[r][1] in frequency_dict:
                            frequency_dict[distance_list[r][1]]+=1
                        else:
                            frequency_dict[distance_list[r][1]]=1
                    max_frequency=max(frequency_dict.values())
                    max_frequency_person=[x for x in frequency_dict if frequency_dict[x]==max_frequency]
                    
                    for x in range(0,len(distance_list)):
                            if distance_list[x][1] in max_frequency_person:
                                closest_person = distance_list[x][1]
                                break                    
                        
                    if closest_person == test_person : correct_predicted += 1
                    prediction_array.append(closest_person)

        return  correct_predicted,prediction_array

# Accuracy Method 4 : Hybrid case of method 1 and method 2
def accuracy_hybrid(test_projection_matrix,train_projection_matrix,test_image_list, train_image_list,top_k_eigenfaces,prediction_array_1,prediction_array_2):
        correct_predicted=0
        prediction_array=[]
        for i in range(0,n_test):
                test_person = test_image_list[i][:(len(test_image_list[i])-7)]
                average_distance_label = prediction_array_1[i]
                closest_match_label = prediction_array_2[i]
                if (closest_match_label != average_distance_label):
                    dist1=0
                    dist2=0
                    count1=0
                    count2=0
                    for j in range(0,n_train):
                        train_person = train_image_list[j][:(len(train_image_list[j])-7)]
                        if(train_person == prediction_array_1[i]):
                            count1+=1
                            for k in range(0,top_k_eigenfaces):
                                dist1 += pow(train_projection_matrix[j][k]-test_projection_matrix[i][k],2)
                        if(train_person == prediction_array_2[i]):
                            count2+=1
                            for k in range(0,top_k_eigenfaces):
                                dist2 += pow(train_projection_matrix[j][k]-test_projection_matrix[i][k],2)
                    dist1 /= dist1/count1
                    dist2 /= dist2/count2

                    if((dist2-dist1)*100/dist2 <10):
                        closest_person=prediction_array_2[i]
                    else:
                        closest_person = prediction_array_1[i]

                else:
                    closest_person = prediction_array_1[i]
                if closest_person == test_person : correct_predicted += 1
                prediction_array.append(closest_person)

        return  correct_predicted,prediction_array

# For loop for running the experiment multiple times to calculate average accuracy                
for iteration in range(total_iteration):
        print("iteration number :",iteration)                
        train_image_list,test_image_list = train_test_listmaker()

        train_data_matrix = data_matrix_maker (train_image_list )
        test_data_matrix = data_matrix_maker (test_image_list)  

        average_face = average_image_maker(train_data_matrix)

        centred_train_data_matrix = centred_matrix_maker(train_data_matrix,average_face)
        centred_test_data_matrix = centred_matrix_maker(test_data_matrix,average_face)

        covariance_matrix = covariance_matrix_maker(centred_train_data_matrix)

        eigen_values,eigen_vector =  eigen_value_and_vector_maker(covariance_matrix)
        top_k_eigenfaces = top_k_eigenfaces_calculator(eigen_values,eigen_vector)
        show_principal_faces(eigen_vector,top_k_eigenfaces)

        train_projection_matrix = projection_matrix_maker(centred_train_data_matrix,eigen_vector,top_k_eigenfaces) 
        test_projection_matrix = projection_matrix_maker(centred_test_data_matrix,eigen_vector,top_k_eigenfaces) 
        
        # Accuracy by measuring the average closest distance from a person.
        correct_predicted,prediction_array_average = accuracy_average_closest_distance(test_projection_matrix, train_projection_matrix,
                                                                                       test_image_list, train_image_list, top_k_eigenfaces)
        accuracy_average_case.append(correct_predicted/n_test)

        #Accuracy by observing the closest matching person (can say k means with k=1)
        correct_predicted,prediction_array_closest_matching = accuracy_closest_matching_person(test_projection_matrix, train_projection_matrix,
                                                                                               test_image_list, train_image_list, top_k_eigenfaces)
        accuracy_closest_matching_case.append(correct_predicted/n_test)

        #Accuracy according to kmeans with parameter no_nearest_neighbor=5
        correct_predicted,prediction_array_knn = accuracy_knn(test_projection_matrix, train_projection_matrix, test_image_list, train_image_list,
                                                              top_k_eigenfaces,KNN_parameter)
        accuracy_knn_case.append(correct_predicted/n_test)

        # Accuracy by hybrid case.
        correct_predicted,prediction_array_hybrid = accuracy_hybrid(test_projection_matrix, train_projection_matrix, test_image_list, train_image_list,
                                                                   top_k_eigenfaces,prediction_array_average,prediction_array_closest_matching)
        accuracy_hybrid_case.append(correct_predicted/n_test)

print("Accuracy by average distance case (method 1) :",accuracy_average_case) 
print("Accuracy by  closest matching face (method 2) :",accuracy_closest_matching_case) 
print("Accuracy by KNN (method 3) :",accuracy_knn_case) 
print("Accuracy by Hybrid case (method 4) :",accuracy_hybrid_case) 

print("\n Average Accuracy by average distance case (method 1) :",sum(accuracy_average_case)/total_iteration) 
print("Average Accuracy by  closest matching face (method 2) :",sum(accuracy_closest_matching_case)/total_iteration) 
print("Average Accuracy by KNN  with parameter k=" +str(KNN_parameter) +"(method 3) :",sum(accuracy_knn_case)/total_iteration) 
print("Average Accuracy by Hybrid case (method 4) :",sum(accuracy_hybrid_case)/total_iteration) 

