# CovidDetectionUsingXRay

##	ABSTRACT	
Coronavirus disease 2019 (COVID-2019), which first appeared in Wuhan, China in 2019 and has swiftly spread over the world since the beginning of 2020, has infected millions of people and killed a large number of them. The fight against the COVID-19 pandemic has emerged as one of the most promising issues in global healthcare. COVID-19 cases must be accurately and quickly diagnosed in order to receive proper medical treatment and limit the pandemic. After that, the virus spread all over the world, with over 4.7 million confirmed cases and over 315000 deaths as of the time of writing this report. Radiologists can employ machine learning algorithms developed on radiography pictures as a decision support mechanism to help them speed up the diagnostic process. 
This project has two objectives. Due to the limited number of images available for analysis, the CNN transfer learning approach was used. We have proposed a simple CNN architecture with a modest number of parameters that successfully differentiates COVID-19 from conventional X-rays. Second, to analyse the data and visualize the results for better understanding of the outcome. The images of the chest X-rays used in this study were obtained from two publically available sources. One COVID-19 X-ray image datasets were used, as well as a huge collection of non-COVID-19 X-rays.



## INTRODUCTION
The COVID-19 outbreak affects all segments of the population and is particularly detrimental to members of those social groups in the most vulnerable situations, continues to affect populations, including people living in poverty situations, older persons, persons with disabilities, youth, and indigenous peoples. Early evidence indicates that the health and economic impacts of the virus are being borne disproportionately by poor people. 
The coronavirus disease 2019 (COVID-19) is profoundly affecting life around the globe. Isolation, contact restrictions and economic shutdown impose a complete change to the psychosocial environment in affected countries. These measures have the potential to threaten the mental health of children and adolescents significantly. People with suspected COVID-19 need to know quickly whether they are infected, so they can receive appropriate treatment, self-isolate, and inform close contacts.
Currently, a formal diagnosis of COVID-19 requires a laboratory test (RT-PCR) of nose and throat samples. RT-PCR requires specialist equipment and takes at least 24 hours to produce a result. It is not completely accurate, and may require a second RT-PCR or a different test to confirm diagnosis.
Clinicians may use chest imaging to diagnose people who have COVID-19 symptoms, while awaiting RT-PCR results or when RT-PCR results are negative, and the person has COVID-19 symptoms.
This project is based on the different approaches that we can take to counter viruses like covid 19, and improving the current models that are countering spread of covid 19. Specially the detection of covid 19 virus with the help of X-Ray images.
X-rays or scans produce an image of the organs and structures in the chest. X-rays (radiography) use radiation to produce a 2-D image. Usually done in hospitals, using fixed equipment by a radiographer; they can also be done on portable machines.
Chest X-rays can be a great way to detect covid-19 virus since, chest x-rays are a fast and inexpensive test that may potentially diagnose COVID-19, the disease caused by the novel coronavirus. However, chest imaging is not a first-line test for COVID-19 due to low diagnostic accuracy and confounding with other viral pneumonias. Recent research using deep learning may help overcome this issue as convolutional neural networks (CNNs) have demonstrated high accuracy of COVID-19 diagnosis at an early stage.
The discovery of X-rays and the invention of CT represented major advances in medicine. X-ray imaging exams are recognized as a valuable medical tool for a wide variety of examinations and procedures. They are used to noninvasively and painlessly help to diagnose disease and monitor therapy; support medical and surgical treatment planning; and guide medical personnel as they insert catheters, stents, or other devices inside the body, treat tumors, or remove blood clots or other blockages.

### Problem Statement

Covid-19 is a big issue around the world right now, affecting millions of individuals. To combat this outbreak, scientists and doctors have opted to adopt real time reverse transcription polymerase chain reaction (RT-PCR) as the only way to detect the virus. However, because RT-PCR isn't available to everyone and can take up to 24 hours to produce results, we use CXR (Chest X-ray) to detect Covid-19, since this procedure is both inexpensive and effective. As a result, we're working on a project to train a CNN model that can detect the presence of Covid 19 using only X-Ray images.


### Objectives 

Collect the statistics from different sources and train a CNN model to predict whether the provided image is Covid-19 positive or not.
Analyze the data and compare the final results of Covid-19, based on the collected data.
Estimate whether chest imaging is accurate enough to diagnose COVID-19 in people with suspected infection.


### Scope of Project 


The areas of study covered under this project are data collection, data analysis, machine learning algorithms, image detection, and data visualization. Data will be collected through publicly available sources like Github and Kaggle. Build a CNN models from scratch to find the best possible accuracy to predict whether the given X-ray image is Covid-19 positive or not. We then visualize the results by using libraries like matplotlib and seaborn to better explain our findings.


## BACKGROUND DETAILS

### Literature Review/Related Work:
Anam-Net, a lightweight CNN based on anamorphic depth embedding, is proposed to segment abnormalities in COVID-19 chest CT images. When compared to the state-of-the-art UNet (or its variations), the suggested Anam-Net has 7.8 times less parameters, making it lightweight and capable of generating inferences in mobile or resource-constrained (point-of-care) systems.[1]

Another work presents a modified MobileNet architecture for COVID-19 CXR image classification and a modified ResNet design for CT image classification to overcome these issues. In addition, utilising CT scans, the suggested modified ResNet is used to classify COVID-19, non-COVID-19 infections, and normal controls.[2]

A bespoke CNN architecture has been presented in this study to enable automated learning of such latent characteristics. For each type of pneumonia, it learns distinct convolutional filter patterns. This is accomplished by limiting some filters in a convolutional layer to respond maximally exclusively to a specific type of pneumonia/COVID-19. Different convolution types are combined in the CNN architecture to provide greater context for learning robust features and to improve gradient flow across layers. The suggested work additionally visualises on the X-ray zones of saliency that have had the greatest impact on CNN's prediction outcome.[3]

They created and trained four CNN-RNN architectures to distinguish coronavirus infection from other infections. A total of 6396 X-ray samples from various sources are used as a dataset to detect coronavirus cases.   Each architecture's performance is measured in terms of area under the receiver operating characteristics (ROC) curve (AUC), accuracy, precision, recall, F1-score, and confusion matrix, and Grad-CAM is used to show the diseased zone of X-rays.[6] 

In terms of offering ANN-based lung segmentation, presenting a hybrid structure comprising a BiLSTM layer with transfer learning, and attaining high classification performance, this study adds to earlier research. To create a CNN-based transfer learning–BiLSTM network for early detection of COVID-19 infection, using ANN-based automatic lung segmentation to get robust features, and to compare the proposed hybrid method against other state-of-the-art models. The suggested model is simple, and it can detect COVID-19 totally on its own.[8].

To identify the virus using chest X-rays, we present a hybrid deep learning model based on a convolutional neural network (CNN) and a gated recurrent unit (GRU).
In this research (CXRs). A CNN is employed to extract features in the proposed model, while a GRU is used as a classifier. The model was trained using 424 CXR pictures divided into three groups (COVID-19, Pneumonia, and Normal).[9]

The data augmentation and CNN hyperparameters for identifying COVID-19 from CXRs are optimized in an article in terms of validation accuracy. The accuracy of common CNN architectures such as the Visual Geometry Group network (VGG-19) and the Residual Neural Network (ResNet-50) is improved by 11.93 percent and 4.97 percent, respectively, thanks to this modification. The CovidXrayNet model was subsequently suggested, which is based on EfficientNet-B0 and our optimization findings. CovidXrayNet was tested on two datasets: the COVIDcxr dataset (960 CXRs) that we developed and the benchmark COVIDx dataset (15,496 CXRs). [10]

## SYSTEM DESIGN AND METHODOLOGY
### System Architecture

The ER diagram of our proposed project model is as follows:

						
### Development Environment.

Hardware:
Central Processing Unit (CPU) — Intel Core i5 6th Generation processor or higher.
RAM — 4 GB minimum, 8 GB or higher is recommended.
Graphics Processing Unit (GPU) — NVIDIA GeForce GTX 960 or higher. 

Software:
Operating System — Ubuntu or Microsoft Windows 10.
High-Level Language — Python.
Jupyter Notebook
Web browser
Google Collaboratory
Google drive cloud



Methodology: Algorithm/Procedures

Planning:  Here we decide the topic of the project, what technology to use, collect research materials and formulate the action plan

Data Collection:  Here we collect the datasets from reliable and authentic sources and compile them.

Data Analysis:  Here we analyse all data collected regarding our research and provide a comprehensive review of the same.

Research:  Here we read and extract ideas from the various research materials collected to get in- depth knowledge about our project topic.

Design:  Here we will draw the outline of the project and the proposed model for the same.

Testing Algorithms: Here we test various machine learning algorithms and try to create a machine learning model with better precision percentages and promising results. 

## IMPLEMENTATION AND RESULTS

### 4.1. Modules/Classes of Implemented Project
The use of Convolutional Nueral Network was decided upon because:
CNN are very satisfactory at picking up on design in the input image, such as lines, gradients, circles, or even eyes and faces.
CNN contains many convolutional layers assembled on top of each other, each one competent of recognizing more sophisticated shapes.
The construction of a convolutional neural network is a multi-layered feed-forward neural network, made by assembling many unseen layers on top of each other in a particular order.

### 4.2. Implementation Detail
 
The first step was to get datasets and upload it on Dropbox. We got the dataset from Kaggle and Github of both Covid-19 positive and Covid-19 negative X-ray images. We then uploaded them to Dropbox and created a link to be used in the Google Colab notebook.

Unzip the file and store the file paths in variables. After that, start importing the libraries. For this project we imported the following libraries:

import numpy
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image

The model type that we will be using is Sequential. Sequential is the easiest way to build a model in Keras. It allows you to build a model layer by layer.

Our first 2 layers are Conv2D layers. These are convolution layers that will deal with our input images, which are seen as 2-dimensional matrices. In the first layer, there are 32 filters and the kernel size is 3X3. The activation used is “relu” and the input shape is 224 X 224 X 3.

After that we add another Conv2D layer with 64 filters and the activation is “relu”. We add a MaxPooling2D layer with the pool_size will be 2 X 2, and finally we add a Dropout layer to decrease the over fitting. 

We finally add our dense layers, starting with a flatten layer which  used to convert all the resultant 2-Dimensional arrays from pooled feature maps into a single long continuous linear vector. Then we add a Dense layer with activation “relu” and another Dropout layer.

Finally we add a Dense layer with 1 output neuron and the activation “sigmoid”. Then we compile the model where we choose loss as binary cross entropy and the optimizer adam.

Then we create a  Data Generator which will preprocess our images:
train_datagen= image.ImageDataGenerator(
    rescale= 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip= True,
)
 
test_dataset = image.ImageDataGenerator(rescale=1./255)

We apply this to both  test set and the validation set. and then finally train the CNN model in 10 epochs.

(i) Here we implemented 2 Layers: First we made a basic CNN model with 2 Conv2D layers, 1 MaxPooling2D layer, and 2 Dense layers, and 2 Dropout layers and achieved the accuracy of 90.62%.



(ii) To improve the accuracy of the model, we added 1 Conv2D layer, 1 MaxPooling2D and one Dropout layer. We achieved the accuracy of 92.50%




(iii)To further improve the accuracy, we added 1 Conv2D layer, 1 MaxPooling2D and one Dropout layer. We achieved the accuracy of 96.25% 




(iv) After adding 1 Conv2D layer, 1 MaxPooling2D and one Dropout layer we noticed that the accuracy dropped to 93.12%. 





### 4.3. Results and Discussion
	
The various accuracies and losses that we get using various number of filtering layers is as follows:

First we made a basic CNN model with 2 Conv2D layers, 1 MaxPooling2D layer, and 2 Dense layers, and 2 Dropout layers and achieved the accuracy of 90.62%.
		
To improve the accuracy of the model, we added 1 Conv2D layer, 1 MaxPooling2D and one Dropout layer. We achieved the accuracy of 92.50% 

To further improve the accuracy, we added 1 Conv2D layer, 1 MaxPooling2D and one Dropout layer. We achieved the accuracy of 96.25% 

After adding 1 Conv2D layer, 1 MaxPooling2D and one Dropout layer we noticed that the accuracy dropped to 93.12%. 

After adding 1 Conv2D layer, 1 MaxPooling2D and one Dropout layer we noticed that the accuracy dropped to 93.12%. 



## CONCLUSION AND FUTURE PLAN

We have concluded that a CNN model with best accuracy was the CNN model with 4 Conv2D layers, 3 MaxPooling2D layer, and 2 Dense layers, and 4 Dropout layers which gave the accuracy of 96.25%. Therefore this model predicts the certainty of being Covid-19 positive, decently. This method can therefore be used by physicians to help the world so that the diagnosis time is reduced by a huge margin, which in turn can save many lives.

## REFERENCES


Anam-Net: Anamorphic Depth Embedding-BasedLightweight CNN for Segmentation of Anomaliesin COVID-19 Chest CT ImagesNaveen Paluru, Aveen Dayal, Håvard Bjørke Jenssen, Tomas Sakinis,Linga Reddy Cenkeramaddi,Senior Member, IEEE,JayaPrakash,and Phaneendra K. Yalavarthy,Senior Member, IEEE
Classification of COVID-19 chest X-Ray and CT images using a type of Dynamic CNN modification Method Guangyu Jia, Hak-Keung Lam*, Yujia Xu
Fuzzy rank‐based fusion of CNN models using Gompertz function for screening COVID‐19 CT‐scans.Rohit Kundu, Hritam Basak, Pawan Kumar Singh, AliAhmadian, Massimiliano Ferrara & Ram Sarkar.
Learning distinctive filters for COVID-19 detection from chest X-ray using shuffled residual CNN. R. Karthik, R. Menaka, Hariharan M.
Diagnosis of COVID-19 from X-rays Using Combined CNN-RNN Architecture with Transfer Learning. Mabrook S. Al-Rakhami, Md. Milon Islam, Md. Zabirul Islam, Amanullah Asraf, Ali Hassan Sodhro, Weiping Ding.
A Light CNN for detecting COVID-19 from CT scans of the chest. Matteo Polsinelli, Luigi Cinque, Giuseppe Placidi.
CNN-based transfer learning– BiLSTM network: A novel approach for COVID-19 infection detection. Muhammet Fatih Aslan, Muhammed Fahri Unlersen, Kadir Sabanci, Akif Durdu.
Deep GRU-CNN model for COVID-19 detection from chest X-rays data PIR MASOOM SHAH, FAIZAN ULLAH, DILAWAR SHAH, ABDULLAH GANI (SENIOR MEMBER, IEEE), CARSTEN MAPLE (MEMBER, IEEE), YULIN WANG, SHAHID, MOHAMMAD ABRAR, SAIF UL ISLAM.
CovidXrayNet: Optimizing data augmentation and CNN hyperparameters for improved COVID-19 Detection from CXR Maram Mahmoud A. Monshia,b,* , Josiah Poon a, Vera Chung a, Fahad Mahmoud Monshi.
