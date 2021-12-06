# Named-Entity-Recogniton-using-Perceptrons-and-Word-Embeddings

Exploration and  analysis of   word similarities and analogies using word embeddings for  binary  and  multi-labelnamed entity disambiguation. 
A perceptron classifier is used for binary and six-way classification and then we build a baseline measure towhich  the  perceptron  is  compared. 

# Data

The  data  used  is  a  corpus  of  named  entitie sand  their  labels.   Each  entry  in  the  dataset  consists of a named entity and a tag that specifies the meaning of the word.  There are in total six tags that an entity can be labeled as:GPE(geo-politicalentity),LOC(location),PERSON,ORG(organiza-tion),DATE, andCARDINAL(cardinal number).

The classification is performed as follows :  In the case of binary classification the labels GPE,LOCwill be considered as LOCATION while la-bels PERSON,ORG,DATE,CARDINAL will  be considered  as NON-LOCATION. 
The dataset is not provided as the project was part of a Master's program, hence restrictions apply. 

