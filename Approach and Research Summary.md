# VQA_Medical


**Data Collection and Preprocessing**

1. Loading the Data set

2. Preprocess the image data:
   Resize images to a uniform size (224x224 pixels).
    Normalize pixel values.
    Augment dataset with techniques, rotation, flipping, and brightness adjustments to increase its diversity.

3. Preprocess the textual questions:
   Tokenize the questions into words.
   Convert words into word embeddings (Word2Vec) to represent text as numerical vectors.
   
4. Create a vocabulary for the question words.

**Data Splitting**

5. Split the question, labels, images into three subsets:
   Training set ( 70% of the data).
   Validation set (15% of the data).
   Test set ( 15% of the data).

**Image Feature Extraction**

6. Choose a pre-trained convolutional neural network (CNN) such as ResNet50. Download the weights and architecture.

7. Modify the pre-trained CNN by removing the classification layers and keep the layers responsible for feature extraction.

8. Extract image features from your preprocessed images using the modified CNN. These features will be used as the image input to your VQA model.

**Model Architecture**
Create the image input layer using the extracted image features using CNN.
Create the text input layer for the tokenized and embedded questions RNN.
Merge the image and text features.
Add one or more LSTM layers to process sequential information in the question.
Include fully connected layers to predict the answer.
     
*note: we can also use existing models such as BERT or GPT-2, but that didn't work in my case. so, made it from scratch**

**Loss Function and Training**

10. Define an appropriate loss function for your model (categorical cross-entropy).

11. Compile the model with the chosen optimizer (Adam) and loss function.

12. Train the model using the training dataset.

**Evaluation**

13. Evaluate the model using the test dataset with relevant evaluation metrics (accuracy)

**Fine-Tuning and Optimization**

14. The model's performance is not satisfactory, so we can consider fine-tuning the architecture, adjusting hyperparameters, or increasing data complexity.
