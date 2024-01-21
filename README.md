# AmericanSignLanguage-CNN

This project is about making a model for American Sign Language (ASL) that recognises hand gestures into letters.

![image](https://github.com/Samhita-kolluri/AmericanSignLanguage-CNN/assets/65637090/d2bd6c40-4f2a-4fd7-b840-ecfc6772a3a1)

The model uses MNIST Dataset from Kaggle, which has about 35,000 images in its dataset. Dataset is split in Train, Validate and Test sets at the ratio of 70:20:10. The training set is further put to Data Augmentation to almost double, or even triple the inputs. Remember to not flip or rotate images as they might lead to misclassification. The data for each letter can be viualised in the below graph.

![image](https://github.com/Samhita-kolluri/AmericanSignLanguage-CNN/assets/65637090/cf1fa05d-44dd-4736-884b-6ed1e7122c52)

The model uses 3 Convolutional Neural Networks and a Dense, Flatten and Dropout layer to form a model. Model is fitted on a batch size of 128 at a learning rate of 0.0001. The output model is saved for further use. You can find the learning curve of the model below.

The loss curve of the model is given below.

![image](https://github.com/Samhita-kolluri/AmericanSignLanguage-CNN/assets/65637090/155b3a92-e31e-4735-90a3-065776f4ecbc)

The saved model for 60 epochs in the Models is evaluated on test and validation set and gives the accuracy of 99.91% and 98.56%, which is better than most state-of-the-art systems. With a python script, a test image can be clicked via webcam and pre-processed and its alphabet can be predicted. A script can be written for live webcam sign language recognition. This will help visualise the project better. Below, the confusion matrix of the predictions show how accurate the trained model is.

![image](https://github.com/Samhita-kolluri/AmericanSignLanguage-CNN/assets/65637090/2f7ba694-fd9e-4a9c-a530-d92f3f8b2bc9)

For future work, NLP can be introduced so the letters can be used to form meaningful words or even sentences. Grammar check can also be added, and this will give sentences as output, instead of single letters.
