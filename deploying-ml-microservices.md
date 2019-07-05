### Deploying Machine Learning to Production (Microservices)


### Main Technologies
- Docker
- Flask
- PySpark
- AWS
- Scikit-Learn
- Spark MLlib
- TensorFlow
- Keras



### Popular Methods of Deploying Machine Learning Models to Production

1. Deploy your machine learning model as a REST API using Docker and AWS services like ECR, Sagemaker and Lambda.

- We’ll then deploy the containerized model to ECR and create a machine learning endpoint in Sagemaker. Then we’ll finish off by creating the REST API endpoint. The model used in this post was made using Scikit Learn, but the approach detailed here will work with any ML framework in which an estimator’s or transformer’s state can be serialized, frozen or saved.
  
You will need to save (serialize, e.g. Joblib and Pickle) any estimators or transformers in addition to your model for the API. e.g. anything done to preprocess the data for the model.


1. Save the Model
2. Creating the Dockerfile




1. Save the Model

The first step in the deployment process will be to prepare and store your model such that it can be easily re-opened elsewhere. This can be achieved through serialization, which freezes the state of your trained classifier and saves it. To do this, we will be using Scikit Learn’s Joblib, a serialization library specifically optimized for storing large numpy array’s, and thus especially suited for Scikit Learn models. If you have more than one Scikit Learn estimator or transformer in your model (for example, a TFIDF preprocessor, like we have), you can save those using Joblib as well. The sentiment analysis model includes two components that we will have to save/freeze: the TFIDF text-preprocessor and the classifier. The following code dumps the classifier and tfidf vectorizer to the folder Model_Artifacts.

```python
from sklearn.externals import joblib
joblib.dump(classifier, 'Model_Artifacts/classifier.pkl')
joblib.dump(tfidf_vectorizer, 'Model_Artifacts/tfidf_vectorizer.pkl')
```

2. Creating the Dockerfile

Once the estimators and transformer are serialized, we can create a Docker image that holds our inference and server environment. Docker makes it possible to package your local environment and use it on any other server/environment/computer without having to worry about technical details. The Docker image will contain all the necessary components that will enable your model to perform predictions and communicate with the outside world. We can define a Docker image with a Dockerfile that specifies the contents of the environment: we’ll want to install Python 3, Nginx for the webserver and various Python packages like Scikit-Learn, Flask and Pandas.











