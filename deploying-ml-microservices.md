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
- Go

=> Train Models in Python, save/serialize models, load it into Go, containerize, and build the API (application). Voila!


-----------
1. Train the Model (in Python) using Keras, TensorFlow, PyTorch, Scikit-Learn or Spark MLlib
2. Build the API - Flask, Flask-RESTful, or Go
3. Test the API
4. Test webserver - Gunicorn web server
5. Load Balancer - Configure NGINX or AWS ELB, [Example: Serve Apps w/ Flask, Gunicorn, & NGINX](https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-16-04)
6. Load/Performance Testing (Recommended: [Locust](https://github.com/locustio/locust))


**Gunicorn**

```python
gunicorn --workers 1 --timeout 300 --bind 0.0.0.0:8000 api:app
- workers (INT): The number of worker processes for handling requests.
- timeout (INT): Workers silent for more than this many seconds are killed and restarted.
- bind (ADDRESS): The socket to bind. [['127.0.0.1:8000']]
- api: The main Python file containing the Flask application.
- app: An instance of the Flask class in the main Python file 'api.py'.
```

**Load Balancer**
You can configure `nginx` to handle all the test requests across all the `gunicorn` workers, where each worker has its own API with the DL model. Refer this resource to understand the setup between `nginx` and `gunicorn`.











### Popular Methods of Deploying Machine Learning Models to Production

1. Deploy your machine learning model as a REST API using Docker and AWS services like ECR, Sagemaker and Lambda.

- We’ll then deploy the containerized model to ECR and create a machine learning endpoint in Sagemaker. Then we’ll finish off by creating the REST API endpoint. The model used in this post was made using Scikit Learn, but the approach detailed here will work with any ML framework in which an estimator’s or transformer’s state can be serialized, frozen or saved.
  
You will need to save (serialize, e.g. Joblib and Pickle) any estimators or transformers in addition to your model for the API. e.g. anything done to preprocess the data for the model.


1. Build the Model. Pick your technology, it doesn't really matter. Python, Go, JavaScript, Java, Scala, etc. The main thing is being able to save the model in a serialized format that can be used anywhere.
2. Create Tests for your model
3. Save the Model in a serialized format

4. Creating the Dockerfile
5. Build and Test the Docker container
6. Figure out the smoke tests/capacity needed (which EC2, ECS, etc size or fleet you need to handle load)
7. Deploy to Production




1. Save the Model

The first step in the deployment process will be to prepare and store your model such that it can be easily re-opened elsewhere. This can be achieved through serialization, which freezes the state of your trained classifier and saves it. To do this, we will be using Scikit Learn’s Joblib, a serialization library specifically optimized for storing large numpy array’s, and thus especially suited for Scikit Learn models. If you have more than one Scikit Learn estimator or transformer in your model (for example, a TFIDF preprocessor, like we have), you can save those using Joblib as well. The sentiment analysis model includes two components that we will have to save/freeze: the TFIDF text-preprocessor and the classifier. The following code dumps the classifier and tfidf vectorizer to the folder Model_Artifacts.

```python
from sklearn.externals import joblib
joblib.dump(classifier, 'Model_Artifacts/classifier.pkl')
joblib.dump(tfidf_vectorizer, 'Model_Artifacts/tfidf_vectorizer.pkl')
```

2. Creating the Dockerfile

Once the estimators and transformer are serialized, we can create a Docker image that holds our inference and server environment. Docker makes it possible to package your local environment and use it on any other server/environment/computer without having to worry about technical details. The Docker image will contain all the necessary components that will enable your model to perform predictions and communicate with the outside world. We can define a Docker image with a Dockerfile that specifies the contents of the environment: we’ll want to install Python 3, Nginx for the webserver and various Python packages like Scikit-Learn, Flask and Pandas.









### Resources
- [A guide to deploying Machine/Deep Learning model(s) in Production](https://blog.usejournal.com/a-guide-to-deploying-machine-deep-learning-model-s-in-production-e497fd4b734a)
- [Exposing Python machine learning models using Flask, Docker and Azure](https://www.martinnorin.se/exposing-python-machine-learning-models-using-flask-docker-and-azure/)
