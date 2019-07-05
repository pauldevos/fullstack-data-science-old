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
  
