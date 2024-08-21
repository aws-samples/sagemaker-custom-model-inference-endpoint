# sagemaker-endpoints-blog


## Description
This repository will walk you through the end-to-end process of deploying a single custom model on SageMaker using the Prithvi model, a temporal Vision transformer developed by IBM and NASA and pre-trained on the Harmonized Landsat Sentinel-2 data collection. The Prithvi model, with its unique dependencies and architecture, is an effective example of how to deploy complex custom models to SageMaker.

## Prerequisites

You need the following prerequisites before you can proceed. For this post, we use the us-east-1 (Northern Virginia) Region:

1. Have access to a POSIX based (Mac/Linux) system or SageMaker Notebooks
2. Ensure you have IAM permissions for SageMaker access, S3 bucket create, read, and putobject access, CodeBuild access Amazon Elastic Container Registry (ECR) repository access, and the ability to create IAM Roles
3. Download [Prithvi model artifacts](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/tree/main) files and [Burn Scar finetuning](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M-burn-scar) files

## Solution Overview
To run a custom model that needs unique packages as an Amazon SageMaker Endpoint you will need to follow these steps:

1. If your model requires additional packages or package versions unavailable from SageMaker’s managed container images you will need to extend one of the container images.
2. Write a python model definition using SageMaker’s inference.py file format. 
3. Define your model artifacts and inference file within a specific file structure, archive your model files as a tar.gz, and upload your files to Amazon Simple Storage Service (S3)
4. With your model code and an extended SageMaker container you will use SageMaker Studio to create a model, endpoint configuration, and endpoint. 
5. Call the inference endpoint to ensure your model is running correctly

## Procedure
1. Extending a SageMaker container image for your model

2. Write a python model definition using SageMaker’s inference.py file format. 

3. Define your model artifacts and inference file within a specific file structure, archive your model files as a tar.gz, and upload your files to Amazon Simple Storage Service (S3)

4.  With your model code and an extended SageMaker container you will use SageMaker Studio to create a model, endpoint configuration, and endpoint. 

5. Call the inference endpoint to ensure your model is running correctly

## Querying the Endpoint
```
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer

payload = "https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-100M-demo/resolve/main/HLS.L30.T13REN.2018013T172747.v2.0.B02.B03.B04.B05.B06.B07_cropped.tif"

predictor = Predictor(endpoint_name=[your endpoint name])
predictor.serializer = NumpySerializer()

predictions = predictor.predict(payload)
```

## Cleaning Up Resources

To clean up the resources from this blog and avoid incurring costs follow these steps:

1. Delete the SageMaker endpoint, endpoint configuration, and model.
2. Delete the ECR image and repository.
3. Delete the model.tar.gz in the S3 bucket that was created.
4. Delete the S3 bucket.


## Support
For any questions reach out to riaidan@amazon.com

## Authors and acknowledgment
Thank to the whole team (Aidan Ricci, Charlotte Fondren, Nate Haynes)

## License
MIT No Attribution

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.aws.dev/wwps_slg_edu_cop_highered_data/sagemaker-endpoints-blog.git
git branch -M main
git push -uf origin main
```