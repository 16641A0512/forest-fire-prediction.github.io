
# coding: utf-8

# # decision tree

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:



import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_5c1e5696391d415098604f7c4c4b7101 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='GbwOV6LFyOf16XYHdkAr3LrGUdmXIP-DWI_JU7Q5QjsH',
    ibm_auth_endpoint="https://iam.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.eu-geo.objectstorage.service.networklayer.com')

body = client_5c1e5696391d415098604f7c4c4b7101.get_object(Bucket='forestfire-donotdelete-pr-kjk2hq5xkicvae',Key='forestfires.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

dataset = pd.read_csv(body)
dataset.head()



# In[3]:


dataset.drop(['X'],axis=1,inplace=True)


# In[4]:


dataset.drop(['Y'],axis=1,inplace=True)


# In[5]:


dataset.drop(['month'],axis=1,inplace=True)


# In[6]:


dataset.drop(['day'],axis=1,inplace=True)


# In[7]:


dataset


# In[8]:


type(dataset)


# In[9]:


dataset.isnull().any()


# In[10]:


x=dataset.iloc[:,0:8].values
x


# In[11]:


y=dataset.iloc[:,8:].values
y


# In[12]:


x.shape


# In[13]:


x.shape


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_test


# In[15]:


x_test.shape


# In[16]:


x_train.shape


# In[17]:


y_test.shape


# In[18]:


y_train.shape


# In[19]:


plt.scatter(x_train[:,4],y_train)


# In[20]:


from sklearn.linear_model import LinearRegression
mr=LinearRegression()


# In[21]:


mr.fit(x_train,y_train)


# In[22]:


x_train[0]


# In[23]:


plt.scatter(x_train[:,2],y_train,color='green')


# In[24]:


y_predict=mr.predict(x_test)


# In[25]:


y_predict


# In[26]:


mr.predict([[86.2,26.2,94.3,5.1,8.2,51,6.7,0.0]])


# In[27]:



body = client_5c1e5696391d415098604f7c4c4b7101.get_object(Bucket='forestfire-donotdelete-pr-kjk2hq5xkicvae',Key='forestfires.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_1 = pd.read_csv(body)
df_data_1.head()

from sklearn.metrics import r2_score
r2_score(y_test,y_predict)


# In[28]:


get_ipython().system('pip install watson-machine-learning-client --upgrade')


# In[29]:


from watson_machine_learning_client import WatsonMachineLearningAPIClient


# In[30]:


wml_credentials={"instance_id": "903db30d-ab06-40e6-929e-4e5337b729ce",
  "password": "e3b3c3af-44c0-40e2-961f-b8617cb54e62",
  "url": "https://eu-gb.ml.cloud.ibm.com",
  "username": "2245893e-e508-40b4-9891-aaff2a2a7b4f"
}


# In[31]:


client=WatsonMachineLearningAPIClient(wml_credentials)


# In[32]:


import json
instance_details=client.service_instance.get_details()
print(json.dumps(instance_details,indent=2))


# In[33]:


models_props={client.repository.ModelMetaNames.AUTHOR_NAME:"siddikha",
             client.repository.ModelMetaNames.AUTHOR_EMAIL:"siddikha208@gmail.com",
             client.repository.ModelMetaNames.NAME:"DesicionTree"}


# In[39]:


model_artifact=client.repository.store_model(mr,meta_props=models_props)


# In[40]:


published_model_uid=client.repository.get_model_uid(model_artifact)


# In[41]:


published_model_uid


# In[42]:


created_deployment=client.deployments.create(published_model_uid,name="DecisionTree")


# In[43]:


scoring_endpoint=client.deployments.get_scoring_url(created_deployment)
scoring_endpoint


# In[44]:


client.deployments.list()

