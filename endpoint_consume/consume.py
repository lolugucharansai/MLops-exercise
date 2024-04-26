import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

def predict_diabetes(pregnancies, plasma_glucose, diastolic_blood_pressure, triceps_thickness, serum_insulin, bmi, diabetes_pedigree, age):
    data =  {
      "input_data": [
        [
          pregnancies,
          plasma_glucose,
          diastolic_blood_pressure,
          triceps_thickness,
          serum_insulin,
          bmi,
          diabetes_pedigree,
          age
        ]
      ],
      "params": {}
    }

    body = str.encode(json.dumps(data))
    url = 'https://aml-diabetes-dev-lmtrj.eastus.inference.ml.azure.com/score'

    api_key = ''  # Replace this with the primary/secondary key or AMLToken for the endpoint
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'diabetes-model-17' }

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        return json.loads(result)  # Assuming the result is in JSON format
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))
        return None