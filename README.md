# EE5111_IOT_Assignment
Implement a simple IoT pipeline with AWS Cloud platform and visualise the data.


1. All the certificates required are included in ProjectDemo.

2. "IOT_Patient1.ipynb" and "IOT_Patient2.ipynb" are used to send and publish data to IoT AWS.

3. "visualize_p1.py" and "visualize_p1.py" are used to visualize the ECG data pulled from DynamoDB.

4. "baseline_classifiers.py" builds basic machine learning classifiers, such as svm and decision tree.

5. "train_model.py" takes the entire ECG data set, and train the model.

6. In the folder named "Data", 
  a) "Myocardial_Infarction.csv" records the ECG signal from a patient who has myocardial infarction disease.
  b) "Healthy_Control.csv" records the ECG signal from a patient who do not have any heart disease.
  c) "Patient1.csv" and "Patient2.csv" are the queried data from DynamoDB which tie to a) and b).
  d) "ecg_mh.csv" and "label_mh.csv" are the randomly selected subset of the entire PTB ECG data (available at PhysioNet),
     which is used to train several baseline machine learning models.

7. Report is available at https://medium.com/@yaojia04/ee5111-special-topics-in-industrial-control-and-instrumentation-57032e45130f
