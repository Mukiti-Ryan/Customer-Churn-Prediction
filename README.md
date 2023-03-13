# Customer-Churn-Prediction
A Machine Learning Model Trained to Predict Customer Churn Built Using Python and Streamlit

This is the initial stage for the dashboard.

After running the command `streamlit run stream_app.py`, this is the first page that will be displayed in the browser.
<img width="1391" alt="Home_page" src="https://user-images.githubusercontent.com/87067381/224629245-5a818a21-df1a-48f9-a375-5770d7334093.png">

The process of a customer approaching a teller for a service is what I will refer to as a Single Customer. The teller should be able to search the account details and have an overview of the churn status of the customer.
<img width="1391" alt="prediction" src="https://user-images.githubusercontent.com/87067381/224629317-6b19a681-fd83-471d-9f00-196a69d7ce27.png">

Since a branch handles several customers, a batch prediction of the churn status can be done by uploading a CSV File with the format indicated above the Upload Button. This CSV file can then be downloaded with additional columns indicating the churn status and the risk score.
<img width="1391" alt="batch_prediction" src="https://user-images.githubusercontent.com/87067381/224629351-d7cb8348-2c3a-47d1-bd92-da18fe4d89f8.png">
