import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64

from PIL import Image

model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

def main():

    image = Image.open('images/icon.png')
    image2 = Image.open('images/image.png')
    st.image(image, use_column_width = True)
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.info('The aim is to assist branches in identifying customers who are likely to churn, and aim at keeping them with the organization.')
    st.sidebar.image(image2)
    st.title("Predicting Customer Churn")
    numerical_cols = ['DAYS_ACC_OPEN', 'CD_TYPE', 'VL_CREDIT_RECENCY', 'VL_DEBIT_RECENCY', 'VL_TENOR', 'AVGCREDITTURNOVER_LY',
    'AVGDEBITTURNOVER_LY', 'AVGCREDIT_TRANS_LY', 'AVGDEBIT_TRANS_LY', 'AVGCREDITTURNOVER_LY2', 'AVGDEBITTURNOVER_LY2', 
    'AVGCREDIT_TRANS_LY2', 'AVGDEBIT_TRANS_LY2']

    if add_selectbox == 'Online':
        days_acc_open = st.number_input('Days Since The Account Was Opened: ', min_value = 0, max_value = None, value = 0)
        cd_type = st.selectbox('Account type: ', [1062, 1063])
        vl_credit_recency = st.number_input('VL Credit Recency: ', min_value = None, max_value = None, value = 0)
        vl_debit_recency = st.number_input('VL Debit Recency: ', min_value = None, max_value = None, value = 0)
        vl_tenor = st.number_input('VL Tenor: ', min_value = None, max_value = None, value = 0)
        avgcreditturnover_ly = st.number_input('Avg Credit Turnover LY: ', min_value = None, max_value = None, value = 0)
        avgdebitturnover_ly = st.number_input('Avg Debit Turnover LY: ', min_value = None, max_value = None, value = 0)
        avgcredit_trans_ly = st.number_input('Avg Credit Trans LY: ', min_value = None, max_value = None, value = 0)
        avgdebit_trans_ly = st.number_input('Avg Debit Trans LY: ', min_value = None, max_value = None, value = 0)
        avgcreditturnover_ly2 = st.number_input('Avg Credit Turnover LY2: ', min_value = None, max_value = None, value = 0)
        avgdebitturnover_ly2 = st.number_input('Avg Debit Turnonver LY2: ', min_value = None, max_value = None, value = 0)
        avgcredit_trans_ly2 = st.number_input('Avg Credit Trans LY2: ', min_value = None, max_value = None, value = 0)
        avgdebit_trans_ly2 = st.number_input('Avg Debit Trans LY2: ', min_value = None, max_value = None, value = 0)
        output = ""
        output_prob = ""
        input_dict = {
            'days_acc_open'          : days_acc_open,
            'cd_type'                : cd_type,
            'vl_credit_recency'      : vl_credit_recency,
            'vl_debit_recency'       : vl_debit_recency,
            'vl_tenor'               : vl_tenor,
            'avgcreditturnover_ly'   : avgcreditturnover_ly,
            'avgdebitturnover_ly'    : avgdebitturnover_ly,
            'avgcredit_trans_ly'     : avgcredit_trans_ly,
            'avgdebit_trans_ly'      : avgdebit_trans_ly,
            'avgcreditturnover_ly2'  : avgcreditturnover_ly2,
            'avgdebitturnover_ly2'   : avgdebitturnover_ly2,
            'avgcredit_trans_ly2'    : avgcredit_trans_ly2,
            'avgdebit_trans_ly2'     : avgdebit_trans_ly2
        }

        if st.button("Predict"):
            numerical_cols = ['days_acc_open', 'cd_type', 'vl_credit_recency', 'vl_debit_recency', 'vl_tenor', 'avgcreditturnover_ly', 
            'avgdebitturnover_ly', 'avgcredit_trans_ly', 'avgdebit_trans_ly', 'avgcreditturnover_ly2', 'avgdebitturnover_ly2', 
            'avgcredit_trans_ly2', 'avgdebit_trans_ly2']
            numerical_data = [input_dict[col] for col in numerical_cols]
            X = np.array(numerical_data).reshape(1, -1)
            y_pred = model.predict_proba(X)[0, 1]
            churn = y_pred >= 0.5
            output_prob = float(y_pred)
            output = bool(churn)
        st.success('Churn Status: {0}, Risk Score: {1}'.format(output, output_prob))

    if add_selectbox == 'Batch':
        st.write("Kindly Follow The Instructions Below For The Best Experience While Using The Application")
        st.write("The Required Format For The CSV File To Be Uploaded is Stated Below: ")
        table_data = np.random.rand(10, 13)
        table_df = pd.DataFrame(table_data, columns = ['days_acc_open', 'cd_type', 'vl_credit_recency', 'vl_debit_recency',
        'vl_tenor', 'avgcreditturnover_ly', 'avgdebitturnover_ly', 'avgcredit_trans_ly', 'avgdebit_trans_ly', 
        'avgcreditturnover_ly2', 'avgdebitturnover_ly2', 'avgcredit_trans_ly2', 'avgdebit_trans_ly2'])
        st.table(table_df)
        second_selectbox = st.selectbox("How would you like the results presented?", ("Pie-Chart", "Bar-Graph"))
        file_upload = st.file_uploader("Upload CSV File for Predictions", type = ["csv"])

        if file_upload is not None:
            df = pd.read_csv(file_upload)
            numerical_data = df[numerical_cols].values
            X = numerical_data
            y_pred = model.predict_proba(X)[:, 1]
            churn = y_pred >= 0.5
            churn = [bool(c) for c in churn]

            # Since we can visualize the results in different ways, let's give the user options
            # Our options will be to view the results as a table, pie-chart, or bar graph

            if second_selectbox == 'Pie-Chart':
                with st.spinner('Generating Pie-Chart...'):
                    fig, ax = plt.subplots(figsize = (12, 10))
                    ax.pie([sum(churn), len(churn) - sum(churn)], labels = ['Likely Churn', 'Unlikely Churn'])
                    st.pyplot(fig)

            if second_selectbox == 'Bar-Graph':
                with st.spinner('Generating Bar-Graph...'):
                    fig, ax = plt.subplots()
                    sns.countplot(x = churn, ax = ax)
                    st.pyplot(fig)
            
            # The results will be added to the uploaded dataset and returned to the user as a downloadable CSV file

            # Let's add a download button for users to download a CSV file
            st.info("Download CSV File With Churn Prediction")

            # Create a new DataFrame with the prediction results
            prediction_df = pd.DataFrame({'Prediction': y_pred, 'Churn': churn})
            final_df = pd.DataFrame(prediction_df, columns = ["Prediction", "Churn"])

            # Merge the prediction DataFrame with the original DataFrame
            df = pd.concat([df, final_df], axis = 1)

            # Download the new DataFrame as a CSV file
            csv = df.to_csv(index = False)
            st.download_button(label = "Download CSV", data = csv, file_name = "prediction_results.csv", mime = "text/csv")

if __name__ == '__main__':
    main()