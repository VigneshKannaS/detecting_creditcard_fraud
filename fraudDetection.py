import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
import smtplib
import socket
import re
import os
import time
import csv
class InputError(Exception):
    pass
'''def simple_mail_transaction(email):
    sender_add = XXXX@gmail.com  # storing the sender's mail id
    #give your mailID for sending alert message to the end user (note: sender's mailID account must be configured to allow 'less secured appa')
    receiver_add = email  # storing the receiver's mail id
    password = XXXX XXXX XXXX XXXX  # storing the password to log in

    try:
        # creating the SMTP server object by giving SMPT server address and port number
        smtp_server = smtplib.SMTP("smtp.gmail.com", 587)
        smtp_server.ehlo()  # setting the ESMTP protocol
        smtp_server.starttls()  # setting up to TLS connection
        smtp_server.ehlo()  # calling the ehlo() again as encryption happens on calling startttls()
        smtp_server.login(sender_add, password)  # logging into out email id
        SUBJECT = 'Urgent: Alert Regarding Fraudulent Transaction Detected'
        
        msg_to_be_sent = ''' '''
Hello {},

We hope this message finds you well. We are reaching out to inform you of a concerning matter regarding your account.

Our system has detected a potentially fraudulent transaction associated with your account.

The transaction in question appears to be inconsistent with your typical spending patterns and may indicate fraudulent activity. Consult with your Card Issuer to investigate the matter and to ensure the safety and security of your account.

Please review your recent transactions and verify any unfamiliar activity. If you recognize the transaction in question or suspect any other unauthorized activity on your account, please contact your Card Issuer immediately or financial experts for further assistance.

Additionally, we recommend taking the following steps to safeguard your account:

. Change your account password immediately, ensuring it is strong and unique.
. Enable two-factor authentication for an added layer of security.
. Regularly monitor your account activity and report any suspicious transactions promptly.
. We apologize for any inconvenience this may cause and appreciate your cooperation in resolving this matter swiftly. Our priority is to ensure the integrity and security of your account.

Thank you for your attention to this matter.

Sincerely,
CreditCardFraud Detection System ''' '''.format(email[0:email.rindex('@')])
        
        msg_to_be_sent = 'Subject: {}\n\n{}'.format(SUBJECT, msg_to_be_sent)
        # sending the mail by specifying the from and to address and the message
        smtp_server.sendmail(sender_add, receiver_add, msg_to_be_sent)
        st.success('Alert message is sent by the detection system')  # priting a message on sending the mail
        smtp_server.quit() #terminating the Server
    except smtplib.SMTPException as e:
        st.error("Something went wrong while sending the Alert Message...")
        st.error(e)  # Printing the error message
    except socket.gaierror as e:
        st.error("Error: unable to resolve hostname to IP address")
        st.error(e)  # Printing the error message
        st.warning("This may occur while you're in offline") '''

def targetVariableSpecification(d,choice,numberOfCol):
    legitfraud = st.radio(
        "for target Variable, 0 specifies",
        ["fraud", "legitimate"],
        index=None,
    )
    if legitfraud == "fraud":
        fraudulent = 0
        legitimate = 1
    else:
        fraudulent = 1
        legitimate = 0
    fraudDetection(d,choice,numberOfCol,fraudulent,legitimate)

def fraudDetection(d,choice,numberOfCol,fraudulent,legitimate):
    data =d

    # separate legitimate and fraudulent transactions
    legit = data[data.Class == legitimate]
    fraud = data[data.Class == fraudulent]
    # undersample legitimate transactions to balance the classes
    if choice=="Detection with Standard Source":
        legit_sample = legit.sample(n=len(fraud), random_state=2)
        data = pd.concat([legit_sample, fraud], axis=0)
    elif choice=="Detection with User's Source":
        st.warning("Before advancing to the prediction phase, it is essential to complete all necessary preprocessing methods.")
    else:
        st.error("something went wrong...")
    # split data into training and testing sets
    X = data.drop(columns="Class", axis=1)
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

    # train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # evaluate model performance

    train_acc = accuracy_score(model.predict(X_train), y_train) # if there's any necessity to display accuracy, it would be helpful
    test_acc = accuracy_score(model.predict(X_test), y_test)

    # web app
    st.title("Credit Card Fraud Detection Model")
    condition_mailid = "^[a-z]+[\._]?[a-z 0-9]+[@]\w+[.]\w{2,3}$"
    email = st.text_input('enter your email address:')
    if re.search(condition_mailid, email):
        try:
            input_df = st.text_input('Enter all input features values[{}]:'.format(numberOfCol))
            input_df_splited = input_df.split(',')
            detect = st.button("detect")
            if detect:
                if len(input_df_splited) != numberOfCol:
                    raise InputError
                features = np.asarray(input_df_splited, dtype=np.float64)
                prediction = model.predict(features.reshape(1, -1))

                if prediction[0] == legitimate:
                    st.subheader("legitimate transaction",anchor=False)
                else:
                    st.subheader("fraudulent transaction",anchor=False)
                    simple_mail_transaction(email)
        except InputError:
            st.warning("this was expected to have {} features, yet it has {} features...".format(numberOfCol,len(input_df_splited)))
        except ValueError:
            st.error("Invalid Input type...")

    elif (email):
        st.error("Invalid Mail Address...")


st.set_page_config(page_title="CreditCard Fraud Detection Model",page_icon=":credit_card:",layout="wide",initial_sidebar_state="expanded")

page_bg_img="""
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://miro.medium.com/max/8002/1*yt0h233ql_VWlSvMI6vJYA.jpeg");
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True)

dataset=["Detection with Standard Source","Detection with User's Source"]
choice=st.sidebar.selectbox("Dataset",dataset)
if choice=="Detection with Standard Source":
    data ='builtInSource/creditcard.csv'
    with open(data) as f:
        dataCol= list(csv.reader(f))
    numberOfCol = len(dataCol[0])-1
    d=pd.read_csv(data)
    legitimate = 0
    fraudulent = 1
    fraudDetection(d,choice,numberOfCol,fraudulent,legitimate)

if choice == "Detection with User's Source":
    try:
        st.subheader("Post Your DataSet")  # user's dataset should have 'Class' variable as its target variable
        st.subheader("Please upload a dataset containing relevant datafields about transactions, with the target variable specified as 'Class'")
        data_file = st.file_uploader("upload csv", type=["csv"])
        if data_file is not None:
            with open(os.path.join("uploadedSource", data_file.name), 'wb') as f:
                f.write(data_file.getbuffer())

            clearData=st.sidebar.button("discard source")
            if clearData:
                os.remove("uploadedSource/"+data_file.name)
                st.success(data_file.name+" has been successfully removed")
                st.stop()

            with st.spinner("loading csv..."):
                time.sleep(3)
            st.success("file saved")
            view_data = st.button("view source")
            with open(os.path.join("uploadedSource", data_file.name), 'r') as f:
                dataCol = list(csv.reader(f))
            numberOfCol = len(dataCol[0]) - 1
            d = pd.read_csv(data_file)
            if view_data:
                st.dataframe(d)
            targetVariableSpecification(d, choice, numberOfCol)
    except AttributeError:
        st.error("There is no target variable with name 'Class'")
hide_st_style="""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style,unsafe_allow_html=True)
