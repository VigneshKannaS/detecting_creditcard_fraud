import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
from streamlit_lottie import st_lottie
import smtplib
import re
import os
import plotly.express as px
import json

class InputError(Exception):
    pass
def simple_mail_transaction(email):
    sender_add = XXXX  # storing the sender's mail id
    receiver_add = email  # storing the receiver's mail id
    password = XXXX  # storing the password to log in
    # creating the SMTP server object by giving SMPT server address and port number
    smtp_server = smtplib.SMTP("smtp.gmail.com", 587)
    smtp_server.ehlo()  # setting the ESMTP protocol
    smtp_server.starttls()  # setting up to TLS connection
    smtp_server.ehlo()  # calling the ehlo() again as encryption happens on calling startttls()
    smtp_server.login(sender_add, password)  # logging into out email id
    SUBJECT = 'Status of your Current transaction'
    msg_to_be_sent = '''ALERT!

                        . Your Current transaction is seems to be fraudalent.
                        . Submitted Credited Card features is predicted to be a fraudalent by this model with it's 95% accuracy.
                        . Kindly Provide all the necessary documents asked by your respective Bank. Generally they ask for copy of front of credit card and FIR copy. Bank will now investigate the transaction and in meantime they will immediately restore the credit limit of your card for fraudulent transaction.After completing their investigation, bank will let you know the outcome.

                        . For further Informations kindly refer below link
                        https://www.financialexpress.com/money/credit-card-fraud-rbi-steps-in-to-protect-customers-but-here-is-what-you-must-do-to-avoid-losses/753759/

                        <Thankyou for Using 21BCM055 S.Vignesh Kanna's Credit Card Fraud detection Model>
                        '''
    msg_to_be_sent = 'Subject: {}\n\n{}'.format(SUBJECT, msg_to_be_sent)
    # sending the mail by specifying the from and to address and the message
    smtp_server.sendmail(sender_add, receiver_add, msg_to_be_sent)
    print('Successfully the mail is sent')  # priting a message on sending the mail
    smtp_server.quit()  # terminating the server

def fraud_detection(d):
    data =d
    # separate legitimate and fraudulent transactions
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]

    # undersample legitimate transactions to balance the classes
    legit_sample = legit.sample(n=len(fraud), random_state=2)
    data = pd.concat([legit_sample, fraud], axis=0)

    # split data into training and testing sets
    X = data.drop(columns="Class", axis=1)
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

    # train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    # evaluate model performance
    train_acc = accuracy_score(model.predict(X_train), y_train)
    test_acc = accuracy_score(model.predict(X_test), y_test)

    # web app
    st.title("Credit Card Fraud Detection Model")
    condition_mailid = "^[a-z]+[\._]?[a-z 0-9]+[@]\w+[.]\w{2,3}$"
    email = st.text_input('enter your email is:')
    if re.search(condition_mailid, email):
        try:
            input_df = st.text_input('Enter all input features values[30]:')
            input_df_splited = input_df.split(',')
            submit = st.button("Submit")
            if submit:
                if len(input_df_splited)!=30:
                    raise InputError
                features = np.asarray(input_df_splited, dtype=np.float64)
                prediction = model.predict(features.reshape(1, -1))

                if prediction[0] == 0:
                    st.write("legitimate transaction")
                else:
                    st.write("fraudulent transaction")
                    simple_mail_transaction(email)
        except InputError:
            st.write("this must be of 30 features...  but it has {} features".format(len(input_df_splited)))
        except ValueError:
            st.write("Invalid Input type...")

    elif (email):
        st.text("Invalid Mail Address...")


pip_img="""
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://t3.ftcdn.net/jpg/04/05/42/40/360_F_405424078_WC4B7won1NJjfzW1ALW19tX1xf9WKWmg.jpg");
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True)

dataset=["Standard detection","Customised detection"]
choice=st.sidebar.selectbox("Dataset",dataset)
if choice=="Standard detection":
    data ='creditcard.csv'
    d=pd.read_csv(data)
    fraud_detection(d)
    # separate legitimate and fraudulent transactions

if choice == "Customised detection":
    st.subheader("Post Your DataSet")
    st.subheader("upload a dataset which should contains the relevant data fields,about the transactions")
    data_file = st.file_uploader("upload csv", type=["csv"])
    if data_file is not None:
        with open(os.path.join("tempDir",data_file.name),'wb') as f:
            f.write(data_file.getbuffer())
        def load_lottiefile(filepath: str):
            with open(filepath, "r") as f:
                return json.load(f)
        lottie_coding = load_lottiefile('''lottie file path''')
        st_lottie(lottie_coding,speed=1,reverse=False,loop=False,quality="high",height=100,width=100,key=None,)
        st.success("file saved")
        view_data=st.button("view data")
        d=pd.read_csv(data_file)
        if view_data:
            st.dataframe(d)
        st.write("Box plot Visualisation to examine Outliers")
        select = (d.columns)
        choices = st.sidebar.selectbox("Box Plot", select)
        d[choices].plot(kind='box', title=choices)
        st.write(plt.gcf())
        fraud_detection(d)

hide_st_style="""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style,unsafe_allow_html=True)
