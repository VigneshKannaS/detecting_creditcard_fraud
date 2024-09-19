import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")

class inputCount(Exception):
    def __init__(self, expected_length, actual_length):
        self.expected_length = expected_length
        self.actual_length = actual_length
        self.message = f"this was expected to have {expected_length} features, yet it has {actual_length} features"
        super().__init__(self.message)

# Standardized the data by applying the Z-score formula.
def ZScoreTransformation(inputData):
    if len(inputData) != 30:
        raise ValueError("Input data must contain exactly 30 features.")
    
    meanValues = [0, 1.16837498e-15, 3.41690805e-16, -1.37953671e-15, 2.07409512e-15, 9.60406632e-16, 1.48731301e-15, -5.55646730e-16, 1.21348136e-16, -2.40633055e-15, 2.23905274e-15, 1.67332693e-15, -1.24701177e-15, 8.19000127e-16,\
                  1.20729421e-15, 4.88745586e-15, 1.43771595e-15, -3.77217069e-16, 9.56414917e-16, 1.03991661e-15, 6.40620363e-16, 1.65406691e-16, -3.56859322e-16, 2.57864790e-16, 4.47326553e-15, 5.34091469e-16, 1.68343720e-15, -3.66009081e-16, -1.22739000e-16, 8.83496193e+01]
    standardDeviationValues = [1, 1.95869237, 1.65130568, 1.51625234, 1.41586609, 1.38024431, 1.33226875, 1.23709143, 1.19435081, 1.09863016, 1.08884785, 1.02071124, 0.99919964, 0.99527248, 0.95859393, 0.9153144, 0.87625135, 0.84933557,\
                                0.83817474, 0.81403907, 0.77092367, 0.73452272, 0.72570029, 0.6244592, 0.605646, 0.52127716, 0.48222617, 0.40363179, 0.33008268, 250.11967014]
    
    standardizedInputData = []
    for i in range(30): 
        value = (inputData[i] - meanValues[i]) / standardDeviationValues[i] # Z = (X - μ) / σ
        standardizedInputData.append(value)
    
    standardizedInputData = np.asarray(standardizedInputData, dtype=np.float64).reshape(1, -1)
    
    return standardizedInputData

def fraudDetection(model, standardizedInputData):
    # Load the specified model from the 'Models/' directory using pickle
    with open('Models/' + model, 'rb') as f:
        loadedModel = pickle.load(f)
    
    # Check if the input data has the expected number of features (30)
    if standardizedInputData.shape[1] != 30:
        raise ValueError(f"Expected 30 features, but got {standardizedInputData.shape[1]} features.")
    
    # Predict the transaction type using the loaded model
    transactionType = loadedModel.predict(standardizedInputData)
    
    if model != "modelSVM.pkl":
        probabilities = loadedModel.predict_proba(standardizedInputData) # Predict probabilities of each class (assuming a binary classification problem)
        likelihood = round(probabilities[0][0],3)*100 if probabilities[0][0] >= probabilities[0][1] else round(probabilities[0][1],3)*100 if probabilities[0][0] < probabilities[0][1] else 0
        unlikelihood = round(probabilities[0][1],3)*100 if probabilities[0][0] >= probabilities[0][1] else round(probabilities[0][0],3)*100 if probabilities[0][0] < probabilities[0][1] else 0
        
        return transactionType, likelihood, unlikelihood
    
    else:
        # SVM output class labels based on decision boundaries, not probability scores.
        return transactionType, 0, 0

def insightsAndVisualization():
    data = { "LR Model 1": [0.952381, 0.909091, 0.090909, 0.187476, 0.954315, 0.253808, 0.943075],
        "LR Model 2": [0.977741, 0.956452, 0.043548, 0.068526, 0.977947, 65.186749, 0.977993],
        "LR Model 3": [0.925924, 0.862066, 0.137934, 0.238982, 0.923758, 30.527738, 0.918871],
        "SVM Model": [0.892655, 0.806122, 0.193878, float('nan'), 0.903553, 55.588042, 0.894313],
        "DecisionTree Model 1": [0.931937, 0.872549, 0.127451, 0.872420, 0.934010, 0.187401, 0.912607],
        "DecisionTree Model 2": [0.971491, 0.944563, 0.055437, 0.079766, 0.971713, 51.568454, 0.971097],
        "Random Forest Model 1": [0.946809, 0.898990, 0.101010, 0.145445, 0.949239, 1.248132, 0.938004],
        "Random Forest Model 2": [0.999921, 0.999842, 0.000158, 0.002642, 0.999921, 2135.215765, 0.999873]
    } # dictionary containing performance metrics for different classification models
    
    compareDf = pd.DataFrame(data, index=["F1 Score", "Jaccard Index", "Jaccard Distance", "Log Loss", "Accuracy Score", "Time Taken", "K-Fold CV"])
    st.dataframe(compareDf)

    ###########################################################################################################################################################33
    # Plot1 : Accuracy and KFold Cross Validation Score

    fig1, ax1 = plt.subplots(figsize=(6, 2.5))
    barwidth = 0.38
    modelLists = compareDf.columns
    x = [i for i in range(len(modelLists))]

    ax1.bar([p - barwidth/2 for p in x], compareDf.loc['Accuracy Score',], width=barwidth, label='Accuracy Score', color="#6B8E23")
    ax1.bar([p + barwidth/2 for p in x], compareDf.loc['K-Fold CV',], width=barwidth, label='K-Fold CV', color="#2E8B57")

    for i, value in enumerate(compareDf.loc['Accuracy Score',]):
        ax1.annotate(f'{value:.4f}', (x[i] - barwidth/2, value), textcoords="offset points", xytext=(0,5), ha='center', fontsize=4, color='white')

    for i, value in enumerate(compareDf.loc['K-Fold CV',]):
        ax1.annotate(f'{value:.4f}', (x[i] + barwidth/2, value), textcoords="offset points", xytext=(0,5), ha='center', fontsize=4, color='white')

    # Customized plot appearance
    ax1.set_ylim(0.85, 1)
    legend = ax1.legend(loc="upper left", fontsize = 4, frameon=False, facecolor='white', edgecolor='white')
    for text in legend.get_texts():
        text.set_color('white')
    ax1.set_xlabel("Classification Models", color = '#ffffff', size = 6)
    ax1.set_ylabel("Accuracy / K-Fold CV Score", color = '#ffffff', size = 6)
    ax1.set_title('Accuracy Score and K-Fold CV Score', color = '#d2d7c6', size = 10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(modelLists, rotation = 45, fontsize = 4, color = '#ffffff')
    ax1.tick_params(axis='x', colors = 'white')
    ax1.tick_params(axis='y', colors = '#969689')

    ax1.spines['top'].set_color('#969689')
    ax1.spines['right'].set_color('#969689')
    ax1.spines['bottom'].set_color('#969689')
    ax1.spines['left'].set_color('#969689')

    fig1.patch.set_alpha(0)
    ax1.patch.set_alpha(0)
    ax1.set_facecolor('black')

    st.pyplot(fig1, clear_figure=True)

    ###########################################################################################################################################################33
    # Plot2 : Comparison of Log Loss across Models

    fig2, ax2 = plt.subplots(figsize=(6, 2.5))
    bars = ax2.bar(compareDf.columns, compareDf.loc['Log Loss'], width=0.8, align='center', color=['#c6a0dc','#b2e2a5','#c6a0dc','#87CEEB','#f6c6c6','#b2e2a5','#c6a0dc','#b2e2a5'])

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{height:.4f}', ha='center', va='bottom', fontsize=4, color='white')
    
    # Customized plot appearance
    ax2.set_title('Comparison of Log Loss Across Models', color='#d2d7c6', fontsize = 10)
    ax2.set_xlabel("Classification Models", color='white', size = 6)
    ax2.set_ylabel("Log Loss", color='white', size = 6)
    ax2.set_xticklabels(compareDf.columns, rotation=45, fontsize=4, color='white')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='#969689')

    ax2.spines['top'].set_color('#969689')
    ax2.spines['right'].set_color('#969689')
    ax2.spines['bottom'].set_color('#969689')
    ax2.spines['left'].set_color('#969689')

    fig2.patch.set_alpha(0)
    ax2.patch.set_alpha(0)
    ax2.set_facecolor('black')

    st.pyplot(fig2, clear_figure=True)

    ###########################################################################################################################################################33
    # Plot3 : Time Taken for Model Training

    fig3, ax3 = plt.subplots(figsize=(6, 2.5))
    ax3.plot(compareDf.columns, compareDf.loc['Time Taken'], marker='.', color="#969689")

    for i, (xi, yi) in enumerate(zip(compareDf.columns, compareDf.loc['Time Taken'])):
        ax3.text(xi, yi + 50, f'({yi:.2f})', fontsize=4, ha='right', va="bottom", color='white')
    
    # Customized plot appearance
    ax3.set_title('Time Taken for Model Training', color='#d2d7c6', fontsize = 10)
    ax3.set_xlabel("Classification Models", color='white', size = 6)
    ax3.set_ylabel("Time Taken in Seconds", color='white', size = 6)
    ax3.set_xticklabels(compareDf.columns, rotation=45, fontsize=4, color='white')
    ax3.tick_params(axis='x', colors='white')
    ax3.tick_params(axis='y', colors='#969689')

    ax3.spines['top'].set_color('#969689')
    ax3.spines['right'].set_color('#969689')
    ax3.spines['bottom'].set_color('#969689')
    ax3.spines['left'].set_color('#969689')

    fig3.patch.set_alpha(0)
    ax3.patch.set_alpha(0)
    ax3.set_facecolor('black')

    st.pyplot(fig3, clear_figure=True)

if __name__ == "__main__":
    st.set_page_config(page_title="CreditCard Fraud Detection Model", page_icon=":credit_card:", layout="wide", initial_sidebar_state="expanded")
    
    task = st.sidebar.selectbox("Pick a Task to Dive Into", ['Fraud Detection', 'Explore Models', 'Insights & Visualization'])
    numberOfFeatures = 30

    modelLists = [ 'Logistic Regression on undersampled data', 'Logistic Regression on oversampled data', 'Decision Tree model on undersampled data',\
                     'Decision Tree on oversampled data', 'Random Forest on undersampled data', 'Random Forest on oversampled data',\
                     'Support Vector Machine on undersampled data' ]
    files = [ 'modelLR1.pkl', 'modelLR2.pkl', 'modelTree1.pkl', 'modelTree2.pkl', 'modelRF1.pkl', 'modelRF2.pkl', 'modelSVM.pkl' ]

    if task == "Fraud Detection":
        st.title('Credit Card Fraud Detection', anchor = False)
        inputData = st.text_input(f'Enter all Input features [{numberOfFeatures}]')
        detect = st.button('Detect')
        if detect:
            try:
                inputData = inputData.split(',')
                if len(inputData) != numberOfFeatures:
                    raise inputCount(numberOfFeatures, len(inputData))
                
                inputData = list(map(lambda x: float(x), inputData))
                standardizedInputData = ZScoreTransformation(inputData)

                legitimate = []
                fraudulent = [[0,0,0,0]]

                for model in files:
                    transactionType, likelihood, unlikelihood = fraudDetection(model, standardizedInputData)
                    transactionType = "legitimate transaction" if transactionType == [0] else "fraudulent transaction" if transactionType == [1] else "fraudulent transaction"
                    if transactionType == "legitimate transaction":
                        legitimate.append([model, transactionType, likelihood, unlikelihood])
                    elif transactionType == "fraudulent transaction":
                        fraudulent.append([model, transactionType, likelihood, unlikelihood])
                
                class0Models = len(legitimate)
                class1Models = len(fraudulent)
                transactionType = "legitimate transaction" if class0Models > class1Models else "fraudulent transaction" if class0Models > class1Models else 'Equal'

                average_accuracy = sum(inner_list[2] for inner_list in legitimate) / class0Models
                fraudulence_confidence = sum(inner_list[2] for inner_list in fraudulent) / (class1Models - 1)
                
                if transactionType == "legitimate transaction" and average_accuracy >= 70:
                    st.success(f"With an impressive average confidence of {average_accuracy}%, we confidently assure you that this transaction is both secure and legitimate, as confirmed by multiple models.")

                elif transactionType == "fraudulent transaction":
                    st.error(f"With an alarming average confidence of {fraudulence_confidence}%, we strongly advise caution regarding this transaction, as multiple models have flagged it as potentially fraudulent.")

                elif transactionType == "legitimate transaction" and average_accuracy < 70:
                    st.warning(f'''Although we affirm the legitimacy of this transaction with {average_accuracy}% average certainty, it remains at the edge of our confidence threshold, with a {fraudulence_confidence}% unlikelihood.
                          It is advisable to remain in the safe zone by taking necessary precautionary measures.''')
                    
                else:
                    st.error("Something went wrong...")
                
            except inputCount as e:
                st.error(e)       

    if task == "Explore Models":
        st.title('Credit Card Fraud Detection', anchor = False)
        st.subheader('Get in the Know: Uncovering Results from Every Model!', anchor = False)
    
        modelChoice = st.sidebar.selectbox('Classification Models', modelLists)
        model = files[modelLists.index(modelChoice)]

        inputData = st.text_input(f'Enter all Input features [{numberOfFeatures}]')
        detect = st.button('Detect')
        if detect:
            try:
                inputData = inputData.split(',')
                if len(inputData) != numberOfFeatures:
                    raise inputCount(numberOfFeatures, len(inputData))
                inputData = list(map(lambda x: float(x), inputData))
                standardizedInputData = ZScoreTransformation(inputData)
                transactionType, likelihood, unlikelihood = fraudDetection(model, standardizedInputData)
            
                transactionType = "legitimate transaction" if transactionType == [0] else "fraudulent transaction" if transactionType == [1] else [1]
                st.subheader(transactionType, anchor = False)
            

                if transactionType == "legitimate transaction" and model == "modelSVM.pkl":
                    st.success('Based on our SVM algorithm, this transaction is situated firmly within the legitimate range, ensuring that it is secure and trustworthy.')

                elif transactionType == "legitimate transaction" and likelihood >= 70 and model != "modelSVM.pkl":
                    st.success(f"With a remarkable {likelihood}% confidence, we assure you that this transaction is both secure and legitimate.")

                elif transactionType == "legitimate transaction" and (likelihood >= 40 and likelihood <= 70) and model != "modelSVM.pkl":
                    st.warning(f'''Although we affirm the legitimacy of this transaction with {likelihood}% certainty, it remains at the edge of our confidence threshold, with a {unlikelihood}% unlikelihood.
                          It is advisable to remain in the safe zone by taking necessary precautionary measures.''')
            
                elif transactionType == "fraudulent transaction":
                    st.error("The algorithm identified patterns and anomalies that indicate the transaction is not legitimate, highlighting a significant deviation from standard transactional behavior.")
            
                else:
                    st.error("Something went wrong...")

            except ValueError as e:
                st.error(str(e))

            except inputCount as e:
                st.error(e)
            
        st.sidebar.subheader("Get Model")
        st.sidebar.download_button( label = "get "+model, data = 'Models/'+model, file_name = model, mime = "application/octet-stream")
    
    st.sidebar.subheader("Description of Dataset Features")
    st.sidebar.markdown("*The dataset consists solely of numerical input variables resulting from a PCA transformation. Due to confidentiality concerns,\
                    the original features and additional background information are not available. Features V1 through V28 represent the \
                    principal components derived from PCA. The features 'Time' and 'Amount' are the only ones not transformed by PCA.\
                     'Time' indicates the number of seconds elapsed between each transaction and the first transaction recorded in the dataset.\
                     'Amount' represents the transaction amount, which can be utilized in cost-sensitive learning approaches.\
                     The 'Class' feature is the response variable, with a value of 1 indicating fraud and 0 otherwise.*")
    st.sidebar.markdown("[Click here to visit Dataset]( https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")

    if task == "Insights & Visualization":
        st.title('Insights & Visualization', anchor = False)
        st.subheader('Visualizing Algorithmic Performance: Insights, Impact, and Key Takeaways')
        insightsAndVisualization()
    
    page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://miro.medium.com/max/8002/1*yt0h233ql_VWlSvMI6vJYA.jpeg");
        background-size: cover;
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    body {color: red;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)
