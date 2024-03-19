import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import plotly.express as px
import streamlit as st
import copy
from model import preprocess

# Function to center-align text and headings
def centered_text(text):
    return f"<h2 style='text-align: center;'>{text}</h2>"

# Set page title and icon

col = st.columns((4, 4), gap='medium')

def visualization(data,target):
    data=preprocess(data,target)
    df=copy.deepcopy(data)
    if df[target].dtype=="object":
        le=LabelEncoder()
        df[target]=le.fit_transform(df[target])
    corr_matrix = df.corr()

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5, ax=ax)
    plt.title("Correlation Heatmap")
    st.pyplot(fig)

    highest_corr_features = corr_matrix[target].sort_values(ascending=False).index[1:6]  # Exclude the target variable itself

    for feature in highest_corr_features:
        st.markdown(centered_text(f"Visualization for {feature}"),unsafe_allow_html=True)
        if df[feature].dtype in ['int64', 'float64']:
            fig, ax = plt.subplots(2,2,figsize=(12, 8))
            sns.histplot(df[feature], kde=True, ax=ax[0,0])
            ax[0,0].set_xlabel(feature)
            ax[0,0].set_ylabel("Frequency")
            ax[0,0].set_title("Histogram of " + feature)


            #fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df, x=feature, y=target, ax=ax[0,1])
            ax[0,1].set_xlabel(feature)
            ax[0,1].set_ylabel(target)
            ax[0,1].set_title("Scatter Plot of " + feature + " vs " + target)


            #fig, ax = plt.subplots(figsize=(8, 6))
            sns.lineplot(data=df, x=feature, y=target, ax=ax[1,0])
            ax[1,0].set_xlabel(feature)
            ax[1,0].set_ylabel(target)
            ax[1,0].set_title("Line Plot of " + feature + " vs " + target)

            #fig, ax = plt.subplots(figsize=(8, 6))
            sns.lineplot(data=df, x=feature, y=target, ax=ax[1,1], ci=None)
            sns.lineplot(data=df, x=feature, y=target, ax=ax[1,1], ci='sd')
            ax[1,1].set_xlabel(feature)
            ax[1,1].set_ylabel(target)
            ax[1,1].set_title("Area Plot of " + feature + " vs " + target)
            plt.subplots_adjust(wspace=0.3,hspace=0.5)
            st.pyplot(fig)

        elif df[feature].dtype == 'object':
            fig, ax = plt.subplots(1,2,figsize=(12, 8))
            #st.write("Bar Plot")
            #bar_plot = px.bar(df[feature].value_counts(), x=df[feature].value_counts().index, y=df[feature].value_counts().values)
            #st.plotly_chart(bar_plot)
            sns.barplot(df[feature].value_counts(), x=df[feature].value_counts().index, y=df[feature].value_counts().values,ax=ax)
            st.pyplot(fig)

            #st.write("Pie Chart")
            pie_chart = px.pie(df[feature].value_counts(), values=df[feature].value_counts().values, names=df[feature].value_counts().index)
            st.plotly_chart(pie_chart)
