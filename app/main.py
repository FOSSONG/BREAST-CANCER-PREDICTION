from ast import main
import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import numpy as np


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


def get_clean_data():
    # Read the data from the CSV file
    data = pd.read_csv(r'C:/Users/GUILIANNO FOSSONG/Downloads/STREAMLIT-APP-CANCER/Data/data.csv')
    # Perform any data cleaning or preprocessing if needed
    
    # Handle non-numeric values
    data.replace({'M': 0, 'B': 1}, inplace=True)  # Replace 'M' with 0 and 'B' with 1
    
    return data

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler):
        self.scaler = scaler
        self.feature_names = None

    def fit(self, X, y=None):
        self.feature_names = X.columns
        self.scaler.fit(X, y)
        return self

    def transform(self, X):
        # Check if feature names are consistent
        if list(X.columns) != list(self.feature_names):
            raise ValueError("The feature names should match those that were passed during fit.")
        
        # Transform X using the scaler
        X_scaled = self.scaler.transform(X)
        # Create DataFrame with original feature names
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        return X_scaled_df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def set_feature_names(self, feature_names):
        self.feature_names = feature_names

def add_sidebar():
    st.sidebar.header("Cell Nuclear Measurements")
    # Call get_clean_data() to retrieve the cleaned data
    data = get_clean_data()

    # Usage example
    scaler = StandardScaler()
    custom_scaler = CustomScaler(scaler)
    scaled_data = custom_scaler.fit_transform(data)
    custom_scaler.set_feature_names(data.columns)  # Set feature names explicitly
    

# Call the add_sidebar() function to start your Streamlit app
add_sidebar()


def add_sidebar():
  st.sidebar.header("Cell Nuclear Measurements")
  data = get_clean_data()

  #print("Column Names:", data.columns)


  slider_labels = [
        ("Radius (mean)", "radius1"),
        ("Texture (mean)", "texture1"),
        ("Perimeter (mean)", "perimeter1 "),
        ("Area (mean)", "area1"),
        ("Smoothness (mean)", "smoothness1"),
        ("Compactness (mean)", "compactness1"),
        ("Concavity (mean)", "concavity1"),
        ("Concave points (mean)", "concave_points1"),
        ("Symmetry (mean)", "symmetry1"),
        ("Fractal dimension (mean)", "fractal_dimension1"),
        ("Radius (se)", "radius2"),
        ("Texture (se)", "texture2"),
        ("Perimeter (se)", "perimeter2"),
        ("Area (se)", "area2"),
        ("Smoothness (se)", "smoothness2"),
        ("Compactness (se)", "compactness2"),
        ("Concavity (se)", "concavity2"),
        ("Concave points (se)", "concave_points2"),
        ("Symmetry (se)", " symmetry2"),
        ("Fractal dimension (se)", "fractal_dimension2"),
        ("Radius (worst)", " radius3"),
        ("Texture (worst)", "texture3 "),
        ("Perimeter (worst)", "perimeter3"),
        ("Area (worst)", "area3"),
        ("Smoothness (worst)", "smoothness3"),
        ("Compactness (worst)", "compactness3"),
        ("Concavity (worst)", "concavity3"),
        ("Concave points (worst)", "concave_points3"),
        ("Symmetry (worst)", "symmetry3"),
        ("Fractal dimension (worst)", " fractal_dimension3 "),
    ]
  
  input_dict = {}
  for label, key in slider_labels:
        if key in data.columns:  # Check if the key exists in DataFrame columns
            input_dict [key] = st.sidebar.slider(
                label,
                min_value=float(0),
                max_value=float(data[key].max()),
                value=float(data[key].mean())
            )
        else:
            st.sidebar.warning(f"Column '{key}' not found in the dataset.")
  return input_dict

input_data = add_sidebar()

def main():
  st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )
  

def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data.drop(['diagnosis'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict
   
def get_radar_chart(input_data):
  
  input_data = get_scaled_values(input_data)
  
  categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius1'], input_data['texture1'], input_data['perimeter1 '],
          input_data['area1'], input_data['smoothness1'], input_data['compactness1'],
          input_data['concavity1'], input_data['concave_points1'], input_data['symmetry1'],
          input_data['fractal_dimension1']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius2'], input_data['texture2'], input_data['perimeter2'], input_data['area2'],
          input_data['smoothness2'], input_data['compactness2'], input_data['concavity2'],
          input_data['concave_points2'], input_data[' symmetry2'],input_data['fractal_dimension2']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data[' radius3'], input_data['texture3 '], input_data['perimeter3'],
          input_data['area3'], input_data['smoothness3'], input_data['compactness3'],
          input_data['concavity3'], input_data['concave_points3'], input_data['symmetry3'],
          input_data[' fractal_dimension3 ']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
  ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
  return fig
   
def add_predictions(input_data):
   
  model = joblib.load("model/model.pkl", "r")
  scaler = joblib.load("model/scaler.pkl", "r") 

  input_array = np.array(list(input_data.values())).reshape(1, -1)
  input_array_scaled = scaler.transform(input_array)

  prediction = model.predict(input_array_scaled)

  st.subheader("Cell cluster prediction")
  st.write("The cell cluster is:")

  if prediction[0] == 0:
    st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
  else:
    st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)

  st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
  st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])

  st.write("This app is meant to assist medical experts in making diagnoses, and should not be used as a substitute for a professional diagnosis.")
  #st.write(prediction)

with open("assets/style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
  
    #input_data = add_sidebar()
  
with st.container():
    st.title("Breast Cancer Predictor")
    st.write("Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")
  
col1, _, col2 = st.columns([3, 0.5, 2])

with col1:
  radar_chart = get_radar_chart(input_data)
  st.plotly_chart (radar_chart)
 
with col2:
  add_predictions(input_data)




#if __name__ == '__main__':
    #main()





