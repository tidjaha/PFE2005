import streamlit as st
import pandas as pd
import joblib
import os
# Load the trained model
from PIL import Image, UnidentifiedImageError
import gdown

from io import BytesIO  # Importation nécessaire
import requests







model_LR_tc=joblib.load("model(tc).pkl")

scaler_tb = joblib.load("scaler tb.pkl")
scaler_tc= joblib.load("scaler tc.pkl")
scaler_MM=joblib.load("scaler MM.pkl")
scaler_d420=joblib.load("scaler d420.pkl")
scaler_n20=joblib.load("scaler n20.pkl")
scaler_ncnh=joblib.load("scaler ncnh.pkl")
scaler_vc=joblib.load("scaler Vc.pkl")
scaler_pc=joblib.load("scaler Pc.pkl")
scaler_Cp=joblib.load("scaler Cp.pkl")

model_RF_pc=joblib.load("model_RF_pc.pkl")

model_LR_vc=joblib.load("model_LR_vc.pkl")

model_LR_Cp = joblib.load("model_LR_Cp.pkl")








# Define function to make predictions

def predict(input_features):

    # Perform any necessary preprocessing on the input_features

    # Make predictions using the loaded model

    tc=model_LR_tc.predict(input_features)
    pc=model_RF_pc.predict(input_features)
    vc=model_LR_vc.predict(input_features)
    Cp=model_LR_Cp.predict(input_features)  

    output_pred=input_features.copy()
    output_pred["Tc(K)"]=tc
    output_pred["Pc(bar)"]=pc
    output_pred["Vc(m3/s)"]=vc
    output_pred["Cp(J/kg/K)"]=Cp  

    vtc=scaler_tc.inverse_transform(output_pred[["Tc(K)"]])
    output_pred["Tc(K)"]=vtc
    vpc=scaler_pc.inverse_transform(output_pred[["Pc(bar)"]])
    output_pred["Pc(bar)"]=vpc
    vvc=scaler_vc.inverse_transform(output_pred[["Vc(m3/s)"]])
    output_pred["Vc(cm3/mol)"]=vvc
    vCp=scaler_Cp.inverse_transform(output_pred[["Cp(J/kg/K)"]])
    output_pred["Cp(J/mol K)"]=vCp

    # Return the predictions

    return output_pred
    #fig, ax = plt.subplots(figsize=(4, 4))
    #ax.bar(["Math","Lecture","Ecriture"],prediction[0] , color=['red','blue','black'], label=[f"Math = {prediction[0][0]:.2f}",f"Lecture = {prediction[0][1]:.2f}",f"Ecriture = {prediction[0][2]:.2f}"])
    #ax.set_ylim(0, 100)
    #ax.set_title('Les trois notes')
    #ax.legend()
    #ax.set_ylabel('Notes')
    #plot=st.pyplot(fig)


 #fonction pour le calcul de Pc:


def main():

    st.title('prediction des valeurs de (Tc,Pc,Vc,Cp) pour un corps pur')

    

    st.write('remplissez les champs pour avoir la prediction')


    # Create input fields for user to enter data tc

    d20=st.number_input("densité à 20°C Kg/m3", min_value=0.0, format="%.2f")
    encodedd20=scaler_d420.transform([[d20]])[0][0]

    n20=st.number_input("indice de refraction à 20°C", min_value=0.0, format="%.6f")
    encodedn20=scaler_n20.transform([[n20]])[0][0]

    Tb=st.number_input("Température d'ébullition en °K", format="%.2f")
    encodedTb=scaler_tb.transform([[Tb]])[0][0]

    MM=st.number_input("masse molaire en g/mol", min_value=0.0, format="%.6f")
    encodedMM=scaler_MM.transform([[MM]])[0][0]

    nbH=st.number_input("nombre d'hydrogène", min_value=1)


    nbC=st.number_input("nombre de carbone", min_value=1)
    data_to_transform = [[nbH, nbC]]

    # Appliquez la transformation
    transformed_data = scaler_ncnh.transform(data_to_transform)

    # Récupérez les valeurs transformées
    encodednbH, encodednbC = transformed_data[0][0], transformed_data[0][1]


    famille=st.selectbox("choisir la famille de votre corps",["aromatiques","i-paraffines","n-paraffines","naphtènes","oléfines", "alcynes"])

        # Add more input fields as needed

    # Combine input features into a DataFrame

    input_data = {

    'd20(Kg/m3)': [encodedd20],
    'n20': [encodedn20],
    'Tb(K)': [encodedTb],
    'MM(g/mole)': [encodedMM],
    'famille': [famille],
    'nbH': [encodednbH],
    'nbC': [encodednbC]

}

# Création du DataFrame
    input_data = pd.DataFrame(input_data)

    #encodage de famille

    family =["famille_aromatiques","famille_i-paraffines","famille_n-paraffines","famille_naphtènes","famille_oléfines"]
    for i in family:
      input_data[i]=0
    for i in range(0,len(input_data)):
      if famille=="n-paraffines":
        input_data["famille_n-paraffines"]=1
      elif famille=="i-paraffines":
        input_data["famille_i-paraffines"]=1
      elif famille=="oléfines":
        input_data["famille_oléfines"]=1
      elif famille=="alcynes":
        continue
      elif famille=="aromatiques":
        input_data["famille_aromatiques"]=1
      elif famille=="naphtènes":
        input_data["famille_naphtènes"]=1
    input_data=input_data.drop("famille",axis=1)



    if st.button('Predictions'):

        prediction = predict(input_data)
        
        tc=prediction.loc[0, "Tc(K)"]
        pc=prediction.loc[0, "Pc(bar)"]
        vc=prediction.loc[0, "Vc(cm3/mol)"]
        cp=prediction.loc[0, "Cp(J/mol K)"]
        st.write("Les Predictions sont :\n\n Tc(K) =" ,tc,"\n\n Pc(bar) =",pc ,"\n\n Vc(cm3/mol) =",vc , "\n\n Cp(J/mol K) =",cp)

        st.write("Copyrights tidjaha 2025 (hamza.tidjani@yahoo.fr) \n\n Link Linkedin : https://www.linkedin.com/in/hamza-tidjani-539b78237" )

        # URL de Google Drive (assurez-vous que c'est un lien de téléchargement direct)
        url = "https://drive.google.com/uc?export=download&id=1mdMdvXYGiowfy3UwNtMBCAllv7wt1DUT"  # Exemple d'ID
        
        # Téléchargement du fichier avec requests
        response = requests.get(url, stream=True)
        
        # Sauvegarder l'image téléchargée dans un fichier temporaire
        image_path = "moi.jpg"
        with open(image_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)
        st.image(image_path)



        #https://drive.google.com/file/d/1mdMdvXYGiowfy3UwNtMBCAllv7wt1DUT/view?usp=sharing



if __name__ == '__main__':

    main()
