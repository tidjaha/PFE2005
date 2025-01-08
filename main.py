import streamlit as st
import pandas as pd
import joblib
import os
# Load the trained model
from PIL import Image, UnidentifiedImageError
import gdown

from io import BytesIO  # Importation nécessaire
import requests






poly_tc =joblib.load("model scaler(Tc).pkl")
model_LR_n2_tc=joblib.load("model(tc).pkl")

scaler_tc_tb = joblib.load("scaler(Tc) tb.pkl")
scaler_tc_tc= joblib.load("scaler(Tc) tc.pkl")
scaler_tc_MM=joblib.load("scaler(Tc) MM.pkl")
scaler_tc_d420=joblib.load("scaler(Tc) d420.pkl")
scaler_tc_n20=joblib.load("scaler(Tc) n20.pkl")
scaler_tc_ncnh=joblib.load("scaler(Tc) ncnh.pkl")


model_RF_pc=joblib.load("model_RF_pc.pkl")



scaler_pc_tb=joblib.load("scaler(Pc) tb.pkl")
scaler_pc_tc=joblib.load("scaler(Pc) tc.pkl")
scaler_pc_MM=joblib.load("scaler(Pc) mm.pkl")
scaler_pc_d20=joblib.load("scaler(Pc) d20.pkl")
scaler_pc_n20=joblib.load("scaler(Pc) n20.pkl")
scaler_pc_nbHnbC=joblib.load("scaler(Pc) nbHnbC.pkl")
scaler_pc_pc=joblib.load("scaler(Pc) pc.pkl")


poly_vc =joblib.load("scaler model vc.pkl")
model_LR_vc_n2=joblib.load("model_LR_n2_vc.pkl")

scaler_vc_tb=joblib.load("scaler(Vc) tb.pkl")
scaler_vc_tc=joblib.load("scaler(Vc) tc.pkl")
scaler_vc_MM=joblib.load("scaler(Vc) mm.pkl")
scaler_vc_d20=joblib.load("scaler(Vc) d20.pkl")
scaler_vc_n20=joblib.load("scaler(Vc) n20.pkl")
scaler_vc_nbHnbC=joblib.load("scaler(Vc) nbHnbC.pkl")
scaler_vc_vc=joblib.load("scaler(Vc) vc.pkl")
scaler_vc_pc=joblib.load("scaler(Vc) pc.pkl")

model_LR_Cp = joblib.load("model_LR_Cp.pkl")

scaler_Cp_tb=joblib.load("scaler(Cp) tb.pkl")
scaler_Cp_tc=joblib.load("scaler(Cp) tc.pkl")
scaler_Cp_MM=joblib.load("scaler(Cp) mm.pkl")
scaler_Cp_d20=joblib.load("scaler(Cp) d20.pkl")
scaler_Cp_n20=joblib.load("scaler(Cp) n20.pkl")
scaler_Cp_nbHnbC=joblib.load("scaler(Cp) nbHnbC.pkl")
scaler_Cp_Cp=joblib.load("scaler(Cp) cp.pkl")
scaler_Cp_vc=joblib.load("scaler(Cp) vc.pkl")






# Define function to make predictions

def predict_tc(input_features_tc):

    # Perform any necessary preprocessing on the input_features

    # Make predictions using the loaded model

    poly = poly_tc.transform(input_features_tc)
    input_features_tc["Tc(K)"] = model_LR_n2_tc.predict(poly)
    st.write(input_features_tc)
    test=input_features_tc[["Tc(K)"]].values.reshape(-1, 1)
    prediction=scaler_tc_tc.inverse_transform(test)

    # Return the predictions

    return prediction
    #fig, ax = plt.subplots(figsize=(4, 4))
    #ax.bar(["Math","Lecture","Ecriture"],prediction[0] , color=['red','blue','black'], label=[f"Math = {prediction[0][0]:.2f}",f"Lecture = {prediction[0][1]:.2f}",f"Ecriture = {prediction[0][2]:.2f}"])
    #ax.set_ylim(0, 100)
    #ax.set_title('Les trois notes')
    #ax.legend()
    #ax.set_ylabel('Notes')
    #plot=st.pyplot(fig)


 #fonction pour le calcul de Pc:

def predict_pc(input_features_pc,vpc):

    # Perform any necessary preprocessing on the input_features

    # Make predictions using the loaded model

    
    input_features_pc["Tc(K)"] = vpc
    
    input_features_pc["Pc(bar)"] = model_RF_pc.predict(input_features_pc)
    st.write(input_features_pc)
    test_pc=input_features_pc[["Pc(bar)"]].values.reshape(-1, 1)
    pred=scaler_pc_pc.inverse_transform(test_pc)

    # Return the predictions

    return pred

# Create the web interface

def main():

    st.title('prediction des valeurs de Tc pour un corps pur')

    st.write('remplissez les champs pour avoir la prediction')


    # Create input fields for user to enter data tc

    d20=st.number_input("densité à 20°C Kg/m3", min_value=0.0, format="%.2f")
    encodedd20=scaler_tc_d420.transform([[d20]])[0][0]

    n20=st.number_input("indice de refraction à 20°C", min_value=0.0, format="%.6f")
    encodedn20=scaler_tc_n20.transform([[n20]])[0][0]

    Tb=st.number_input("Température d'ébullition en °K", format="%.2f")
    encodedTb=scaler_tc_tb.transform([[Tb]])[0][0]

    MM=st.number_input("masse molaire en g/mol", min_value=0.0, format="%.6f")
    encodedMM=scaler_tc_MM.transform([[MM]])[0][0]

    nbH=st.number_input("nombre d'hydrogène", min_value=1)


    nbC=st.number_input("nombre de carbone", min_value=1)
    data_to_transform = [[nbH, nbC]]

    # Appliquez la transformation
    transformed_data = scaler_pc_nbHnbC.transform(data_to_transform)

    # Récupérez les valeurs transformées
    encodednbH, encodednbC = transformed_data[0][0], transformed_data[0][1]


    famille=st.selectbox("choisir la famille de votre corps",["aromatiques","i-paraffines","n-paraffines","naphtènes","oléfines", "alcynes"])

        # Add more input fields as needed

    # Combine input features into a DataFrame

    input_data_tc = {

    'd20(Kg/m3)': [encodedd20],
    'n20': [encodedn20],
    'Tb(K)': [encodedTb],
    'MM(g/mole)': [encodedMM],
    'famille': [famille],
    'nbH': [encodednbH],
    'nbC': [encodednbC]

}

# Création du DataFrame
    input_data_tc = pd.DataFrame(input_data_tc)

    #encodage de famille

    family =["famille_aromatiques","famille_i-paraffines","famille_n-paraffines","famille_naphtènes","famille_oléfines"]
    for i in family:
      input_data_tc[i]=0
    for i in range(0,len(input_data_tc)):
      if famille=="n-paraffines":
        input_data_tc["famille_n-paraffines"]=1
      elif famille=="i-paraffines":
        input_data_tc["famille_i-paraffines"]=1
      elif famille=="oléfines":
        input_data_tc["famille_oléfines"]=1
      elif famille=="alcynes":
        continue
      elif famille=="aromatiques":
        input_data_tc["famille_aromatiques"]=1
      elif famille=="naphtènes":
        input_data_tc["famille_naphtènes"]=1
    input_data_tc=input_data_tc.drop("famille",axis=1)



# Create input fields for user to enter data pc

    
    encodedd20_pc=scaler_pc_d20.transform([[d20]])[0][0]

    
    encodedn20_pc=scaler_pc_n20.transform([[n20]])[0][0]

    
    encodedTb_pc=scaler_pc_tb.transform([[Tb]])[0][0]

    
    encodedMM_pc=scaler_pc_MM.transform([[MM]])[0][0]

    


    
    data_to_transform_pc = [[nbH, nbC]]

    # Appliquez la transformation
    transformed_data_pc = scaler_pc_nbHnbC.transform(data_to_transform_pc)

    # Récupérez les valeurs transformées
    encodednbH_pc, encodednbC_pc = transformed_data_pc[0][0], transformed_data_pc[0][1]


    

        # Add more input fields as needed

    # Combine input features into a DataFrame

    input_data_pc = {

    'd20(Kg/m3)': [encodedd20_pc],
    'n20': [encodedn20_pc],
    'Tb(K)': [encodedTb_pc],
    'MM(g/mole)': [encodedMM_pc],
    'famille': [famille],
    'nbH': [encodednbH_pc],
    'nbC': [encodednbC_pc],
    'Tc(K)':[0]

}

# Création du DataFrame
    input_data_pc = pd.DataFrame(input_data_pc)

    #encodage de famille

    family =["famille_aromatiques","famille_i-paraffines","famille_n-paraffines","famille_naphtènes","famille_oléfines"]
    for i in family:
      input_data_pc[i]=0
    for i in range(0,len(input_data_pc)):
      if famille=="n-paraffines":
        input_data_pc["famille_n-paraffines"]=1
      elif famille=="i-paraffines":
        input_data_pc["famille_i-paraffines"]=1
      elif famille=="oléfines":
        input_data_pc["famille_oléfines"]=1
      elif famille=="alcynes":
        continue
      elif famille=="aromatiques":
        input_data_pc["famille_aromatiques"]=1
      elif famille=="naphtènes":
        input_data_pc["famille_naphtènes"]=1
    input_data_pc=input_data_pc.drop("famille",axis=1)




    if st.button('Predictions'):

        prediction_tc = predict_tc(input_data_tc)
        prediction_pc= predict_pc(input_data_pc,prediction_tc)

        st.write('Les Predictions sont :\n\n', prediction_tc,prediction_pc)



if __name__ == '__main__':

    main()
