from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Charger le modèle depuis le fichier .pkl
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Fonction de prétraitement des nouvelles données
def preprocess_input_data(input_df):
    # Encoder les variables catégorielles
    input_df = pd.get_dummies(input_df, columns=['carrier', 'origin', 'dest'])
    
    # Convertir les colonnes booléennes (False/True) en 0/1
    bool_columns = input_df.select_dtypes(include=['bool']).columns
    input_df[bool_columns] = input_df[bool_columns].astype(int)
    
    # Identification des colonnes de type date/heure
    datetime_columns = input_df.select_dtypes(include=['object']).apply(pd.to_datetime, errors='coerce').notna().any()
    
    # Conversion des colonnes date/heure en composantes numériques
    for col in datetime_columns[datetime_columns].index:
        input_df[col] = pd.to_datetime(input_df[col], errors='coerce')
        input_df[col + '_year'] = input_df[col].dt.year
        input_df[col + '_month'] = input_df[col].dt.month
        input_df[col + '_day'] = input_df[col].dt.day
        input_df[col + '_hour'] = input_df[col].dt.hour
        input_df[col + '_minute'] = input_df[col].dt.minute
        input_df = input_df.drop(columns=[col])  # Supprimer la colonne originale

    return input_df

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # Récupérer les données du formulaire
        data = request.form.to_dict()
        data = {key: [float(value)] if key not in ['carrier', 'origin', 'dest', 'tailnum', 'time_hour'] else [value] for key, value in data.items()}
        
        # Convertir les données en DataFrame
        new_data = pd.DataFrame(data)
        
        # Assigner 'tailnum' comme index
        new_data.set_index('tailnum', inplace=True)
        
        # Prétraiter les nouvelles données
        new_data = preprocess_input_data(new_data)
        
        # Aligner les colonnes de l'entrée avec celles utilisées lors de l'entraînement
        all_columns = model.feature_names_in_
        for col in all_columns:
            if col not in new_data.columns:
                new_data[col] = 0

        new_data = new_data[all_columns]

        # Faire des prédictions
        prediction = model.predict(new_data)[0]

         # Déterminer le message basé sur la prédiction
        if prediction == 1:
            message = 'The flight will be delayed'
        else:
            message = 'The flight will not be delayed'

    return render_template('index.html', prediction=message)

if __name__ == "__main__":
    app.run(debug=True)
