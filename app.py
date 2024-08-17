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
    
    return input_df

@app.route('/', methods=['GET', 'POST'])
def home():
    message = None  # Initialize the message variable
    prediction = None
    if request.method == 'POST':
        # Récupérer les données du formulaire
        data = request.form.to_dict()
        data = {key: [float(value)] if key not in ['carrier', 'origin', 'dest', 'tailnum', 'time_hour'] else [value] for key, value in data.items()}
        
        # Reconstituer les champs dep_time et sched_dep_time à partir des heures et minutes
        data['dep_time'] = [int(data['dep_time'][0])]
        data['sched_dep_time'] = [int(data['sched_dep_time'][0])]
        
        # Convertir les données en DataFrame
        new_data = pd.DataFrame(data)
        
        # Supprimer les colonnes inutiles
        columns_to_drop = ['hour', 'minute', 'time_hour']
        new_data = new_data.drop(columns=columns_to_drop, errors='ignore')
        
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

         
        if prediction == 1:
            message = 1
        else:
            message = 0

    return render_template('index.html', prediction=message)

if __name__ == "__main__":
    app.run(debug=True)
