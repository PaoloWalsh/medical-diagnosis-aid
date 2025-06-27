import streamlit as st
import requests
import json
import os # Importa il modulo os per accedere alle variabili d'ambiente

# --- Configurazione API Flask ---
# L'applicazione Streamlit ora legge l'URL dell'API Flask dalla variabile d'ambiente FLASK_API_URL.
# Se la variabile non è impostata (es. esecuzione locale diretta senza Docker Compose),
# userà http://localhost:5001 come fallback.
FLASK_API_URL = os.getenv("FLASK_API_URL", "http://localhost:5001")

# --- Funzioni per chiamare l'API ---

def get_available_models():
    """
    Recupera i modelli disponibili dall'endpoint /models della Flask API.
    """
    try:
        response = requests.post(f"{FLASK_API_URL}/model_list", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("available_models", [])
    except requests.RequestException as e:
        st.warning(f"Errore nella richiesta all'API: {e}")
        return []
    except ValueError:
        st.warning("Risposta API non è un JSON valido.")
        return []
    
def predict_data(model_name, data):
    """
    Invia i dati all'endpoint /predict della Flask API per ottenere una previsione.
    """
    endpoint = f"{FLASK_API_URL}/predict"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model_name": model_name,
        "data": [data]  # L'API Flask si aspetta una lista di liste per i dati
    }
    try:
        response = requests.post(endpoint, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Lancia un'eccezione per codici di stato HTTP errati
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Errore di connessione all'API Flask. Assicurati che l'API sia in esecuzione su {FLASK_API_URL}")
        return {"error": "Connessione all'API fallita."}
    except requests.exceptions.HTTPError as e:
        st.error(f"Errore HTTP durante la previsione: {e.response.status_code} - {e.response.json().get('error', 'Errore sconosciuto')}")
        return {"error": e.response.json().get('error', 'Errore sconosciuto')}
    except Exception as e:
        st.error(f"Errore durante la chiamata all'API /predict: {e}")
        return {"error": "Errore sconosciuto durante la previsione."}

def get_model_performance(model_name):
    """
    Recupera le metriche di performance di un modello dall'endpoint /models della Flask API.
    """
    endpoint = f"{FLASK_API_URL}/models"
    headers = {"Content-Type": "application/json"}
    payload = {"model_name": model_name}
    try:
        response = requests.post(endpoint, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Errore di connessione all'API Flask. Assicurati che l'API sia in esecuzione su {FLASK_API_URL}")
        return {"error": "Connessione all'API fallita."}
    except requests.exceptions.HTTPError as e:
        st.error(f"Errore HTTP durante il recupero delle performance: {e.response.status_code} - {e.response.json().get('error', 'Errore sconosciuto')}")
        return {"error": e.response.json().get('error', 'Errore sconosciuto')}
    except Exception as e:
        st.error(f"Errore durante la chiamata all'API /models: {e}")
        return {"error": "Errore sconosciuto durante il recupero delle performance."}

# --- Interfaccia Utente Streamlit ---

st.set_page_config(page_title="Medical Diagnosis Aid", layout="centered")

st.title("👨‍⚕️ Strumento di Supporto alla Diagnosi Medica")
st.markdown("Questa app si interfaccia con un'API Flask per eseguire previsioni di diagnosi.")

# Recupera i modelli disponibili all'avvio dell'app
available_models = get_available_models()

if not available_models:
    st.warning("Nessun modello disponibile dall'API Flask. Assicurati che l'API sia in esecuzione e che i modelli siano caricati correttamente.")
else:
    # Selezione del modello
    st.sidebar.header("Configurazione Modello")
    selected_model = st.sidebar.selectbox(
        "Seleziona il Modello di Previsione:",
        available_models,
        help="Scegli il modello di Machine Learning da utilizzare per la diagnosi."
    )

    # Visualizza le performance del modello selezionato
    if selected_model:
        st.sidebar.subheader("Performance del Modello Selezionato")
        with st.spinner(f"Recupero performance per {selected_model}..."):
            performance_data = get_model_performance(selected_model)
            if performance_data and "error" not in performance_data:
                st.sidebar.markdown(f"**Accuratezza:** {float(performance_data.get('accuracy', 0)):.4f}")
                st.sidebar.markdown(f"**Recall:** {float(performance_data.get('recall', 0)):.4f}")
                st.sidebar.markdown(f"**ROC AUC:** {float(performance_data.get('roc_auc', 0)):.4f}")
            elif performance_data and "error" in performance_data:
                st.sidebar.error(f"Impossibile recuperare le performance: {performance_data['error']}")
            else:
                st.sidebar.info("Performance non disponibili per questo modello.")


    st.header("Inserisci i Dati del Paziente")
    st.markdown("Si prega di inserire i valori delle caratteristiche pertinenti per ottenere una previsione.")

    # Inizializza la lista per i valori delle feature
    # La lista sarà popolata nell'ordine esatto richiesto per il modello one-hot encoded
    feature_values = []

    st.subheader("Informazioni Demografiche e Cliniche:")

    # SEX (sex_Female, sex_Male)
    sex_selected = st.radio("Sesso", ["Maschio", "Femmina"], index=0, key="sex_input_ohe")
    feature_values.append(1.0 if sex_selected == "Femmina" else 0.0) # sex_Female
    feature_values.append(1.0 if sex_selected == "Maschio" else 0.0) # sex_Male

    # CHEST_PAIN_TYPE (chest_pain_type_asymptomatic, chest_pain_type_atypical angina, chest_pain_type_non-anginal, chest_pain_type_typical angina)
    chest_pain_selected = st.selectbox("Tipo di dolore al petto", 
                                       ["Angina tipica", "Angina atipica", "Dolore non anginoso", "Asintomatico"], 
                                       index=3, # Default: Asintomatico (come nell'esempio)
                                       key="chest_pain_input_ohe")
    feature_values.append(1.0 if chest_pain_selected == "Asintomatico" else 0.0)
    feature_values.append(1.0 if chest_pain_selected == "Angina atipica" else 0.0)
    feature_values.append(1.0 if chest_pain_selected == "Dolore non anginoso" else 0.0)
    feature_values.append(1.0 if chest_pain_selected == "Angina tipica" else 0.0)

    # FASTING_BLOOD_SUGAR (fasting_blood_sugar_False, fasting_blood_sugar_True)
    fasting_blood_sugar_selected = st.radio("Glicemia a digiuno > 120 mg/dl?", ["Sì", "No"], index=1, key="fbs_input_ohe") # Sì = True, No = False
    feature_values.append(1.0 if fasting_blood_sugar_selected == "No" else 0.0) # fasting_blood_sugar_False
    feature_values.append(1.0 if fasting_blood_sugar_selected == "Sì" else 0.0) # fasting_blood_sugar_True

    # ECG_RESTING (ecg_resting_lv hypertrophy, ecg_resting_normal, ecg_resting_st-t abnormality)
    ecg_resting_selected = st.selectbox("Risultati ECG a riposo", 
                                        ["Ipertrofia ventricolare sinistra", "Normale", "Anormalità dell'onda ST-T"], 
                                        index=0, # Default: Ipertrofia ventricolare sinistra
                                        key="ecg_resting_input_ohe")
    feature_values.append(1.0 if ecg_resting_selected == "Ipertrofia ventricolare sinistra" else 0.0)
    feature_values.append(1.0 if ecg_resting_selected == "Normale" else 0.0)
    feature_values.append(1.0 if ecg_resting_selected == "Anormalità dell'onda ST-T" else 0.0)

    # EXERCISE_INDUCED_ANGINA (exercise_induced_angina_False, exercise_induced_angina_True)
    exercise_induced_angina_selected = st.radio("Angina indotta da esercizio?", ["Sì", "No"], index=1, key="eia_input_ohe") # Sì = True, No = False
    feature_values.append(1.0 if exercise_induced_angina_selected == "No" else 0.0) # exercise_induced_angina_False
    feature_values.append(1.0 if exercise_induced_angina_selected == "Sì" else 0.0) # exercise_induced_angina_True

    # ST_SLOPE_TYPE (st_slope_type_downsloping, st_slope_type_flat, st_slope_type_upsloping)
    st_slope_selected = st.selectbox("Pendenza del segmento ST al picco dell'esercizio", 
                                     ["In discesa", "Piatto", "In salita"], 
                                     index=0, # Default: In discesa
                                     key="st_slope_input_ohe")
    feature_values.append(1.0 if st_slope_selected == "In discesa" else 0.0)
    feature_values.append(1.0 if st_slope_selected == "Piatto" else 0.0)
    feature_values.append(1.0 if st_slope_selected == "In salita" else 0.0)

    # THAL_DEFECT_TYPE (thal_defect_type_fixed defect, thal_defect_type_normal, thal_defect_type_reversable defect)
    thal_defect_selected = st.selectbox("Tipo di difetto alla risonanza (Thal)", 
                                        ["Difetto fisso", "Normale", "Difetto reversibile"], 
                                        index=2, # Default: Difetto reversibile
                                        key="thal_defect_input_ohe")
    feature_values.append(1.0 if thal_defect_selected == "Difetto fisso" else 0.0)
    feature_values.append(1.0 if thal_defect_selected == "Normale" else 0.0)
    feature_values.append(1.0 if thal_defect_selected == "Difetto reversibile" else 0.0)

    st.subheader("Valori Numerici:")

    # AGE
    age = st.number_input("Età (anni)", min_value=0.0, max_value=120.0, value=45.0, step=1.0, format="%.0f", key="age_input_num")
    feature_values.append(age)

    # BLOOD_PRESSURE_RESTING
    blood_pressure_resting = st.number_input("Pressione sanguigna a riposo (mm Hg)", min_value=70.0, max_value=200.0, value=142.0, step=1.0, format="%.1f", key="bp_resting_input_num")
    feature_values.append(blood_pressure_resting)

    # CHOLESTEROL
    cholesterol = st.number_input("Colesterolo (mg/dl)", min_value=100.0, max_value=600.0, value=309.0, step=1.0, format="%.1f", key="cholesterol_input_num")
    feature_values.append(cholesterol)

    # MAX_HEART_RATE
    max_heart_rate = st.number_input("Frequenza cardiaca massima (bpm)", min_value=60.0, max_value=220.0, value=147.0, step=1.0, format="%.1f", key="max_hr_input_num")
    feature_values.append(max_heart_rate)

    # ST_DEPRESSION_EXERCISE
    st_depression_exercise = st.number_input("Depressione ST indotta da esercizio", min_value=0.0, max_value=6.0, value=0.0, step=0.1, format="%.1f", key="st_dep_input_num")
    feature_values.append(st_depression_exercise)

    # MAJOR_VESSELS_COLORED
    major_vessels_colored = st.number_input("Numero di vasi maggiori colorati (0-3)", min_value=0.0, max_value=3.0, value=3.0, step=1.0, format="%.0f", key="mvc_input_num")
    feature_values.append(major_vessels_colored)

    st.markdown("---")
    if st.button("Ottieni Diagnosi"):
        if selected_model and feature_values:
            with st.spinner("Effettuando la previsione..."):
                prediction_result = predict_data(selected_model, feature_values)

                if prediction_result and "error" not in prediction_result:
                    st.subheader("Risultati della Previsione:")
                    st.success(f"Modello utilizzato: **{prediction_result.get('model_used')}**")
                    result = 'Sano' if prediction_result.get('predictions', ['N/A'])[0] == 0 else 'Malato'

                    st.write(f"Previsione: Il paziente è **{result}**")

                    if "probabilities" in prediction_result:
                        st.markdown("---")
                        st.subheader("Probabilità:")
                        
                        # [{'0': 0.09090909361839294, '1': 0.9090909361839294}]
                        # Quindi, dobbiamo estrarre il dizionario e poi accedere alle sue chiavi.
                        probabilities_list_of_dict = prediction_result["probabilities"]
                        
                        # Estrai il dizionario di probabilità
                        probabilities_dict = probabilities_list_of_dict[0]
                        
                        prob_class_0 = probabilities_dict.get('0')
                        prob_class_1 = probabilities_dict.get('1')
                        
                        if prob_class_0 is not None and prob_class_1 is not None:
                            st.write(f"Probabilità di Classe 0 (sano): **{prob_class_0:.4f}**")
                            st.write(f"Probabilità di Classe 1 (malato): **{prob_class_1:.4f}**")
                        else:
                            st.warning("Le probabilità di Classe 0 o Classe 1 non sono state trovate nel dizionario.")
            
                elif prediction_result and "error" in prediction_result:
                    st.error(f"Errore durante la previsione: {prediction_result['error']}")
                else:
                    st.error("Nessuna risposta valida dall'API di previsione.")
        else:
            st.warning("Per favore, seleziona un modello e inserisci tutti i valori delle caratteristiche.")
