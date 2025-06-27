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
    Recupera i modelli disponibili dall'endpoint home della Flask API.
    Nota: L'endpoint home restituisce una stringa HTML, quindi dobbiamo parsare per trovare i nomi dei modelli.
    Un endpoint '/models_list' dedicato nella tua API Flask sarebbe più robusto e consigliato.
    """
    try:
        response = requests.get(FLASK_API_URL + '/')
        response.raise_for_status() # Lancia un'eccezione per codici di stato HTTP errati (4xx o 5xx)
        # Parsare la risposta per ottenere i nomi dei modelli
        html_content = response.text
        if "Available models:" in html_content:
            parts = html_content.split("Available models:")[1].split("<br>")[0]
            model_names_str = parts.strip()
            if model_names_str and model_names_str != 'None':
                return [m.strip() for m in model_names_str.split(',')]
            else:
                return []
        return []
    except requests.exceptions.ConnectionError:
        st.error(f"Errore di connessione all'API Flask. Assicurati che l'API sia in esecuzione su {FLASK_API_URL}")
        return []
    except Exception as e:
        st.error(f"Errore nel recupero dei modelli disponibili: {e}")
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

    # Input delle caratteristiche (ho ipotizzato 5 caratteristiche numeriche)
    # Puoi aggiungere più input o personalizzarli in base ai tuoi dati reali
    feature_values = []
    st.subheader("Caratteristiche:")
    num_features = 5 # Modifica questo valore in base al numero di feature del tuo modello

    # Creazione dinamica degli input numerici
    for i in range(num_features):
        value = st.number_input(
            f"Caratteristica {i+1}",
            min_value=0.0,  # Imposta un valore minimo ragionevole
            max_value=100.0, # Imposta un valore massimo ragionevole
            value=0.0,      # Valore predefinito
            step=0.1,       # Passo di incremento/decremento
            format="%.2f",  # Formato di visualizzazione
            key=f"feature_{i+1}" # Chiave unica per ogni input
        )
        feature_values.append(value)

    # Pulsante per la previsione
    st.markdown("---")
    if st.button("Ottieni Diagnosi"):
        if selected_model and feature_values:
            with st.spinner("Effettuando la previsione..."):
                prediction_result = predict_data(selected_model, feature_values)

                if prediction_result and "error" not in prediction_result:
                    st.subheader("Risultati della Previsione:")
                    st.success(f"Modello utilizzato: **{prediction_result.get('model_used')}**")
                    st.write(f"Previsione: **{prediction_result.get('predictions', ['N/A'])[0]}**") # Assumiamo una singola previsione

                    if "probabilities" in prediction_result:
                        st.markdown("---")
                        st.subheader("Probabilità:")
                        probabilities = prediction_result["probabilities"][0] # Assumiamo singola riga di probabilità
                        # Visualizza le probabilità in modo più leggibile
                        if len(probabilities) == 2:
                            st.write(f"Probabilità di Classe 0 (negativo): **{probabilities[0]:.4f}**")
                            st.write(f"Probabilità di Classe 1 (positivo): **{probabilities[1]:.4f}**")
                        else:
                            st.write("Probabilità per classe:")
                            for i, prob in enumerate(probabilities):
                                st.write(f"Classe {i}: **{prob:.4f}**")
                elif prediction_result and "error" in prediction_result:
                    st.error(f"Errore durante la previsione: {prediction_result['error']}")
                else:
                    st.error("Nessuna risposta valida dall'API di previsione.")
        else:
            st.warning("Per favore, seleziona un modello e inserisci tutti i valori delle caratteristiche.")

