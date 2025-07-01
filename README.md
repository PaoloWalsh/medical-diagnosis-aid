# Medical Diagnosis Aid

Questo progetto sviluppa un sistema di supporto alla diagnosi medica basato su machine learning. Analizza dati relativi a malattie cardiache, addestra diversi modelli di classificazione e li espone tramite un'API REST per predizioni in tempo reale.

Il sistema è containerizzato con Docker per garantire una facile installazione e scalabilità.

## 🌟 Caratteristiche Principali

-   **Analisi Dati**: Analisi esplorativa e pre-elaborazione del dataset "Heart Disease UCI".
-   **Addestramento Modelli**: Addestramento e tuning di iperparametri per più modelli di classificazione:
    -   K-Nearest Neighbors (KNN)
    -   Logistic Regression
-   **Valutazione Approfondita**: Valutazione dei modelli tramite metriche come Accuracy, Recall, ROC AUC e matrici di confusione.
-   **Interpretabilità**: Analisi dell'importanza delle feature tramite SHAP (SHapley Additive exPlanations).
-   **API RESTful**: Un'API basata su Flask per servire i modelli addestrati (in formato ONNX) e ottenere predizioni.
-   **Containerizzazione**: L'intera applicazione è containerizzata con Docker per una facile distribuzione e portabilità.
-   **Analisi Comparativa**: Confronto delle performance dei modelli con predizioni generate da un LLM (GPT) e da un operatore umano.

## 📂 Struttura del Progetto

```
medical-diagnosis-aid/
├── data/                   # Contiene i dataset (raw, puliti, predizioni esterne)
├── models/                 # Contiene i modelli addestrati (.onnx) e le metriche di performance
├── notebooks/              # Jupyter Notebooks per l'analisi, l'addestramento e la valutazione
├── plots/                  # Grafici e visualizzazioni salvate
├── src/
│   ├── data_acquisition/   # Script per l'acquisizione delle predizioni del LLM e della studentessa
│   ├── flask/              # Codice sorgente dell'applicazione Flask (API)
│   └── streamlit/          # Codice sorgente della dashboard interattiva (UI)
├── docker-compose.yml      # File per orchestrare i container (API e UI)
└── README.md               # Questo file
```

## 🚀 Iniziare

Segui questi passaggi per mettere in funzione il progetto localmente.

### Prerequisiti

-   [Git](https://git-scm.com/)
-   [Docker](https://www.docker.com/get-started)
-   [Docker Compose](https://docs.docker.com/compose/install/)

### Installazione

1.  **Clona la repository:**
    ```sh
    git clone https://github.com/PaoloWalsh/medical-diagnosis-aid.git
    cd medical-diagnosis-aid
    ```

2.  **Scarica i Dataset:**
    I dataset non sono inclusi nella repository. Scaricali dai seguenti link e posizionali nella cartella `data/`:
    -   **Heart Disease Data**: [https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
        -   Salva il file come `data/heart_disease_uci.csv`.
    -   *(Nota: il notebook di pulizia dati genererà `heart_disease_clean.csv`)*
  
3.  **Avvia l'applicazione con Docker:**
    Il modo più semplice per avviare l'API è usare Docker Compose.
    ```sh
    docker-compose up --build
    ```
    Questo comando costruirà l'immagine Docker e avvierà il container. L'API sarà disponibile all'indirizzo `http://localhost:5001`.

## 🎮 Utilizzo dell'API

Una volta che l'applicazione è in esecuzione, puoi interagire con i seguenti endpoint.

#### 1. Ottenere la lista dei modelli disponibili

Restituisce i nomi dei modelli che possono essere utilizzati per le predizioni.

**Richiesta:**
```sh
curl -X POST http://localhost:5001/model_list
```

**Risposta:**
```json
{
  "available_models": [
    "K-Nearest Neighbors",
    "Logistic Regression",
  ]
}
```

#### 2. Ottenere le performance di un modello

Restituisce le metriche di performance (Accuracy, Recall, ROC AUC) per un modello specifico.

**Richiesta:**
```sh
curl -X POST http://localhost:5001/models \
     -H "Content-Type: application/json" \
     -d '{"model_name": "Logistic Regression"}'
```

**Risposta:**
```json
{
    "accuracy": "0.838",
    "model_name": "Logistic Regression",
    "recall": "0.831",
    "roc_auc": "0.908"
}
```

#### 3. Eseguire una predizione

Esegue una predizione utilizzando il modello e i dati forniti.

**Richiesta:**
```sh
curl -X POST http://localhost:5001/predict \
     -H "Content-Type: application/json" \
     -d '{
             "model_name": "Logistic Regression",
             "data": [
                 [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 45.0, 142.0, 309.0, 147.0, 0.0, 3.0]
             ]
         }'
```
*(Nota: il vettore `data` deve contenere 25 feature, corrispondenti ai dati dopo il one-hot encoding).*

**Risposta:**
```json
{
  "model_used": "Logistic Regression",
  "predictions": [
    1
  ],
  "probabilities": [
    {
      "0": 9.623169898986816e-05,
      "1": 0.9999037981033325
    }
  ]
}
```

## 📓 Notebooks di Analisi

La cartella `notebooks/` contiene i Jupyter Notebooks che documentano l'intero processo di analisi, addestramento e valutazione dei modelli. Puoi esplorarli per comprendere in dettaglio ogni fase del progetto.

Per eseguire i notebook localmente, è consigliabile creare un ambiente virtuale Python per gestire le dipendenze.

### Setup per i Notebook

1.  **Crea e attiva un ambiente virtuale:**
    Dal terminale, nella directory principale del progetto:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # Su Windows usa: venv\Scripts\activate
    ```

2.  **Installa le dipendenze:**
    ```sh
    pip install -r requirements.txt
    ```

3.  **Avvia Jupyter Lab:**
    ```sh
    jupyter notebook
    ```
    Questo aprirà un'interfaccia nel tuo browser. Naviga fino alla cartella `notebooks/` per aprire ed eseguire i file `.ipynb`.
