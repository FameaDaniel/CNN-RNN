# Proyecto de Deep Learning: CNN y RNN

## CNN

### 1. Estructura del Módulo CNN
```
cnn/
├── data/
│   ├── extract_samples.py           # Extracción de imágenes de prueba desde la caché
│   └── fashion_samples/             # Muestra de imágenes extraídas en formato PNG
├── models/
│   ├── export_cnn.py                # Script de entrenamiento y exportación de artefactos
│   └── cnn_fashion_mnist.keras      # Archivo binario con el modelo CNN entrenado
├── tests/
│   ├── baseline-accuracy/           # Evaluación de la línea base (Random Guessing)
│   ├── testing-kfold/               # Test riguroso de partición y validación de data leakage
│   ├── sesgo-varianza/              # Generador de curvas de pérdida y precisión (Underfitting/Overfitting)
│   └── testing-unitario/            # Validación matemática de reducción espacial (Pooling)
├── infer.py                         # Script de inferencia evaluando el conjunto de testing
└── cnn.py                           # Definición de arquitectura y validación cruzada (Stratified K-Fold)
```

### 2. Ejecución de Inferencia y Tests

#### A. Inferencia con el Modelo Entrenado (CLI)
Para ejecutar la inferencia utilizando el modelo CNN exportado:

```bash
# Ejecución por defecto (utiliza cnn/data/fashion_samples/ y los pesos exportados)
python cnn/infer.py
```

#### B. Tests y Utilidades
```bash
# Evaluar la línea base (Baseline de clasificación aleatoria)
python cnn/tests/baseline-accuracy/test_random_baseline.py

# Validar ausencia de fuga de información (Data Leakage) con Stratified K-Fold
python cnn/tests/testing-kfold/test_group_kfold.py

# Generar curvas de aprendizaje (Accuracy y Loss) para análisis de sesgo/varianza
python cnn/tests/sesgo-varianza/plot_metrics.py

# Ejecutar test de arquitectura (validación de dimensiones de capas convolucionales y pooling)
python cnn/tests/testing-unitario/test_architectures.py

# Extraer muestras visuales del dataset Fashion MNIST a formato físico (PNG)
python cnn/data/extract_samples.py

# Reentrenar y exportar el modelo (.keras) de forma limpia
python cnn/models/export_cnn.py
```

---

## RNN (Red Neuronal Recurrente)

### 1. Estructura del Módulo RNN
```
rnn/
├── data/
│   ├── DailyDelhiClimateTrain.csv   # Conjunto de entrenamiento (2013-2016)
│   ├── DailyDelhiClimateTest.csv    # Conjunto de evaluación (2017)
│   └── sample_30days.csv            # Muestra de 30 días para inferencia rápida
├── models/
│   ├── export_gru16.py              # Script de entrenamiento y exportación de artefactos
│   ├── gru16_delhi.keras            # Archivo binario con el modelo GRU-16 entrenado
│   └── scaler_delhi.joblib          # Escalador MinMaxScaler ajustado
├── tests/
│   ├── MAE-ingenuo/                 # Evaluación de la línea base ingenua
│   ├── testing-unitario/            # Comparativas arquitectónicas y test de delta en GRU-16
│   └── resultados-unitarios/        # Curvas de pérdida y predicciones (Tandas 1, 2 y 3)
├── infer.py                         # Script de inferencia por línea de comandos (CLI)
└── rnn.py                           # Definición de arquitecturas y pipeline central
```

### 2. Ejecución de Inferencia y Tests

#### A. Inferencia con el Modelo Entrenado (CLI)
Para ejecutar la inferencia utilizando el modelo GRU-16 exportado:

```bash
# Ejecución por defecto (utiliza rnn/data/sample_30days.csv y los pesos exportados)
python rnn/infer.py

# Ejecución con archivo de datos personalizado
python rnn/infer.py --data rnn/data/DailyDelhiClimateTest.csv
```

#### B. Tests y Benchmarks
```bash
# Evaluar la línea base ingenua (Naive Persistence Baseline)
python rnn/tests/MAE-ingenuo/test_naive_mae.py

# Ejecutar la comparativa de arquitecturas (SimpleRNN vs LSTM vs GRU)
python rnn/tests/testing-unitario/test_architectures.py

# Ejecutar el experimento de predicción de delta en GRU-16
python rnn/tests/testing-unitario/test_gru16_delta.py
```
