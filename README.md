# Proyecto de Deep Learning: CNN y RNN

## CNN

<!-- La documentación del módulo CNN será agregada aquí por el compañero de equipo -->


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
