# Proyecto de Deep Learning: CNN y RNN

## CNN

<!-- La documentación del módulo CNN será agregada aquí por el compañero de equipo -->


---

## RNN (Red Neuronal Recurrente)

### 1. Descripción del Problema y Dataset
Este módulo implementa una red neuronal recurrente en Keras para resolver el problema de pronóstico secuencial multivariado de temperatura a partir de mediciones climáticas diarias de Delhi (`rnn/data/DailyDelhiClimateTrain.csv` y `rnn/data/DailyDelhiClimateTest.csv`).

- **Variables de entrada (4):** `meantemp` (temperatura media), `humidity` (humedad), `wind_speed` (velocidad del viento), `meanpressure` (presión atmosférica).
- **Variable objetivo:** Tasa de cambio diaria de temperatura ($\Delta y_{t+1} = y_{t+1} - y_t$) para predecir la temperatura media del día siguiente ($\hat{y}_{t+1} = y_t + \widehat{\Delta y}_{t+1}$).
- **Calidad de datos y saneamiento:** Los valores atípicos (*outliers*) de presión fuera del rango físico atmosférico plausible [950, 1.050 hPa] fueron saneados mediante interpolación lineal temporal, preservando estrictamente el intervalo uniforme $\Delta t = 1 \text{ día}$.
- **Partición de datos (*Time Split*):** Partición cronológica estricta (Entrenamiento: 2013-2016, Evaluación: 2017) ajustando el escalador `MinMaxScaler` únicamente con el conjunto de entrenamiento para garantizar que el futuro no contamine el pasado (*data leakage*).

### 2. Arquitectura Seleccionada: GRU-16
Tras realizar un análisis comparativo formal de 12 experimentos (SimpleRNN, LSTM, GRU-16 y GRU-8 evaluadas en tandas de 20, 50 y 100 épocas), la arquitectura **GRU con 16 unidades** demostró ser el balance óptimo entre sesgo y varianza:
- **Mecanismo de compuertas:** Las compuertas de reseteo ($r_t$) y actualización ($z_t$) mitigan la desaparición del gradiente durante BPTT (*Backpropagation Through Time*) mediante una autopista aditiva de memoria.
- **Eficiencia de parámetros:** Posee 1.073 parámetros (frente a 4.769 en LSTM-32 y 1.217 en SimpleRNN-32), evitando el sobreajuste (*overfitting*) en datasets de tamaño moderado.
- **Estrategia de optimización:** Optimizador Adam con `ReduceLROnPlateau` (factor=0,5, paciencia=5) para ajuste fino del mínimo local y `EarlyStopping` (paciencia=15, restaurando los mejores pesos alcanzados).
- **Rendimiento:** MAE reconstruido en Test de **1,22 °C**, superando a la línea base ingenua de persistencia (*Naive Baseline* de $1,23 \text{ °C}$) con un *Skill Score* positivo de **+0,52%**.

### 3. Estructura del Módulo RNN
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

### 4. Ejecución de Inferencia (Línea de Comandos)
Para ejecutar la inferencia utilizando el modelo GRU-16 exportado:

```bash
# Ejecución por defecto (utiliza rnn/data/sample_30days.csv y los pesos exportados)
python rnn/infer.py

# Ejecución con archivo de datos personalizado
python rnn/infer.py --data rnn/data/DailyDelhiClimateTest.csv
```

### 5. Ejecución de Tests y Benchmarks
```bash
# Evaluar la línea base ingenua (Naive Persistence Baseline)
python rnn/tests/MAE-ingenuo/test_naive_mae.py

# Ejecutar la comparativa de arquitecturas (SimpleRNN vs LSTM vs GRU)
python rnn/tests/testing-unitario/test_architectures.py

# Ejecutar el experimento de predicción de delta en GRU-16
python rnn/tests/testing-unitario/test_gru16_delta.py
```
