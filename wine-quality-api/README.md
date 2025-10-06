# Wine Quality Prediction API

API para predecir la calidad de vinos basándose en características químicas.

## Autor
**Diana Erazo**

## Dataset
Wine Quality Dataset - UCI Machine Learning Repository

## Modelo
- Algoritmo: Random Forest Classifier
- Accuracy: ~85%
- Features: 11 características químicas

## Endpoints

### GET /
Información general de la API

### GET /health
Health check del servicio

### GET /example
Ejemplo de datos de entrada

### POST /predict
Realiza una predicción

**Request:**
```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}

**Request:**

{
  "quality": "low",
  "probability_low": 0.85,
  "probability_high": 0.15,
  "confidence": 0.85
}

Uso Local
pip install -r requirements.txt
python app.py

URL de Producción
https://TU-USUARIO.pythonanywhere.com

Fecha de Deployment
[05/10/2025]

---

## 🔧 Parte 4: Obtener los Modelos Pre-entrenados

### Opción A: Entrenar el Modelo (Opcional)

Si quieres entrenar tu propio modelo, crea `train_model.py`:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Cargar dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# Preparar datos
# Convertir calidad a binario: >=6 = high (1), <6 = low (0)
df['quality_binary'] = (df['quality'] >= 6).astype(int)

X = df.drop(['quality', 'quality_binary'], axis=1)
y = df['quality_binary']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluar
accuracy = model.score(X_test_scaled, y_test)
print(f"Accuracy: {accuracy:.2%}")

# Guardar modelo y scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Modelo guardado exitosamente!")

**Ejecutar**

pip install pandas scikit-learn joblib
python train_model.py

Opción B: Descargar Modelos Pre-entrenados
Los modelos ya están disponibles en el repositorio base.
