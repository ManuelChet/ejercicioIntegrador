import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Cargar datos
data = pd.read_csv("comentarios.csv")

# 2. Dividir datos
X = data['comentario']
y = data['etiqueta']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Crear pipeline: vectorizador + clasificador
modelo = Pipeline([
    ('vectorizador', CountVectorizer()),
    ('clasificador', MultinomialNB())
])

# 4. Entrenar modelo
modelo.fit(X_train, y_train)

# 5. Evaluar modelo
y_pred = modelo.predict(X_test)
print("Precisión:", accuracy_score(y_test, y_pred))
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# 6. Guardar modelo entrenado
joblib.dump(modelo, 'modelo_sentimientos.pkl')
print("✅ Modelo guardado como 'modelo_sentimientos.pkl'")
