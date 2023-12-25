from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

# Ajustamos un Random Forest
modelo_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
modelo_rf.fit(X_train, y_train)

# Realizamos predicciones sobre el conjunto de test
predicciones_rf = modelo_rf.predict(X_test)

# Calculamos la matriz de confusión
matriz_confusion = confusion_matrix(y_test, predicciones_rf)
print(f'Matriz de Confusión:\n{matriz_confusion}')

# Calculamos el F1-Score
f1 = f1_score(y_test, predicciones_rf)
print(f'F1-Score: {f1}')

# Comparamos F1-Score con el Accuracy
accuracy_rf = accuracy_score(y_test, predicciones_rf)
print(f'Accuracy del Random Forest: {accuracy_rf}')

# El F1-Score es útil para evaluar modelos en conjuntos de datos desbalanceados, 
# ya que tiene en cuenta tanto falsos positivos como falsos negativos, 
# proporcionando una medida más equilibrada del rendimiento del modelo.
