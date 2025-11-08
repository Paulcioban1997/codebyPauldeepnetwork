# cnn_super_simple.py
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import keras
from keras.callbacks import EarlyStopping

# --- 1. Données simples ---
noms = np.array([['Paul'], ['Guillaume'], ['Mathias'], ['Ashiana'],
                 ['Annie'], ['John'], ['Xavier'], ['Reda']])

professions = np.array([['Programmeur'], ['Docteur'], ['Camionneur'], ['Infirmière'],
                        ['Designer'], ['Professeur'], ['Mécanicien'], ['Analyste']])

salaires = np.array([[100000], [120000], [75000], [80000],
                     [95000], [85000], [90000], [99000]])

# --- 2. Encodage des textes ---
encoder = OneHotEncoder(sparse_output=False)
X = encoder.fit_transform(np.concatenate((noms, professions), axis=1))

# Normalisation du salaire (entre 0 et 1)
scaler = MinMaxScaler()
y = scaler.fit_transform(salaires)

# --- 3. Modèle CNN simple ---
model = keras.models.Sequential([
    keras.layers.Input(shape=(X.shape[1], 1, 1)),     # entrée simple
    keras.layers.Conv2D(8, (1, 1), activation='relu'), # couche de convolution
    keras.layers.MaxPooling2D(pool_size=(1, 1)),       # couche de pooling
    keras.layers.Flatten(),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='linear')         # sortie = salaire
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')

# --- 4. Propagation avant et rétropropagation ---
# Le modèle apprend sur tout le dataset (rétropropagation automatique)
early_stop = EarlyStopping(monitor='loss', patience=25, restore_best_weights=True)
model.fit(X.reshape(X.shape[0], X.shape[1], 1, 1), y, epochs=60, verbose=1, callbacks=[early_stop])

# --- 5. Prédiction : salaire de Paul ---
X_paul = encoder.transform(np.concatenate(([['Paul']], [['Programmeur']]), axis=1))
pred = model.predict(X_paul.reshape(X_paul.shape[0], X_paul.shape[1], 1, 1))
salaire_pred = scaler.inverse_transform(pred)[0][0]

print(f"\nPrédiction du salaire de Paul : {salaire_pred:.2f} $")

# --- 6. Précision très simple (exactitude) ---
pred_train_scaled = model.predict(X.reshape(X.shape[0], X.shape[1], 1, 1), verbose=0)
pred_train = scaler.inverse_transform(pred_train_scaled)
y_true = salaires

# Précision avec tolérance (± 10 000 $)
tol = 10000
pourcent = int((abs(pred_train - y_true) <= tol).mean() * 100)
print("Précision (±10 000 $) :", pourcent, "%")   #Precision sure a 100 % si au dessus de 10 000 $
