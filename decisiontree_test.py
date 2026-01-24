import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Daten laden
df = pd.read_csv('data/DatenDecisionTree.csv', sep=';')

# 2. Features definieren (Ihre Auswahl)
feature_cols = ['SD05', 'SD01', 'SD02_01', 'SD03_09', 'PG02', 'PG01', 'PW01']
X_raw = df[feature_cols].copy()

# Vorverarbeitung: Kategorische Texte in Zahlen umwandeln (One-Hot Encoding)
X = pd.get_dummies(X_raw, drop_first=True)

# 3. Zielvariable definieren (Beispiel: Knollenblätterpilz PA19)
# Laut Confusion Matrix: Korrekt = giftig, Alles andere = falsch/unsicher [1]
y = df['PA19'].apply(lambda x: 1 if str(x).startswith('giftig') else 0)

# 4. Decision Tree Modell trainieren
# Wir begrenzen die Tiefe, um die "Erklärbarkeit" zu maximieren
clf = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=42)
clf.fit(X, y)

# 5. Visualisierung: Faktoren der Fehlidentifikation erklären
plt.figure(figsize=(20,10))
plot_tree(clf, 
          feature_names=X.columns, 
          class_names=['Falsch/Unsicher', 'Korrekt'], 
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title("Faktorenanalyse für die Identifikation des Knollenblätterpilzes (PA19)")
plt.show()

# 6. Wichtigste Faktoren ausgeben
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top Faktoren für die Identifikationsleistung:")
print(importances.head(5))