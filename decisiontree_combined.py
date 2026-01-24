import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# 1. Daten laden
df = pd.read_csv('data/DatenDecisionTree.csv', sep=';')

# 2. Berechnung des Expertise-Scores (0 bis 12 Punkte)
# Definition der korrekten Antworten laut Confusion Matrix [1]
toxic_species = ['PA03', 'PA05', 'PA07', 'PA09', 'PA17', 'PA19']
edible_species = ['PA11', 'PA13', 'PA15', 'PA21', 'PA23', 'PA25']

def calculate_id_score(row):
    score = 0
    # Punkte für korrekt erkannte Giftpilze (Antwort enthält "giftig")
    for s in toxic_species:
        if 'giftig' in str(row[s]).lower(): score += 1
    # Punkte für korrekt erkannte Speisepilze (Antwort enthält "essbar")
    for s in edible_species:
        if 'essbar' in str(row[s]).lower(): score += 1
    return score

df['Total_Score'] = df.apply(calculate_id_score, axis=1)

# 3. Definition der Zielvariable (Target)
# Der Durchschnitt liegt bei 2,0 [3]. Wir definieren "Expertise" als Score > 2.
y = (df['Total_Score'] > 2).astype(int)

# 4. Features definieren und aufbereiten
# Auswahl der Faktoren aus Ihren Anforderungen
feature_cols = ['SD05', 'SD01', 'SD02_01', 'PG02', 'PG01', 'PW01']
X_raw = df[feature_cols].copy()

# One-Hot Encoding für kategoriale Variablen
X = pd.get_dummies(X_raw, drop_first=True)

# 5. Expertise-Baum trainieren
clf = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=42)
clf.fit(X, y)

# 6. Visualisierung des Expertise-Baums
plt.figure(figsize=(20,10))
plot_tree(clf, 
          feature_names=X.columns, 
          class_names=['Geringes Wissen', 'Hohe Expertise'], 
          filled=True, 
          rounded=True, 
          fontsize=12)
plt.title("Expertise-Baum: Faktoren für allgemeine Pilz-Artenkenntnis (Zusammenfassung)")
plt.show()

# 7. Wichtigste Faktoren für allgemeines Wissen ausgeben
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Ranking der Faktoren für allgemeine Pilzkompetenz:")
print(importances.head(5))