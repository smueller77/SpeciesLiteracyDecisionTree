import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Daten laden
df = pd.read_csv('data/DatenDecisionTree.csv', sep=';')

# 2. Definition der korrekten Bestimmungen laut Confusion Matrix [3]
toxic_species = ['PA03', 'PA05', 'PA07', 'PA09', 'PA17', 'PA19']
edible_species = ['PA11', 'PA13', 'PA15', 'PA21', 'PA23', 'PA25']

def check_determination_success(row):
    correct_count = 0
    # Korrekte Bestimmung von Giftpilzen
    for s in toxic_species:
        if 'giftig' in str(row[s]).lower(): correct_count += 1
    # Korrekte Bestimmung von Speisepilzen
    for s in edible_species:
        if 'essbar' in str(row[s]).lower(): correct_count += 1
    
    # Zielvariable: 1 wenn über dem Durchschnitt von 3.0 korrekt bestimmt wurde [1]
    return 1 if correct_count > 3 else 0

# 3. Zielvariable (y) und Merkmale (X) definieren
y = df.apply(check_determination_success, axis=1)

# Auswahl der Prädiktoren basierend auf der Regressionsanalyse der Studie [4]
feature_cols = ['SD05', 'SD01', 'SD02_01', 'PG02', 'PG01', 'PW01']
X = pd.get_dummies(df[feature_cols], drop_first=True)

# 4. Decision Tree Modell trainieren
clf = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=42)
clf.fit(X, y)

# 5. Visualisierung
plt.figure(figsize=(20,10))
plot_tree(clf, 
          feature_names=X.columns, 
          class_names=['Falsch/Unsicher (<=3)', 'Korrekt bestimmt (>3)'], 
          filled=True, rounded=True, fontsize=10)
plt.title("Faktorenanalyse für die Bestimmungs-Kompetenz (Speisewert aller 12 Pilzarten)")
plt.show()
print("")