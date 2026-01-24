import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Step 1: Load and Prepare Data ---
# Load the dataset using the separator found in the sources [1]
df = pd.read_csv('data/DatenDecisionTree.csv', sep=';')

# Filter to keep only male/female records
df = df[df['SD01'].isin(['männlich', 'weiblich'])].reset_index(drop=True)

print(f"Dataset loaded with {len(df)} records (male/female only)")

# Feature: Gender only
features = ['SD01']

# Feature description mapping
feature_descriptions = {
    'SD01': 'Gender'
}

# Species mapping for more informative visualization
species_names = {
    'PA03': 'Fly Agaric (Fliegenpilz)',
    'PA05': 'Spring False Morel (Frühjahrs-Giftlorchel)',
    'PA07': 'Bitter Bolete (Gallenröhrling)',
    'PA09': 'Satan\'s Bolete (Satansröhrling)',
    'PA11': 'Red-Footed Bolete (Rotfußröhrling)',
    'PA13': 'Birch Bolete (Birken-Rotkappe)',
    'PA15': 'Chestnut Bolete (Maronenröhrling)',
    'PA17': 'Scaly Hedgehog Fungus (Bauchweh-Koralle)',
    'PA19': 'Death Cap/Green Death Cap (Grüner Knollenblätterpilz)',
    'PA21': 'Morel (Speisemorchel)',
    'PA23': 'Spruce Bolete (Fichten-Steinpilz)',
    'PA25': 'Meadow Mushroom (Wiesen-Champignon)'
}

# Define all target species codes
targets = ['PA03', 'PA05', 'PA07', 'PA09', 'PA11', 'PA13', 
           'PA15', 'PA17', 'PA19', 'PA21', 'PA23', 'PA25']

# Define which answers are correct for each species
toxic_inedible = ['PA03', 'PA05', 'PA07', 'PA09', 'PA17', 'PA19']
edible = ['PA11', 'PA13', 'PA15', 'PA21', 'PA23', 'PA25']

# --- Step 2: Data Preprocessing ---
X = df[features].copy()

# Encode categorical variables using Label Encoding
le = LabelEncoder()
X['SD01'] = le.fit_transform(X['SD01'].astype(str))

print(f"\nGender encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# --- Step 3: Model Training Loop ---
results = {}

for species in targets:
    # Create a binary outcome: 1 if correctly identified, 0 otherwise
    if species in toxic_inedible:
        y = df[species].apply(lambda x: 1 if 'giftig' in str(x).lower() else 0)
    else:
        y = df[species].apply(lambda x: 1 if 'essbar' in str(x).lower() else 0)
    
    # Count correct and incorrect identifications
    correct_count = (y == 1).sum()
    incorrect_count = (y == 0).sum()
    accuracy = correct_count / len(y) if len(y) > 0 else 0
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Decision Tree
    clf = DecisionTreeClassifier(max_depth=2, random_state=42, criterion='entropy')
    clf.fit(X_train, y_train)
    
    # Get training and test accuracy
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    
    # Store results
    results[species] = {
        'correct': correct_count,
        'incorrect': incorrect_count,
        'accuracy': accuracy,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'feature_importance': clf.feature_importances_[0],
        'model': clf,
        'feature_names': features,
        'le_classes': le.classes_,
        'X_test': X_test,
        'y_test': y_test
    }

# --- Summary Statistics ---
print("\n" + "="*80)
print("GENDER-BASED MISIDENTIFICATION ANALYSIS - SUMMARY")
print("="*80)

summary_data = []
for species in targets:
    species_label = species_names.get(species, species)
    summary_data.append({
        'Species': species,
        'Species Name': species_label,
        'Correct': results[species]['correct'],
        'Incorrect': results[species]['incorrect'],
        'Accuracy': f"{results[species]['accuracy']*100:.1f}%",
        'Test Acc': f"{results[species]['test_accuracy']*100:.1f}%"
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# --- Feature Importance ---
print("\n" + "="*80)
print("GENDER FEATURE IMPORTANCE FOR EACH SPECIES")
print("="*80)

importance_data = []
for species in targets:
    importance_data.append({
        'Species': species,
        'Species Name': species_names.get(species, species),
        'Gender Importance': f"{results[species]['feature_importance']:.4f}"
    })

importance_df = pd.DataFrame(importance_data).sort_values('Gender Importance', ascending=False, key=abs)
print(importance_df.to_string(index=False))

# --- Visualization for selected species ---
print("\n" + "="*80)
print("GENERATING DECISION TREE VISUALIZATIONS")
print("="*80)

# Create subplots for multiple species
species_to_plot = ['PA03', 'PA19', 'PA21', 'PA25']  # Examples: toxic and edible

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()

for idx, species_code in enumerate(species_to_plot):
    if idx < len(axes):
        species_label = species_names.get(species_code, species_code)
        clf = results[species_code]['model']
        
        ax = axes[idx]
        plot_tree(clf, 
                  feature_names=['Gender (männlich=0, weiblich=1)'], 
                  class_names=['Incorrect', 'Correct'], 
                  filled=True,
                  ax=ax)
        ax.set_title(f"Decision Tree: {species_label}\n(Test Accuracy: {results[species_code]['test_accuracy']*100:.1f}%)")

plt.tight_layout()
plt.savefig('gender_decision_trees.png', dpi=300, bbox_inches='tight')
print("Saved: gender_decision_trees.png")
plt.show()

print("\nAnalysis complete!")
