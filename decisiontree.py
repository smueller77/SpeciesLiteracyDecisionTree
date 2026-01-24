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

# Define features (Predictors) requested [2-4]
features = [
    'SD05',    # Wohnort (längster Zeitraum)
    'SD01',    # Geschlecht
    'SD02_01', # Alter
    'SD03_09', # Bildungsabschluss
    'PG01',    # Essverhalten Pilze
    'PG02',    # Sammelverhalten Pilze
    'PG03',    # Gründe für das Nichtsammeln
    'PW01'     # Behandlung im schulischen Kontext
]

# Feature description mapping for better readability
feature_descriptions = {
    'SD05': 'Residence Location (Longest Period)',
    'SD01': 'Gender',
    'SD02_01': 'Age',
    'SD03_09': 'Education Level',
    'PG01': 'Mushroom Consumption Habits',
    'PG02': 'Mushroom Foraging Behavior',
    'PG03': 'Reasons for Not Foraging',
    'PW01': 'School Context Experience'
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

# Define all target species codes [1, 5]
targets = ['PA03', 'PA05', 'PA07', 'PA09', 'PA11', 'PA13', 
           'PA15', 'PA17', 'PA19', 'PA21', 'PA23', 'PA25']

# Define which answers are correct for each species based on source categories [5]
toxic_inedible = ['PA03', 'PA05', 'PA07', 'PA09', 'PA17', 'PA19']
edible = ['PA11', 'PA13', 'PA15', 'PA21', 'PA23', 'PA25']

# --- Step 2: Data Preprocessing ---
X = df[features].copy()

# Handle missing values often coded as -9 in these datasets [6]
X = X.replace('-9', pd.NA).fillna('Unknown')

# Convert metric columns to numeric
X['SD02_01'] = pd.to_numeric(X['SD02_01'], errors='coerce').fillna(0)

# Encode categorical variables using Label Encoding and store mappings
le_dict = {}  # Store label encoders for each column
categorical_cols = ['SD05', 'SD01', 'SD03_09', 'PG01', 'PG02', 'PG03', 'PW01']
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le  # Store the encoder for later use

# --- Step 3 & 4: Model Training and Factor Importance Loop ---
# We loop through every species to identify its specific drivers of misidentification
results = {}

# Function to create human-readable decision rules from tree
def get_tree_rules(tree_model, feature_names, le_mappings, feature_descriptions):
    """Extract and format decision tree rules in human-readable text"""
    tree = tree_model.tree_
    feature_name = [
        feature_descriptions[feature_names[i]] if i != -2 else "undefined!"
        for i in tree.feature
    ]
    
    def recurse(node, depth, rules_text):
        indent = "  " * depth
        
        # Check if node index is valid
        if node >= len(tree.feature):
            return rules_text
            
        if tree.feature[node] != -2:  # Not a leaf node
            name = feature_name[node]
            threshold = tree.threshold[node]
            feature_col = feature_names[tree.feature[node]]
            
            # Try to decode categorical values
            if feature_col in le_mappings:
                le = le_mappings[feature_col]
                try:
                    # Get the value that corresponds to this threshold
                    class_name = le.classes_[int(threshold)]
                    rules_text += f"{indent}if {name} == '{class_name}':\n"
                except (IndexError, ValueError):
                    rules_text += f"{indent}if {name} <= {threshold:.2f}:\n"
            else:
                # Numeric column (Age)
                rules_text += f"{indent}if {name} <= {threshold:.2f}:\n"
            
            rules_text = recurse(2 * node + 1, depth + 1, rules_text)
            rules_text += f"{indent}else:\n"
            rules_text = recurse(2 * node + 2, depth + 1, rules_text)
        else:  # Leaf node
            # Get class distribution
            value = tree.value[node][0]
            class_pred = "Correct" if tree.value[node][0][1] > tree.value[node][0][0] else "Incorrect"
            rules_text += f"{indent}Prediction: {class_pred} (samples: {int(tree.n_node_samples[node])})\n"
        
        return rules_text
    
    return recurse(0, 0, "")

for species in targets:
    # Create a binary outcome: 1 if correctly identified, 0 otherwise [5, 7]
    if species in toxic_inedible:
        y = df[species].apply(lambda x: 1 if 'giftig' in str(x).lower() else 0)
    else:
        y = df[species].apply(lambda x: 1 if 'essbar' in str(x).lower() else 0)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Decision Tree [Source-derived plan]
    clf = DecisionTreeClassifier(max_depth=3, random_state=42, criterion='entropy')
    clf.fit(X_train, y_train)
    
    # Store feature importance
    results[species] = pd.Series(clf.feature_importances_, index=features)

# --- Visualization for a specific critical species (e.g., Death Cap PA19) ---
# The Death Cap has a 67.85% misidentification/unknown rate [5]
species_code = 'PA19'
species_label = species_names.get(species_code, species_code)

# Print human-readable decision rules
print("\n" + "="*80)
print(f"DECISION RULES FOR {species_label}")
print("="*80)
rules = get_tree_rules(clf, features, le_dict, feature_descriptions)
print(rules)

plt.figure(figsize=(20,10))
feature_labels = [feature_descriptions[f] for f in features]
plot_tree(clf, feature_names=feature_labels, class_names=['Incorrect', 'Correct'], filled=True)
plt.title(f"Decision Tree for {species_label} Identification Factors\n(Key Factors Determining Correct Species Recognition)")
plt.tight_layout()
plt.show()

# Print average importance across all species to find general drivers
importance_df = pd.DataFrame(results).mean(axis=1).sort_values(ascending=False)
importance_with_descriptions = pd.DataFrame({
    'Technical Name': importance_df.index,
    'Description': importance_df.index.map(feature_descriptions),
    'Average Importance': importance_df.values
})
print("Average Factor Importance Across All Species:")
print(importance_with_descriptions.to_string(index=False))