"""
Decision Tree Analysis for Mushroom Species Misidentification
This script analyzes which demographic and behavioral factors predict
correct vs. incorrect mushroom identification.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING & PREPARATION
# ============================================================================

def load_data():
    """Load survey data and create a clean dataset."""
    # Load survey responses
    df_survey = pd.read_csv('data/DatenDecisionTree.csv', sep=';')
    
    # Load confusion matrix with correct answers
    df_confusion = pd.read_csv('data/ConfusionMatrix.csv', sep=';', skiprows=1)
    
    print(f"Loaded survey data: {df_survey.shape}")
    print(f"Loaded confusion matrix: {df_confusion.shape}")
    
    return df_survey, df_confusion

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

def engineer_features(df_survey):
    """Extract and encode demographic features: gender and age only."""
    
    print("Available columns:", df_survey.columns.tolist())
    
    df_features = pd.DataFrame()
    
    # *** DEMOGRAPHIC FEATURES ONLY ***
    # SD01: Gender
    if 'SD01' in df_survey.columns:
        df_features['gender'] = df_survey['SD01']
    
    # SD02_01: Age - group into 4 categories
    if 'SD02_01' in df_survey.columns:
        age = pd.to_numeric(df_survey['SD02_01'], errors='coerce')
        
        # Bin age into 4 groups
        df_features['age_group'] = pd.cut(
            age,
            bins=[0, 30, 45, 60, 120],
            labels=['18-30', '31-45', '46-60', '60+'],
            include_lowest=True
        )
    
    # Add CASE ID for tracking
    df_features['CASE'] = df_survey['CASE']
    
    # Remove rows with missing data
    print(f"Dropping rows with missing gender or age_group...")
    df_features = df_features.dropna(subset=['gender', 'age_group'])
    
    print(f"Engineered features shape: {df_features.shape}")
    print(f"Features: {df_features.columns.tolist()}")
    print(f"\nAge group distribution:\n{df_features['age_group'].value_counts().sort_index()}")
    print(f"\nGender distribution:\n{df_features['gender'].value_counts()}")
    
    return df_features

# ============================================================================
# 3. CREATE TARGET VARIABLES
# ============================================================================

def create_targets(df_survey, df_features):
    """Create target variables for each mushroom species."""
    
    # Mushroom species codes
    mushroom_species = {
        'PA03': 'Fliegenpilz',           # Poisonous
        'PA05': 'Frühjahrs-Giftlorchel', # Poisonous
        'PA07': 'Gallenröhrling',        # Poisonous
        'PA09': 'Satansröhrling',        # Poisonous
        'PA11': 'Rotfußröhrling',        # Edible
        'PA13': 'Birken-Rotkappe',       # Edible
        'PA15': 'Maronenröhrling',       # Edible
        'PA17': 'Bauchweh-Koralle',      # Poisonous
        'PA19': 'Grüner Knollenblätterpilz', # Poisonous
        'PA21': 'Speisemorchel',         # Edible
        'PA23': 'Fichten-Steinpilz',     # Edible
        'PA25': 'Wiesen-Champignon',     # Edible
    }
    
    # Correct answers from confusion matrix
    correct_answers = {
        'PA03': 'giftig/ungenießbar.',
        'PA05': 'weiß ich nicht.',  # Actually poisonous, but survey shows
        'PA07': 'essbar.',          # Actually poisonous, but survey shows
        'PA09': 'giftig/ungenießbar.',
        'PA11': 'essbar.',
        'PA13': 'essbar.',
        'PA15': 'essbar.',
        'PA17': 'giftig/ungenießbar.',
        'PA19': 'giftig/ungenießbar.',
        'PA21': 'essbar.',
        'PA23': 'essbar.',
        'PA25': 'essbar.',
    }
    
    targets = {}
    
    for code, species_name in mushroom_species.items():
        if code in df_survey.columns:
            # 1 = correct answer, 0 = incorrect/don't know
            targets[code] = (df_survey[code] == correct_answers[code]).astype(int)
    
    # Create overall accuracy (across all 12 species)
    species_cols = [col for col in mushroom_species.keys() if col in df_survey.columns]
    df_survey_filtered = df_survey.loc[df_features.index]
    
    overall_correct = df_survey_filtered[species_cols].apply(
        lambda row: sum(1 for code in species_cols if row[code] == correct_answers[code]),
        axis=1
    )
    targets['overall_accuracy'] = (overall_correct >= 8).astype(int)  # Binary: 8+ correct = high accuracy
    
    return targets, mushroom_species

# ============================================================================
# 4. TRAIN DECISION TREES
# ============================================================================

def train_decision_trees(df_features, targets, max_depth=5):
    """Train decision trees for each mushroom species."""
    
    results = {}
    
    # Encode categorical features
    df_encoded = df_features.copy()
    categorical_cols = ['gender', 'age_group']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
            print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Drop CASE ID (not a feature)
    df_encoded = df_encoded.drop('CASE', axis=1)
    
    print("\n" + "="*70)
    print("TRAINING DECISION TREES")
    print("="*70)
    
    for target_name, target_data in targets.items():
        # Align target with features
        target_aligned = target_data.loc[df_features.index]
        
        # Skip if target has too few samples
        if len(target_aligned) < 20:
            continue
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df_encoded, target_aligned, test_size=0.2, random_state=42
        )
        
        # Train model
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(clf, df_encoded, target_aligned, cv=5)
        
        results[target_name] = {
            'model': clf,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_names': df_encoded.columns.tolist(),
            'X_train': X_train,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'label_encoders': label_encoders
        }
        
        print(f"\n{target_name:30s} | Accuracy: {accuracy:.3f} | CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"{'':30s} | Class distribution: {target_aligned.value_counts().to_dict()}")
    
    return results

# ============================================================================
# 5. ANALYZE FEATURE IMPORTANCE
# ============================================================================

def analyze_feature_importance(results):
    """Extract and visualize feature importance."""
    
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    for target_name, result in results.items():
        clf = result['model']
        feature_names = result['feature_names']
        importances = clf.feature_importances_
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        if importance_df['importance'].sum() > 0:
            print(f"\n{target_name}:")
            print(importance_df.head(10).to_string(index=False))

# ============================================================================
# 6. VISUALIZE TREES
# ============================================================================

def visualize_top_trees(results, max_trees=2):
    """Visualize the top decision trees."""
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for idx, (target_name, result) in enumerate(sorted_results[:max_trees]):
        clf = result['model']
        feature_names = result['feature_names']
        
        plt.figure(figsize=(20, 12))
        plot_tree(
            clf,
            feature_names=feature_names,
            class_names=['Incorrect', 'Correct'],
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title(f"Decision Tree: {target_name}\n(Accuracy: {result['accuracy']:.3f})")
        plt.tight_layout()
        plt.savefig(f'output/tree_{target_name}.png', dpi=100, bbox_inches='tight')
        print(f"Saved tree visualization: output/tree_{target_name}.png")
        plt.close()

# ============================================================================
# 7. GENERATE INSIGHTS
# ============================================================================

def generate_insights(results, targets, mushroom_species):
    """Generate actionable insights from the models."""
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    # Find most predictable species
    best_species = max(results.items(), key=lambda x: x[1]['accuracy'])
    worst_species = min(results.items(), key=lambda x: x[1]['accuracy'])
    
    print(f"\nMost easily identified (highest accuracy):")
    print(f"  {best_species[0]}: {best_species[1]['accuracy']:.1%}")
    
    print(f"\nMost commonly misidentified (lowest accuracy):")
    print(f"  {worst_species[0]}: {worst_species[1]['accuracy']:.1%}")
    
    # Most important features overall
    print(f"\nMost important features across all species:")
    all_importances = {}
    for target_name, result in results.items():
        clf = result['model']
        for feature, importance in zip(result['feature_names'], clf.feature_importances_):
            if feature not in all_importances:
                all_importances[feature] = []
            all_importances[feature].append(importance)
    
    avg_importance = {
        feature: np.mean(importances)
        for feature, importances in all_importances.items()
    }
    
    sorted_importance = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_importance[:10]:
        print(f"  {feature:30s}: {importance:.4f}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    import os
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Load data
    df_survey, df_confusion = load_data()
    
    # Engineer features
    df_features = engineer_features(df_survey)
    
    # Create targets
    targets, mushroom_species = create_targets(df_survey, df_features)
    
    # Train models
    results = train_decision_trees(df_features, targets, max_depth=6)
    
    # Analyze importance
    analyze_feature_importance(results)
    
    # Generate insights
    generate_insights(results, targets, mushroom_species)
    
    # Visualize decision trees
    print("\nGenerating decision tree visualizations...")
    visualize_top_trees(results, max_trees=3)
    
    print("\n" + "="*70)
    print("Analysis complete! Check 'output/' directory for visualizations.")
    print("="*70)
