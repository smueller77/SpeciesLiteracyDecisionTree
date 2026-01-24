import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Step 1: Load and Prepare Data ---
# Load the dataset using the separator found in the sources [1]
df = pd.read_csv('data/DatenDecisionTree.csv', sep=';')
