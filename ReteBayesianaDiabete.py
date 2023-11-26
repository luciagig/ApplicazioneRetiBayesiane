import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import warnings
import networkx as nx
from glob import glob
from pgmpy.models import BayesianModel
from pgmpy.estimators import PC
from pgmpy.estimators import ParameterEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import MmhcEstimator
from pgmpy.estimators import BDeuScore
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import BDeuScore, K2Score, BicScore
from pgmpy.inference import VariableElimination
warnings.filterwarnings("ignore")


def map_gender_value(gender):
    mapping = {
        'Female': 0,
        'Male': 1
    }
    return mapping.get(gender, -1)

def map_smoking_history(smoking_status):
    mapping = {
        'not current': 0,
        'former': 1,
        'No Info': 2,
        'current': 3,
        'never': 4,
        'ever': 5
    }
    return mapping.get(smoking_status, -1)

def map_bmi_category(bmi):
    if bmi < 18.5:
        return 0  
    elif 18.5 <= bmi <= 24.9:
        return 1 
    elif 25 <= bmi <= 29.9:
        return 2  
    elif bmi >= 30:
        return 3  
    else:
        return -1



dfs = pd.concat((pd.read_csv(file) for file in glob('C:\\Users\\hp\\Desktop\\CartellaTesiLuciaGigliotti\\dati\\diabete\\*.csv')),
                ignore_index=True)

dfs['gender'] = dfs['gender'].apply(map_gender_value)
dfs['smoking_history_numeric'] = dfs['smoking_history'].map(map_smoking_history)
dfs['bmi_Category'] = dfs['bmi'].apply(map_bmi_category)
dfs = dfs.drop(['gender', 'smoking_history','bmi'], axis=1)


for col in dfs.columns:
    plt.figure(figsize=(8, 5))
    value_counts=dfs[col].value_counts().sort_index()
    value_counts.plot(kind='bar', color='black')
    plt.title(f"Distribuzione di '{col}'")
    plt.xlabel(col)
    plt.ylabel("Frequenza")
    plt.show()

    

hill_climb = HillClimbSearch(dfs)
best_model = hill_climb.estimate(scoring_method=K2Score(dfs))
bayesian_model = BayesianModel(best_model.edges())
G = nx.DiGraph()
G.add_edges_from(bayesian_model.edges())
pos = nx.spring_layout(G) 
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='purple', font_size=8, font_color='black', arrowsize=10)
plt.show()

print(bayesian_model.edges())

print("Scoring functions")
bdeu=BDeuScore(dfs,equivalent_sample_size=5)
k2=K2Score(dfs)
bic=BicScore(dfs)
print(bdeu.score(bayesian_model))
print(k2.score(bayesian_model))
print(bic.score(bayesian_model))

bayesian_model.fit(dfs,estimator=BayesianEstimator)
dfs['age_group'] = pd.cut(dfs['age'], bins=[0, 20, 40, 60, 80], labels=['0-20', '21-40', '41-60', '61-80'])
infer = VariableElimination(bayesian_model)
for age_group in dfs['age_group'].unique():
    subset_data = dfs[dfs['age_group'] == age_group]
    bayesian_model.fit(subset_data,estimator=BayesianEstimator)
    cpd_bmi_diabetes=bayesian_model.get_cpds('bmi_Category')
    print(f"CPD for 'bmi_Category' in age group{age_group}:\n{cpd_bmi_diabetes}")

cpd_diabetes_HbA1c = infer.query(variables=['HbA1c_level'], evidence={'diabetes': 1})
cpd_diabetes_glucose = infer.query(variables=['blood_glucose_level'], evidence={'diabetes': 1})
print("CPD for 'HbA1c_level' given 'diabetes':\n", cpd_diabetes_HbA1c)
print("\nCPD for 'blood_glucose_level' given 'diabetes':\n", cpd_diabetes_glucose)

warnings.filterwarnings("default")
