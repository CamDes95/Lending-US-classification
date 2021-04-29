#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import classification_report_imbalanced
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


df = pd.read_csv("lending.csv", low_memory = False)
df_1 = df.iloc[:40000,:]

print(df_1.info(),"\n")
print(df_1.describe(),"\n")



#### Mise en forme du dataset
df_loan = df_1[["acc_now_delinq", "addr_state","annual_inc", "application_type","avg_cur_bal","bc_util","delinq_2yrs",
       "dti", "emp_length", "emp_title", "fico_range_high", "fico_range_low", "fico_range_low", "funded_amnt",
       "funded_amnt_inv", "grade", "home_ownership", "id", "inq_fi", "inq_last_6mths", "installment", "int_rate",
       "issue_d", "last_pymnt_amnt", "last_pymnt_d","loan_amnt", "loan_status", "mort_acc", "mths_since_recent_bc",
       "mths_since_recent_inq", "next_pymnt_d", "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts",
       "open_acc", "purpose", "term", "title", "tot_cur_bal", "tot_hi_cred_lim", "total_acc", "total_bal_ex_mort",
       "total_bal_il", "total_bc_limit", "total_cu_tl", "total_pymnt", "total_pymnt_inv", "total_rec_int", "total_rec_late_fee",
       "total_rec_prncp", "verification_status"]]


# Suppression des colonnes avec trop de valeurs manquantes ou pas d'intérêt
df = df_loan.drop(df_loan[["next_pymnt_d", "total_bal_il","inq_fi","total_cu_tl",
                   "mths_since_recent_inq","mths_since_recent_bc", "emp_title", "last_pymnt_d",
                   "addr_state", "acc_now_delinq", "application_type","fico_range_high","fico_range_low",
                   "home_ownership"]], axis = 1)

print(df.isna().sum())

# Remplacement des Nans par mode 
df["title"] = df.fillna(df["title"].mode())
df["emp_length"] = df.fillna(df["emp_length"].mode())
df["inq_last_6mths"] = df.fillna(df["inq_last_6mths"].mode())

# Remplacement des Nans par moyenne
#df["bc_util"] = df.fillna(df["bc_util"].mean())
df["dti"] = df.fillna(df["dti"].mean())

df = df.dropna()
print(df.isna().sum())



print(df["loan_status"].value_counts())
print(df["loan_status"].value_counts(normalize=True))

# Statuts considérés commes bons : Fully Paid, Current 
    # Le prêt est complétement remboursé ou pas de retards sur les remboursement (à jour)
    # Ces clients sont plus susceptibles de rembourser leur prêt dans les temps
    
# Statuts considérés comme mauvais : Charged Off, Default, Late(16-30days), Late(31-120days)
    # Tout ces statuts représentent un défaut de paiement ou non respect des délais de remboursement
    # Ces clients sont susceptibles de ne pas rembourser le prêt dans les temps impartis


# Définition variable cible
good_loan = df[(df["loan_status"] == "Fully Paid") + (df["loan_status"] == "Current")]
good_loan["loan_status"] = 1

bad_loan = df[(df["loan_status"] == "Charged Off") +
              (df["loan_status"] =="Default") +
              (df["loan_status"] =="Late (16-30 days)") +
              (df["loan_status"] =="Late (31-120 days)") +
              (df["loan_status"] =="In Grace Period")]
bad_loan["loan_status"] = 0

# Assemblage des deux df en target 
df_loan = pd.concat([good_loan, bad_loan], axis = 0)
target = df_loan["loan_status"]

print(target.value_counts(normalize = True))

# Classe 1 est majoritaire, => désequilibre des classes ! utilisation UnderSampling



# Proportions des notes dans le dataset 
df_loan["grade"].value_counts()

# grouper par status
df_1 = df_loan["loan_status"].groupby(df_loan["grade"])

print(df_1.value_counts(), "\n")
# Les prêts notés A présentent peu de risques, ils sont trés peu non remboursés
# B sont majoritairement remboursés mais environ 10% ne le sont pas
# C et D sont majoritairement remboursés mais partie conséquente non remboursée 
# Prêts E, F, G à risque (environ la moitié non remboursés)


# Définition des variables explicatives
data = df.drop(["loan_status","funded_amnt", "funded_amnt_inv", "last_pymnt_amnt", "total_rec_int", 
                    "total_rec_late_fee", "total_rec_prncp", "total_pymnt","total_pymnt_inv", "issue_d",
                    "verification_status","purpose", "term", "emp_length", "title"], axis = 1)
print(data.head())
print(data.info())



# transformations des var non num en variables numériques
from sklearn.preprocessing import LabelEncoder

data["id"] = data["id"].astype(int)
data["dti"] = data["dti"].astype(int)
data["bc_util"] = data["bc_util"].astype(int)
data["inq_last_6mths"] = data["inq_last_6mths"].astype(int)

data["grade"] = LabelEncoder().fit_transform(data["grade"]).astype(int)

data = data.drop("id", axis = 1)
print(data.grade.value_counts())



# Séparation des données en train, valid, test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(data, target, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)




from sklearn.preprocessing import MinMaxScaler

mmscaler = MinMaxScaler().fit(X_train)
X_train_scaled = mmscaler.transform(X_train)

mmscaler = MinMaxScaler().fit(X_test)
X_test_scaled = mmscaler.transform(X_test)

mmscaler = MinMaxScaler().fit(X_val)
X_val_scaled = mmscaler.transform(X_val)


###### Première approche : utilisation d'un ensemble de validation ######
# Entrainement et recherche HP sur train
# Evaluation sur val
# Evaluation sur test


# entraînement LR sur X_train
clf_lr = LogisticRegression(random_state=2, max_iter=1000)
clf_lr.fit(X_train_scaled, y_train) 

# grille de recherche lr sur X_val
param_lr = {"solver":["liblinear", 'lbfgs'], "C": np.logspace(-4, 2, 9), "max_iter":[1000] }
grid_clf_lr = GridSearchCV(estimator = clf_lr, param_grid = param_lr)
grille_lr = grid_clf_lr.fit(X_train_scaled, y_train)
print(pd.DataFrame.from_dict(grille_lr.cv_results_).loc[:,["params","mean_test_score"]],"\n")

print("Meilleur hyperparamètres de LR : {} ".format(grid_clf_lr.best_params_))



# Prédiction sur les données de validation
lr_final = LogisticRegression(C=3.1622776601683795, max_iter=1000, solver="lbfgs")

lr_final.fit(X_train_scaled, y_train)
y_pred = lr_final.predict(X_val_scaled)

print(pd.crosstab(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Taux de bonnes prédictions LR  :", lr_final.score(X_val_scaled, y_val)*100, "%")



# Prédiction sur les données tests 
lr_final.fit(X_train_scaled, y_train)
y_pred = lr_final.predict(X_test_scaled)

print(pd.crosstab(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Taux de bonnes prédictions LR  :", lr_final.score(X_test_scaled, y_test)*100, "%")



# Mauvaise prédiction de la classe minoritaire 0 et surprédiction de la classe majoritaire 1
# Algorithmes classiques baisés par la surprésence de classe majoritaire 1
# Traitement possible de la classe minoritaire comme des valeurs abérrantes



###### Deuxième approche : UnderSampling ######
rus = RandomUnderSampler()
(X_ru, y_ru) = rus.fit_resample(X_train_scaled, y_train)


# Entraînement LR sur les 2 ensembles resamplés
clf_lr = LogisticRegression(random_state=2, max_iter=1000)
clf_lr.fit(X_ru, y_ru) 

# grille de recherche lr sur X_val
param_lr = {"solver":["liblinear", 'lbfgs'], "C": np.logspace(-4, 2, 9), "max_iter":[1000] }
grid_clf_lr = GridSearchCV(estimator = clf_lr, param_grid = param_lr)
grille_lr = grid_clf_lr.fit(X_train_scaled, y_train)
print(pd.DataFrame.from_dict(grille_lr.cv_results_).loc[:,["params","mean_test_score"]],"\n")

print("Meilleur hyperparamètres de LR : {} ".format(grid_clf_lr.best_params_))


# Prédiction sur les données tests
lr_final = LogisticRegression(C=3.1622776601683795, max_iter=1000, solver="lbfgs")

lr_final.fit(X_ru, y_ru)
y_pred = lr_final.predict(X_test_scaled)

print(pd.crosstab(y_test, y_pred))
print(classification_report_imbalanced(y_test, y_pred))
print("Taux de bonnes prédictions LR  :", lr_final.score(X_test_scaled, y_test)*100, "%")


# meilleur prédiction de la classe minoritaire 0 que la première approche!
# Trés peu de FP mais augmentation des FN
# 2303 obs classifiées 0 alors que 1
# 24 obs classifiées 1 alors que 0

# score f1 faible => performances de rappel et précision mal équilibré 
# (bonne détection de la classe 0 mais inclut des observations de classe 1)


###### Troisième approche : NestedCV ######
# Détermination meilleur HP sur X_train
# Selection du meilleur estimateur
# Prédictions sur X_test


# Instanciation des modéles
clf_lr = LogisticRegression(random_state = 1)
clf_rf = RandomForestClassifier(random_state = 1, n_jobs=-1)
clf_svm = SVC(random_state = 1)


# hyperparamètres à rechercher pour chaque modèle
param_lr = {"solver":["liblinear", 'lbfgs'], "C": np.logspace(-4, 2, 9), "max_iter":[1000] }
param_rf = [{"n_estimators": [10,50,100,500], 
             "min_samples_leaf": [1,3,5], 
             'max_features':["sqrt", "log2"]}]
param_svm = {"kernel":["rbf", "linear"], "C": [0.1, 1, 10]}


# Instanciation GridSearch 
gridcvs = {}
for pgrid, clf, name in zip((param_lr, param_rf, param_svm),
                           (clf_lr, clf_rf, clf_svm),
                           ("LogisticRegression", "RandomForest", "SVM")):
    gvc = GridSearchCV(clf, pgrid, cv=5, refit=True)
    gridcvs[name]=gvc
    
# Création StratifiedKFold
outer_cv = StratifiedKFold(n_splits = 3, shuffle=True)


# score de validation croisée pour chaque couple sur l'ensemble train
outer_scores = {}
for name, gs in gridcvs.items():
    nested_score = cross_val_score(gs, X_train_scaled, y_train, cv=outer_cv)
    outer_scores[name] = nested_score
    print("{} outer accuracy {} +/- {}".format(name, (100*nested_score.mean()), (100*nested_score.std())))


# Selection et entraînement du meilleur algo sur ensemble test
# Les 3 algo fournissent des performances relativement proches

final_clf = gridcvs["RandomForest"]
final_clf.fit(X_train_scaled, y_train)

train_acc = accuracy_score(y_true = y_train, y_pred = final_clf.predict(X_train_scaled))
test_acc = accuracy_score(y_true = y_test, y_pred = final_clf.predict(X_test_scaled))

print("LR train : ", train_acc*100)
print("LR test : ", test_acc*100, "\n")
print(confusion_matrix(y_test, final_clf.predict(X_test_scaled)))
print(classification_report(y_test, final_clf.predict(X_test_scaled)))


# RandomForest effectue une prédiction intéressante de la classe minoritaire !
# /r UnderSampling: plus de FP mais diminution des FN
# 505 obs classifiées 0 alors que 1
# 369 obs classifiées 1 alors que 0

# meilleures prédictions qu'avec un UnderSampling ou ensemble de validation !

# Même résultats qu'avec utilisation d'un ensemble de validation mais temps d'éxecution trés long !


###### Quatrième approche : Nested CV avec UnderSampling ######
# test de combinaison des deux techniques pour améliorer les résultats 


rus = RandomUnderSampler()
(X_ru, y_ru) = rus.fit_resample(X_train_scaled, y_train)


# Utilisation des mêmes modèles et HP que pour NestedCV
# Utilisation du même GridSearchCV 

# score de validation croisée pour chaque couple sur l'ensemble train
outer_scores = {}
for name, gs in gridcvs.items():
    nested_score = cross_val_score(gs, X_ru, y_ru, cv=outer_cv)
    outer_scores[name] = nested_score
    print("{} outer accuracy {} +/- {}".format(name, (100*nested_score.mean()), (100*nested_score.std())))


# Selection et entraînement du meilleur algo sur ensemble test
# Les 3 algo fournissent des performances relativement proches

final_clf = gridcvs["RandomForest"]
final_clf.fit(X_ru, y_ru)

train_acc = accuracy_score(y_true = y_train, y_pred = final_clf.predict(X_train_scaled))
test_acc = accuracy_score(y_true = y_test, y_pred = final_clf.predict(X_test_scaled))

print("LR train : ", train_acc*100)
print("LR test : ", test_acc*100, "\n")
print(confusion_matrix(y_test, final_clf.predict(X_test_scaled)))
print(classification_report_imbalanced(y_test, final_clf.predict(X_test_scaled)))



import matplotlib.pyplot as plt

final_clf = RandomForestClassifier(n_jobs=-1, max_features="sqrt", min_samples_leaf = 1, n_estimators=50)
final_clf.fit(X_train_scaled, y_train)
importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': final_clf.feature_importances_
})
importances = importances.sort_values(by='Importance', ascending=False)

plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation='vertical')
plt.show()


# Bonne prédiction de la classe minoritaire 0, quasi-identique au résultat du NestedCV seul sans UnderSampling
# 695 obs classifiées 0 alors que 1
# 295 obs classifiées 1 alors que 0

# score de précision et recall élevé => bonne prédiction des classes minoritaires et majoritaires
# F1 score sur les deux classes assez élevé, meilleur prédiction de la classe majoritaire



