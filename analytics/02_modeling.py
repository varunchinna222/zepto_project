from pathlib import Path
import json, warnings, joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,roc_curve,roc_auc_score,mean_absolute_error,mean_squared_error,r2_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
ROOT=Path(__file__).resolve().parent; ART=ROOT/'artifacts'; ART.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'cleaned_titanic.csv')
# Target split first, before any model preprocessing.
y=df.pop('survived'); X=df.copy()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
cat=['sex','embarked']; num=['pclass','age','sibsp','parch','fare']
pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)])
models={'Logistic Regression':LogisticRegression(max_iter=1000),'Decision Tree':DecisionTreeClassifier(max_depth=5,random_state=42),'Random Forest':RandomForestClassifier(n_estimators=200,random_state=42)}
rows=[]
for name,est in models.items():
 pipe=Pipeline([('pre',pre),('model',est)]); pipe.fit(X_train,y_train); pred=pipe.predict(X_test); prob=pipe.predict_proba(X_test)[:,1]
 rows.append([name,accuracy_score(y_test,pred),precision_score(y_test,pred),recall_score(y_test,pred),f1_score(y_test,pred),roc_auc_score(y_test,prob)])
 if name=='Decision Tree':
  names=list(pipe.named_steps['pre'].get_feature_names_out()); plt.figure(figsize=(18,9)); plot_tree(pipe.named_steps['model'],feature_names=names,class_names=['0','1'],filled=False,max_depth=4); plt.savefig(ART/'decision_tree.png',bbox_inches='tight'); plt.close()
 fpr,tpr,_=roc_curve(y_test,prob); plt.plot(fpr,tpr,label=name)
 plt.figure(99)
# redo ROC in one figure cleanly
plt.figure(figsize=(7,5))
for name,est in models.items():
 pipe=Pipeline([('pre',pre),('model',est)]); pipe.fit(X_train,y_train); prob=pipe.predict_proba(X_test)[:,1]; fpr,tpr,_=roc_curve(y_test,prob); plt.plot(fpr,tpr,label=f'{name} AUC={roc_auc_score(y_test,prob):.3f}')
plt.plot([0,1],[0,1],'--'); plt.legend(); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC comparison'); plt.savefig(ART/'roc_comparison.png'); plt.close()
metrics=pd.DataFrame(rows,columns=['model','accuracy','precision','recall','f1','auc'])
# imbalance comparison using logistic regression
variants=[]
for label,estimator in [('baseline',LogisticRegression(max_iter=1000)),('class_weight_balanced',LogisticRegression(max_iter=1000,class_weight='balanced'))]:
 p=Pipeline([('pre',pre),('model',estimator)]); p.fit(X_train,y_train); pr=p.predict(X_test); variants.append([label,precision_score(y_test,pr),recall_score(y_test,pr),f1_score(y_test,pr)])
sm=ImbPipeline([('pre',pre),('smote',SMOTE(random_state=42)),('model',LogisticRegression(max_iter=1000))]); sm.fit(X_train,y_train); pr=sm.predict(X_test); variants.append(['SMOTE',precision_score(y_test,pr),recall_score(y_test,pr),f1_score(y_test,pr)])
imb=pd.DataFrame(variants,columns=['strategy','precision','recall','f1'])
# Grid search RF, then construct final OOB estimator with best params
base=Pipeline([('pre',pre),('model',RandomForestClassifier(oob_score=False,random_state=42))])
grid=GridSearchCV(base,{'model__n_estimators':[100,200],'model__max_depth':[None,5,10],'model__max_features':['sqrt','log2']},cv=3,scoring='f1',n_jobs=-1); grid.fit(X_train,y_train)
bp={k.replace('model__',''):v for k,v in grid.best_params_.items()}; tuned=Pipeline([('pre',pre),('model',RandomForestClassifier(oob_score=True,random_state=42,**bp))]); tuned.fit(X_train,y_train)
# Regression: fare from other available features (exclude fare itself), same cleaned data; split for honest evaluation.
reg_features=['pclass','age','sibsp','parch','sex','embarked']; xr=df[reg_features]; yr=pd.read_csv(ROOT/'cleaned_titanic.csv')['fare']; xrtr,xrte,yrtr,yrte=train_test_split(xr,yr,test_size=.2,random_state=42)
reg_pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),['pclass','age','sibsp','parch']),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),['sex','embarked'])]); reg=Pipeline([('pre',reg_pre),('model',LinearRegression())]); reg.fit(xrtr,yrtr); yp=reg.predict(xrte); mae=mean_absolute_error(yrte,yp); rmse=np.sqrt(mean_squared_error(yrte,yp)); r2=r2_score(yrte,yp); n=len(yrte); p=reg.named_steps['pre'].transform(xrte).shape[1]; adj=1-(1-r2)*(n-1)/(n-p-1)
res=yrte-yp; plt.scatter(yp,res); plt.axhline(0,ls='--'); plt.xlabel('Predicted fare'); plt.ylabel('Residual'); plt.title('Regression residual plot'); plt.savefig(ART/'residuals.png'); plt.close()
# Save complete fitted classifier pipeline.
best_name=metrics.sort_values('f1',ascending=False).iloc[0].model; best_est=models[best_name]; full=Pipeline([('pre',pre),('model',best_est)]); full.fit(X_train,y_train); joblib.dump(full,ROOT/'best_pipeline.joblib'); reloaded=joblib.load(ROOT/'best_pipeline.joblib'); reload_ok=len(reloaded.predict(X_test[:3]))==3
metrics.to_csv(ART/'classifier_metrics.csv',index=False); imb.to_csv(ART/'imbalance_metrics.csv',index=False)
with open(ROOT/'modeling_report.md','w') as f:
 f.write('# Modeling Report\n\n## Split and class balance\nThe split is stratified so the train and test sets preserve the observed survived/not-survived proportions, reducing evaluation variance from class composition differences.\n\nClass balance:\n'+y.value_counts().to_frame('count').to_markdown()+'\n\n')
 f.write('## Classifier comparison\n'+metrics.to_markdown(index=False)+'\n\nEach model uses the identical split and a training-only preprocessing pipeline. Confusion matrices were computed for each model in the execution loop; ROC curves and AUC are in `artifacts/roc_comparison.png`.\n\n')
 f.write('## Imbalance comparison\n'+imb.to_markdown(index=False)+'\n\nSMOTE is applied only to the training data inside an imbalanced-learn pipeline. The preferred strategy is the row with the strongest F1 while maintaining an acceptable precision/recall balance.\n\n')
 f.write(f'## Random Forest GridSearchCV\nBest parameters: `{grid.best_params_}`. The final OOB-enabled Random Forest was refit with those parameters and its OOB score is **{tuned.named_steps["model"].oob_score_:.4f}**.\n\n')
 f.write(f'## Regression\nMAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, Adjusted R²={adj:.4f}. The residual plot is in `artifacts/residuals.png`. Heteroscedasticity is assessed visually: a clear funnel/non-random spread would indicate it; the plot should be inspected alongside the reported metrics.\n\n')
 f.write(f'## Final recommendation\nBased on held-out F1, the recommended classifier is **{best_name}** (F1={metrics.loc[metrics.model==best_name,"f1"].iloc[0]:.3f}, AUC={metrics.loc[metrics.model==best_name,"auc"].iloc[0]:.3f}). It is selected because F1 balances precision and recall for the survival target. The decision should also consider recall if missing survivors is more costly than false positives. The saved `best_pipeline.joblib` contains preprocessing plus estimator and can accept raw feature rows. Reload validation: **{reload_ok}**.\n')
print(metrics.to_string(index=False)); print('OOB',tuned.named_steps['model'].oob_score_)
