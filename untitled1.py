import cmapPy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file=cmapPy.pandasGEXpress.parse_gct.parse(r"C:\Users\ASUS\Downloads\PAAD.gct")

data=file.data_df
#row=file.row_metadata_df
column=file.col_metadata_df.T


df=pd.concat([column,data]).T

y0=df["histological_type"]
y1=df["histological_type_other"]

X0=df.iloc[:,124:]

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="mean")
X1=imputer.fit_transform(X0)

from sklearn.preprocessing import StandardScaler
X1 = StandardScaler().fit_transform(X1)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X1)


l=[i for i in range(183)]
y0.index=l
y1.index=l



principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf=pd.concat([principalDf,y1],axis=1) 
finalDf["histological_type_other"].fillna("adenocarcinoma",inplace=True)
X=finalDf.iloc[:,0:2]
y=finalDf.iloc[:,2]

finalDf.replace("invasive adenocarcinoma","adenocarcinoma",inplace=True)
finalDf.replace("invasive, well-differentiated","adenocarcinoma",inplace=True)
finalDf.replace("poorly differentiated adenocarcinoma","adenocarcinoma",inplace=True)
finalDf.replace("neuroendocrine carcinoma nos","neuroendocrine",inplace=True)
finalDf.replace("82463 neuroendocrine carcinoma nos","neuroendocrine",inplace=True)
finalDf.replace("neuroendocrine carcinoma","neuroendocrine",inplace=True)
finalDf.replace("adenocarcinoma, nos","adenocarcinoma",inplace=True)
finalDf.replace("poorly differentiated pancreatic adenocarcinoma","adenocarcinoma",inplace=True)
finalDf.replace("not specified","adenocarcinoma",inplace=True)
finalDf.replace("intraductal tubulopapillary neoplasm","adenocarcinoma",inplace=True)
finalDf.replace("adenocarcinoma- nos","adenocarcinoma",inplace=True)

     
from sklearn.preprocessing import LabelEncoder
lb0=LabelEncoder()
finalDf["histological_type_other"]=lb0.fit_transform(finalDf["histological_type_other"].astype(str))
y=finalDf["histological_type_other"]

plt.scatter(principalDf["principal component 1"][y==0],principalDf["principal component 2"][y==0],label="adenocarcinoma",color="r")
plt.scatter(principalDf["principal component 1"][y==1],principalDf["principal component 2"][y==1],label='moderately differentiated ductal adenocarcinoma 60% + neuroendocrine 40%',color="b")
plt.scatter(principalDf["principal component 1"][y==2],principalDf["principal component 2"][y==2],label="neuroendocrine",color="g")
plt.legend(loc="best")
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.show()

                                    
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        