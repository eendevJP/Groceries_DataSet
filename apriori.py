import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

colunas_nomes = ['Item1','Item2','Item3','Item4','Item5','Item6','Item7','Item8','Item9','Item10','Item11','Item12','Item13','Item14','Item15','Item16','Item17','Item18','item19','Item20','Item21','Item22','Item23','Item24','Item25','Item26','Itemr27','Item28','Item29','Item30','Item31','Item32']
arquivo = pd.read_csv('Supermarket.csv', names = colunas_nomes)
#arquivo - #output

arquivo.shape #output
arquivo.dtypes #output

records = []
for i in range (0, 9835):
    records.append([str(arquivo.values[i, j]) for j in range(0, 32)])
#records - #outuput
#O output é gigantesco

transaction = TransactionEncoder()
transaction_array = transaction.fit_transform(records)
#transaction_array - #output

transaction_df = pd.DataFrame(transaction_array, columns = transaction.columns_)
#transaction_df - #output

for col in transaction_df.columns:
    print(col)

arquivo_clean= transaction_df.drop(['nan'], axis = 1)
#arquivo_clean - #output

contador = arquivo_clean.loc[:,:].sum()
#contador - #output

itemPopular = contador.sort_values(0, ascending = False).head(10)
#itemPopular - #outuput

itemPopular = itemPopular.to_frame().reset_index()
#itemPopular - #output

itemPopular = itemPopular.rename(columns = {'index': 'itens', 0: 'contador'})
#itemPopular - #output

sns.barplot('Contador','Itens',data = itemPopular) #Bugado :(

#Apriori
itensFrequentes = apriori(arquivo_clean, min_support=0.06, use_colnames= True)
#itensFrequentes - #output

regras = association_rules(itensFrequentes, metric='lift', min_threshold=0.7)
#regras - #output

#Gráficos 
plt.scatter(regras['support'],regras['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

itensFrequentes2 = apriori(arquivo_clean, min_support=0.01, use_colnames= True)
#intesFrequentes2 - #output

regras2 = association_rules(itensFrequentes2, metric='lift', min_threshold=0.8)
#regras2 - #output

regras2[regras2.lift > 1]

#Gráficos 
plt.scatter(regras2['support'],regras2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

menorSuporte = apriori(arquivo_clean, min_support=0.01, use_colnames = True)
#menorSuporte - #output

maiorSuporte = apriori(arquivo_clean, min_support=1.0, use_colnames= True)
#maiorSuporte - #output

defaultSuporte = apriori(arquivo_clean, min_support=0.04, use_colnames= True)
#defaultSuporte - #output

defaultSuporteConfidence3 = association_rules(defaultSuporte, metric='confidence', min_threshold = 0.3)
#defaultSuporteConfidence3

defaultSuporteConfidence5 = association_rules(defaultSuporte, metric='confidence', min_threshold=0.5)
#defaultSuporteConfidence5

defaultSuporteConfidence7 = association_rules(defaultSuporte, metric='confidence', min_threshold=0.7)
#defaultSuporteConfidence7

liftCalculo = apriori(arquivo_clean, min_support=0.01, use_colnames= True)
#liftCalculo - #output

liftCalculoRegra = association_rules(liftCalculo, metric='lift', min_threshold=3)
#liftCalculoRegra - #output

liftMenorQ1 = apriori(arquivo_clean, min_support=0.01, use_colnames= True)
#liftMenorQ1 - #output

lifeMenorQ1Calculo = association_rules(liftMenorQ1, metric='lift', min_threshold=0.5)
#lifeMenorQ1Calculo - #output
