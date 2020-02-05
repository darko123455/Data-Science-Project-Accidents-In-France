import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from sklearn.preprocessing import LabelBinarizer
import sklearn.preprocessing as prep
import sklearn.metrics as metrics

#Ucitavamo CSV file koji je skroz sredjen, obradjen.
df = pd.read_csv('./to_je_to.csv', encoding = "ISO-8859-1")

#izdvajamo kolone koje su nam bitne, od znacaja.
bitne = df[['sat_minut', 'osvetljenje', 'stepen_povrede', 'postanski_broj_opstine']]

#Radi provere, printamo prvih 5 redova tih kolona.
print(bitne.head())


#Posto je osvetljenje kategoricki tip podatka, sa njim ne mozemo da radimo tako, nego mroamo da 
#ga binarizujemo.

lb = LabelBinarizer()
lb_osvetljenje = lb.fit_transform(bitne["osvetljenje"])

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
scaler = prep.MinMaxScaler().fit(bitne[['sat_minut', 'postanski_broj_opstine']])
#U ovom delu koda, sve kolone koje ne treba da se binarizuju skaliramo na vrednost izmedju 0 i 1
x = pd.DataFrame(scaler.transform(bitne[['sat_minut', 'postanski_broj_opstine']]))
x.columns = ["sat_minut", "postanski_broj_opstine"]


#Spajamo skalirane kolone sa binarizovanim Osvetljenjem (kolonom osvetljenje koja se 
#prosirila na 4 kolone)
tmp = pd.concat([pd.DataFrame(lb_osvetljenje, columns = lb.classes_), x], axis = 1)

#binarizujemo kolonu stepen povrede
lb2 = LabelBinarizer()
lb_stepen_povrede = lb2.fit_transform(bitne["stepen_povrede"])

#Pomocu pd.concat spajamo i stepen_povrede sa ostalim kolonama
n_df = pd.concat([pd.DataFrame(lb_stepen_povrede, columns = lb2.classes_), tmp], axis = 1)


#radi provere printamo prvih pet redova.
print(n_df.head())



#Algoritam DBSCAN prima dva argumenta, epsilon i min_samples,
#mi cemo u for petlji tri puta pokrenuti pomenuti algoritam, sa razlicitim epsilon parametrima.
for eps in [0.048, 0.044, 0.046]:
    estDbscan = DBSCAN(eps = eps, min_samples = 100)
    estDbscan.fit(n_df)
    n_df['labels'] = estDbscan.labels_    
    num_clusters = len(n_df['labels'].unique())
   	#ispisujemo broj klastera, senka koeficijent i epsilon 
   	print("epsilon: %f " % eps)
    print("Broj klastera: %d" % num_clusters)
    print("Senka koeficijent: %f " % silhouette_score(n_df, estDbscan.labels_))
    print()

    #crtamo figuru, odnosno dijagram za svaki odradjen DBSCAN algoritam. 
    fig = plt.figure(figsize=(9,7))
    for j in range(-1,num_clusters):
    	cluster = n_df.loc[n_df['labels'] == j]
    	plt.scatter(cluster['labels'], cluster['sat_minut'], color = 'blue')
    plt.xlabel("Klaster", fontsize = 20)
    plt.ylabel("sat_minut", fontsize = 20)
    plt.title("Nesrece po klasterima", fontsize = 30)
    plt.tight_layout()
    fig.savefig("nesrece_klaster" + str(eps) + ".png")

