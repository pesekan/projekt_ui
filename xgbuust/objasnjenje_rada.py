# U pozadini XGBoost radi iterativno trenirajući i kombinirajući stabla odlučivanja na aditivan način.
# Točnije, počinje fittanjem jednog stabla odlučivanja u testne podatke, a zatim izračunava pogreške između
# predviđenih vrijednosti i pravih vrijednosti. Algoritam zatim prilagođava drugo stablo odlučivanja greškama
# i dodaje predviđanja iz drugog stabla predviđanjima iz prvog stabla.
# Ovaj se proces ponavlja, pri čemu se svako dodatno stablo trenira za predviđanje pogrešaka iz prethodnih stabala,
# sve dok algoritam ne dosegne unaprijed određeni broj stabala ili dok ne postigne određenu razinu točnosti.

from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression

# Generate a sample dataset
X, y = make_regression(n_samples=10, n_features=4, random_state=16)
print("y:", y, "\n")

# Initialize a list to store the predictions from each tree
tree_preds = []

# Set the number of trees to use
n_trees = 10

# Loop over the number of trees
for i in range(n_trees):
    # Initialize a new decision tree model
    tree = DecisionTreeRegressor(random_state=16)

    # Fit the decision tree to the training data
    tree.fit(X, y)

    # Make predictions with the decision tree
    tree_pred = tree.predict(X)

    # Add the predictions to the list of tree predictions
    tree_preds.append(tree_pred)

    # Update the training data by subtracting the predictions from the target variable
    y = y - tree_pred

# Combine the predictions from all the trees
ensemble_preds = sum(tree_preds)

# Print the ensemble predictions
print("final predicions:", ensemble_preds)


# DecisionTreeRegressor je algoritam strojnog učenja koji se koristi za rješavanje problema regresije.
# Djeluje tako da konstruira model odluka i njihovih mogućih posljedica u obliku stabla. Svaki unutarnji
# čvor stabla predstavlja test ulazne značajke, a svaka grana predstavlja ishod testa.
# Čvorovi listova stabla predstavljaju predviđenu izlaznu vrijednost.
#
# Regresor stabla odlučivanja radi tako da rekurzivno dijeli ulazni prostor na manja područja, a zatim
# prilagođava jednostavan model, kao što je konstantna funkcija, svakom području. Ulazni prostor podijeljen
# je u područja na temelju vrijednosti ulaznih značajki, a regije su odabrane da minimiziraju varijance
# izlaznih vrijednosti unutar svake regije.
# Za konstruiranje stabla odlučivanja, algoritam počinje s cijelim skupom podataka za obuku u korijenskom
# čvoru stabla. Zatim odabire značajku za testiranje na korijenskom čvoru i dijeli podatke u podskupove
# na temelju vrijednosti odabrane značajke. Ovaj se proces ponavlja rekurzivno na svakom podskupu sve dok se
# ne postigne kriterij zaustavljanja, kao što je kada se dosegne najveća dubina stabla ili kada sve instance
# u podskupu imaju istu izlaznu vrijednost. Prilikom donošenja predviđanja pomoću regresora stabla odlučivanja,
# ulaz se prenosi kroz stablo odlučivanja, počevši od korijenskog čvora i prateći put odluka dok se ne dosegne
# lisnati čvor. Predviđena izlazna vrijednost tada je vrijednost povezana s lisnim čvorom.