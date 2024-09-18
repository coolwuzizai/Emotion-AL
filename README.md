# Emotion-AL

Projekat je rađen kao tema za seminarski rad za kurs "Računarska Inteligencija" na četvrtoj godini Matematičkog fakulteta. Tema je **Prepoznavanje facijalnih ekspresija i emocija korišćenjem dubokih neuronskih mreža**.

Tema projekta je rađena po uzoru na [Kaggle challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/overview), odakle je i korišćen skup podataka za treniranje neuronskih mreža.

## Korišćene biblioteke

1. [Python](https://www.python.org/)
2. [PyTorch](https://pytorch.org)
3. [Matplotlib](https://matplotlib.org/)

## Metodologije

U sklopu ovog projekta je demostrirano projektovanje, treniranje i evaluacija nekih dubokih neuronskih mreža cije su mogućnosti testirane u realnoj primeni.

## Testiranje i razvoj

U koliko biste želeli da pokrenete i testirate kodove iz ovog projekta to možete uraditi na neki od sledećih načina

### Pokretanje iz Google Collab okruženja

1. Potrebno je da sve `.py` fajlove iskopirate u lokalno okruženje Google Collab-a.
2. Ako želite da isprobate treniranje svoje, ili da ponovo trenirate postojeće modele, potrebno je da se prekopira i direktorijum koji sadrži skup podataka (ima oko 50MB ceo "data" direktorijum)
3. Nakon toga je neophodno da instalirate biblioteke koje se koriste a nisu podrazumevano dostupne na Google Collab-u. Ovo možete uraditi tako što na samom vrhu ubacite novu ćeliju i u njoj pokrenete sledeći kod.

```bash
# Ako je requrements.txt fajl iskopiran zajedno sa kodovima
pip install -r requirements.txt
```

4. Nakon ovoga možete pokretati kodove iz projekta i testirate postojeće ili vaše nove funkcije.

### Pokretanje lokalno na vašoj mašini

1. Ako na računaru nemate instaliran jupyter kernel potrebno je da se on instalira. Ovo se može automatski uraditi preko VSCode razvojnog okruženje (VSCode će sam predložiti instalaciju kernela čim se otvori `.ipynb` fajl u editoru). Ukoliko koristite neki drugi editor ili želite samostalno to da urdite možete pokrenuti sledeću komandu u terminalu.

```
pip install ipython ipykernel jupyter_client jupyter_core
```

##### NAPOMENA: Preporučujemo da sve instalacije vršite u virtualnom python okruženju (venv).

2. Nakon instalacije jupyter-a potrebno je da se instaliraju neophodne biblioteke. Ovo se može uraditi sledećom komandom iz terminala.

```
pip install -r requirements.txt
```

3. Sada se mogu pokretati i testirati kodovi.

## Demo

![Selection_001](https://github.com/user-attachments/assets/50ae38e1-5e42-4b93-ac11-d179638f3479)

