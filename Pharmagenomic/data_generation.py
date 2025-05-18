import random

# Possibili diagnosi cliniche simulate
diagnoses = ["Depressione", "Ipertensione", "Asma"]

# Varianti genetiche per due geni farmacogenetici noti
genes = {
    "CYP2D6": ["PM", "IM", "EM", "UM"],     # Poor, Intermediate, Extensive, Ultra metabolizer
    "CYP2C19": ["PM", "IM", "EM", "UM"]
}

# Farmaci simulati per la raccomandazione
drugs = ["Sertralina", "Nortriptilina", "Paroxetina"]

# Funzione che genera un singolo campione paziente
def generate_sample():
    diagnosis = random.choice(diagnoses)
    g1 = random.choice(genes["CYP2D6"])
    g2 = random.choice(genes["CYP2C19"])

    # Logica farmacogenomica fittizia
    if diagnosis == "Depressione":
        if g1 == "PM":
            drug = "Sertralina"
        elif g2 == "UM":
            drug = "Nortriptilina"
        else:
            drug = "Paroxetina"
    elif diagnosis == "Ipertensione":
        drug = "Paroxetina"
    else:  # Asma
        drug = "Nortriptilina"

    return {
        "diagnosis": diagnosis,
        "CYP2D6": g1,
        "CYP2C19": g2,
        "drug": drug
    }

# Funzione che genera un dataset di n pazienti
def generate_dataset(n=1000):
    return [generate_sample() for _ in range(n)]

# Test veloce del modulo
if __name__ == "__main__":
    samples = generate_dataset(n=5)
    for s in samples:
        print(s)
