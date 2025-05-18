# Dizionario per la codifica delle diagnosi
diagnosis2idx = {
    "Depressione": 0,
    "Ipertensione": 1,
    "Asma": 2
}

# Codifica per i metabolizzatori dei due geni
gene_encoding = {
    "PM": 0,  # Poor Metabolizer
    "IM": 1,  # Intermediate Metabolizer
    "EM": 2,  # Extensive Metabolizer
    "UM": 3   # Ultra Metabolizer
} 

# Farmaci codificati
drug2idx = {
    "Sertralina": 0,
    "Nortriptilina": 1,
    "Paroxetina": 2
}

idx2drug = {v: k for k, v in drug2idx.items()}

def encode_sample(sample):
    """
    Converte un campione da formato testuale a numerico.
    Output: (input_vector, target_label), target_label può essere None se non presente.
    """
    diagnosis_code = diagnosis2idx[sample["diagnosis"]]
    cyp2d6_code = gene_encoding[sample["CYP2D6"]]
    cyp2c19_code = gene_encoding[sample["CYP2C19"]]
    
    input_vector = [diagnosis_code, cyp2d6_code, cyp2c19_code]

    drug_label = sample.get("drug")
    if drug_label in drug2idx:
        target = drug2idx[drug_label]
    else:
        target = None  # nessun target se è solo input da predire

    return input_vector, target

# Test
if __name__ == "__main__":
    sample = {
        "diagnosis": "Depressione",
        "CYP2D6": "PM",
        "CYP2C19": "EM",
        "drug": "Sertralina"
    }
    x, y = encode_sample(sample)
    print("Input codificato:", x)
    print("Target codificato:", y)