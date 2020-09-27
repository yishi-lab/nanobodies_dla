from sklearn.preprocessing import LabelEncoder
import numpy as np

aa = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

hydrophobicity_color = {
    'Hydrophilic': 'blue',
    'Neutral': 'green',
    'Hydrophobic': 'black'
}

hydrophobicity = {
    'A': hydrophobicity_color["Neutral"],
    'C': hydrophobicity_color["Hydrophobic"],
    'D': hydrophobicity_color["Hydrophilic"],
    'E': hydrophobicity_color["Hydrophilic"],
    'F': hydrophobicity_color["Hydrophobic"],
    'G': hydrophobicity_color["Neutral"],
    'H': hydrophobicity_color["Neutral"],
    'I': hydrophobicity_color["Hydrophobic"],
    'K': hydrophobicity_color["Hydrophilic"],
    'L': hydrophobicity_color["Hydrophobic"],
    'M': hydrophobicity_color["Hydrophobic"],
    'N': hydrophobicity_color["Hydrophilic"],
    'P': hydrophobicity_color["Neutral"],
    'Q': hydrophobicity_color["Hydrophilic"],
    'R': hydrophobicity_color["Hydrophilic"],
    'S': hydrophobicity_color["Neutral"],
    'T': hydrophobicity_color["Neutral"],
    'V': hydrophobicity_color["Hydrophobic"],
    'W': hydrophobicity_color["Hydrophobic"],
    'Y': hydrophobicity_color["Hydrophobic"],
}

chemistry_color = {
    'Polar': 'green',
    'Neutral': 'purple',
    'Basic': 'blue',
    'Acidic': 'red',
    'Hydrophobic': 'black',
}

chemistry = {
    'A': chemistry_color["Hydrophobic"],
    'C': chemistry_color["Polar"],
    'D': chemistry_color["Acidic"],
    'E': chemistry_color["Acidic"],
    'F': chemistry_color["Hydrophobic"],
    'G': chemistry_color["Polar"],
    'H': chemistry_color["Basic"],
    'I': chemistry_color["Hydrophobic"],
    'K': chemistry_color["Basic"],
    'L': chemistry_color["Hydrophobic"],
    'M': chemistry_color["Hydrophobic"],
    'N': chemistry_color["Neutral"],
    'P': chemistry_color["Hydrophobic"],
    'Q': chemistry_color["Neutral"],
    'R': chemistry_color["Basic"],
    'S': chemistry_color["Polar"],
    'T': chemistry_color["Polar"],
    'V': chemistry_color["Hydrophobic"],
    'W': chemistry_color["Hydrophobic"],
    'Y': chemistry_color["Polar"],
}

chemistry_color_aromatic = {
    'Polar': 'grey',
    'Neutral': 'purple',
    'Basic': 'blue',
    'Acidic': 'red',
    'Hydrophobic': 'black',
    'Sulfuric': 'green',
    'Aromatic': [199/256., 182/256., 0.,1.]#'gold'
}

chemistry_aromatic = {
    'A': chemistry_color_aromatic["Hydrophobic"],
    'C': chemistry_color_aromatic["Sulfuric"],
    'D': chemistry_color_aromatic["Acidic"],
    'E': chemistry_color_aromatic["Acidic"],
    'F': chemistry_color_aromatic["Aromatic"],
    'G': chemistry_color_aromatic["Polar"],
    'H': chemistry_color_aromatic["Basic"],
    'I': chemistry_color_aromatic["Hydrophobic"],
    'K': chemistry_color_aromatic["Basic"],
    'L': chemistry_color_aromatic["Hydrophobic"],
    'M': chemistry_color_aromatic["Hydrophobic"],
    'N': chemistry_color_aromatic["Neutral"],
    'P': chemistry_color_aromatic["Hydrophobic"],
    'Q': chemistry_color_aromatic["Neutral"],
    'R': chemistry_color_aromatic["Basic"],
    'S': chemistry_color_aromatic["Polar"],
    'T': chemistry_color_aromatic["Polar"],
    'V': chemistry_color_aromatic["Hydrophobic"],
    'W': chemistry_color_aromatic["Aromatic"],
    'Y': chemistry_color_aromatic["Aromatic"],
}

charge_color = {
    'Positive': 'blue',
    'Neutral': 'black',
    'Negative': 'red',
}

charge = {
    'A': charge_color["Neutral"],
    'C': charge_color["Neutral"],
    'D': charge_color["Negative"],
    'E': charge_color["Negative"],
    'F': charge_color["Neutral"],
    'G': charge_color["Neutral"],
    'H': charge_color["Positive"],
    'I': charge_color["Neutral"],
    'K': charge_color["Positive"],
    'L': charge_color["Neutral"],
    'M': charge_color["Neutral"],
    'N': charge_color["Neutral"],
    'P': charge_color["Neutral"],
    'Q': charge_color["Neutral"],
    'R': charge_color["Positive"],
    'S': charge_color["Neutral"],
    'T': charge_color["Neutral"],
    'V': charge_color["Neutral"],
    'W': charge_color["Neutral"],
    'Y': charge_color["Neutral"],
}

# https://medium.com/@h_76213/efficient-dna-embedding-with-tensorflow-ffce5f499083
embedding_values = np.zeros([ord('Z'), 4], np.float32)
embedding_values[ord('A')] = np.array([1, 0, 0, 0])
embedding_values[ord('C')] = np.array([0, 1, 0, 0])
embedding_values[ord('D')] = np.array([0, 0, 1, 0])
embedding_values[ord('E')] = np.array([0, 0, 0, 1])
embedding_values[ord('F')] = np.array([.5, 0, 0, .5])
embedding_values[ord('G')] = np.array([0, .5, .5, 0])
embedding_values[ord('H')] = np.array([.5, .5, 0, 0])
embedding_values[ord('I')] = np.array([0, 0, .5, .5])
embedding_values[ord('K')] = np.array([.5, 0, .5, 0])
embedding_values[ord('L')] = np.array([0, .5, 0, .5])
embedding_values[ord('M')] = np.array([0, 1 / 3, 1 / 3, 1 / 3])
embedding_values[ord('N')] = np.array([1 / 3, 0, 1 / 3, 1 / 3])
embedding_values[ord('P')] = np.array([1 / 3, 1 / 3, 0, 1 / 3])
embedding_values[ord('Q')] = np.array([1 / 3, 1 / 3, 1 / 3, 0])
embedding_values[ord('R')] = np.array([.25, .25, .25, 0])
embedding_values[ord('S')] = np.array([.25, .25, 0, .25])
embedding_values[ord('T')] = np.array([.25, 0, .25, .25])
embedding_values[ord('V')] = np.array([0, .25, .25, .25])
embedding_values[ord('W')] = np.array([.5, .5, .5, .5])
embedding_values[ord('Y')] = np.array([1, 1, 1, 1])

aa_list = list(aa.values())
le = LabelEncoder()
le.fit(aa_list)

le_align = LabelEncoder()
aa_align= list(aa.values())
aa_align.append('-')
le_align.fit(aa_align)

list_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',  'W','Y','$\\boxminus$']
amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',  'W', 'Y','-']
aadict = {amino_acids[k]:k for k in range(len(amino_acids))}