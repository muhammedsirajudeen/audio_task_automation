import spacy
from spacy.training import Example
import random
# Load the base model
nlp = spacy.load("en_core_web_sm")

# Create a blank NER model
ner = nlp.get_pipe("ner")

# Add the label for PERSON
ner.add_label("PERSON")

# Prepare the training data
training_data = [
    ("Adhnan", {"entities": [(0, 6, "PERSON")]}),
    ("Amal Jahan", {"entities": [(0, 10, "PERSON")]}),
    ("Rashidh", {"entities": [(0, 7, "PERSON")]}),
    ("Abhinand", {"entities": [(0, 8, "PERSON")]}),
    ("Anto John", {"entities": [(0, 9, "PERSON")]}),
    ("Lijo", {"entities": [(0, 4, "PERSON")]}),
    ("Mathew", {"entities": [(0, 6, "PERSON")]}),
    ("Minhaj", {"entities": [(0, 6, "PERSON")]}),
    ("Muhammed Hashim", {"entities": [(0, 15, "PERSON")]}),
    ("Nihal", {"entities": [(0, 5, "PERSON")]}),
    ("Rafan", {"entities": [(0, 5, "PERSON")]}),
    ("Rashidh", {"entities": [(0, 7, "PERSON")]}),
    ("Riyaz KP", {"entities": [(0, 8, "PERSON")]}),
    ("Sadin", {"entities": [(0, 5, "PERSON")]}),
    ("Salman", {"entities": [(0, 6, "PERSON")]}),
    ("Shahanah", {"entities": [(0, 8, "PERSON")]}),
    ("Shamir", {"entities": [(0, 6, "PERSON")]}),
    ("Sharafath", {"entities": [(0, 9, "PERSON")]}),
    ("Siyad", {"entities": [(0, 5, "PERSON")]}),
    ("Thaha", {"entities": [(0, 5, "PERSON")]}),
    ("Uvais", {"entities": [(0, 5, "PERSON")]}),
    ("Vishnu", {"entities": [(0, 6, "PERSON")]}),
    ("Roshan", {"entities": [(0, 6, "PERSON")]}),
]

# Disable other pipes to train only the NER model
with nlp.select_pipes(enable="ner"):
    for itn in range(30):  # Adjust the number of iterations
        print(f"Iteration {itn + 1}")
        random.shuffle(training_data)
        for text, annotations in training_data:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example])

# Save the model
nlp.to_disk("custom_name_model")

