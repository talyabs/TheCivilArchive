from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

# tokenizer = AutoTokenizer.from_pretrained("yam-peleg/Hebrew-Gemma-11B")
# model = AutoModelForCausalLM.from_pretrained("yam-peleg/Hebrew-Gemma-11B", device_map="auto", max_length=900)
# outputs = model.generate(input_ids=input_ids, max_length=900)  # Adjust this value based on your needs
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("yam-peleg/Hebrew-Mistral-7B-200K")
model = AutoModelForCausalLM.from_pretrained(
    "yam-peleg/Hebrew-Mistral-7B-200K", device_map="auto", max_length=1200
)


input_text = """Your task is to find entities in thr following hebrew text. The required entitites are: PER (person), ORG (organization), LOC (location), DATE (date), TIME (time). please return the entities in the following format: (entity, entity_type)
Text: '״יש 130 אלף תושבים שגרים בכפרים הבלתי מוכרים. 95 אחוז מהכפרים בנויים ממתכת פשוטה ופח. ואין אפילו מיגונית אחת", אומר עטייה אלאעסם ראש המועצה לכפרים הלא מוכרים.

"כשניסינו לשים, המדינה הציבה צווי הריסה. אין דרך אחרת לומר זאת- הופקרנו״.'"""
input_text = """המשימה שלך היא למצא ישויות בטקסט המצורף: אנשים, מקומות, תאריכים. זה הטקסט: ׳אני טליה בן שטרית ואני גרה בתל אביב׳. תחזיר את הישויות בפורמט הבא (ישות, סוג ישות)"""
# input_text = """Your task is to find entities in the following hebrew text. The required entitites are: PER (person), ORG (organization), LOC (location), DATE (date), TIME (time). please return the entities in the following format: (entity, entity_type)
# Text: נסעתי אתמול למודיעין ופגשתי את דודו שפירא"""
# input_text = "שלום! מה שלומך היום?"
print(input_text)
print("Tokenizing text...")
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

print("Generating text...")
outputs = model.generate(**input_ids)
print("Decoding text...")
output = tokenizer.decode(outputs[0])
print(tokenizer.decode(outputs[0]))
# save to csv file all entities
# df = pd.DataFrame(tokenizer.decode(outputs[0]))
df = pd.DataFrame({"Input": [input_text], "Output": [output]})
df.to_csv("entities.csv", index=False)
