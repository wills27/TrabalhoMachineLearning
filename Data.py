import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Criando data set e removendo colunas com baixa correlação e duplicatas

dataset = pd.read_csv('wine-quality.csv', delimiter=';')

dataset.drop(columns=["fixed acidity", "residual sugar", "free sulfur dioxide", "pH"])

dataset.drop_duplicates()

classification = []

for x, y in enumerate(dataset["quality"]):
  if y > 5:
    classification.append(1)
  else:
    classification.append(0)

dataset["class"] = classification

# Standartization

std_data = scaler.fit_transform(dataset)

std_df = pd.DataFrame(std_data)

for x, name in enumerate(dataset.columns):
  std_df.rename(columns={x: name}, inplace=True)

x = std_df.drop(columns=["class", "quality"])
y = dataset["class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Salvando como .csv

with open("x_train_data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(x_train.columns.values)
    writer.writerows(x_train.values.tolist())
    file.close()
with open("x_test_data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(x_test.columns.values)
    writer.writerows(x_test.values.tolist())
    file.close()
with open("y_train_data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["class"])
    for x in y_train:
      writer.writerow([x])
    file.close()
with open("y_test_data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["class"])
    for x in y_test:
      writer.writerow([x])
    file.close()