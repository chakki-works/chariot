import chariot.transformer as ct


formatter = ct.formatter.Padding(padding=0, length=5)

data = [
    [1, 2],
    [3, 4, 5],
    [1, 2, 3, 4, 5]
]

padded = formatter.transform(data)
print(padded)
