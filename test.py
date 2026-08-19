def change(a):
    a.append(4)

    return a


a = [1,2,3]
b = a.copy()
print(f"A: {a}")
print(f"B: {b}")

change(b)
print(f"A: {a}")
print(f"B: {b}")