s = """I am Learning
Python String on GeeksforGeeks   """
print(s)

print(s.strip())

name = "Deeps"
age = 10
print(f"Name: {name}, Age: {age}")

text = "Deeps"
print(text[::-1])

#List

a = [1, 2, 2, "Test"]
ab = list((1, 2,"Test" ))
print(a[0])   # index-based
print(ab)
print(ab[-1])
a.pop()
print(a)

#Tuple

tup = tuple("Deeps")
print(tup[0])
print(tup[1:4])  
print(tup[:3])

t = tuple(('1',56,[1,4,5], 8.5))
print(t)
print(t[2])

#Set
#key in s	containment check
#key not in s	non-containment check
#s1 == s2	s1 is equivalent to s2
#s1 != s2	s1 is not equivalent to s2
#s1 <= s2	s1 is subset of s2
#s1 < s2	s1 is proper subset of s2
#s1 >= s2	s1 is superset of s2
#s1 > s2	s1 is proper superset of s2
#s1 | s2	the union of s1 and s2
#s1 & s2	the intersection of s1 and s2
#s1 - s2	the set of elements in s1 but not s2
#s1 ˆ s2	the set of elements in precisely one of s1 or s2

s = {"Geeks", "for", 10, 52.7, True}
print(s)

s = set(["a", "b", "c"])
print("Normal Set:", s)

fs = frozenset(["e", "f", "g"])
print("Frozen Set:", fs)

#Dictionary

d = {"name": "Deeps", "age": 11}
b = dict(name="Rio", age=35)
print(b)
print(d["name"])     # Access using key
print(d.get("age"))  # Access using get()

# If condition
a = 200
b = 33
if b > a:
  print("b is greater than a")
elif a == b:
  print("a and b are equal")
else:
  print("a is greater than b")
  
 # For Loop
fruits = ["apple", "banana", "cherry"]
for x in fruits:
  if x == "banana":
    continue
  print(x)
  
#Functions:

def evenOdd(x):
    if (x % 2 == 0):
        return "Even"
    else:
        return "Odd"

print(evenOdd(16))
print(evenOdd(7))