from pprint import pprint

arr = []
my_obj = {}

f = open('df', 'r')
lines = f.readlines()
print(lines)

for line in lines:
    arr.append(int(line))

print(arr)
len_arr = len(arr)
print(len_arr)
a_counter = 0
b_counter = 0
c_counter = 0
d_counter = 0
e_counter = 0
for elem in arr:
    if 60 <= elem <= 66:
        e_counter += 1
    if 67 <= elem <= 74:
        d_counter += 1
    if 75 <= elem <= 81:
        c_counter += 1
    if 82 <= elem <= 89:
        b_counter += 1
    if 90 <= elem <= 100:
        a_counter += 1

my_obj['A'] = a_counter/len_arr*100
my_obj['B'] = b_counter/len_arr*100
my_obj['C'] = c_counter/len_arr*100
my_obj['D'] = d_counter/len_arr*100
my_obj['E'] = e_counter/len_arr*100

print(my_obj)