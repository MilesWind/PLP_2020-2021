import random as r
import math as m
from threading import Thread
set_length = int(input("set_length:"))

def order_set(input_set):
    return_set = [input_set[0]]
    input_set.pop(0)
    momentum = None
    index = 0
    for x in input_set:
        index = m.floor((len(return_set)-1)/2)
        bottom = return_set[index]
        if x > bottom:
            momentum = 1
        else:
            momentum = -1
        while True:
            index += momentum
            if index == len(return_set):
                break
            if index == -1:
                index = 0
                break
            bottom = return_set[index]
            if bottom > x and momentum == 1:
                break
            if bottom < x and momentum == -1:
                index += 1
                break
        return_set.insert(index, x)
    return return_set

def quicksort(array):
    if len(array) < 2:
        return array
    low, same, high = [], [], []
    pivot = array[r.randint(0, len(array) - 1)]
    for item in array:
        if item < pivot:
            low.append(item)
        elif item == pivot:
            same.append(item)
        elif item > pivot:
            high.append(item)
    return quicksort(low) + same + quicksort(high)

def py_sort():
    global sorting_set_a
    sorting_set_a = quicksort(sorting_set_a)
    print("py_sort")
    
def my_sort():
    global sorting_set_b
    sorting_set_b = order_set(sorting_set_b)
    print("my_sort")

global sorting_set_a
global sorting_set_b
sorting_set_a = []
sorting_set_b = []
for i in range(set_length):
    random_value = r.uniform(-10, 10)
    sorting_set_a.append(random_value)
    sorting_set_b.append(random_value)

Thread(target = my_sort).start()
Thread(target = py_sort).start()
