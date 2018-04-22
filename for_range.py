import random

step_range = 10
step_choices = range(-1*step_range,step_range+1)
rand_walk = [random.choice(step_choices) for x in range(100)]
#print(rand_walk)
zeros = [0 for i in range(len(rand_walk))]
print(len(zeros))

