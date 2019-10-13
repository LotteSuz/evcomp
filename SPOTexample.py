 # Module sys has to be imported:
import sys

m= sys.argv[1]
m= eval(m.split()[0])

#here you can call your optimization algorithm
#e.g. evolve_program(m[0],m[1],m[2])
#return fitness by printing to console

# Example: set fitness to value of last paramter
fitness= sum(m)
print(fitness)
