import random
import matplotlib.pyplot as plt
import imp
import pandas as pd

class Perceptron:
    def __init__(self, input_number, step_size=0.1):
        self._ins = input_number
        self._w = [random.random() for _ in range(input_number)]
        self._eta = step_size
        
    
    def predict(self, inputs):
        calculo = sum(w * entrada for w, entrada in zip(self._w, inputs))
        if calculo > 0:
            return 1
        return 0
    
    
    def train(self, inputs, ex_output):
        output = self.predict(inputs)
        error = ex_output - output
        if error != 0:
            self._w = [w+self._eta*error*x for w, x in zip(self._w, inputs)]
        return error

    def w(self):
        return self._w
    
#Conjunto de entradas
#data = [[170, 56, 1], #mujer de 1.7m y 56kg
#        [172, 63, 0], #hombre de 1.72m y 63kg
#        [160, 50, 1],
#        [170, 63, 0],
#        [174, 66, 0],
#        [158, 55, 1],
#        [183, 80, 0],
#        [182, 70, 0],
#        [165, 54, 1]]

datacv = pd.read_csv("antropo-latinos.csv")

dataaux = datacv.copy()
dataaux['peso'] = datacv['estatura']
dataaux['estatura'] = datacv['peso']
dataaux.columns = ['estatura', 'peso', 'sexo']

data = dataaux.values.tolist()

pr = Perceptron(3,0.1)

#Lista de pesos inicial
weights = []
#Lista de errores inicial
errores = []

#Fase de entrenamiento
for _ in range(100):
    for persona in data:
        #La salida es la variable sexo (último dato de la lista)
        output = persona[-1]
        #Bias +x1 + x2 + ...
        inp = [1] + persona[0:-1]
        weights.append(pr.w)
        err = pr.train(inp, output)
        errores.append(err)
        

#Valores para evaluar la respuesta del perceptrón
estatura = 1.70
peso = 71.0

if pr.predict([1, estatura, peso]) == 1:
    print('Mujer')
else:
    print('Hombre')
    
puede_graficar = True
try:
    imp.find_module('matplotlib')
except:
    puede_graficar= False

if not puede_graficar:
    print('No es posible graficar los resultados porque hace fala el modulo matplotlib')
    sys.exit(0)
    pass

plt.plot(errores)
plt.show()
