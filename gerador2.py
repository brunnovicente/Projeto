import numpy as np
import pandas as pd

nomes = ['Balaroti','Leroy Merlin','Tamara Materiais','Base Forte','Telha Norte','Casa Show','Companhia da Obra',
'C&C Casa e Construção','Mapa da Obra','Arquitetando Materiais','Casa Construção','Construindo Materiais',
'Cimento & Tudo','Tijolar','Residência – Materiais de Construção','CT Materiais – Cimento & Tijolo',
'Constrular','Construcasa','Real Construções','Construmais','Central Construções','Tudo Construções',
'Construloja','Construponto','Massa Construções','Paraíso Construções']

enderecos = ["P.O. Box 394, 5916 Sodales. Avenue",
            "P.O. Box 960, 7019 Dolor Avenue",
            "P.O. Box 106, 4953 Magnis Rd.",
            "P.O. Box 349, 916 Cras Av.",
            "P.O. Box 184, 4074 Netus Ave",
            "P.O. Box 265, 8050 Pede. Rd.",
            "P.O. Box 375, 6491 Ullamcorper Ave",
            "3554 In, Street",
            "P.O. Box 953, 5851 Etiam Rd.",
            "P.O. Box 707, 2926 Vehicula Ave",
            "8096 Non, Road",
            "P.O. Box 104, 6553 Semper Avenue",
            "Ap #890-3992 A, Rd.",
            "P.O. Box 815, 4192 Quis Road",
            "P.O. Box 441, 4832 Velit. Rd.",
            "P.O. Box 707, 8351 Est Av.",
            "P.O. Box 375, 8652 Magna. Rd.",
            "P.O. Box 340, 5762 Sed Av.",
            "P.O. Box 730, 9489 Faucibus. St.",
            "5345 Et, Avenue",
            "P.O. Box 208, 5362 Eu, Road",
            "P.O. Box 575, 4446 Purus Ave",
            "P.O. Box 550, 2089 Aliquam Road",
            "780-8630 Ut, Street",
            "P.O. Box 101, 5520 Sem. Rd.",
            "P.O. Box 788, 6885 Mi. Street"]

def gerarCNPJ():
    #34.342.432/4324-32
    p1 = np.random.randint(10,99)
    p2 = np.random.randint(100,999)
    p3 = np.random.randint(100,999)
    p4 = np.random.randint(1000,9999)
    p5 = np.random.randint(10,99)
    
    cnpj = str(p1)+'.'+str(p2)+'.'+str(p3)+'/'+str(p4)+'-'+str(p5)
    return cnpj

def gerarEmail(nome):
    n = nome.split(' ')
    email = n[0]+'@gmail.com'
    return email

def gerarTelefone():
    p1 = np.random.randint(88888,99999)
    p2 = np.random.randint(1000,9999)
    telefone = '(86)'+str(p1)+'-'+str(p2)
    return telefone

fornecedores = []
SQL = ''
for i,nome in enumerate(nomes):
    cnpj = gerarCNPJ()
    email = gerarEmail(nome)
    endereco = enderecos[i]
    status = '1'
    telefone1 = gerarTelefone()
    telefone2 = gerarTelefone()
    sql = 'INSERT INTO fornecedores VALUES('+str(i+1)+',\"'+cnpj+'\",\"'+email+'\",\"'+endereco+'\",\"'+nome+'\",'+status+','+'\"'+telefone1+'\",\"'+telefone2+'\");'    
    fornecedores.append(sql)
    SQL += sql+'\n'
    
    







