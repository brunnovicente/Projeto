import numpy as np
import pandas as pd

nomes = ['Telha', 'Bloco de Vidro', 'Tinta', 'Cerâmica', 'Massa', 'Adesivo', 'Cola', 'Prego', 'Parafuso', 'Veda Rosca'
         'Cimento', 'Tijolo', 'Areia', 'Escada', 'Joelho', 'Torneira', 'Chuveiro', 'Pia', 'Ducha', 'Vaso Sanitário', 'Tubo',
         'Ralo','Lâmpada','Luminária','Cabo', 'Tomada','Interruptor','Fechadura','Mangueira','Cadeado']

complementos = [['Tropical 5 mm','Ondulada 6mm','em Cerâmica','Translúcida','Ecológica'],
               ['Ondulado','Plano'],
               ['Acrílica','para Gesso','Látex'],
               ['Brilhante','anti Aderente','de Parede'],
               ['Acrílica Branca','Corrida PVA','Corrida Acrílica','PVA','Corrida 25','Acríclica 18'],
               ['Azulejo','Parede', 'Decorativo','Banheiro', 'Piso'],
               ['Contato','Cano','PVC','Papel','Parede'],
               ['Com cabeça 12x12mm','Com cabeça 15x15mm','Com Cabeça em Aço 12x12mm','Com cabeça em Aço 15x15mm','para Madeira 12x12mm','em Aço para Madeira 12x12mm'],
               ['Telha','Fixação','8 mm','Com Cabeça Chata','Com Arruela'],
               ['18 mm','20 mm','15 mm','18','25'],
               ['50 kg','25 kg'],
               ['Baiano','Vidro','Lajota','Vazado'],
               ['Fina','Grossa'],
               ['Alumínio 5 degraus','Articulada 5 degraus','Alumínio 10 degraus','Articulada 10 degraus'],
               ['90 graus Soldável','45 graus Soldável','90 graus em PVC', '45 graus em PVC'],
               ['de Cozinha','de Banheiro','de Cozinha com Filtro'],
               ['de Plástico','de Aço','de Alumínio','Elétrico de Metal', 'Elétrico de Plástico'],
               ['de Aço para Cozinha','de Mármore para Banheiro','de Porcelada para Banheiro','de Mármore para Cozinha'],
               ['de Plástico','de Metal'],
               ['com Caixa Acoplada','sem Caixa Acoplada'],
               ['Esgoto 100 mm','Esgoto 150 mm','PVC 100 mm', 'PVC 150 mm'],
               ['Linear','Inox','de Plástico','de Aço'],
               ['Fluorescente 60W','Led 60W','Fluorescente 90W','Led 90W'],
               ['Fluorescente 60W','Led 60W','Fluorescente 90W','Led 90W'],
               ['Flexível 25','Elétrico 10mm','Flexível 30', 'Elétrico 15mm','Elétrico 6mm','Elétrico 8 mm'],
               ['Simples','com Interruptor', 'Dupla Simples', 'Dupla com Interruptor'],
               ['Simples','Duplo','para Ar Condicionado','para Contador Elétrico'],
               ['Externa de Porta Alavanca','Externa de Porta Redonda','Interna'],
               ['10 m','15 m', '30 m', '40 m'],
               ['Em Latão Clássico','com Segredo']
               ]

marcas = ['Weder','Votoran','Vonder','Utimil','Twb','Tigre','Seven','Santa Luzia','Pulvitec','Prosteel','A Brazilian',
          'Aminox','Aquainox','Astra','Blukit','Censi','Costa Navarro','Deca','Docol','Esteves','Fani Metais','Lorenzetti',
          'Moldenox','Montana','Nagano','Oderm','Hydra','Moldenox','Montana','Wog','Dattiti','Silvana','Aliança','Papaiz',
          'Tramontina','Fischer']

produtos = []
codigos = np.random.choice(np.arange(123456789012, 252839503896576,100000000), 5000, replace=False)
np.random.shuffle(codigos)
k = 0

for i,nome in enumerate(nomes):
    
    complemento = complementos[i]
    
    for c in complemento:
        descricao = nome + ' ' + c
        m = np.random.choice(np.arange(np.size(marcas)), 25, replace=False)
        for j in m:
            sql = 'INSERT INTO produtos VALUES('
            codigobarras = str(codigos[k])
            estoque = str(np.random.randint(5,10))
            minimo = '5'
            preco = str(np.round(np.random.rand()+np.random.randint(10), 2))
            status = '1'
            sql += str(k+1)+",\""+codigobarras+"\",\""+descricao+"\","+estoque+",\""+marcas[j]+"\","+minimo+","+preco+","+status+")"
            k+=1

            produtos.append(sql)

SQL = ''
#np.random.shuffle(produtos)

for p in produtos:
    SQL += p+';\n'
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    