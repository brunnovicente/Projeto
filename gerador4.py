import numpy as np

produtos = np.arange(2850)+1
fornecedores = np.arange(26)+1

k = 1
SQL = ''
for p in produtos:
    
    n = np.random.randint(1,5)+1
    forn = np.random.choice(fornecedores, n, replace=False)
    
    for f in forn:
        sql = 'INSERT INTO produto_fornecedor VALUES('+str(k)+','+str(f)+','+str(p)+');'
        SQL += sql + '\n';
        k += 1