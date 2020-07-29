import numpy as np

nomes = ['Wayne Whitley','Justine Alvarado','Darrel Sweet','Kitra Ewing',
         'Felix Church','Deacon Larson','Kuame Cannon','Amela Michael',
         'Melanie Michael','Odysseus Alford','Aubrey Beach','Tatyana Hardin',
         'Chester Battle','Eric Jacobson','Cody Malone','Travis Delacruz',
         'Cairo Santana','Orli Conner','Adara Horton','Portia Burt','Dakota Bartlett',
         'Deirdre Charles','Courtney Kim','Troy Russell','Louis Zamora','Leigh Shepherd',
         'Elmo Fields','Nelle Murray','Kimberley Mccullough','Patience Christensen',
         'Quintessa Patrick','Leonard York','Kennedy Glover','Driscoll Mclaughlin',
         'Sydney Mueller','Carson Reilly','Damon Patton','Iona Carrillo','Reese Wilkerson',
         'Latifah Barlow','Maia Phillips','Myles Vazquez','Yoshio Bartlett','Madeline Merritt',
         'Elijah Snider','Herman Haley','Kerry Chandler','Marsden Steele','Cheyenne Harmon',
         'Brenda Melendez','Kibo Tyson','Cadman Finley','Adara Acosta','Reuben Sherman',
         'Fritz Austin','Vance Whitfield','Charlotte Torres','Gage Bender','Nola Berger',
         'Evan Church','Cherokee Cunningham','Emery Hurley','Rose Chapman','Mechelle Howard',
         'Deanna Gonzalez','Claire Gonzales','Ross Knapp','Sydney Farmer','Gary Orr',
         'Willow Brewer','Chaney Donovan','Garrison Luna','Rhonda Sutton','Yuli Cortez',
         'Lucian Dawson','Tatum Meyer','Thor Cameron','Otto Hernandez','Ria Nixon',
         'Angela Allison','Paula Underwood','Riley Barnett','Shea Boone','Tana Estes',
         'Rigel Perry','Wylie Mcconnell','Baxter Mcintosh','Ezekiel Hardy','Eve Casey',
         'Malik Workman','Chase Kirkland','Wendy Hernandez','Aidan Clarke','Belle Black',
         'Kadeem Allison','Gage Johns','Xenos Baker','Cooper Atkins','Kermit Sampson',
         'Preston Jacobs']

enderecos = ['P.O. Box 603, 6901 Sed Rd.','Ap #627-8472 Eget Ave','485 Libero. Av.',
             '7579 Velit. St.','993 Eget, Rd.','5600 Varius Road','P.O. Box 421, 1983 Iaculis St.',
             '5455 Nunc Rd.','777-7385 Ligula Avenue','P.O. Box 954, 1806 Montes, Rd.',
             '5503 Quam Avenue','P.O. Box 899, 2685 Sodales Street','374 Vitae Road','115-2030 Ut Street',
             'Ap #380-172 Et Road','P.O. Box 972, 6979 Luctus St.','340 Praesent Rd.','734-4140 Metus Avenue',
             'Ap #673-9053 Morbi Road','Ap #801-1392 Aliquam, Avenue','P.O. Box 651, 5102 Enim, Road',
             'Ap #343-8656 Facilisis. Road','627-3352 Porttitor St.','Ap #901-2058 Penatibus Street',
             'P.O. Box 609, 8713 Quis Road','4954 Primis Avenue','213 Risus Rd.','9937 In Rd.',
             'Ap #668-5621 Risus. Rd.','P.O. Box 712, 5103 Fringilla St.','Ap #800-8093 Phasellus St.',
             '564-1813 Pellentesque St.','P.O. Box 373, 536 Porttitor Ave','652-4315 Orci Ave',
             'P.O. Box 812, 2428 Duis Rd.','8761 Pede, Rd.','Ap #360-8942 Ultrices Avenue',
             'Ap #655-7673 Nonummy Street','835-9716 Eleifend. Street','1427 Facilisis Avenue',
             '436 Tristique Avenue','P.O. Box 991, 4072 Id Rd.','P.O. Box 109, 9510 Semper Av.',
             '7352 Eu Street','Ap #949-7464 Id Street','Ap #113-3217 Aliquet Avenue','1304 In, St.',
             'Ap #531-2452 Vivamus St.','P.O. Box 471, 6151 Urna Ave','Ap #284-9371 Metus Avenue',
             'P.O. Box 540, 7251 Non Avenue','2611 Purus. Street','306-7650 Mollis Av.',
             'P.O. Box 296, 1753 Arcu. Rd.','1690 Rutrum Ave','P.O. Box 497, 5824 Quisque Street',
             'P.O. Box 159, 4168 Ultrices Av.','605-3342 Posuere Av.','P.O. Box 206, 2753 Aenean St.',
             'P.O. Box 238, 6817 Lacinia. Av.','404-9275 Molestie Street','2013 Sed Avenue',
             'P.O. Box 992, 2939 Eu Rd.','P.O. Box 556, 3597 Gravida Ave','579-6386 Mi Street',
             '3731 Donec Ave','525-5003 Aliquet Rd.','121-936 Varius Rd.','854-4820 Eget, Av.',
             '6450 Pellentesque Rd.','P.O. Box 969, 4193 Suspendisse Av.','3147 Mi Avenue',
             'Ap #928-897 Sed, Avenue','9784 Ipsum Road','P.O. Box 120, 6973 Lorem Rd.','2733 Ipsum. Ave',
             'P.O. Box 619, 9012 Donec St.','519 Sit Avenue','P.O. Box 387, 9498 Eu St.',
             'Ap #429-6285 Sed, Street','373-8986 Magna, Rd.','Ap #698-7210 Sit St.',
             '1227 Ultricies Avenue','P.O. Box 187, 7176 Orci Avenue',
             'P.O. Box 342, 942 Neque. Rd.','P.O. Box 706, 6923 Donec Ave','P.O. Box 601, 2716 Elementum, Av.',
             'Ap #726-5349 Ridiculus Road','502 Dignissim Ave','4006 Sed Street',
             'P.O. Box 396, 7093 Egestas. Ave','719 Semper St.','P.O. Box 777, 9149 Cum Ave',
             '7076 Magna. Rd.','Ap #501-2809 Donec Av.','426-1937 Sed Av.','764-9159 Aliquam St.',
             'Ap #153-4369 Eros. Street','P.O. Box 379, 9299 Mollis Ave','P.O. Box 740, 145 Mi Rd.']

def gerarEmail(nome):
    n = nome.split(' ')
    email = n[0]+'@gmail.com'
    return email

def gerarTelefone():
    p1 = np.random.randint(88888,99999)
    p2 = np.random.randint(1000,9999)
    telefone = '(86)'+str(p1)+'-'+str(p2)
    return telefone

def gerarCPF():
    #34.342.432/4324-32
    p1 = np.random.randint(100,999)
    p2 = np.random.randint(100,999)
    p3 = np.random.randint(100,999)
    p4 = np.random.randint(10,99)
    
    cpf = str(p1)+'.'+str(p2)+'.'+str(p3)+'-'+str(p4)
    return cpf

SQL = ''
for i, nome in enumerate(nomes):
    cpf = gerarCPF()
    email = gerarEmail(nome)
    endereco = enderecos[i]
    status = '1'
    telefone1 = gerarTelefone()
    telefone2 = gerarTelefone()
    
    sql = 'INSERT INTO clientes VALUES('+str(i+2)+',\"'+cpf+'\",\"'+email+'\",\"'+endereco+'\",\"'+nome+'\",'+status+',\"'+telefone1+'\",\"'+telefone2+'\");'
    SQL += sql+'\n'











