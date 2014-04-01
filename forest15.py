""" Writing my first randomforest code.
Author : AstroDave
Date : 23rd September, 2012
please see packages.python.org/milk/randomforests.html for more

""" 

def train():

    import pdb
    import copy
    import numpy as np
    import csv as csv
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn import svm
    
    from sklearn import tree
    csv_file_object = csv.reader(open('csv/train_noid.csv', 'rb')) #Load in the training csv file
    header = csv_file_object.next() #Skip the fist line as it is a header
    train_data=[] #Creat a variable called 'train_data'
    for row in csv_file_object: #Skip through each row in the csv file
        train_data.append(row) #adding each row to the data variable
    train_data = np.array(train_data) #Then convert from a list to an array

    #normaliza class
    train_data[train_data[0::,1] == '1',1] = 1.0
    train_data[train_data[0::,1] == '2',1] = 2.0
    train_data[train_data[0::,1] == '3',1] = 3.0
    
    #I need to convert all strings to integer classifiers:
    #Male = 1, female = 0:
    
    train_data[train_data[0::,3]=='male',3] = 1.0
    train_data[train_data[0::,3]=='female',3] = 0.0
    #embark c=0, s=1, q=2
    train_data[train_data[0::,10] =='C',10] = 1.0
    train_data[train_data[0::,10] =='S',10] = 2.0
    train_data[train_data[0::,10] =='Q',10] = 3.0

    
    #Create 2 more columns from name: family name, title
    train_data = np.append(train_data,train_data[:,[0,1]],1)

    last_names = {}
    tickets = {}
    titles = {"Mr":0,"Mrs":1,"Miss":2,"Master":3}
    name_count = 0
    ticket_count = 0
    
    for i in range(train_data.shape[0]):
        name = train_data[i,2]
        val = name.replace(".",",").split(",")
        if val[1].strip() == "Don" or val[1].strip() == "Rev" or val[1].strip() == "Dr"  or \
                                      val[1].strip() == "Major"  or val[1].strip() == "Sir" or \
                                      val[1].strip() == "Col"   or val[1].strip() == "Capt" or \
                                      val[1].strip() == "Jonkheer": val[1] = "Mr"
        if val[1].strip() == "Mme" or val[1].strip() == "Ms"  or val[1].strip() == "Lady" or \
                                      val[1].strip() == "the Countess"  or val[1].strip() == "Dona": val[1] = "Mrs"
        if val[1].strip() == "Mlle": val[1] = "Miss"  
        
        if val[0].strip() not in last_names.keys():
            last_names[val[0].strip()] = name_count
            name_count += 1
        #titles
        train_data[i,12] = titles[val[1].strip()]#/3.0
        #ticket
        if train_data[i,7] not in tickets.keys():
            tickets[train_data[i,7]] = ticket_count
            ticket_count += 1    
    
    for i in range(train_data.shape[0]):
        name = train_data[i,2]
        val = name.replace(".",",").split(",")     
        train_data[i,11] = last_names[val[0].strip()]#/float(len(last_names.keys())-1)
        train_data[i,7] = tickets[train_data[i,7]]#/float(len(tickets.keys())-1)
   
    
    # create a joint feature of tiket and last_name
    #for i in tickets.keys():
    #    min_family = np.min(train_data[(train_data[:,7] == str(tickets[i])),11].astype(np.float))
    #    train_data[(train_data[:,7] == str(tickets[i])),11] = min_family


    

    #I need to fill in the gaps of the data and make it complete.
    #So where there is no price, I will assume price on median of that class
    #Where there is no age I will give median of all ages

    #All the ages with no data make the median of the data among the passengers with the same sex, number of sibsp and parch
    #decision to predict age
    train_age_exist = train_data[(train_data[:,4] != ''),:]
    train_age_exist = train_age_exist[:,[4,3,5,6,11,12]]
    train_age_missing = train_data[(train_data[:,4] == ''),:]
    train_age_missing = train_age_missing[:,[3,5,6,11,12]]
    print 'Training missing age values'

    tree_reg = tree.DecisionTreeRegressor()
    tree_reg = tree_reg.fit(train_age_exist[0::,1::],train_age_exist[0::,0])

    print 'Predicting missing age values'
    output = tree_reg.predict(train_age_missing)

    train_data[(train_data[:,4] == ''),4] = output


    #All missing ebmbarks just make them embark from most common place among the passengers of the same class

    for i in range(train_data.shape[0]):
        if train_data[i,10] == "":
            train_data[i,10] = np.round(np.mean(train_data[(train_data[0::,10]\
                                                       != '') * (train_data[0::,1] == train_data[i,1]),10].astype(np.float)))
  
 
    #All the missing prices assume median of their respective class
    for i in xrange(np.size(train_data[0::,0])):
        if train_data[i,8] == '':
            train_data[i,8] = np.median(train_data[(train_data[0::,8] != '') &\
                                                 (train_data[0::,0] == train_data[i,1])\
                ,8].astype(np.float))   
    """
    #normalize ticket price
    max_ticket = np.amax(train_data[:,8].astype(np.float))
    train_data[:,8] = np.divide(train_data[:,8].astype(np.float),max_ticket)
    #normalize age
    max_age = np.amax(train_data[:,4].astype(np.float))
    train_data[:,4] = np.divide(train_data[:,4].astype(np.float),max_age)    
    #normalize sibsp
    max_sibsp = np.amax(train_data[:,5].astype(np.float))
    train_data[:,5] = np.divide(train_data[:,5].astype(np.float),max_sibsp)       
    #normalize parch
    max_parch = np.amax(train_data[:,6].astype(np.float))
    train_data[:,6] = np.divide(train_data[:,6].astype(np.float),max_parch)  
    """
    train_data = np.delete(train_data,[2,7,9,11,12],1) #remove the name data, cabin


    with open("csv/train_data_work.csv", "wb") as f:
        writer = csv.writer(f)
        for i in range(train_data.shape[0]):
            writer.writerow(train_data[i,:])

            
    #I need to do the same with the test data now so that the columns are in the same
    #as the training data

    
    test_file_object = csv.reader(open('csv/test_noid.csv', 'rb')) #Load in the test csv file
    header = test_file_object.next() #Skip the fist line as it is a header
    test_data=[] #Creat a variable called 'test_data'
    for row in test_file_object: #Skip through each row in the csv file
        test_data.append(row) #adding each row to the data variable
    test_data = np.array(test_data) #Then convert from a list to an array

    #I need to convert all strings to integer classifiers:

    #normaliza class
    test_data[test_data[0::,0] == '1',0] = 1.0
    test_data[test_data[0::,0] == '2',0] = 2.0
    test_data[test_data[0::,0] == '3',0] = 3.0
    
    #Male = 1, female = 0:
    test_data[test_data[0::,2]=='male',2] = 1.0
    test_data[test_data[0::,2]=='female',2] = 0.0
    #ebark c=0, s=1, q=2
    test_data[test_data[0::,9] =='C',9] = 1.0 #Note this is not ideal, in more complex 3 is not 3 tmes better than 1 than 2 is 2 times better than 1
    test_data[test_data[0::,9] =='S',9] = 2.0
    test_data[test_data[0::,9] =='Q',9] = 3.0



    #Create 2 more columns from name: family name, title
    test_data = np.append(test_data,test_data[:,[0,1]],1)

    last_names = {}
    tickets = {}
    titles = {"Mr":0,"Mrs":1,"Miss":2,"Master":3}
    name_count = 0
    ticket_count = 0
    
    for i in range(test_data.shape[0]):
        name = test_data[i,1]
        val = name.replace(".",",").split(",")
        if val[1].strip() == "Don" or val[1].strip() == "Rev" or val[1].strip() == "Dr"  or \
                                      val[1].strip() == "Major"  or val[1].strip() == "Sir" or \
                                      val[1].strip() == "Col"   or val[1].strip() == "Capt" or \
                                      val[1].strip() == "Jonkheer": val[1] = "Mr"
        if val[1].strip() == "Mme" or val[1].strip() == "Ms"  or val[1].strip() == "Lady" or \
                                      val[1].strip() == "the Countess"   or val[1].strip() == "Dona": val[1] = "Mrs"
        if val[1].strip() == "Mlle": val[1] = "Miss"  
        
        if val[0].strip() not in last_names.keys():
            last_names[val[0].strip()] = name_count
            name_count += 1
        test_data[i,11] = titles[val[1].strip()]# /3.0  
        #ticket
        if test_data[i,6] not in tickets.keys():
            tickets[test_data[i,6]] = ticket_count
            ticket_count += 1
 
    for i in range(test_data.shape[0]):
        name = test_data[i,1]
        val = name.replace(".",",").split(",")     
        test_data[i,10] = last_names[val[0].strip()]#/float(len(last_names.keys())-1)
        test_data[i,6] = tickets[test_data[i,6]]#/float(len(tickets.keys())-1) 

    # create a joint feature of tiket and last_name
    #for i in tickets.keys():
    #   min_family = np.min(test_data[(test_data[:,6] == str(tickets[i])),10].astype(np.float))
    #   test_data[(test_data[:,6] == str(tickets[i])),10] = min_family
        
    #All the ages with no data make the median of the data among the passengers with the same sex, number of sibsp and parch
    #decision tree to predict age
    test_age_exist = test_data[(test_data[:,3] != ''),:]
    test_age_exist = test_age_exist[:,[3,2,4,5,10,11]]
    test_age_missing = test_data[(test_data[:,3] == ''),:]
    test_age_missing = test_age_missing[:,[2,4,5,10,11]]
    print 'Training missing age values'

    tree_reg = tree.DecisionTreeRegressor()
    tree_reg = tree_reg.fit(test_age_exist[0::,1::],test_age_exist[0::,0])

    print 'Predicting missing age values'
    output = tree_reg.predict(test_age_missing)

    test_data[(test_data[:,3] == ''),3] = output



    #All missing ebmbarks just make them embark from most common place among the passengers with the same class

    for i in range(test_data.shape[0]):
        if test_data[i,9] == "":
            test_data[i,9] = np.round(np.mean(test_data[(test_data[0::,9]\
                                                       != '') * (test_data[0::,0] == test_data[i,0]),9].astype(np.float)))


    #All the missing prices assume median of their respectice class
    for i in xrange(np.size(test_data[0::,0])):
        if test_data[i,7] == '':
            test_data[i,7] = np.median(test_data[(test_data[0::,7] != '') &\
                                                 (test_data[0::,0] == test_data[i,0])\
                ,7].astype(np.float))

    """            
    #normalize ticket price
    max_ticket = np.amax(test_data[:,7].astype(np.float))
    test_data[:,7] = np.divide(test_data[:,7].astype(np.float),max_ticket)
  
    #normalize age
    max_age = np.amax(test_data[:,3].astype(np.float))
    test_data[:,3] = np.divide(test_data[:,3].astype(np.float),max_age)      
    #normalize sibsp
    max_sibsp = np.amax(test_data[:,4].astype(np.float))
    test_data[:,4] = np.divide(test_data[:,4].astype(np.float),max_sibsp)       
    #normalize parch
    max_parch = np.amax(test_data[:,5].astype(np.float))
    test_data[:,5] = np.divide(test_data[:,5].astype(np.float),max_parch) 
    """
    
    test_data = np.delete(test_data,[1,6,8,10,11],1) #remove the name data, cabin

    with open("csv/test_data_work.csv", "wb") as f:
        writer = csv.writer(f)
        for i in range(test_data.shape[0]):
            writer.writerow(test_data[i,:])
    
    
    #The data is now ready to go. So lets train then test!
    
    # Here we will train two random forests, once for each sex
    

    print 'Training '
    forest_m = RandomForestClassifier(n_estimators=100, random_state = 0)
    forest_f = RandomForestClassifier(n_estimators=100, random_state = 0)
    #forest = ExtraTreesClassifier(n_estimators=500, random_state = 0)
    #forest = DecisionTreeClassifier(random_state = 0)    
    #forest = GradientBoostingClassifier(n_estimators=500, random_state = 0, learning_rate=0.5, max_depth=1)
    #forest = SGDClassifier() 
    #forest = svm.SVC()  

    
    #split train data into two subsets
    train_data_m = np.delete(train_data[(train_data[:,2] == '1.0'),:],[2],1)
    train_data_f = np.delete(train_data[(train_data[:,2] == '0.0'),:],[2],1)
    
    # First train a tree for males
       
    # k-fold cross-validation
    score_m = []
    k = 10
    test_length_m = train_data_m.shape[0]/k
    print test_length_m
    for i in range(k):
        train = train_data_m[:,:]
        train = np.delete(train_data_m,slice(i*test_length_m,(i+1)*test_length_m),0)
        test = train_data_m[i*test_length_m:(i+1)*test_length_m,:]
        #training
        forest_m = forest_m.fit(train[0::,1::],\
                        train[0::,0])  
        output = forest_m.predict(test[:,1::])
        score_m.append(forest_m.score(test[:,1::],test[:,0]))            

    print "Cross-validation error (males): ", np.mean(score_m)
      
    # k-fold cross-validation
    score_f = []
    k = 10
    test_length_f = train_data_f.shape[0]/k
    print test_length_f
    for i in range(k):
        train = train_data_f[:,:]
        train = np.delete(train_data_f,slice(i*test_length_f,(i+1)*test_length_f),0)
        test = train_data_f[i*test_length_f:(i+1)*test_length_f,:]
        #training
        forest_f = forest_f.fit(train[0::,1::],\
                        train[0::,0])  
        output = forest_f.predict(test[:,1::])
        score_f.append(forest_f.score(test[:,1::],test[:,0]))            

    print "Cross-validation error (females): ", np.mean(score_f)      
        
 
    print "Cross-validation error (total): ", (np.mean(score_f)*test_length_f + np.mean(score_m)*test_length_m)/(test_length_m + test_length_f)   
        
    #pdb.set_trace()
        
    print 'Final predicting'


    forest_m = forest_m.fit(train_data_m[0::,1::],\
                        train_data_m[0::,0])
                        
    forest_f = forest_f.fit(train_data_f[0::,1::],\
                        train_data_f[0::,0])                       

    #split test data into two subsets
    test_data_m = np.delete(test_data[(test_data[:,1] == '1.0'),:],[1],1)
    test_data_f = np.delete(test_data[(test_data[:,1] == '0.0'),:],[1],1)                        
                                             
    output_m = forest_m.predict(test_data_m)
    output_f = forest_f.predict(test_data_f)

 
    #restore output
    output = test_data[:,0] # initialize
    output[(test_data[:,1] == '1.0')] = output_m
    output[(test_data[:,1] == '0.0')] = output_f   
 
    #get train error

    #train_output = forest.predict(train_data[:,1::])

    #print " my score: ", sum(train_data[:,0] == train_output)/float(train_data.shape[0])

    #mean_accuracy = forest.score(train_data[:,1::],train_data[0::,0])

    #print "mean accuracy train data: ", mean_accuracy
    open_file_object = csv.writer(open("csv/forest15.csv", "wb"))
    test_file_object = csv.reader(open('csv/test.csv', 'rb')) #Load in the csv file


    test_file_object.next()
    open_file_object.writerow(["PassengerId","Survived"])
    i = 0
    for row in test_file_object:
        open_file_object.writerow([row[0],output[i].astype(np.uint8)])
        i += 1



if __name__ == '__main__':
    train()