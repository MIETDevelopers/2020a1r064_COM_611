import csv
num_attributes = 3
a = []
print('\nThe Given Training Data Set : \n')
with open("C:\\Users\\DIKSHA\\Documents\\exp1.csv","r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        a.append(row)
        print(row)
        
print('\nThe Initial Value of Hypothesis : ')
hypothesis = ['0'] * num_attributes
print(hypothesis)

for j in range(0,num_attributes):
    hypothesis[j] = a[0][j]
    
print('\nFind-S Algorithm : Finding a Maximally Spcific Hypothesis')

for i in range(0,len(a)):
    if a[i][num_attributes] == 'Positive':
        for j in range(0,num_attributes):
            if a[i][j] != hypothesis[j]:
                hypothesis[j] = '?'
            else:
                hypothesis[j] = a[i][j]
                
    print('\nFor Training Instance No. {0} the Hypothesis is : '.format(i),hypothesis)
print('\nThe Maximally Specific Hypothesis for a given Training Example : ')
print(hypothesis)