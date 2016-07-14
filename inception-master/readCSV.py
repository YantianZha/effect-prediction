import csv

reader = csv.reader(open('./features.csv','r+'))
mydict = dict(reader)
print mydict[1]
