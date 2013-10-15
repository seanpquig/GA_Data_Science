# Import required libraries
import sys
import datetime

"""
Class that is used to access and store data
for a unique age, gender, signed_in combo
"""
class AggregateData:
    def __init__(self, clicks, imps):
        self.count = 1
        self.clickSum = clicks
        self.clickMax = clicks
        self.impSum = imps
        self.impMax = imps
    def update(self, clicks, imps):
        self.count += 1
        self.clickSum += clicks
        self.impSum += imps
        if clicks > self.clickMax:
            self.clickMax = clicks
        if imps > self.impMax:
            self.impMax = imps



# Track program runtime
start_time = datetime.datetime.now()

# Get lines from input csv file, skip 1st header line
lines = sys.stdin.readlines()
lines.pop(0)

# hash structure with tuple keys
aggregates = {}

# For each input line get data and store in hash structure
print "importing data..."
for line in lines:
    age = int(line.strip().split(',')[0])
    gender = int(line.strip().split(',')[1])
    imps = int(line.strip().split(',')[2])
    clicks = int(line.strip().split(',')[3])
    signedIn = int(line.strip().split(',')[4])

    key = (age, gender, signedIn)
    if key in aggregates:
        aggregates[key].update(clicks, imps)
    else:
        aggregates[key] = AggregateData(clicks, imps)


# output csv file
print "exporting data..."
f = open('aggregates2.csv', 'w')
f.write('age,gender,signed_in,avg_click,' 
        'avg_impressions,max_click,max_impressions\n')

# write data from hash structure to csv output
for i in aggregates:
    age = i[0]
    gender = i[1]
    signedIn = i[2]
    count = aggregates[i].count
    clickSum = aggregates[i].clickSum
    impSum = aggregates[i].impSum
    avgClick = float(clickSum)/count
    avgImp = float(impSum)/count
    clickMax = aggregates[i].clickMax
    impMax = aggregates[i].impMax
    f.write(`age` + ',' + `gender` + ',' + `signedIn` + ',' + `avgClick`
            + ',' + `avgImp` + ',' + `clickMax` + ',' + `impMax` + '\n')


#Check final runtime
end_time = datetime.datetime.now()
print "runtime:  ", end_time - start_time
