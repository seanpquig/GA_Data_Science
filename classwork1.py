#!/usr/bin/python
# Import required libraries
import sys

# Start a counter and store the textfile in memory
count = 0
count_non0 = 0
age_sum = 0
imp_sum = 0
click_sum = 0
max_age = 0
age_set = set()

lines = sys.stdin.readlines()
lines.pop(0)

# For each line, find the sum of various columns in the list.
for line in lines:
  count = count + 1
  imp_sum = imp_sum + int(line.strip().split(',')[2])
  click_sum = click_sum + int(line.strip().split(',')[3])
  age = int(line.strip().split(',')[0])
  if age != 0:
    age_sum = age_sum + age
    count_non0 = count_non0 + 1
  # Check if current age is oldest so far
  if age > max_age:
    max_age = age
  age_set.add(age)

# Write to console
print "\nNum records:  ", count
print "Num nonzero age records:  ", count_non0
print "Age sum:  ", age_sum
print "Impression sum:  ", imp_sum
print "Click sum:  ", click_sum
print "Average age:  ", float(age_sum) / float(count_non0)
print "CTR:  ", float(click_sum) / float(imp_sum)
print "Max age:  ", max_age, '\n'
print len(age_set)
