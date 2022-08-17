import re


with open("requirements.txt", "r") as fp:
	lines = fp.read().splitlines()

lines = [re.sub('==', '>=', s) for s in lines]
print (lines)