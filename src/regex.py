import sys, re, os



if len(sys.argv) < 2:
	sys.exit("./poeni .html")




try:
	with open(sys.argv[1], "r") as f:
		data = f.read()
except IOError:
	sys.exit("File open failed")


ri = re.compile(r"(.?)\,\s(.?)")
delovi= []
tmp = ri.search(data)

new_data = re.sub(ri, r'\1 \2', data)
with open(sys.argv[1], 'w') as f:
	f.write(new_data)

