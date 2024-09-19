


if __name__=="__main__":

	import sys
	method = sys.argv[1]

	if method=="remove_h5":
		# remove unecesary h5 files
		from tools.preprocess import remove_unneeded_h5_files
		animal = sys.argv[2]
		date = sys.argv[3]
		remove_unneeded_h5_files(animal, date)
	elif method=="remove_h5_batch":
		# remove h5 looping over animals and dates
		from tools.preprocess import remove_unneeded_h5_files
		animal = sys.argv[2]
		date_first = int(sys.argv[3]) # YYMMDD
		date_last = int(sys.argv[4])

		for date in range(date_first, date_last+1):
			remove_unneeded_h5_files(animal, date)	




