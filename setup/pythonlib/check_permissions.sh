#!/bin/bash

# Collect all lines that have WARNING
grep -nr --exclude=\*.{txt,sh} "WARNING" * > corrupted_lines.txt

# OLD VERSION, below is for one file. this may be useful
# if want to loop over files
# See https://unix.stackexchange.com/questions/360540/append-to-a-pipe-and-pass-on
# https://unix.stackexchange.com/questions/238522/extract-all-lines-from-a-file-starting-with-some-sequence-and-then-output-it-to
#f="2020-05-23_data1"
#fout="corrupted_lines.txt"
#touch test.txt
#echo "FILE:$f" >> ${fout}
#echo "$(grep -nr WARNING $f)/test" >> ${fout}

# List only the filenames
# https://stackoverflow.com/questions/16956810/how-do-i-find-all-files-containing-specific-text-on-linux
# https://unix.stackexchange.com/questions/333121/how-to-find-lines-containing-a-string-and-then-printing-those-specific-lines-and
grep -rnwl --exclude=\*.{txt,sh} "permitted" * > permissions_error_files.txt
