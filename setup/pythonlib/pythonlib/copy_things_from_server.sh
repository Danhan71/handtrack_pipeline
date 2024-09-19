# To copy things from server (originally from bonobo) to gorilla.

# analyses
rsync --verbose -avz /mnt/Freiwald/ltian/backup/bonobo/data2/analyses/database /gorilla1/analyses/
#rsync --verbose -avz /mnt/bonobo/data2/analyses/database/BEH /gorilla1/analyses/database/
rsync --verbose -avz /mnt/bonobo/data2/analyses /gorilla1/

# neural preprocess
rsync --verbose -avz /mnt/Freiwald/ltian/backup/bonobo/data3/recordings /gorilla1/neural_preprocess/

# monkeylogic pkl processed files
# NOT WORKING
# rsync -rav0z --include-from=<(ls /mnt/hopfield_data01/ltian/backup/bonobo/data2/animals/Pancho/ | awk -F. '$1>220500') --include '*/' --include '*.pkl' --exclude '*' /mnt/hopfield_data01/ltian/backup/bonobo/data2/animals/Pancho/ /gorilla1/test/
# rsync -rav0z --include '*/' --include '*.pkl' --exclude '*' /mnt/hopfield_data01/ltian/backup/bonobo/data2/animals/Pancho/ /gorilla1/test/

# This filters to only take pkl, but doesnt filter by date...
rsync -avz --include '*/' --include '*.pkl' --exclude '*' --prune-empty-dirs /mnt/Freiwald/ltian/backup/bonobo/data2/animals/Pancho /gorilla1/animals/
rsync -avz --include '*/' --include '*.pkl' --exclude '*' --prune-empty-dirs /mnt/bonobo/data2/animals/Pancho /gorilla1/animals/
rsync -avz --include '*/' --include '*/figures/*' --include '*.pkl' --exclude '*' --prune-empty-dirs /mnt/bonobo/data2/animals/Pancho /gorilla1/animals/

