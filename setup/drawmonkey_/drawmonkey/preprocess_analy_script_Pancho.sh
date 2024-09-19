# ALL
# dates=(231011 231013 231018 231023 231024 231026 231101 231102 231103 231106 231109 231110 231118 231120 231121 231122 231128 231129 231201)

# SPLIT INTO SECTIONS
# dates=(231011 231013 231018 231023 231024 231026 231101 231102 231103 231106 231109 231110) # all up to 231110
# dates=(231118 231120 231121 231122 231128 231129 231201) # gramdirstimpancho3b (from 231118 onwards)
dates=(231208 231211 231214 231215)


# IGNORE
# dates=(231013 231018 231023 231024) # dir vs. AnBm

animal=Pancho
for d in "${dates[@]}"
do
  logfile=preprocess_logs/preprocess_script_${d}_${animal}
  echo ${logfile}
  python -m tools.preprocess ${animal} $d $d Y g & 2>&1 | tee ${logfile}
  sleep 2s
done
