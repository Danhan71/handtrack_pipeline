dates=(231025 231026 231027 231029 231101 231102 231103 231106 231107 231108 231109 231110 231113 231114 231115)
# dates=(231106 231107)
animal=Diego
for d in "${dates[@]}"
do
  logfile=preprocess_logs/preprocess_script_${d}_${animal}
  echo ${logfile}
  python -m tools.preprocess ${animal} $d $d Y g & 2>&1 | tee ${logfile}
  sleep 5s
done
