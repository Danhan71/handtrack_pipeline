#!/bin/bash

this_dir="$(dirname "$0")"
config_file=${this_dir}/../config
source $config_file
if [[ $? -ne 0 ]]; then 
	echo "no config file."
	exit 1
fi

expt='220814_neuralbiasdir3c'
animal='Pancho'

# Output log file
LOGFILE="logs/${animal}/${expt}_resource_usage.log"
OLOG="logs/${animal}/${expt}_out.log"

# Interval between logs (in seconds)
INTERVAL=5

# Function to log resource usage
log_usage() {
    # Get timestamp
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

    # GPU memory usage (using nvidia-smi)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{ sum += $1 } END { print sum " MB" }')

    # CPU and RAM usage (using top)
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4"%"}')
    RAM_USAGE=$(free -m | awk '/Mem:/ {print $3" MB / "$2" MB"}')

    # Log output
    echo "$TIMESTAMP - GPU Memory: $GPU_MEM, CPU Usage: $CPU_USAGE, RAM Usage: $RAM_USAGE" >> "$LOGFILE"
}

# Run resource monitoring in the background
(
    while true; do
        log_usage
        sleep $INTERVAL
    done
) &

# Get the PID of the background process (monitoring script)
MONITOR_PID=$!

# Run the target script
${pipe_path}/notrain_run_script.sh -b -e ${expt} -a ${animal} -g > ${OLOG}

# Once the target script finishes, kill the monitoring process
kill $MONITOR_PID

# Optional: Inform user that monitoring has finished
echo "Resource monitoring finished, log saved to $LOGFILE"
