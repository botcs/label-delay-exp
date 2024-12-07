#!/bin/bash

directory=$1  # Replace with your designated directory

# Function to measure write time
measure_write_time() {
  local file="$1"
  local start_time=$(date +%s%N)
  dd if=/dev/zero of="$file" bs=1M count=1 status=none
  local end_time=$(date +%s%N)
  local elapsed_time=$((end_time - start_time))
  local elapsed_time_ms=$(awk "BEGIN { printf \"%.2f\", $elapsed_time / 1000000 }")
  echo "$elapsed_time_ms"
}

# Function to measure read time
measure_read_time() {
  local file="$1"
  local start_time=$(date +%s%N)
  dd if="$file" of=/dev/null bs=1M count=1 status=none
  local end_time=$(date +%s%N)
  local elapsed_time=$((end_time - start_time))
  local elapsed_time_ms=$(awk "BEGIN { printf \"%.2f\", $elapsed_time / 1000000 }")
  echo "$elapsed_time_ms"
}

# Function to display progress bar
progress_bar() {
  local width=50
  local percent=$1
  local num_chars=$((percent * width / 100))
  local bar=$(printf "%${num_chars}s" | tr ' ' '#')
  local empty=$(printf "%$((width - num_chars))s" | tr ' ' ' ')
  printf "\r[%s%s] %d%%" "$bar" "$empty" "$percent"
}

# Perform measurements
declare -a write_times
declare -a read_times

echo "Measuring write and read times..."
for ((i=1; i<=100; i++))
do
  temp_file="$directory/temp_$i.txt"
  write_times[$i]=$(measure_write_time "$temp_file")
  read_times[$i]=$(measure_read_time "$temp_file")
  rm "$temp_file"

  progress=$((100 * i / 100))
  progress_bar "$progress"
done

echo  # Move to the next line after the progress bar

# Calculate min, average, and max times
min_write_time=$(echo "${write_times[*]}" | tr " " "\n" | sort -n | head -n 1)
avg_write_time=$(echo "${write_times[*]}" | awk '{sum+=$1} END {printf "%.2f", sum/NR}')
max_write_time=$(echo "${write_times[*]}" | tr " " "\n" | sort -n | tail -n 1)

min_read_time=$(echo "${read_times[*]}" | tr " " "\n" | sort -n | head -n 1)
avg_read_time=$(echo "${read_times[*]}" | awk '{sum+=$1} END {printf "%.2f", sum/NR}')
max_read_time=$(echo "${read_times[*]}" | tr " " "\n" | sort -n | tail -n 1)

# Print the results
echo "Write times (in milliseconds):"
echo "Minimum: $min_write_time"
echo "Average: $avg_write_time"
echo "Maximum: $max_write_time"
echo
echo "Read times (in milliseconds):"
echo "Minimum: $min_read_time"
echo "Average: $avg_read_time"
echo "Maximum: $max_read_time"
