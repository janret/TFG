#!/bin/bash

# Default values (empty)
subject_dir=""
num_timepoints=""

# Process command-line flags
while getopts ":d:t:h" opt; do
  case $opt in
    d) subject_dir="$OPTARG" ;;
    t) num_timepoints="$OPTARG" ;;
    h) 
      echo "Usage: $0 -d <subject_directory> -t <num_timepoints>"
      echo "Example: $0 -d /path/to/subject -t 20"
      exit 0
      ;;
    \?) 
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :) 
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Validate mandatory parameters
if [ -z "$subject_dir" ] || [ -z "$num_timepoints" ]; then
  echo "ERROR: Both parameters are required"
  echo "Usage: $0 -d <subject_directory> -t <num_timepoints>" >&2
  exit 1
fi

# Validate directory exists
if [ ! -d "$subject_dir" ]; then
  echo "ERROR: Directory not found: $subject_dir" >&2
  exit 1
fi

# Validate timepoints is a positive integer
if ! [[ "$num_timepoints" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: Timepoints must be a positive integer" >&2
  exit 1
fi

# Build freeview command
freeview_cmd="freeview -layout 2 -viewport coronal "

# Add template images
freeview_cmd+="-v ${subject_dir}/sub-001_template.mgz "
freeview_cmd+="-v ${subject_dir}/sub-001_template_seg.mgz:colormap=lut:opacity=0.5 "

# Add timepoints and segmentations
for tp in $(seq -f "%03g" 1 "${num_timepoints}"); do
    freeview_cmd+="-v ${subject_dir}/sub-001_tp${tp}.mgz "
    freeview_cmd+="-v ${subject_dir}/sub-001_tp${tp}_seg.mgz:colormap=lut:opacity=0.5 "
done

# Execute command
eval "${freeview_cmd}"


# # Basic usage
# ./script.sh -d /path/to/subject -t 15

# # Help message
# ./script.sh -h