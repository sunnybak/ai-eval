
# run the portal.sh script
# hash of the evals folder
# hash of the current commit
# results/scores pickle

#!/bin/bash

# Function to calculate the hash of a directory
calculate_directory_hash() {
  local directory=$1
  find "$directory" -type f -print0 | sort -z | xargs -0 sha1sum | sha1sum | awk '{print $1}'
}

# Function to get the hash of the latest commit in the repository
get_latest_commit_hash() {
  local repo_path=$1
  cd "$repo_path"
  git rev-parse HEAD
}

# Set the path to your local directory
local_directory="$(pwd)/evals"

# Set the path to your repository
repo_path="/Users/shikharbakhda/Desktop/root/ai-lab/ai-eval"

# Calculate the hash of the local directory
directory_hash=$(calculate_directory_hash "$local_directory")

# Get the hash of the latest commit in the repository
latest_commit_hash=$(get_latest_commit_hash "$repo_path")

# Concatenate the directory hash and the latest commit hash
combined_hash="$directory_hash-$latest_commit_hash"

echo "Combined hash: $combined_hash"