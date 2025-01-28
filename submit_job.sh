#!/bin/bash


#SBATCH -A demelo                       # Account name
#SBATCH --partition sorcery             # Partition name
#SBATCH --time=32:00:00                 # Set a time limit of 1 hour
#SBATCH -C GPU_MEM:40GB                   # Request 1 GPU
#SBATCH --mem=32GB                      # Request 50 GB of system memory
#SBATCH --gpus=1

#timestamp=$(date +%Y%m%d_%H%M%S)
#SBATCH --output=/hpi/fs00/home/afsana.mimi/llama_project/output/job_output_hugging_face_data.txt  # Standard output

 
# Activate the Python environment
source /hpi/fs00/home/afsana.mimi/llama_project/icl/bin/activate

echo "done"

# Run the Python script
python -u /hpi/fs00/home/afsana.mimi/llama_project/more_test_example.py  #added u to see output
