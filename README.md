------------------- code setup :rocket:-----------------------------
1. Code for experimenting with hugging face data : code/new_data_experiment.py /n
2. Code for experimenting with new synthetic generated data : code/more_test_example.py //
3. Code for experimenting with hugging face data but only 1 example: code/memory_eff_backup.py//


-------------------------- plots :rocket:-------------------------
1. The result for the experimenting with hugging faces 300 data is available on: new_sbatch_plots //

(epoch_i_precision_scores.png = for i incontext example how the rouge score precision is changing for the 300 test data(from hugging face) //
(average rouge_scores vs the number of examples.png = Our final result :star: After increasing the number of incontext example how the average rouge precesion score of the 300 test data is chnaging)

------------------------ slurm file :rocket: ----------------------

submit_job.sh has the all necessary setup to put a job in the cloud. The runtime is set to 32h and 40Gb GPU memory is being requested.

Note: For interactive job (For debugging purpose)
Use salloc command:

for accessing gx21 use: salloc -A demelo -p sorcery --gpus=1 --mem=80G
for accessing gx01-gx05 use: salloc -A demelo -p sorcery --gres=gpu:a100:1 --mem=80G

-------------------- Some basic command in the cloud :fire:-----------------
1. squeue -u afsana.mimi (shows the job that is running on the cloud along with the node)
2. scontrol show job <job_id> (shows more details including when the submitted slurm file will start running)
3. scancel -u afsana.mimi (cancels all the jobs at once running under the user name)
4. scancel -u <job_id> (cancels only a particular job)

------------------ Environment creatrion :fire: --------------------
1. use python3 -m venv icl for creating a new envrionment called icl
2. pip install -r requirements.txt
3. For activing the environment use : source icl/bin/activate
4. For deactivating : deactivate





   
