#!/usr/bin/env bash
####
################################## START OF EMBEDDED PBS COMMANDS ##########################
#PBS -S /bin/bash  #### Default Shell to be Used
#PBS -N XNet_GenerateValOutputs  #### Job Name to be listed in qstat
#PBS -o /scratch/hasm/Data/Lesion/X-net_Test/logs/$PBS_JOBNAME_$PBS_JOBID.stdout  #### stdout default path
#PBS -e /scratch/hasm/Data/Lesion/X-net_Test/logs/$PBS_JOBNAME_$PBS_JOBID.stderr  #### stderr default path
#PBS -M sungminha@wustl.edu  #### email address to nofity with following options/scenarios
#PBS -m abe ####abort, end notifications - see below lines for more options
#PBS -l nodes=1:ppn=1,mem=16gb,walltime=4:00:00 #### 1 node, 1 processor, 1 gpu, 8GB of memory, 15 hours of wall time requests
############################## END OF DEFAULT EMBEDDED PBS COMMANDS #######################

parent_dir="/home/hasm/comp_space/Data/Lesion/X-net_Test/X-Net_20210401_CompleteDataSet_3Folds";
script="${parent_dir}/generate_test_output_for_input.py";
if [ ! -e "${script}" ];
then
  echo -e "ERROR: script ( ${script} ) does not exist.";
  return;
fi;

echo -e "\n\n \
export HDF5_USE_FILE_LOCKING='FALSE'";
export HDF5_USE_FILE_LOCKING='FALSE';

echo -e "\n\n \
conda activate py3_7_xnet";
conda activate py3_7_xnet;

cd "${parent_dir}";
echo -e "\n\n \
pwd";
pwd;

python ${script};