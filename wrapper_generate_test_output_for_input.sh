#!/usr/bin/env bash
####
################################## START OF EMBEDDED PBS COMMANDS ##########################
#PBS -S /bin/bash  #### Default Shell to be Used
#PBS -N XNet_GenerateValOutputs  #### Job Name to be listed in qstat
#PBS -o /scratch/hasm/Data/Lesion/X-net_Test/logs/\${PBS_JOBNAME}_\${PBS_JOBID}.stdout  #### stdout default path
#PBS -e /scratch/hasm/Data/Lesion/X-net_Test/logs/\${PBS_JOBNAME}_\${PBS_JOBID}.stderr  #### stderr default path
#PBS -M sungminha@wustl.edu  #### email address to nofity with following options/scenarios
#PBS -m abe ####abort, end notifications - see below lines for more options
#PBS -l nodes=1:ppn=1,mem=16gb,walltime=4:00:00 #### 1 node, 1 processor, 1 gpu, 8GB of memory, 15 hours of wall time requests
############################## END OF DEFAULT EMBEDDED PBS COMMANDS #######################

parent_dir="/home/hasm/comp_space/Data/Lesion/X-net_Test/X-Net_20210401_CompleteDataSet_3Folds";
script="${parent_dir}/generate_test_output_for_input.py";
VERBOSE=0; #choose 0 or 1

if [ ! -e "${script}" ];
then
  echo -e "ERROR: script ( ${script} ) does not exist.";
  return;
fi;

if [ "${VERBOSE}" == "1" ];
then
  echo -e "\n\n \
  module avail";
  module avail;

  echo -e "\n\n \
  module list";
  module list;
fi;

echo -e "\n\n \
module load cuda-10.0";
module load cuda-10.0;

if [ "${VERBOSE}" == "1" ];
then
  echo -e "\n\n \
  module list";
  module list;

  echo -e "\n\n \
  nvidia-smi";
  nvidia-smi;
fi;

echo -e "\n\n \
export HDF5_USE_FILE_LOCKING='FALSE'";
export HDF5_USE_FILE_LOCKING='FALSE';


if [ "${VERBOSE}" == "1" ];
then
  echo -e "\n\n \
  conda info --envs;";
  conda info --envs;
fi;

echo -e "\n\n \
source activate py3_7_xnet";
source activate py3_7_xnet;

cd "${parent_dir}";
echo -e "\n\n \
pwd";
pwd;

python ${script};