#!/usr/bin/env bash
####
################################## START OF EMBEDDED PBS COMMANDS ##########################
#PBS -S /bin/bash  #### Default Shell to be Used
#PBS -N XNet_GenerateValOutputs  #### Job Name to be listed in qstat
#PBS -o /scratch/hasm/Data/Lesion/X-net_Test/logs/\${PBS_JOBNAME}_\${PBS_JOBID}.stdout  #### stdout default path
#PBS -e /scratch/hasm/Data/Lesion/X-net_Test/logs/\${PBS_JOBNAME}_\${PBS_JOBID}.stderr  #### stderr default path
#PBS -M sungminha@wustl.edu  #### email address to nofity with following options/scenarios
#PBS -m abe ####abort, end notifications - see below lines for more options
#PBS -l nodes=1:ppn=1:gpus=1:K20,mem=32gb,walltime=4:00:00 #### 1 node, 1 processor, 1 gpu, 8GB of memory, 15 hours of wall time requests
############################## END OF DEFAULT EMBEDDED PBS COMMANDS #######################

#HELP
#This function geenrates npy (numpy array) predictions using generate_test_output_for_input.py, then proceeds to generate corresponding nifti outputs using visualize_test_output_for_input.py

git_dir="/scratch/hasm/git/WUSTL_2021A_ESE_5934_XNet";
parent_dir="/scratch/hasm/Data/Lesion";
xnet_dir="${parent_dir}/X-net_Test/X-Net_20210401_CompleteDataSet_3Folds";
generate_script="${git_dir}/generate_test_output_for_input.py";
visualize_script="${git_dir}/visualize_test_output_for_input.py";
VERBOSE=0; #choose 0 or 1
list="${parent_dir}/ATLAS_R1.1_Lists/Sample_Visualization_Site_ID_Timepoint.csv";
data_dir="${parent_dir}/ATLAS_R1.1/Sample_Visualization";
pretrained_weight_file="${xnet_dir}/fold_0/trained_final_weights.h5";
output_dir="${xnet_dir}/output_visualization/fold_0";

if [ ! -d "${data_dir}" ];
then
  echo -e "ERROR: data_dir ( ${data_dir} ) does not exist.";
  return;
fi;

if [ ! -d "${output_dir}" ];
then
  echo -e "ERROR: output_dir ( ${output_dir} ) does not exist.";
  return;
fi;

if [ ! -e "${generate_script}" ];
then
  echo -e "ERROR: generate_script ( ${generate_script} ) does not exist.";
  return;
fi;

if [ ! -e "${pretrained_weight_file}" ];
then
  echo -e "ERROR: pretrained_weight_file ( ${pretrained_weight_file} ) does not exist.";
  return;
fi;

if [ ! -e "${list}" ];
then
  echo -e "ERROR: list ( ${list} ) does not exist.";
  return;
fi;

if [ ! -e "${visualize_script}" ];
then
  echo -e "ERROR: visualize_script ( ${visualize_script} ) does not exist.";
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

# cd "${parent_dir}";
# echo -e "\n\n \
# pwd";
# pwd;

for i in `sed 1d ${list}`;
do
  site=`echo ${i} | cut -d, -f1`;
  id=`echo ${i} | cut -d, -f2`;
  timepoint=`echo ${i} | cut -d, -f3`;
  echo -e "\n\n \
  ${i}\t|\t${num_subject} - ${site} | ${id} | ${timepoint}";

  data_file="${data_dir}/${site}/${id}/${timepoint}/train.h5";
  if [ ! -e "${data_file}" ];
  then
    echo -e "\n\n \
    ERROR: data_file ( ${data_file} ) does not exist.";
    continue;
  fi;

  echo -e "\n\n \
  python ${generate_script} \
  --data-file-path \"${data_file}\" \
  --num-patients 1 \
  --num-slices 189 \
  --xnet-pretrained-weights-file \"${pretrained_weight_file}\" \
  --output-dir \"${output_dir}\"";

  python ${generate_script} \
  --data-file-path "${data_file}" \
  --num-patients 1 \
  --num-slices 189 \
  --xnet-pretrained-weights-file "${pretrained_weight_file}" \
  --output-dir "${output_dir}";

  echo -e "\n\n \
  python ${visualize_script} \
  --num-patients 1 \
  --output-dir \"${output_dir}\";";
  python ${visualize_script} \
  --num-patients 1 \
  --output-dir "${output_dir}";
done
