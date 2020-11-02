#! /bin/bash
model_dir=json_model

mkdir -p ${model_dir}
#download model
wget -p https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/resnet50/float/model-stride16.json -O ${model_dir}/saved_model.json
#download weights
for i in $(seq 1 23);
do
	wget -p https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/resnet50/float/group1-shard${i}of23.bin -O ${model_dir}/group1-shard${i}of23.bin
done