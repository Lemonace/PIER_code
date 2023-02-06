#!/bin/bash

echo ""
# 共建模型服务模版的路径 ------------------------ 修改模版配置 ----------------------
template_path="../model_template/"

# 待上传模型pb文件的路径 ------------------------ 修改模型配置 ----------------------
model_path="../model/"
model_name=$1
ep=$2
model_file=${model_path}${model_name}/${ep}
echo ${model_file}
target_path="../model_zip/"
if [[ ! -d ${target_path} ]]; then
        mkdir ${target_path}
fi
target_name=${model_name}_${ep}
target_file=${target_path}${target_name}
target_name_zip=${target_name}.zip
echo ${target_name}
echo "---> 待上传模型的目标路径: "${target_file}
if [[ -d ${target_file} ]]; then
    cmd_del="rm -fr ${target_file}"
    echo "---> 删除已存在的目标文件: "${cmd_del}
    ${cmd_del}
fi
if [[ ! -d ${target_file} ]]; then
        mkdir ${target_file}
fi

cmd_cp_model_pb="cp -fr ${model_file} ${target_file}/${target_name}"
echo "---> 拷贝模型pb文件到目标目录: "${cmd_cp_model_pb}
${cmd_cp_model_pb}

cmd_mv1="cp ${template_path}/model.properties ${target_file}/${target_name}.properties"
cmd_mv2="cp ${template_path}/model.xml ${target_file}/${target_name}.xml"
cmd_mv3="cp ${template_path}/tensors.xml ${target_file}/tensors.xml"
cmd_mv4="cp ${template_path}/poi_feature.xml ${target_file}/poi_feature.xml"
cmd_mv5="cp ${template_path}/poi_feature.tensor ${target_file}/poi_feature.tensor"
echo "---> 修改properties文件名: "${cmd_mv1}
echo "---> 修改xml文件名: "${cmd_mv2}
${cmd_mv1}
${cmd_mv2}
${cmd_mv3}
${cmd_mv4}
${cmd_mv5}

cmd_zip1="cd ${target_path}"
cmd_zip2="zip -r ${target_name_zip} ${target_name}"
echo "---> 压缩上传模型文件: "${cmd_zip1} ${cmd_zip2}
${cmd_zip1}
${cmd_zip2}
cd -

echo ${target_name}.zip
