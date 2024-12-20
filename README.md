# ArgMax

## 使用msopgen创建项目
${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。例如，若安装的Ascend-cann-toolkit软件包，则安装后文件存储路径为：$HOME/Ascend/ascend-toolkit/latest。

${INSTALL_DIR}/python/site-packages/bin/msopgen gen -i {*.json} -f {framework type} -c {Compute Resource} -lan cpp -out {Output Path}

AI处理器的型号<soc_version>请通过如下方式获取：

- 在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，获取Chip Name信息。实际配置值为AscendChip Name，例如Chip Name取值为xxxyy，实际配置值为Ascendxxxyy。