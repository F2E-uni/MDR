import os

# 注意s3上的东西会自动映射到cur_dir上，而反过来不会，则需要用mox.file.copy(拷贝单文件)或mox.file.copyparallel（拷贝文件夹）
cur_dir = os.getcwd().replace('\\', '/') + '/myMDR'  # 机器上的地址 '/home/ma-user/modelarts/user-job-dir/myMDR'
s3_dir = 's3://bucket-852/w50022420/myMDR'  # s3桶上的地址

