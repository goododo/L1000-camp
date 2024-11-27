import argparse
import scanpy as sc
import os
from scipy.sparse import issparse
import numpy as np

# 定义一个函数来处理数据检查
def check_adata(file_path):
    print(f"\n正在检查文件: {file_path}")
    
    # 读取数据
    adata1 = sc.read_h5ad(file_path)

    # 初始化一个变量来跟踪是否所有检查都通过
    all_checks_passed = True

    # 1. 检查 obs.cell_iname.value_counts() 的长度是否为 1
    if adata1.obs['cell_iname'].value_counts().shape[0] == 1:
        print("adata1.obs.cell_iname 的唯一值通过检查")
    else:
        print("Error: adata1.obs.cell_iname 的唯一值数量不为1")
        all_checks_passed = False

    # 2. 检查 var 中是否包含指定的列
    required_columns = ['features', 'gene_id', 'ensembl_id', 'gene_title', 'gene_type', 'src', 'feature_space']
    missing_columns = [col for col in required_columns if col not in adata1.var.columns]

    if len(missing_columns) == 0:
        print("adata1.var 包含所有指定的列")
    else:
        print(f"Error: adata1.var 缺少列: {missing_columns}")
        all_checks_passed = False

    # 3. 确保 var 的 index 除了列名以外和 'features' 列的内容一致
    if all(adata1.var.index == adata1.var['features']):
        print("adata1.var 的 index 和 'features' 列的内容一致")
    else:
        print("Error: adata1.var 的 index 和 'features' 列的内容不一致")
        all_checks_passed = False

    # 4. 检查 X 是否为稀疏矩阵并且类型为 numpy.float64
    if issparse(adata1.X) and adata1.X.dtype == np.float64:
        print("adata1.X 是一个稀疏矩阵，并且类型为 numpy.float64")
    else:
        print("Error: adata1.X 不是稀疏矩阵，或类型不是 numpy.float64")
        all_checks_passed = False

    # 5. 检查 canonical_smiles 列是否存在 NA 值
    if 'canonical_smiles' in adata1.obs.columns:
        if adata1.obs['canonical_smiles'].isna().sum() == 0:
            print("adata1.obs['canonical_smiles'] 列中没有 NA 值")
        else:
            print(f"Error: adata1.obs['canonical_smiles'] 列中存在 NA 值，总数为 {adata1.obs['canonical_smiles'].isna().sum()}")
            all_checks_passed = False
    else:
        print("Error: 'canonical_smiles' 列不存在于 adata1.obs")
        all_checks_passed = False

    # 最后根据检查结果输出
    if all_checks_passed:
        print(f"文件 {file_path} 通过检测")
    else:
        print(f"文件 {file_path} 未通过检测，请检查上面的错误信息")
    
    return all_checks_passed


# 使用 argparse 接收文件夹路径
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='对文件夹中的所有 h5ad 数据文件进行检查')
    parser.add_argument('folder_path', type=str, help='包含 h5ad 文件的文件夹路径')

    args = parser.parse_args()

    # 遍历文件夹，查找所有 .h5ad 文件并进行检查
    folder_path = args.folder_path
    all_files_passed = True  # 初始化变量，跟踪所有文件的检查结果

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.h5ad'):
            file_path = os.path.join(folder_path, file_name)
            if not check_adata(file_path):
                all_files_passed = False  # 如果有任何文件未通过检查，将该变量设置为 False

    # 输出最终结果
    if all_files_passed:
        print("\n所有文件都通过了检测")
    else:
        print("\n有文件未通过检测，请检查上面的错误信息")

