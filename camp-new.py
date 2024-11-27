#############################################
# Date: 2024-11-27
# Analysis of L1000 camp
#############################################

import os
import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp  # 可以从这个命名空间中直接使用 csr_matrix 和 issparse
from anndata import AnnData
import cmapPy.pandasGEXpress as pg
from cmapPy.pandasGEXpress.parse_gctx import parse

import warnings
warnings.filterwarnings('ignore')

# Load Data ----------------------------------
info = pd.read_csv("/home/lushi02/project/sl_data/LINCS/instinfo_beta.txt", sep="\t", header=0, low_memory=False)
info_cp = info[info['pert_type'] == "trt_cp"] # LINCS数据中化合物perturb的数据

cellinfo = pd.read_csv("/home/lushi02/project/sl_data/LINCS/cellinfo_beta.csv", header=0) # 细胞文件
compinfo = pd.read_csv("/home/lushi02/project/sl_data/LINCS/compoundinfo_beta.csv", header=0) # 化合物文件
geneinfo = pd.read_csv("/home/lushi02/project/sl_data/LINCS/geneinfo_beta.txt", sep="\t", header=0)  # 基因文件
siginfo = pd.read_csv("/home/lushi02/project/sl_data/LINCS/geneinfo_beta.csv", header=0)  # 基因文件

smilesinfo = pd.read_csv("/home/lushi02/project/sl_data/LINCS/cmap_smiles_cleaned.csv", header=0)
smilesinfo.columns = ["product_name", "SMILES", "canonical_smiles"]

check_data = pd.read_csv("/home/qcao02/gaozy/camp/merged_cell_counts_update.csv", header=0) # merged_cell文件

## Load perturb GCTX file 
my_ds = parse("/home/lushi02/project/sl_data/LINCS/level3_beta_trt_cp_n1805898x12328.gctx") # trt

## Load control GCTX file
cl_ds = parse("/home/lushi02/project/sl_data/LINCS/level3_beta_ctl_n188708x12328.gctx") # ctl

# 创建基于 'cmap_name' 的频率表并按频率排序 ----------------------------------
compN = info_cp['cmap_name'].value_counts().reset_index()
compN.columns = ['cmap_name', 'freq']

# 删除 'canonical_smiles' 列中的空白字符 ----------------------------------
compinfo['canonical_smiles'] = compinfo['canonical_smiles'].str.strip()

## 筛选出 'canonical_smiles' 列非空的行
compinfox = compinfo[compinfo['canonical_smiles'] != ""]

## 将频率表与筛选后的化合物信息合并
compNx = pd.merge(compN, compinfox, on='cmap_name', how='left')

## 筛选出 'canonical_smiles' 列非NaN的行
compNx = compNx[compNx['canonical_smiles'].notna()]

## 基于 'cmap_name' 去除重复项
unique_compNx = compNx.drop_duplicates(subset='cmap_name')

# 筛选 'info_cp' DataFrame 中 'pert_idose' 包含 '10 uM' 的行 ----------------------------------
infox = info_cp[info_cp['pert_idose'].str.contains("10 uM", na=False)]

# 进一步筛选 'pert_itime' 包含 '24 h' 的行
infox = infox[infox['pert_itime'].str.contains("24 h", na=False)]

# 创建基于 'cell_iname' 的频率表 ----------------------------------
cellN = infox['cell_iname'].value_counts().reset_index()
cellN.columns = ['cell_iname', 'Freq']

## 将频率表与细胞信息合并
cellN = pd.merge(cellN, cellinfo, on='cell_iname', how='left')

## 按频率降序排序得到的 DataFrame
cellN = cellN.sort_values(by='Freq', ascending=False)

# 将 'rowx' (my_ds中的'gene_id') 与 'geneinfo' DataFrame 合并 ----------------------------------
rowx = pd.DataFrame(my_ds.data_df.index.values, columns=['gene_id'])
rowx['gene_id'] = rowx['gene_id'].astype(str)
geneinfo['gene_id'] = geneinfo['gene_id'].astype(str)
rowx = pd.merge(rowx, geneinfo, on='gene_id', how='left')
rowx = rowx[~rowx['gene_id'].str.contains("117153")]

# 筛选 'my_ds' 中的数据 ----------------------------------
mask = ~my_ds.data_df.index.str.contains("117153")
my_ds.filtered_data_df = my_ds.data_df[mask].copy()
my_ds.filtered_row_metadata_df = my_ds.row_metadata_df[mask].copy()

## 确保 my_ds.filtered_data_df 索引与 rowx['gene_id'] 一致
my_ds.filtered_data_df = my_ds.filtered_data_df.loc[rowx['gene_id'],:]

gene_id_index = pd.Index(rowx['gene_id'])
my_ds.filtered_data_df.index.equals(gene_id_index)

## my_ds.filtered_data_df 索引 以 rowx['gene_symbol'] 命名
my_ds.filtered_data_df.index = rowx['gene_symbol'].values

# Control data -----------------------------------------
info_x = info[(info['pert_type'] == "ctl_vehicle") & (info['pert_id'] == "DMSO")]

# Control data 创建基于 'cell_iname' 的频率表 ----------------------------------
cellN_ctl = info_x['cell_iname'].value_counts().reset_index()
cellN_ctl.columns = ['cell_iname', 'Freq']

## 将频率表与细胞信息合并
cellN_ctl = pd.merge(cellN_ctl, cellinfo, on='cell_iname', how='left')

## 按频率降序排序得到的 DataFrame
cellN_ctl = cellN_ctl.sort_values(by='Freq', ascending=False)

# 定义一个函数来处理AnnData对象创建
def create_ann_data(input_matrix, input_meta, input_var, project_name):
    """
    创建一个AnnData对象从一个给定的矩阵和元数据。

    参数:
    - input_matrix: pd.DataFrame, 转录数据矩阵，行名为样本ID，列名为基因名。
    - input_meta: pd.DataFrame, 每个样本的元数据。
    - project_name: str, 项目的名称，用于在AnnData对象中标识。
    
    返回:
    - output_data: AnnData, 创建的AnnData对象。
    """
    
    # 计算每个样本的RNA计数总数
    nCount_RNA = input_matrix.sum(axis=0)  # sum per column

    # 计算每个样本的检测到的特征（基因）数
    nFeature_RNA = (input_matrix > 0).sum(axis=0)  # count non-zero entries per column

    # 将计算的值添加到元数据中
    input_meta['orig.ident'] = project_name
    input_meta['nCount_RNA'] = nCount_RNA
    input_meta['nFeature_RNA'] = nFeature_RNA
    input_meta['sample_id'] = input_meta.index.values
    
    # 创建 AnnData 对象
    output_data = sc.AnnData(X=input_matrix.T, obs=input_meta, var=input_var, dtype=np.float64)
    output_data.uns['project'] = project_name

    return output_data

# 定义一个函数来处理数据check
def check_adata(file_path):
    # 读取数据
    adata1 = sc.read_h5ad(file_path)
    
    adata1 = sc.read_h5ad(filename)
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
    if sp.issparse(adata1.X) and adata1.X.dtype == np.float64:
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
        print("\n数据通过检测")
    else:
        print("\n数据未通过检测，请检查上面的错误信息")

#warnings.filterwarnings('default') # re-active warnings

# Initialize a loop to process cell lines
for i in range(len(cellN)-1, -1, -1):
  cell_iname = cellN.iloc[i]['cell_iname']
  
  # 提取 'sample_id'
  infox_sub = infox.loc[infox['cell_iname'] == cell_iname]
  sample_id1 = infox_sub['sample_id'].values
  
  # 根据 'sample_id1' 筛选矩阵的列
  matching_mat = my_ds.filtered_data_df.loc[:, my_ds.filtered_data_df.columns.isin(sample_id1)]
  
  # 准备 'compin' DataFrame，选取特定列
  compin = unique_compNx[['cmap_name', 'canonical_smiles']].copy()
  compin_agg = compin.dropna(subset=['canonical_smiles']).drop_duplicates(subset=['cmap_name'])
  compin_agg['canonical_smiles'] = compin_agg['canonical_smiles'].str.strip()
  
  # 将 'meta' 与 'compin_agg' 合并
  meta = infox_sub[infox_sub['sample_id'].isin(sample_id1)].copy()
  metax = pd.merge(meta, compin_agg, on="cmap_name", how="left")
  metax.set_index('sample_id', inplace=True)
  metax = metax.reindex(matching_mat.T.index)  # 确保 metax 索引与 matching_mat.T 的索引相同
  
  print(f'\n    1. metax 索引与 matching_mat.T 的索引是否相同：{metax.index.equals(matching_mat.T.index)}')
  
  
  # 检查 'matching_mat' 中是否有重复的基因名称
  duplicate_genes = matching_mat.index[matching_mat.index.duplicated()]
  
  if not duplicate_genes.empty:
      print("Duplicate features found:")
      print(duplicate_genes.unique())
  else:
      print("All features are unique.")
  
  # 确保 rowx 索引与 matching_mat.T 的索引相同
  rowx.index = rowx['gene_symbol']
  rowx = rowx.loc[matching_mat.T.columns,:]
  
  # 创建 AnnData 对象 
  select_cell = create_ann_data(matching_mat,metax,rowx,cell_iname)
  
  print(f'\n    2. select_cell 索引与 matching_mat 的索引是否相同：{select_cell.var.index.equals(pd.Index(matching_mat.T.columns))}')
  
  # 筛选 'info' DataFrame 以获取 DMSO control
  cl2 = info_x[info_x['cell_mfc_name'] == cell_iname]
  
  # control GCTX file
  rowx_ctl = pd.DataFrame(cl_ds.data_df.index.values, columns=['gene_id'])
  rowx_ctl['gene_id'] = rowx_ctl['gene_id'].astype(str)
  rowx_ctl = pd.merge(rowx_ctl, geneinfo, on='gene_id', how='left')
  rowx_ctl = rowx_ctl[~rowx_ctl['gene_id'].str.contains("117153")]
  
  # 筛选 'cl_ds' 中的数据
  mask = ~cl_ds.data_df.index.str.contains("117153")
  cl_ds.filtered_data_df = cl_ds.data_df[mask].copy()
  cl_ds.filtered_row_metadata_df = cl_ds.row_metadata_df[mask].copy()
  
  ## 确保 cl_ds.filtered_data_df 索引与 rowx_ctl['gene_id'] 一致
  cl_ds.filtered_data_df = cl_ds.filtered_data_df.loc[rowx_ctl['gene_id'],:]
  
  print(f'\n    3. cl_ds.filtered_data_df 索引与 rowx_ctl["gene_id"] 是否相同：{cl_ds.filtered_data_df.index.equals(pd.Index(rowx_ctl["gene_id"]))}')
  
  ## cl_ds.filtered_data_df 索引 以 rowx_ctl['gene_symbol'] 命名
  cl_ds.filtered_data_df.index = rowx_ctl['gene_symbol'].values
  
  # 筛选矩阵以仅包含与 'sample_id' 匹配的列
  matching_cl2 = cl_ds.filtered_data_df.loc[:, cl_ds.filtered_data_df.columns.isin(cl2['sample_id'])]
  matching_cl2 = matching_cl2.loc[matching_cl2.index.isin(rowx_ctl['gene_symbol']), :]
  
  # 筛选 meta_cl 并与 'compin_agg' 合并
  meta_cl = info[info['sample_id'].isin(cl2['sample_id'])]
  metax_cl = pd.merge(meta_cl, compin_agg, on='cmap_name', how='left')
  metax_cl.index = metax_cl['sample_id'].values
  
  # 筛选满足要求的 metax_cl 并更新 matching_cl2
  filtered_metax_cl = metax_cl[metax_cl['project_code'].isin(metax['project_code'])]
  filtered_matching_cl2 = matching_cl2.loc[:, matching_cl2.columns.isin(filtered_metax_cl.index)]
  
  # 确保 filtered_metax_cl 索引与 filtered_matching_cl2 的索引相同
  filtered_metax_cl = filtered_metax_cl.loc[filtered_matching_cl2.columns.values,:]
  
  filtered_metax_cl.index.equals(pd.Index(filtered_matching_cl2.columns))
  
  # 确保 rowx_ctl 索引与 filtered_matching_cl2  的索引相同
  rowx_ctl.index = rowx_ctl['gene_symbol']
  rowx_ctl = rowx_ctl.loc[filtered_matching_cl2.T.columns,:]
  
  # 为对照创建一个 AnnData 对象
  vehicle = create_ann_data(filtered_matching_cl2,filtered_metax_cl,rowx_ctl,"vehicle")
  
  print(f'\n    4. vehicle 索引与 filtered_matching_cl2 的索引是否相同：{vehicle.var.index.equals(pd.Index(filtered_matching_cl2.T.columns))}')
  
  # 合并两个 AnnData 对象
  merged_select_cell = vehicle.concatenate(select_cell, batch_key="pert_type", join='inner')
  merged_select_cell.var.rename(columns={'gene_symbol': 'features'}, inplace=True)
  merged_select_cell = merged_select_cell[merged_select_cell.obs['cell_iname'] == cell_iname, :]
  
  # double check 细胞数量
  ## 将value_counts转换为DataFrame
  cell_counts = merged_select_cell.obs['orig.ident'].value_counts().reset_index()
  cell_counts.columns = ['cell_iname', 'expected_count']
  
  # 合并数据集并检查频率匹配
  result = pd.merge(cell_counts.iloc[0:1], check_data, on='cell_iname', how='left')
  result['matches'] = result['expected_count'] == result['Count_cp']
  discrepancies = result[~result['matches']]
  
  # 输出结果
  if discrepancies.empty:
      print(f"    5. Perturb 样本数与 check_data 中一致。")
  else:
      print(f"    5. Perturb 样本数与 check_data 中不一致。")
      print(discrepancies[['cell_iname', 'count', 'Freq']])
  
  # 移除 'canonical_smiles' 为 NA 的行
  merged_select_cell = merged_select_cell[~merged_select_cell.obs['canonical_smiles'].isna(), :]
  
  # 将数组转换为压缩稀疏行矩阵
  denseX = sp.csr_matrix(merged_select_cell.X.toarray(), dtype=np.float64)
  new_merged_select_cell = sc.AnnData(X=denseX, obs=merged_select_cell.obs, var=merged_select_cell.var)
  
  # 构建文件名并将 AnnData 对象保存为 .h5ad 文件
  filename = f"{cell_iname}_cmap_{rowx.shape[0]}.h5ad"
  
  # 检查 'failure_mode' 中非字符串类型
  if new_merged_select_cell.obs['failure_mode'].apply(lambda x: not isinstance(x, str)).any():
      new_merged_select_cell.obs['failure_mode'] = new_merged_select_cell.obs['failure_mode'].astype(str)
  
  # 遍历 obs 中的所有列，如果列全是 NaN，则将其替换为字符串 'nan'
  for col in new_merged_select_cell.obs.columns:
      if new_merged_select_cell.obs[col].isna().all():
          new_merged_select_cell.obs[col] = 'nan'
  
  # 保存文件
  new_merged_select_cell.write(filename)
  
  # Check and print the number of genes and cells
  #adata = sc.read_h5ad(filename)
  print(f"Loop {i} - {cell_iname}:")
  #print(f"  - Number of genes: {adata.n_vars}")
  #print(f"  - Number of cells: {adata.n_obs}")
  print(f"  - Saved file: {filename}")
  check_adata(filename)
  print(f"------------------------------------------  {cell_iname} is Done. ------------------")
