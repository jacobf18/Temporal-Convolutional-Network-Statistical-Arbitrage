import subprocess
import os

stock_datasets = [
    # ['stock_A_B.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/13580'],
    # ['stock_C_D.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/13585'],
    # ['stock_E_F.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/13590'],
    # ['stock_G_H.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/13595'],
    # ['stock_I_J.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/13600'],
    # ['stock_K_L.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/13605'],
    # ['stock_M_N.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/12509'],
    # ['stock_O_P.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/13610'],
    # ['stock_Q_R.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/13615'],
    # ['stock_S_T.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/13620'],
    # ['stock_U_V.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/13625'],
    # ['stock_W_X.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/13630'],
    # ['stock_Y_Z.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/13635']
]

etf_datasets = [
    ['etf_A_C.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/13502'],
    ['etf_D_F.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/13507'],
    ['etf_G_L.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/13512'],
    ['etf_M_N.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/13517'],
    ['etf_O_S.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/13522'],
    ['etf_T_Z.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/13527']
]

us_index_datasets = [
    # ['indexs_all.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/12248']
]

futures_datasets = [
    # ['futures_all70.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/13738']
]

crypto_datasets = [
    ['crypto_all50.zip', 'https://firstratedata.com/datafile/ZpKb77M8hkSKYkSFg8mY-w/12627']
]

datasets = [
    stock_datasets,
    etf_datasets,
    us_index_datasets,
    futures_datasets,
    crypto_datasets
]

def download(output, url):
    subprocess.run(['wget', '-O', output, url])

def unzip(zipfile, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    subprocess.run(['unzip', '-o', zipfile, '-d', output_dir])

def main():
    """"""
    filepath = os.path.abspath(__file__)
    folder = os.path.dirname(filepath)

    unzip_folder = os.path.join(folder, 'unzip')
    for ds in datasets:
        for item in ds:
            zipfile = os.path.join(folder, item[0])
            download(zipfile, item[1])
            target_folder = item[0].split('_')[0]
            target_folder = os.path.join(unzip_folder, target_folder)
            unzip(zipfile, target_folder)


if __name__ == "__main__":
    main()
