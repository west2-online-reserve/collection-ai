import zipfile
with zipfile.ZipFile('/tmp/pycharm_project_570/Caltech101.zip','r')as zip_ref:
    zip_ref.extractall('/tmp/pycharm_project_570/Caltech101')