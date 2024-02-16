@echo off

rem 确认mnist数据集已经下载到本地
set "have_mnist=true"
if not exist "mnist\t10k-images-idx3-ubyte.gz" (
    set "have_mnist=false"
)
if not exist "mnist\t10k-labels-idx1-ubyte.gz" (
    set "have_mnist=false"
)
if not exist "mnist\train-images-idx3-ubyte.gz" (
    set "have_mnist=false"
)
if not exist "mnist\train-labels-idx1-ubyte.gz" (
    set "have_mnist=false"
)

if "%have_mnist%"=="false" (
    echo mnist数据集未下载到本地
    echo 请先将数据集放入mnist文件夹中
    md mnist
    cd.> mnist\数据集放这里.txt
    pause
    exit
)

rem pip
pip install -r requirements.txt

rem 清洗数据
python data_clean.py

rem 训练模型

python bayes.py

pause