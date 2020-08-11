#!/bin/bash

apt-get install -y python3-tk bsdmainutils less python3-matplotlib &> /dev/null
pip3 install tabulate &> /dev/null
pip install tabulate &> /dev/null

DIR="log"
if [ -d "$DIR" ]; then
  # Take action if $DIR exists. #
  echo "Start to do training test"
else
  mkdir -p log
  echo "Start to do training test"
fi

# check GPU status
echo "Check GPU status with pynvml"
python3 pynvml/pynvml_status.py &> log/pynvml_result
cat log/pynvml_result
GPU_num=`cat log/pynvml_result | grep "GPU" | awk '{print $2}'`
GPU_num=${GPU_num: -1}
echo "Has $((GPU_num+1)) GPU"
echo "=========================================="

# check tensorflow
echo "check tensorflow with resnet50 model and parameter_server in tf_cnn_benchmark "
ten_start=`date +%s`
python3 tensorflow/tf_cnn_benchmarks.py --num_gpus=$((GPU_num+1)) --batch_size=6 --model=resnet50 --variable_update=parameter_server &> log/tensoflow_result
ten_end=`date +%s`
ten_runtime=$((ten_end-ten_start))
TEN_RESULT=$?
if [ $TEN_RESULT -eq 0 ]; then
  echo success
  cat log/tensoflow_result | tail -n 3
  echo "tensorflow: "$ten_runtime" sec"
else
  echo failed
fi
echo "=========================================="

# check pytorch with benchmark
echo "check pytorch with benchmark"
tor_start=`date +%s`
python3 torch/pytorch-gpu-benchmark/benchmark_models_via.py --NUM_GPU=$((GPU_num+1)) &> /dev/null
tor_end=`date +%s`
tor_runtime=$((tor_end-tor_start))
column -t -s, -n results/test.csv | less -F -S -X -K > log/pytorch_result 
TOR_RESULT=$?
if [ $TOR_RESULT -eq 0 ]; then
  echo success
  head -2 log/pytorch_result
  echo "pytorch: "$tor_runtime" sec"
else
  echo failed
fi
echo "=========================================="

# check keras with MNIST training
echo "Check keras with MNIST training"
ker_start=`date +%s`
python3 keras/keras_training.py &> log/keras_result
ker_end=`date +%s`
ker_runtime=$((ker_end-ker_start))
KER_RESULT=$?
if [ $KER_RESULT -eq 0 ]; then
  echo success
  cat log/keras_result | tail -n 1
  echo "keras: "$ker_runtime" sec"
else
  echo failed
fi
echo "=========================================="

# check sklearn with MNIST training
echo "check sklearn with MNIST training"
mkdir -p ~/scikit_learn_data/mldata
cd ~/scikit_learn_data/mldata
wget https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat
cd -
skl_start=`date +%s`
python3 sklearn/sklearn_train.py &> log/sklearn_result
skl_end=`date +%s`
skl_runtime=$((skl_end-skl_start))
SKL_RESULT=$?
if [ $SKL_RESULT -eq 0 ]; then
  echo success
  cat log/sklearn_result | tail -n 2
  echo "sklearn: "$skl_runtime" sec"
else
  echo failed
fi
echo "=========================================="

# check caffe with MNIST training
echo "check caffe with MNIST training"
curr=$PWD
log_taget=$curr"/log/caffe_result"
cd caffe/caffe
caf_start=`date +%s`
caffe train -solver=examples/mnist/lenet_solver.prototxt &> $log_taget
caf_end=`date +%s`
caf_runtime=$((caf_end-caf_start))
cd $curr
CAF_RESULT=$?
if [ $CAF_RESULT -eq 0 ]; then
  echo success
  cat log/caffe_result | tail -n 4 | head -2 | awk '{$1=$2=$3=$4=""; print $0}'
  echo "caffe: "$caf_runtime" sec"
else
  echo failed
fi
echo "=========================================="

# check Caffe2 with CIFAR10 training
echo "check caffe2 with CIFAR10 training"
curr=$PWD
log_taget=$curr"/log/caffe2_result"
cd caffe2_test
caf_start=`date +%s`
python3 caffe2_cifar10.py --batch_size 6 --epochs 3 --eval_freq 3 &> $log_taget
#caffe train -solver=examples/mnist/lenet_solver.prototxt &> $log_taget
caf_end=`date +%s`
caf_runtime=$((caf_end-caf_start))
cd $curr
CAF_RESULT=$?
if [ $CAF_RESULT -eq 0 ]; then
  echo success
  cat log/caffe2_result | grep tr_loss
  cat log/caffe2_result | tail -n 10 | head -1
  #cat log/caffe_result | tail -n 4 | head -2 | awk '{$1=$2=$3=$4=""; print $0}'
  echo "caffe2: "$caf_runtime" sec"
else
  echo failed
fi
echo "=========================================="

# check pycuda performance
echo "check pycuda performance"
python3 pycuda/performance.py &> log/pycuda_result
PYC_RESULT=$?
if [ $PYC_RESULT -eq 0 ]; then
  echo success
  cat log/pycuda_result
else
  echo failed
fi
echo "=========================================="
