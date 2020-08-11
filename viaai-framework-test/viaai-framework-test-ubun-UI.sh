 #!/bin/bash
# VIA AI Test Tool has following options
# Show GPU status
# Test Tensorflow with Resnet Model
# Test Pytorch
# Test Keras
# Test SKlearn
# Test Caffe
# Test Caffe2
# Test PYcuda
# Test ALL items
# Exit
# As per option do the job
# -----------------------------------------------
# This script is licensed under GNU GPL version 2.0 or above
# -------------------------------------------------------------------------

# pre install packages
pre(){
apt-get install -y python3-tk bsdmainutils less python3-matplotlib &> /dev/null
pip3 install tabulate &> /dev/null
pip install tabulate &> /dev/null

DIR="log"
if [ -d "$DIR" ]; then
  echo "Start to do training test"
else
  mkdir -p log
  echo "Start to do training test"
fi
}

# check GPU status
pynvml(){
echo "Check GPU status with pynvml"
python3 pynvml/pynvml_status.py &> log/pynvml_result
cat log/pynvml_result
GPU_num=`cat log/pynvml_result | grep "GPU" | awk '{print $2}'`
GPU_num=${GPU_num: -1}
echo "Has $((GPU_num+1)) GPU"
}

# check tensorflow
tensorflow(){
echo "check tensorflow with resnet50 model and parameter_server in tf_cnn_benchmark "
GPU_num=`cat log/pynvml_result | grep "GPU" | awk '{print $2}'`
GPU_num=${GPU_num: -1}
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
}

# check pytorch with benchmark
pytorch(){
echo "check pytorch with benchmark"
GPU_num=`cat log/pynvml_result | grep "GPU" | awk '{print $2}'`
GPU_num=${GPU_num: -1}
tor_start=`date +%s`
python3 torch/pytorch-gpu-benchmark/benchmark_models_via.py --NUM_GPU=$((GPU_num+1)) &> /dev/null
tor_end=`date +%s`
tor_runtime=$((tor_end-tor_start))
cat results/test.csv > log/pytorch_result
TOR_RESULT=$?
if [ $TOR_RESULT -eq 0 ]; then
  echo success
  head -2 log/pytorch_result
  echo "pytorch: "$tor_runtime" sec"
else
  echo failed
fi
}

# check keras with MNIST training
keras(){
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
}

# check sklearn with MNIST training
sklearn(){
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
}

# check caffe with MNIST training
caffe_test(){
  echo "check caffe with MNIST training"
  curr=$PWD
  log_taget=$curr"/log/caffe_result"
  cd caffe/caffe
  caf_start=`date +%s`
  caffe train -solver=examples/mnist/lenet_solver.prototxt &> $log_taget
  #caffe train -solver=examples/mnist/lenet_solver.prototxt
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
}

# check Caffe2 with CIFAR10 training
caffe2_test(){
  echo "check caffe2 with CIFAR10 training"
  curr=$PWD
  log_taget=$curr"/log/caffe2_result"
  cd caffe2_test
  caf_start=`date +%s`
  python3 caffe2_cifar10.py --batch_size 6 --epochs 3 --eval_freq 3 &> $log_taget
  caf_end=`date +%s`
  caf_runtime=$((caf_end-caf_start))
  cd $curr
  CAF_RESULT=$?
  if [ $CAF_RESULT -eq 0 ]; then
    echo success
    cat log/caffe2_result | grep tr_loss
    cat log/caffe2_result | tail -n 10 | head -1
    echo "caffe2: "$caf_runtime" sec"
  else
    echo failed
  fi
}

# check pycuda performance
pycuda(){
echo "check pycuda performance"
python3 pycuda/performance.py &> log/pycuda_result
PYC_RESULT=$?
if [ $PYC_RESULT -eq 0 ]; then
  echo success
  cat log/pycuda_result
else
  echo failed
fi
}

# Main function 
pre;
while :
do
 clear
 #nvidia-smi
 echo "   V I A A I T E S T - M E N U"
 echo "1. Show GPU Resource"
 echo "2. Tensorflow"
 echo "3. Pytorch"
 echo "4. Keras"
 echo "5. SKlearn"
 echo "6. Caffe"
 echo "7. Caffe2"
 echo "8. PyCUDA"
 echo "9. ALL"
 echo "10. Exit"
 echo -n "Please enter option [1 - 10]"
 read opt
 case $opt in
  1) echo "************ GPU Resource *************";
     pynvml;
     echo "Press [enter] key to continue. . .";
     read enterKey;;
  2) echo "************ Tensorflow *************";
     pynvml;
     tensorflow;
     echo "Press [enter] key to continue. . .";
     read enterKey;;
  3) echo "************ Pytorch *************";
     pynvml;
     pytorch;
     echo "Press [enter] key to continue. . .";
     read enterKey;;
  4) echo "************ Keras *************";
     pynvml;
     keras;
     echo "Press [enter] key to continue. . .";
     read enterKey;;
  5) echo "************ SKlearn *************";
     pynvml;
     sklearn;
     echo "Press [enter] key to continue. . .";
     read enterKey;;
  6) echo "************ Caffe *************";
     pynvml;
     echo $PWD
     caffe_test;
     echo "Press [enter] key to continue. . .";
     read enterKey;;
  7) echo "************ Caffe2 *************";
     pynvml;
     caffe2_test;
     echo "Press [enter] key to continue. . .";
     read enterKey;;
  8) echo "************ PyCUDA *************";
     pynvml;
     pycuda;     
     echo "Press [enter] key to continue. . .";
     read enterKey;;
  9) echo "************ ALL *************";
     pynvml;
     echo "************ Tensorflow *************";
     tensorflow;
     echo "************ Pytorch *************";
     pytorch;
     echo "************ Keras *************";
     keras;
     echo "************ SKlearn *************";
     sklearn;
     echo "************ Caffe *************";
     caffe_test;
     echo "************ Caffe2 *************";
     caffe2_test;
     echo "************ PyCUDA *************";
     pycuda;
     echo "Press [enter] key to continue. . .";
     read enterKey;;
  10) echo "Good Bye";
     exit 1;;
  *) echo "$opt is an invaild option. Please select option between 1-10 only";
     echo "Press [enter] key to continue. . .";
     read enterKey;;
esac
done
