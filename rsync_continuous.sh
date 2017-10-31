
while true
do
rsync -r -e "ssh -i $HOME/.ssh/id_rsa" tgan4199@muspelheim.cs.usyd.edu.au:docker-tf/AdaCNN/cifar100-inc-cont-stationary-1gpu-lifelong/ada_cnn_tensorboard_data/ ./tensorboard_sync
sleep 30
done
