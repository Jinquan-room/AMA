# Transferable adversarial attack method based on attention mechanism and multi-model integration optimization
The vulnerability of deep neural networks (DNNs) has been widely demonstrated in adversarial attack scenarios. In a black-box attack scenario, since the internal parameters of the target model are invisible, the attacker needs to use a proxy model to approximate the function and generate adversarial samples based on it. However, the limitations of a single proxy model can easily cause the generated samples to fall into local optimal limit, significantly reducing their ability to transfer between models. In view of this, this study proposes an adversarial sample generation framework (AMA) based on the attention mechanism and multi-model feature integration. Specifically, the attention weight is extracted from the middle feature layer of the agent model, the shared features across models are found by integrating the attention graphs of multiple agent models, and optimization algorithms are used to destroy these key features, thus breaking the local optimal limit of a single agent model. Experimental results show that the attack success rate of our proposed method significantly outperforms existing methods on multiple cutting-edge benchmarks.

<img width="916" height="333" alt="fig2" src="https://github.com/user-attachments/assets/a2f456b8-a582-4de7-84c7-2801b24e9db6" />


# Inception_v3模型
http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
