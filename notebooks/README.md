# Some knowledge from transformers paper

## <font color="lightgreen"> Models' architecture</font>

The network consisted of Transformer
modules as well as operations of Positional Embedding.
We also designed methods that combined the CNN module
and the Transformer module. CNN was included because
of its good properties for feature representation [5].

In the implementation, we built a total of **five** Transformer-based
models in which two models only relied on the Transformer
without including the CNN and three models used network
architecture of combined CNN and Transformer. After the
CNN and the Transformer modules, we included a fully
connected layer.

In this study, we employed **h = 8 parallel attention layers
(so-called 8 attention heads)**, and solely embedded the encoder
part of Transformer into the EEG classification.

The Transformer module had two submodules. The
first submodule included a multi-head attention layer followed
by a normalization layer. The second submodule included a
position-wise fully connected feed-forward layer followed by
a normalization layer. The residual connection was employed
around each of the two submodules.

We explored the influence of the number of Transformer modules on the classification results. The number of Transformer modules was tested from 1 to 6. When the number of 3 was chosen, the classification achieved the best results. We therefore included three Transformer modules in our models

## <font color="lightgreen"> Training parameter settings</font>

- Empirically, the number of head in each multi-head attention layer was set to 8
- The dropout rate was set to 0.3.\*
- The parameter of the position-wise fully connected feed-forward layer with a ReLU activation was set to 512.
- The weight attenuation was 0.0001.
- All the models used the Adam optimizer. The training epoch was set to 50.
- We set the number of training epochs to 10
- The EEG data were transformed into 3D tensors (N, C, T), where N is the number of trials, C is the number of channels, and T is the time points.
- In our Transformer-based models, we set dk = dv = 64, which was the same size as EEG channel numbers.
