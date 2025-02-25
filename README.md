<div align="center">
<h2>Kinase-substrate prediction using an autoregressive models</h2>

**Farzaneh Esmaili**<sup>1</sup> · **Yongfanf Qin**<sup>1 </sup> · **Duolin Wang** <sup>1</sup> · **Dong Xu**<sup>1*</sup>

<sup>1</sup>University of Missouri

<p align="center" style="text-align:justify">
<strong>Abstract</strong>: 
In this study, we introduce an innovative approach leveraging an autoregressive model to predict kinase-substrate pairs. Unlike traditional methods focused on predicting site-specific phosphorylation, our approach addresses kinase-specific protein substrate prediction at the protein level rather than specific phosphorylation sites. We redefine this problem as a special type of protein-protein interaction prediction task.
Our model integrates the protein large language model ESM-2 as the encoder and employs an autoregressive decoder to classify protein-kinase interactions as either "connected" or "disconnected." Our hard negative strategy, based on kinase embedding distances generated from ESM-2, compels the model to effectively distinguish positive from negative data.
While the independent test results demonstrate overall performance, real-world applications often require ranking multiple potential kinases. Therefore, we conducted a top‑k analysis to assess how well our model can prioritize the most likely kinase candidates.
Our method is also capable of zero-shot prediction, meaning it can predict substrates for a kinase even when no known substrates are available, which cannot be achieved by conventional methods. Our model’s robust generalization to novel kinase and underrepresented groups showcases its versatility and broad utility.
 </p>

### This model is on top of our previous mode "Prot2token:  A multi-task framework for protein language processing using autoregressive language modeling"
