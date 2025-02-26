
<h2>Kinase-substrate prediction using an autoregressive models</h2>

**Farzaneh Esmaili**<sup>1</sup> · **Yongfanf Qin**<sup>1 </sup> · **Duolin Wang** <sup>1</sup> · **Dong Xu**<sup>1*</sup>

<sup>1</sup>University of Missouri

<p style="text-align:justify">
<strong>Abstract</strong>: 
In this study, we introduce an innovative approach leveraging an autoregressive model to predict kinase-substrate pairs. Unlike traditional methods focused on predicting site-specific phosphorylation, our approach addresses kinase-specific protein substrate prediction at the protein level rather than specific phosphorylation sites. We redefine this problem as a special type of protein-protein interaction prediction task.
Our model integrates the protein large language model ESM-2 as the encoder and employs an autoregressive decoder to classify protein-kinase interactions as either "connected" or "disconnected." Our hard negative strategy, based on kinase embedding distances generated from ESM-2, compels the model to effectively distinguish positive from negative data.
While the independent test results demonstrate overall performance, real-world applications often require ranking multiple potential kinases. Therefore, we conducted a top‑k analysis to assess how well our model can prioritize the most likely kinase candidates.
Our method is also capable of zero-shot prediction, meaning it can predict substrates for a kinase even when no known substrates are available, which cannot be achieved by conventional methods. Our model’s robust generalization to novel kinase and underrepresented groups showcases its versatility and broad utility.
 </p>

<h3>This model builds upon our previous model, "Prot2token: A Multi-Task Framework for Protein Language Processing using Autoregressive Language Modeling." It is a complete tool that provides various tasks, with kinase-substrate interaction prediction being one of them.</h3>
 
<p align="center"><img src="./src/fig1_full_abstract.svg" alt=""></p>

# Usage

To use Prot2Token, you can either install the package directly for quick prediction usage or set up the full repository
for development.

# Package
## Installation

To install the package, you can run the following command:

```commandline
pip install prot2token
```

## Prediction Tutorial
To use the kinase-substrate interaction model for making predictions, follow the example code snippet below:

```python
from prot2token.models import prepare_models

net = prepare_models(name='kinase_interaction', device='cuda', compile_model=True)

substrate_sequence = ["MEETMKLATMEDTVEYCLFLIPDESRDSDKHKEILQKYIERIITRFAPMLVPYIWQNQPFNLKYKPGKGGVPAHMFGVTKFGDNIEDEWFIVYVIKQITKEFPELVARIEDNDGEFL"]
kinase_sequence = ["MSGPVPSRARVYTDVNTHRPREYWDYESHVVEWGNQDDYQLVRKLGRGKYSEVFEAINITNNEKVVVKILKPVKKKKIKREIKILENLRGGPNIITLADIVKDPVSRTPALVFEHVNNTDFKQLYQTLTDYDIRFYMYEILKALDYCHSMGIMHRDVKPHNVMIDHEHRKLRLIDWGLAEFYHPGQEYNVRVASRYFKGPELLVDYQMYDYSLDMWSLGCMLASMIFRKEPFFHGHDNYDQLVRIAKVLGTEDLYDYIDKYNIELDPRFNDILGRHSRKRWERFVHSENQHLVSPEALDFLDKLLRYDHQSRLTAREAMEHPYFAQQ"]

results = net.run(substrate_sequence,kinase_sequence, merging_character='')
print(results)

```

And the results will be like the following:

```commandline
["MEETMKLATMEDTVEYCLFLIPDESRDSDKHKEILQKYIERIITRFA<EOS>MSGPVPSRARVYTDVNTHRPREYWDYESHVVEWGNQDDYQLVRKLGRGKYS", "<task_kinase_interaction>", "connected"]
```