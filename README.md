# fed-SURF

This repository contains the code needed to reproduce the results of <a href="https://bit.ly/3MBtRfH">Stochastic Unrolled Neural Networks</a>. To cite the paper, please use the bib entry:
```
@inproceedings{hadou2024stochastic,
      title={Stochastic Unrolled Neural Networks}, 
      author={Samar Hadou, and Navid NaderiAlizadeh and Alejandro Ribeiro},
      booktitle={Conference on Parsimony and Learning (CPAL)},
      year={2026},
      url={https://openreview.net/pdf?id=hQQMk0IOPv}, 
}
```

----

To run the main experiment in the paper:
```
python3 main.py --constrained --createMetaDataset
```

To see the options to configure:
```
python3 main.py --help
```

Or, simply, you can run the files under scripts:
```
bash scripts/SCRIPT_FILE.sh
```

To install dependencies:
```
pip install -r requirements.txt
```
