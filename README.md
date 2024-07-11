# SIGEVO Summer School 2024 Modelling Projects 
PROJECT: 3D PRINTING

## IMPLEMENTED SO FAR:
- Code foundation
- Greedy Construction
- First-improvement local search
- Best-improvement local search
- Evaluator compatible

## TODO:
- heuristic construction
- ant system
- beam search
- grasp
- ils
- mmas
- tls
- sa
- other solvers


## HOW TO USE:
Use the debug mode to see the steps of the algorithm.

From pwd: `src/`
- For the greedy construction:
```bash 
python base.py --log-level debug --input-file '../data/3d-printing/sample.txt' --csearch greedy
```
```bash 
python base.py --input-file '../data/3d-printing/sample.txt' --csearch greedy
```

- For the local search with debug mode (first-improvement):
```bash
python base.py --log-level debug --input-file '../data/3d-printing/sample.txt' --lsearch fi
```

  - first-improvement local search:
```bash
python base.py --input-file '../data/3d-printing/sample.txt' --lsearch fi
```

  - best-improvement local search:
```bash
python base.py --input-file '../data/3d-printing/sample.txt' --lsearch bi
```

- For the evaluator:
```bash 
python base.py --log-level debug --input-file '../data/3d-printing/sample.txt' --lsearch bi --output-file '../data/3d-printing/my_sample_output.txt'
```
```bash 
python evaluators/3d_printing.py '../data/3d-printing/sample.txt' '../data/3d-printing/my_sample_output.txt'
```

```bash 
python base.py --input-file '../data/3d-printing/wt40-1.txt' --lsearch bi --output-file '../data/3d-printing/my_wt40-1_output.txt'
```
```bash 
python evaluators/3d_printing.py '../data/3d-printing/wt40-1.txt' '../data/3d-printing/my_wt40-1_output.txt'
```

```bash 
python base.py --input-file '../data/3d-printing/wt50-1.txt' --lsearch bi --output-file '../data/3d-printing/my_wt50-1_output.txt' && python evaluators/3d_printing.py '../data/3d-printing/wt50-1.txt' '../data/3d-printing/my_wt50-1_output.txt'
```

```bash 
python base.py --input-file '../data/3d-printing/wt100-1.txt' --lsearch bi --output-file '../data/3d-printing/my_wt100-1_output.txt' && python evaluators/3d_printing.py '../data/3d-printing/wt100-1.txt' '../data/3d-printing/my_wt100-1_output.txt'
```

## PROJECT INFORMATION

This repository contains all relevant files for the SIGEVO Summer
School 2024 modelling projects. It is organized as follows:

- In the `documents` folder you can find the slides of the "Constructive
  search" and "Local search" presentations given during the summer school,
  and the project statement, which includes the description of the
  problems and the modelling API documentation.
- In the `src` folder you can find all the code, including the API
  code, a simple TSP example model, and several evaluators to validate
  solutions obtained for the problems.
- In the `data` folder you can find some problem instances that you
  can use to test the models.

## Project Mentors

- Carlos M. Fonseca, University of Coimbra, Portugal
- Diederick Vermetten, Leiden Institute of Advanced Computer Science, Netherlands

## Acknowledgements

This work is partially funded by the FCT - Foundation for Science and
Technology, I.P./MCTES through national funds (PIDDAC), within the
scope of CISUC R&D Unit -- UIDB/00326/2020 or project code
UIDP/00326/2020.

